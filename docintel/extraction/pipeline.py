"""Extract the seven target fields from one filing using cited BM25 retrieval.

This is a first executable baseline, not an evaluated accuracy claim. Model
answers are untrusted until their schema and verbatim chunk spans validate.
Ground-truth data is deliberately not consulted by this module.
"""

import json
import re
import sys
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, TypeAdapter

from docintel.extraction.schema import (
    Citation,
    CompanyAccounts,
    DocumentExtraction,
    Extracted,
    Failed,
    Found,
    Money,
    NotStated,
    Scale,
)
from docintel.gold import GoldField
from docintel.llm.base import LLMError, LLMProvider
from docintel.retrieval.chunker import Chunk, chunk_document, verify_chunks
from docintel.retrieval.retriever import BM25Retriever, Hit, Retriever

DEFAULT_TOP_K = 4


class CitationError(ValueError):
    """The model's claimed evidence cannot be located unambiguously."""


# These queries are independent of the gold labelling ANCHORS table. They are
# an un-tuned lexical baseline, not claims about the best retrieval queries.
FIELD_QUERIES: dict[GoldField, str] = {
    GoldField.COMPANY_NAME: "registrant company legal name annual report",
    GoldField.PERIOD_START: "fiscal year period beginning started",
    GoldField.PERIOD_END: "for the fiscal year ended annual report",
    GoldField.REVENUE: "consolidated statements of operations revenue net sales",
    GoldField.NET_INCOME: "consolidated statements of operations net income",
    GoldField.EMPLOYEE_COUNT: "number of employees full time worldwide",
    GoldField.AUDITOR_NAME: "report of independent registered public accounting firm auditor",
}

FIELD_VALUE_FORMAT: dict[GoldField, str] = {
    GoldField.COMPANY_NAME: 'a string, e.g. "Example Corporation"',
    GoldField.PERIOD_START: "a date string in YYYY-MM-DD format",
    GoldField.PERIOD_END: "a date string in YYYY-MM-DD format",
    GoldField.REVENUE: 'an object {"amount":"391035000000","currency":"USD","scale":"millions"}',
    GoldField.NET_INCOME: 'an object {"amount":"93736000000","currency":"USD","scale":"millions"}',
    GoldField.EMPLOYEE_COUNT: "a non-negative integer",
    GoldField.AUDITOR_NAME: 'a string, e.g. "Example Audit LLP"',
}


# The report heading occurs in the contents and the auditor's actual opinion.
# A generic SEC 10-K auditor's report usually ends with a signed accounting-firm
# name. The regex looks for that document structure, never for any gold label.
_SIGNED_ACCOUNTING_FIRM = re.compile(
    r"(?im)^\s*/s/\s*[^\r\n]{2,100}(?:\bLLP\b|\bLLC\b|\bCPAs?\b|P\.C\.)"
)
_AUDITOR_BACKUP_QUERY = "independent auditors accounting firm LLP LLC CPA opinion"


def retrieve_field_hits(field: GoldField, *, retriever: Retriever, top_k: int) -> list[Hit]:
    """Select evidence without consulting gold labels or changing the retriever.

    Auditor reports have many repeated section headings, while the actual
    signer can rank far lower. Rank signed-accounting-firm evidence ahead of
    the generic headings, but leave other fields' baseline retrieval alone.
    This is a structural heuristic to validate on additional unseen filings.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = FIELD_QUERIES[field]
    if field is not GoldField.AUDITOR_NAME:
        return retriever.search(query, limit=top_k)

    # Asking for all BM25-positive candidates does NOT send them to the LLM.
    # At most top_k chunks reach the prompt. This avoids an arbitrary rank
    # threshold fitted to one company's filing.
    candidates = retriever.search(query, limit=sys.maxsize)
    primary = candidates[:top_k]
    signatures = [hit for hit in candidates if _SIGNED_ACCOUNTING_FIRM.search(hit.chunk.text)]
    if not signatures:
        more = retriever.search(_AUDITOR_BACKUP_QUERY, limit=sys.maxsize)
        signatures = [hit for hit in more if _SIGNED_ACCOUNTING_FIRM.search(hit.chunk.text)]
    if not signatures:
        return primary
    signature = signatures[0]
    if any(hit.chunk.chunk_id == signature.chunk.chunk_id for hit in primary):
        return primary
    return [signature, *primary[: top_k - 1]]


_FIELD_INSTRUCTIONS = {
    GoldField.EMPLOYEE_COUNT: (
        "Extract the total headcount for the filing's reporting year. "
        "Do not cite division headcounts or the previous year's comparison. "
        "The quote must include the exact reported total that supports your value."
    ),
    GoldField.PERIOD_START: (
        "The field is the *beginning* of the annual reporting interval. "
        "A balance-sheet 'as of' date, fiscal-year-end date, or the first day "
        "you calculated by subtracting a year is NOT a stated start date. "
        "Only return found when the excerpt explicitly identifies the start "
        "of that reporting interval. Otherwise return not_stated."
    ),
    GoldField.AUDITOR_NAME: (
        "Identify the accounting firm signing the actual independent auditor's "
        "opinion, not a firm guessed from memory or a table-of-contents heading. "
        "If the excerpt includes a '/s/' signature, quote that exact line. "
        "If the signer is not established by the excerpts, return not_stated."
    ),
}

# A date printed in a filing is not automatically the start of its reporting
# period. A found result for period_start must cite an explicit start statement.
_START_DATE_CONTEXT = re.compile(
    r"(?i)\b(?:began|begins|beginning|started|starts|starting|commenced|commencing|from)\b"
)


def build_prompt(field: GoldField, chunks: list[Chunk]) -> str:
    """Explain the evidence contract to the model with stable chunk coordinates."""
    snippets = "\n\n".join(
        f"=== chunk {chunk.chunk_id} ===\n{chunk.text}\n=== end chunk ===" for chunk in chunks
    )
    return (
        "Extract one factual field from the SEC filing excerpts below.\n"
        f"Field: {field.value}\n"
        f"Value type: {FIELD_VALUE_FORMAT[field]}\n"
        "Return ONLY one JSON object, no Markdown. Choose exactly one response:\n"
        '{"status":"not_stated"}\n'
        'or {"status":"found","value":VALUE,"citation":'
        '{"chunk_id":"EXACT_CHUNK_ID","quote":"VERBATIM_QUOTE"}}\n'
        "The quote must be an exact contiguous substring that appears ONCE in its chunk. "
        "The program, not you, will calculate its character offsets. "
        "Cite the actual figure/name/date rather than an unrelated mention. "
        "For money, use the heading's currency and magnitude, converting amount "
        "to whole currency units; never assume a scale when it is missing. "
        "If the supplied excerpts do not establish the answer, return not_stated. "
        "Do not guess, use other knowledge, or invent a quote or span.\n"
        f"{_FIELD_INSTRUCTIONS.get(field, '')}\n\n"
        f"{snippets}"
    )


def _typed_result(field: GoldField, decoded: object) -> BaseModel:
    """Validate the model output with the schema for the requested field."""
    if field in {GoldField.COMPANY_NAME, GoldField.AUDITOR_NAME}:
        return TypeAdapter(Extracted[str]).validate_python(decoded)
    if field in {GoldField.PERIOD_START, GoldField.PERIOD_END}:
        return TypeAdapter(Extracted[date]).validate_python(decoded)
    if field in {GoldField.REVENUE, GoldField.NET_INCOME}:
        return TypeAdapter(Extracted[Money]).validate_python(decoded)
    return TypeAdapter(Extracted[int]).validate_python(decoded)


# Post-validation ensures that the cited characters contain the claimed value,
# rather than merely existing somewhere in the retrieved context. This is a
# conservative syntactic check, not a substitute for accuracy evaluation.
_NUMBERS = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
_MONTH_DATE = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
)
_SCALE_FACTORS = {
    Scale.UNITS: Decimal("1"),
    Scale.THOUSANDS: Decimal("1000"),
    Scale.MILLIONS: Decimal("1000000"),
}


def _claim_appears_in_quote(field: GoldField, value: object, quote: str) -> bool:
    if field in {GoldField.COMPANY_NAME, GoldField.AUDITOR_NAME}:
        if not isinstance(value, str):
            return False
        return " ".join(value.lower().split()) in " ".join(quote.lower().split())
    if field in {GoldField.PERIOD_START, GoldField.PERIOD_END}:
        if not isinstance(value, date):
            return False
        for match in re.finditer(
            r"(?:[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}"
            r"|\d{4}[-/]\d{2}[-/]\d{2})",
            quote,
        ):
            for fmt in _MONTH_DATE:
                try:
                    if datetime.strptime(match.group(), fmt).date() == value:
                        return True
                except ValueError:
                    continue
        return False
    numbers: list[Decimal] = []
    for raw in _NUMBERS.findall(quote):
        negative = raw.startswith("(") and raw.endswith(")")
        clean = raw.strip("()").replace(",", "")
        try:
            number = Decimal(clean)
        except ArithmeticError:
            continue
        numbers.append(-number if negative else number)
    if field is GoldField.EMPLOYEE_COUNT:
        return isinstance(value, int) and Decimal(value) in numbers
    if field in {GoldField.REVENUE, GoldField.NET_INCOME}:
        return isinstance(value, Money) and value.amount / _SCALE_FACTORS[value.scale] in numbers
    return False


def _complete_citation(decoded: object, selected: dict[str, Chunk]) -> object:
    """Resolve a unique verbatim quote to offsets instead of trusting model maths."""
    if not isinstance(decoded, dict) or decoded.get("status") != "found":
        return decoded
    raw_citation = decoded.get("citation")
    if not isinstance(raw_citation, dict):
        raise CitationError("found result requires a citation object")
    chunk_id = raw_citation.get("chunk_id")
    quote = raw_citation.get("quote")
    if not isinstance(chunk_id, str) or not isinstance(quote, str) or not quote.strip():
        raise CitationError("citation requires a non-empty chunk_id and quote")
    if set(raw_citation) != {"chunk_id", "quote"}:
        raise CitationError("citation must supply only chunk_id and quote, not offsets")
    chunk = selected.get(chunk_id)
    if chunk is None:
        raise CitationError("citation names a chunk outside retrieved context")
    start = chunk.text.find(quote)
    if start < 0:
        raise CitationError("citation quote not found verbatim in chunk")
    if chunk.text.find(quote, start + 1) >= 0:
        raise CitationError("citation quote occurs more than once; add context")
    return {
        **decoded,
        "citation": {
            "chunk_id": chunk_id,
            "quote": quote,
            "start": start,
            "end": start + len(quote),
        },
    }


def extract_field(
    field: GoldField,
    *,
    retriever: Retriever,
    provider: LLMProvider,
    top_k: int = DEFAULT_TOP_K,
) -> BaseModel:
    """Return a typed field result; never collapse model errors into abstention."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hits = retrieve_field_hits(field, retriever=retriever, top_k=top_k)
    if not hits:
        # A retrieval miss cannot prove that the source did not state a fact.
        return Failed(error="retrieval_no_hits")
    selected = {hit.chunk.chunk_id: hit.chunk for hit in hits}
    chunks = [hit.chunk for hit in hits]
    prompt = build_prompt(field, chunks)
    original_value: object | None = None

    # One bounded evidence-only correction if the model states a value but
    # chooses a real quote that does not contain that value. A second model
    # call cannot change the value or turn the error into an abstention.
    for attempt in range(2):
        try:
            response = provider.complete(prompt)
        except LLMError as exc:
            return Failed(error=f"provider_error: {exc}")

        try:
            decoded: object = json.loads(response)
            parsed = _typed_result(field, _complete_citation(decoded, selected))
        except CitationError as exc:
            return Failed(error=f"invalid_citation: {exc}")
        except (ValueError, TypeError) as exc:
            return Failed(error=f"invalid_model_output: {exc}")

        if isinstance(parsed, Failed):
            return Failed(error="invalid_model_output: model must not emit failed status")
        if isinstance(parsed, NotStated):
            if original_value is not None:
                return Failed(
                    error="invalid_citation: evidence correction cannot become abstention"
                )
            return parsed
        if not isinstance(parsed, Found):
            return Failed(error="invalid_model_output: unexpected result type")
        if original_value is not None and parsed.value != original_value:
            return Failed(error="invalid_citation: evidence correction changed the claimed value")

        citation: Citation = parsed.citation
        cited_chunk = selected.get(citation.chunk_id)
        if cited_chunk is None:
            return Failed(error="invalid_citation: chunk was not in retrieved context")
        if citation.end > len(cited_chunk.text):
            return Failed(error="invalid_citation: span exceeds chunk length")
        if cited_chunk.text[citation.start : citation.end] != citation.quote:
            return Failed(error="invalid_citation: quote does not match source text")
        if field is GoldField.PERIOD_START and not _START_DATE_CONTEXT.search(citation.quote):
            return Failed(error="invalid_citation: quote does not establish reporting-period start")
        if not _claim_appears_in_quote(field, parsed.value, citation.quote):
            if attempt == 0:
                original_value = parsed.value
                rendered_value = json.dumps(parsed.model_dump(mode="json")["value"])
                # This is deliberately appended AFTER the retrieved evidence:
                # the local model can attend to the specific failure without
                # fetching new passages or seeing evaluation gold labels.
                prompt += (
                    "\n\nEVIDENCE CORRECTION ONLY. Your previous answer claimed value "
                    f"{rendered_value}, but its citation did not contain that value. "
                    "Keep the exact same value. Find an exact verbatim quote in "
                    "one of the supplied chunks containing that value as printed. "
                    "Return a new found JSON answer with the same value, quote and "
                    "exact chunk_id. Never cite a component count, an earlier year "
                    "or another figure. Do not invent characters in the quote. "
                    'If you cannot support the value, reply {"status":"not_stated"}. '
                    "The program will record that correction as a failure, not "
                    "as a correct abstention.\n"
                )
                continue
            return Failed(error="invalid_citation: claimed value is absent from quote")
        return parsed

    # All normal branches above return. Keep this explicit for type checking.
    return Failed(error="invalid_citation: citation correction could not be completed")


def extract_document(
    text: str,
    *,
    document_id: str,
    provider: LLMProvider,
    top_k: int = DEFAULT_TOP_K,
) -> DocumentExtraction:
    """Chunk once, retrieve per field, return exactly seven typed statuses."""
    chunks = chunk_document(text, document_id=document_id)
    problems = verify_chunks(text, chunks)
    if problems:
        raise ValueError(f"chunk integrity failure: {problems}")
    retriever = BM25Retriever(chunks)
    field_results = {
        field.value: extract_field(field, retriever=retriever, provider=provider, top_k=top_k)
        for field in GoldField
    }
    return DocumentExtraction(
        document_id=document_id,
        accounts=CompanyAccounts.model_validate(field_results),
    )

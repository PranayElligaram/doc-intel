"""The pipeline must preserve missing/failed/found and reject invented spans."""

import json

from docintel.extraction.pipeline import (
    build_prompt,
    extract_document,
    extract_field,
    retrieve_field_hits,
)
from docintel.extraction.schema import Failed, Found, NotStated
from docintel.gold import GoldField
from docintel.llm.base import LLMProvider
from docintel.llm.fake import FakeLLM
from docintel.retrieval.chunker import Chunk, chunk_document
from docintel.retrieval.retriever import BM25Retriever, Hit

FILING = (
    "Consolidated statements of operations\n"
    "Total net sales 391,035\n"
    "Net income 93,736\n"
    "Employees worldwide: 2,400\n"
    "For the fiscal year ended December 31, 2024\n"
    "Report of Independent Registered Public Accounting Firm\n"
)


def field_test(provider: LLMProvider) -> object:
    chunks = chunk_document(FILING, document_id="cik/accession")
    return extract_field(
        GoldField.EMPLOYEE_COUNT,
        retriever=BM25Retriever(chunks),
        provider=provider,
    )


def test_verified_citation_is_returned() -> None:
    quote = "2,400"
    response = {
        "status": "found",
        "value": 2400,
        "citation": {
            "chunk_id": "cik/accession@0",
            "quote": quote,
        },
    }
    result = field_test(FakeLLM(default=json.dumps(response)))
    assert isinstance(result, Found)
    assert result.value == 2400
    assert result.citation.quote == quote


def test_rejects_invented_quote() -> None:
    response = {
        "status": "found",
        "value": 2400,
        "citation": {"chunk_id": "cik/accession@0", "quote": "2,999"},
    }
    result = field_test(FakeLLM(default=json.dumps(response)))
    assert isinstance(result, Failed)
    assert "not found verbatim" in result.error


def test_rejects_hallucinated_chunk() -> None:
    response = {
        "status": "found",
        "value": 2400,
        "citation": {"chunk_id": "other@0", "quote": "2,400"},
    }
    result = field_test(FakeLLM(default=json.dumps(response)))
    assert isinstance(result, Failed)
    assert "outside retrieved context" in result.error


def test_model_failure_is_not_abstention() -> None:
    result = field_test(FakeLLM(fail_with="connection refused"))
    assert isinstance(result, Failed)
    assert "provider_error" in result.error


def test_malformed_json_is_not_abstention() -> None:
    result = field_test(FakeLLM(default="{bad json"))
    assert isinstance(result, Failed)
    assert "invalid_model_output" in result.error


def test_explicit_abstention_is_preserved() -> None:
    result = field_test(FakeLLM(default='{"status":"not_stated"}'))
    assert isinstance(result, NotStated)


def test_retrieval_miss_is_not_scored_as_abstention() -> None:
    provider = FakeLLM(default='{"status":"not_stated"}')
    chunks = chunk_document("xyzzy", document_id="d")
    result = extract_field(
        GoldField.REVENUE,
        retriever=BM25Retriever(chunks),
        provider=provider,
    )
    assert isinstance(result, Failed)
    assert result.error == "retrieval_no_hits"
    assert provider.calls == []


def test_seven_fields_return_typed_results() -> None:
    result = extract_document(
        FILING,
        document_id="cik/accession",
        provider=FakeLLM(default='{"status":"not_stated"}'),
    )
    assert result.document_id == "cik/accession"
    assert isinstance(result.accounts.employee_count, NotStated)
    assert isinstance(result.accounts.revenue, NotStated)
    assert isinstance(result.accounts.period_end, NotStated)


def test_rejects_value_not_in_cited_quote() -> None:
    response = {
        "status": "found",
        "value": 9999,
        "citation": {"chunk_id": "cik/accession@0", "quote": "2,400"},
    }
    result = field_test(FakeLLM(default=json.dumps(response)))
    assert isinstance(result, Failed)
    assert "claimed value is absent" in result.error


def test_date_value_must_occur_in_quote() -> None:
    response = {
        "status": "found",
        "value": "2024-12-31",
        "citation": {
            "chunk_id": "cik/accession@0",
            "quote": "For the fiscal year ended December 31, 2024",
        },
    }
    chunks = chunk_document(FILING, document_id="cik/accession")
    result = extract_field(
        GoldField.PERIOD_END,
        retriever=BM25Retriever(chunks),
        provider=FakeLLM(default=json.dumps(response)),
    )
    assert isinstance(result, Found)
    assert result.value.isoformat() == "2024-12-31"


def test_money_value_is_checked_against_printed_figure() -> None:
    response = {
        "status": "found",
        "value": {"amount": "391035000000", "scale": "millions", "currency": "USD"},
        "citation": {"chunk_id": "cik/accession@0", "quote": "Total net sales 391,035"},
    }
    chunks = chunk_document(FILING, document_id="cik/accession")
    result = extract_field(
        GoldField.REVENUE,
        retriever=BM25Retriever(chunks),
        provider=FakeLLM(default=json.dumps(response)),
    )
    assert isinstance(result, Found)
    assert result.value.currency == "USD"
    assert result.value.amount == 391035000000


def test_repeated_quote_is_rejected_as_ambiguous() -> None:
    text = "Worldwide employees 2,400, and employees 2,400."
    chunks = chunk_document(text, document_id="d")
    response = {
        "status": "found",
        "value": 2400,
        "citation": {"chunk_id": "d@0", "quote": "2,400"},
    }
    result = extract_field(
        GoldField.EMPLOYEE_COUNT,
        retriever=BM25Retriever(chunks),
        provider=FakeLLM(default=json.dumps(response)),
    )
    assert isinstance(result, Failed)
    assert "more than once" in result.error


def test_auditor_retrieval_promotes_signed_opinion_below_top_four() -> None:
    """Synthetic firm not present in the project's four gold labels."""
    chunks = [
        Chunk(
            chunk_id=f"d@{i * 100}",
            start=i * 100,
            text=f"Report of Independent Registered Public Accounting Firm index {i}",
        )
        for i in range(4)
    ]
    signed = Chunk(
        chunk_id="d@500", start=500, text="Our audit opinion is signed:\n/s/ KPMG LLP\nPCAOB ID 185"
    )

    class StubRetriever:
        def search(self, query: str, *, limit: int) -> list[Hit]:
            ranked = [
                *(Hit(chunk, float(100 - i)) for i, chunk in enumerate(chunks)),
                Hit(signed, 10.0),
            ]
            return ranked[:limit]

    hits = retrieve_field_hits(GoldField.AUDITOR_NAME, retriever=StubRetriever(), top_k=4)
    assert len(hits) == 4
    assert hits[0].chunk.chunk_id == "d@500"
    assert [h.chunk.chunk_id for h in hits[1:]] == ["d@0", "d@100", "d@200"]


def test_auditor_retrieval_falls_back_if_no_signed_opinion() -> None:
    chunks = chunk_document(
        "Report of Independent Registered Public Accounting Firm", document_id="d"
    )
    hits = retrieve_field_hits(GoldField.AUDITOR_NAME, retriever=BM25Retriever(chunks), top_k=4)
    assert [h.chunk.chunk_id for h in hits] == ["d@0"]


def test_unrelated_fields_keep_standard_retrieval() -> None:
    chunks = chunk_document(FILING, document_id="d")
    retriever = BM25Retriever(chunks)
    expected = retriever.search("number of employees full time worldwide", limit=4)
    actual = retrieve_field_hits(GoldField.EMPLOYEE_COUNT, retriever=retriever, top_k=4)
    assert actual == expected


def test_period_start_prompt_does_not_confuse_as_of_with_beginning() -> None:
    prompt = build_prompt(GoldField.PERIOD_START, [])
    assert "balance-sheet 'as of' date" in prompt
    assert "return not_stated" in prompt


def test_period_start_rejects_valid_date_from_balance_sheet() -> None:
    text = "Balance sheet as of December 31, 2024. Fiscal year period beginning started."
    chunks = chunk_document(text, document_id="d")
    response = json.dumps(
        {
            "status": "found",
            "value": "2024-12-31",
            "citation": {"chunk_id": "d@0", "quote": "as of December 31, 2024"},
        }
    )
    result = extract_field(
        GoldField.PERIOD_START,
        retriever=BM25Retriever(chunks),
        provider=FakeLLM(default=response),
    )
    assert isinstance(result, Failed)
    assert "does not establish reporting-period start" in result.error


def test_period_start_accepts_explicit_start() -> None:
    text = "The reporting period began on January 1, 2024."
    chunks = chunk_document(text, document_id="d")
    response = json.dumps(
        {"status": "found", "value": "2024-01-01", "citation": {"chunk_id": "d@0", "quote": text}}
    )
    result = extract_field(
        GoldField.PERIOD_START,
        retriever=BM25Retriever(chunks),
        provider=FakeLLM(default=response),
    )
    assert isinstance(result, Found)
    assert result.value.isoformat() == "2024-01-01"


class SequentialProvider:
    """A deterministic first-answer / correction-answer provider for tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "sequential"

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        return self.responses[len(self.calls) - 1]


def _test_evidence_correction(second: dict[str, object]) -> tuple[object, SequentialProvider]:
    doc = "As of 2024, there were 6,000 employees worldwide. In 2023, there were 5,500 employees."
    chunks = chunk_document(doc, document_id="doc")
    old_quote = "In 2023, there were 5,500 employees."
    initial = {
        "status": "found",
        "value": 6000,
        "citation": {"chunk_id": "doc@0", "quote": old_quote},
    }
    provider = SequentialProvider([json.dumps(initial), json.dumps(second)])
    result = extract_field(
        GoldField.EMPLOYEE_COUNT,
        retriever=BM25Retriever(chunks),
        provider=provider,
    )
    return result, provider


def test_evidence_retry_accepts_correct_quote_without_changing_value() -> None:
    result, provider = _test_evidence_correction(
        {
            "status": "found",
            "value": 6000,
            "citation": {
                "chunk_id": "doc@0",
                "quote": "As of 2024, there were 6,000 employees worldwide.",
            },
        }
    )
    assert isinstance(result, Found)
    assert result.value == 6000
    assert len(provider.calls) == 2
    assert "EVIDENCE CORRECTION ONLY" in provider.calls[1]
    assert "6,000 employees" in provider.calls[1]
    assert "5,500 employees" in provider.calls[1]


def test_evidence_retry_rejects_changed_value_even_if_cited() -> None:
    result, provider = _test_evidence_correction(
        {
            "status": "found",
            "value": 5500,
            "citation": {"chunk_id": "doc@0", "quote": "In 2023, there were 5,500 employees."},
        }
    )
    assert isinstance(result, Failed)
    assert "changed the claimed value" in result.error
    assert len(provider.calls) == 2


def test_evidence_retry_cannot_turn_failure_into_abstention() -> None:
    result, provider = _test_evidence_correction({"status": "not_stated"})
    assert isinstance(result, Failed)
    assert "cannot become abstention" in result.error
    assert len(provider.calls) == 2


def test_evidence_retry_rejects_quote_with_no_claimed_value() -> None:
    result, provider = _test_evidence_correction(
        {
            "status": "found",
            "value": 6000,
            "citation": {"chunk_id": "doc@0", "quote": "In 2023, there were 5,500 employees."},
        }
    )
    assert isinstance(result, Failed)
    assert "claimed value is absent from quote" in result.error
    assert len(provider.calls) == 2


def test_valid_initial_evidence_requires_no_retry() -> None:
    provider = SequentialProvider(
        [
            json.dumps(
                {
                    "status": "found",
                    "value": 2400,
                    "citation": {"chunk_id": "cik/accession@0", "quote": "2,400"},
                }
            )
        ]
    )
    result = field_test(provider)
    assert isinstance(result, Found)
    assert len(provider.calls) == 1


def test_employee_prompt_requires_current_total_and_supporting_quote() -> None:
    prompt = build_prompt(GoldField.EMPLOYEE_COUNT, [])
    assert "filing's reporting year" in prompt
    assert "previous year's comparison" in prompt
    assert "quote must include the exact reported total" in prompt

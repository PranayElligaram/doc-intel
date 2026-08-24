"""The hand-made answer key, and how a model's citation is compared against it.

A gold label says: in this document, this field has this value, and here is the
character span of the derived text that says so.

WHY A GOLD LABEL IS NOT A Citation. `Citation` in docintel/extraction/schema.py
carries a `chunk_id`, because a model reads chunks and cites the chunk it read.
Chunks do not exist yet - the chunker is Phase 3 - and more to the point they
are not stable: chunk size is one of the things Phase 3 sets out to ablate. A
gold label carrying a chunk id would be invalidated by every change to the
chunker, so the 252 labels would have to be remade to answer the question the
ablation exists to ask. Instead a label indexes into the document's derived
text, which changes only when `DERIVATION_VERSION` changes, and the model's
chunk-relative citation is resolved into that same coordinate system at
evaluation time. Re-chunk at 500 characters or at 2,000 and the gold set does
not move.

HOW THE COMPARISON RUNS, per (entry_id, field). Phase 3's chunker must be
offset-preserving: every chunk records the offset in the derived text where it
starts, and `chunk.text` must be a verbatim slice from there - no re-wrapping,
no whitespace tidying, no joining across a gap. Given that, `resolve_citation`
maps a chunk-relative citation to a `DocumentSpan`, and the eval then runs four
gates in order:

1. VERSION. The label's `derivation_version` against the manifest row's. They
   disagree only when the corpus moved under the labels; see below.
2. GOLD INTEGRITY. `derived_text[span.start:span.end] == span.quote`. Failing
   this while the versions agree means the text on disk was edited underneath
   the label - a harness fault, not a model result.
3. GROUNDING. The same check against the model's resolved span. The chunk
   slice identity is checked once per chunk first, so a chunker that is not
   offset-preserving is reported as a harness fault rather than charged to the
   model as 252 ungrounded citations.
4. AGREEMENT. `not_stated` against NotStated is a correct abstention. A stated
   label against a Found is compared with `spans_agree` - one span must contain
   the other - and the value is scored separately.

WHEN THE VERSIONS DISAGREE, evaluation must refuse to score and stop, naming
every stale label. Not "skip": that shrinks the denominator and quietly
improves the score. Not "count it wrong": that charges the model for a change
to the corpus. Both produce a number that looks like a measurement and is not.
`add_label` cannot create the situation - it stamps the version from the
manifest row and refuses if that row disagrees with the code's current
DERIVATION_VERSION - so a disagreement means a derivation bump happened after
labelling, and `verify_labels` is what narrows it down to the labels whose
quote actually moved. Equal versions do not prove equal text, which is why both
the write path and the verify path also check the text hash.
"""

import hashlib
import json
import os
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator

from docintel.extraction.schema import Citation, Currency, Money, Scale
from docintel.ingest.html_text import DERIVATION_VERSION
from docintel.ingest.manifest import Disposition, Manifest, ManifestEntry, load_manifest


class LabellingError(Exception):
    """Something about a labelling request is wrong and a human must fix it.

    One class rather than a hierarchy: every one of these ends the same way -
    printed to stderr, nothing written, the labeller reads it and tries again.
    The distinctions live in the message, which is the part a person reads.
    """


class GoldField(StrEnum):
    """The seven fields that get hand-labelled.

    Seven, not the eight CompanyAccounts once had: the CIK appears in none of
    the derived texts, so labelling it would produce 36 not_stated labels
    measuring only whether a model learned to abstain on one named field. See
    CompanyAccounts in docintel/extraction/schema.py.
    """

    COMPANY_NAME = "company_name"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    REVENUE = "revenue"
    NET_INCOME = "net_income"
    EMPLOYEE_COUNT = "employee_count"
    AUDITOR_NAME = "auditor_name"


class DocumentSpan(BaseModel):
    """A span of a document's derived text, with what it says there.

    The gold-side twin of `Citation`, minus the chunk id and plus nothing. It
    deliberately does not share a base class with Citation: they hold the same
    three numbers for opposite reasons - a Citation is a claim a model makes
    about a chunk it was shown, a DocumentSpan is a fact about the document -
    and a shared base would invite a future change to one to be applied to
    both. The invariants below are duplicated on purpose.
    """

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _span_must_match_quote(self) -> "DocumentSpan":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        span_length = self.end - self.start
        if len(self.quote) != span_length:
            raise ValueError(
                f"quote is {len(self.quote)} chars but span [{self.start}, {self.end}) "
                f"is {span_length}"
            )
        if not self.quote.strip():
            raise ValueError("quote is only whitespace, which cannot ground a value")
        return self


class TextValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


class DateValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["date"] = "date"
    date: date


class CountValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["count"] = "count"
    count: int = Field(ge=0)


class MoneyValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["money"] = "money"
    money: Money


# Tagged rather than a bare `str | int | date | Money` union, because Pydantic's
# smart mode resolves a union member by exact type match: the JSON string
# "2024-12-31" would bind to str and never to date, and the label would come
# back off disk as a different type from the one written. The tag makes the
# round trip exact, and it is the same idiom the extraction schema uses for
# `status`.
type GoldValue = Annotated[TextValue | DateValue | CountValue | MoneyValue, Discriminator("kind")]

# Which shape of value each field must carry. A gold set where period_end
# sometimes holds a string is not comparable to anything.
FIELD_VALUE_KIND: dict[GoldField, str] = {
    GoldField.COMPANY_NAME: "text",
    GoldField.PERIOD_START: "date",
    GoldField.PERIOD_END: "date",
    GoldField.REVENUE: "money",
    GoldField.NET_INCOME: "money",
    GoldField.EMPLOYEE_COUNT: "count",
    GoldField.AUDITOR_NAME: "text",
}

SCALE_MULTIPLIER: dict[Scale, int] = {
    Scale.UNITS: 1,
    Scale.THOUSANDS: 1_000,
    Scale.MILLIONS: 1_000_000,
}


class GoldStated(BaseModel):
    """The document states this field, here, and this is what it means."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["stated"] = "stated"
    span: DocumentSpan
    value: GoldValue


class GoldNotStated(BaseModel):
    """The document does not state this field. See Rule 1 in the policy."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_stated"] = "not_stated"


# Two states, where an extraction has three. There is no `failed` gold label: a
# crash is something a model does, never something a document is.
type GoldAnswer = Annotated[GoldStated | GoldNotStated, Discriminator("status")]


class GoldLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(min_length=1)
    field: GoldField
    # The derivation the span was measured against. Stamped from the manifest
    # row at write time; the reason a stale label is diagnosable rather than
    # mysterious.
    derivation_version: int = Field(ge=1)
    answer: GoldAnswer
    labelled_at: datetime

    @model_validator(mode="after")
    def _value_kind_must_match_the_field(self) -> "GoldLabel":
        if isinstance(self.answer, GoldStated):
            expected = FIELD_VALUE_KIND[self.field]
            if self.answer.value.kind != expected:
                raise ValueError(
                    f"{self.field.value} takes a {expected} value, got {self.answer.value.kind}"
                )
        return self


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    labels: list[GoldLabel] = Field(default_factory=list)

    def by_key(self) -> dict[tuple[str, GoldField], GoldLabel]:
        return {(label.entry_id, label.field): label for label in self.labels}


def gold_path(gold_root: Path) -> Path:
    return gold_root / "labels.json"


def load_gold_set(gold_root: Path) -> GoldSet:
    path = gold_path(gold_root)
    if not path.exists():
        return GoldSet()
    return GoldSet.model_validate_json(path.read_text(encoding="utf-8"))


def write_gold_set(gold_set: GoldSet, gold_root: Path) -> Path:
    """Writes the gold set atomically, with a stable byte-for-byte layout.

    Same shape as write_manifest, for the same reason: this file is committed
    and reviewed, so a diff should show the label that changed rather than
    where the JSON encoder felt like putting things. Adding one label to a file
    of 252 should be a one-hunk diff.
    """
    ordered = gold_set.model_copy(
        update={
            "labels": sorted(gold_set.labels, key=lambda label: (label.entry_id, label.field.value))
        }
    )
    payload = json.dumps(ordered.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    path = gold_path(gold_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --- reading the corpus ----------------------------------------------------


def resolve_entry_id(manifest: Manifest, token: str) -> str:
    """Expands a fragment of an entry id to the whole thing, or refuses.

    An entry id is "0000019617/0001628280-26-008131". Typing that 252 times by
    hand is a source of errors that all look like "document not found", so any
    unambiguous fragment will do. Ambiguity is refused rather than resolved to
    the first match, for the same reason an ambiguous quote is: a coin flip
    here writes a label against the wrong document.

    A fragment only ever matches a document that is in the corpus. An exact id
    is looked up across every row, so naming an excluded document tells you why
    it was excluded - but the manifest also holds rows for things that are not
    documents at all, and a fragment that happens to be a CIK would collide
    with that company's row for one of them every time. Nothing in this module
    should be able to arrive at those, not even to print their names.
    """
    if token in {entry.entry_id for entry in manifest.entries}:
        return token

    matches = sorted(
        entry.entry_id
        for entry in manifest.entries
        if entry.disposition is Disposition.IN_CORPUS and token in entry.entry_id
    )
    if not matches:
        raise LabellingError(f"no document matches {token!r}")
    if len(matches) > 1:
        listed = "\n  ".join(matches)
        raise LabellingError(f"{token!r} matches {len(matches)} documents:\n  {listed}")
    return matches[0]


def corpus_entry(manifest: Manifest, entry_id: str) -> ManifestEntry:
    entry = manifest.by_id().get(entry_id)
    if entry is None:
        raise LabellingError(f"no manifest row for {entry_id}")
    if entry.disposition is not Disposition.IN_CORPUS:
        raise LabellingError(
            f"{entry_id} is {entry.disposition.value} ({entry.reason.value}), not in the corpus"
        )
    return entry


def document_text(cache_root: Path, entry: ManifestEntry) -> str:
    """The derived text a label's offsets index into, checked against its hash.

    Read through `read_text`, which translates line endings, because that is
    the string the offsets were measured in: `derive_text` produces LF, the
    manifest hashes the LF form, and on Windows the file lands on disk as CRLF.
    Reading the bytes directly would shift every offset after the first
    newline on one platform and not the other.
    """
    if entry.text_path is None:
        raise LabellingError(f"{entry.entry_id} has no derived text on disk")

    path = cache_root / entry.text_path
    if not path.exists():
        raise LabellingError(f"{path} is missing - run scripts/fetch_corpus.py")

    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != entry.text_sha256:
        raise LabellingError(
            f"{path} does not match the manifest hash. The text has been edited or "
            "re-derived without updating the manifest; run scripts/fetch_corpus.py "
            "--recheck before labelling."
        )
    return text


def find_quote_offsets(text: str, quote: str) -> list[int]:
    """Every offset where `quote` appears, including overlapping occurrences.

    Overlapping matters: "aa" occurs twice in "aaa", and a labeller who cannot
    see that would be told their quote was unique when it is not.
    """
    offsets: list[int] = []
    start = text.find(quote)
    while start != -1:
        offsets.append(start)
        start = text.find(quote, start + 1)
    return offsets


def quote_context(text: str, offset: int, length: int, width: int = 60) -> str:
    """The quote with its surroundings, on one line, for disambiguating."""
    before = text[max(0, offset - width) : offset]
    after = text[offset + length : offset + length + width]
    middle = text[offset : offset + length]
    return f"...{before}[[{middle}]]{after}...".replace("\n", " ")


# --- writing a label -------------------------------------------------------


def parse_value(
    field: GoldField,
    raw: str,
    *,
    scale: Scale | None = None,
    currency: Currency | None = None,
) -> GoldValue:
    """Turns what the labeller typed into the typed value for this field.

    The money branch does the arithmetic on purpose. A filing prints "391,035"
    under a "$ in millions" heading, and asking a person to type 391035000000
    at two in the morning is asking for a wrong answer that looks right.
    """
    kind = FIELD_VALUE_KIND[field]

    if kind == "text":
        if not raw.strip():
            raise LabellingError(f"{field.value} needs a non-empty value")
        return TextValue(text=raw.strip())

    if kind == "date":
        try:
            return DateValue(date=date.fromisoformat(raw))
        except ValueError:
            raise LabellingError(
                f"{field.value} needs an ISO date like 2024-12-31, got {raw!r}"
            ) from None

    if kind == "count":
        try:
            return CountValue(count=int(raw.replace(",", "").strip()))
        except ValueError:
            raise LabellingError(f"{field.value} needs a whole number, got {raw!r}") from None

    if scale is None or currency is None:
        raise LabellingError(f"{field.value} needs --scale and --currency as well as --amount")
    return MoneyValue(
        money=Money(
            amount=parse_amount(raw) * SCALE_MULTIPLIER[scale],
            scale=scale,
            currency=currency,
        )
    )


def parse_amount(raw: str) -> Decimal:
    """Reads a figure as printed in a filing.

    Parentheses are how a filing writes a negative, and the labeller is pasting
    what is on the page: "(15,000)" is -15000. Getting the sign wrong on a loss
    is a silent scoring error, so it is handled here rather than left to be
    typed by hand.

    An em dash, which a filing prints for a nil, is left to fail as not a
    number. Whether a dash is a stated zero or an absence is a labelling
    decision under Rule 1, and it is not one this function should make.
    """
    cleaned = raw.strip().replace(",", "").replace("$", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1].strip()
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        raise LabellingError(f"{raw!r} is not a number") from None
    if not amount.is_finite():
        raise LabellingError(f"{raw!r} is not a finite number")
    return -amount if negative else amount


def add_label(
    *,
    cache_root: Path,
    gold_root: Path,
    entry_id: str,
    field: GoldField,
    quote: str | None,
    value: GoldValue | None,
    replace: bool = False,
    now: datetime | None = None,
) -> GoldLabel:
    """Locates the quote, computes the span and writes the label.

    Refuses a quote that occurs zero times or more than once. Ambiguity is a
    labelling error the person needs to see and fix by quoting more text; a
    tool that picked the first match would write a label pointing at the MD&A
    discussion of a figure rather than the statement it comes from, and nothing
    downstream could tell.
    """
    manifest = corpus_manifest(cache_root)
    entry = corpus_entry(manifest, entry_id)

    if entry.derivation_version != DERIVATION_VERSION:
        raise LabellingError(
            f"{entry_id} was derived at version {entry.derivation_version} but the code is at "
            f"version {DERIVATION_VERSION}. Run scripts/fetch_corpus.py --recheck first; "
            "labelling against text the next recheck will replace is how a gold set goes stale "
            "before it is finished."
        )
    text = document_text(cache_root, entry)

    answer: GoldAnswer
    if quote is None:
        if value is not None:
            raise LabellingError("a not_stated label carries no value")
        answer = GoldNotStated()
    else:
        if value is None:
            raise LabellingError(f"{field.value} needs a value as well as a quote")
        offsets = find_quote_offsets(text, quote)
        if not offsets:
            raise LabellingError(
                f"quote not found in {entry_id}. It must appear verbatim in the derived "
                "text - check for a curly apostrophe, a line break inside the quote, or a "
                "run of spaces that the derivation collapsed to one."
            )
        if len(offsets) > 1:
            found = "\n  ".join(
                f"{offset}: {quote_context(text, offset, len(quote))}" for offset in offsets[:10]
            )
            more = "" if len(offsets) <= 10 else f"\n  ... and {len(offsets) - 10} more"
            raise LabellingError(
                f"quote appears {len(offsets)} times in {entry_id}. Extend it until it is "
                f"unique - an ambiguous quote is a labelling error, not a coin flip:\n  "
                f"{found}{more}"
            )
        answer = GoldStated(
            span=DocumentSpan(start=offsets[0], end=offsets[0] + len(quote), quote=quote),
            value=value,
        )

    gold_set = load_gold_set(gold_root)
    existing = gold_set.by_key().get((entry_id, field))
    if existing is not None and not replace:
        raise LabellingError(
            f"{entry_id} {field.value} is already labelled "
            f"({existing.answer.status}). Pass --replace to overwrite it."
        )

    label = GoldLabel(
        entry_id=entry_id,
        field=field,
        derivation_version=DERIVATION_VERSION,
        answer=answer,
        labelled_at=now or datetime.now(UTC),
    )
    kept = [
        other for other in gold_set.labels if (other.entry_id, other.field) != (entry_id, field)
    ]
    write_gold_set(gold_set.model_copy(update={"labels": [*kept, label]}), gold_root)
    return label


def corpus_manifest(cache_root: Path) -> Manifest:
    """The manifest, refusing an empty one rather than reporting 36 absences."""
    manifest = load_manifest(cache_root)
    if not manifest.entries:
        raise LabellingError(f"no manifest at {cache_root} - run scripts/fetch_corpus.py first")
    return manifest


# --- what is left to do ----------------------------------------------------


def unlabelled(manifest: Manifest, gold_set: GoldSet) -> list[tuple[str, GoldField]]:
    labelled = gold_set.by_key()
    return [
        (entry.entry_id, field)
        for entry in sorted(manifest.entries, key=lambda entry: entry.entry_id)
        if entry.disposition is Disposition.IN_CORPUS
        for field in GoldField
        if (entry.entry_id, field) not in labelled
    ]


def progress_report(manifest: Manifest, gold_set: GoldSet) -> str:
    """A grid of documents by field, so the next thing to label is on screen.

    252 labels is several sittings, and tracking which of them are done in your
    head across those sittings is the part that goes wrong.
    """
    corpus = [
        entry
        for entry in sorted(manifest.entries, key=lambda entry: entry.entry_id)
        if entry.disposition is Disposition.IN_CORPUS
    ]
    labelled = gold_set.by_key()
    fields = list(GoldField)

    # Numbered columns rather than truncated names: period_start and period_end
    # both abbreviate to "peri", and a grid whose two date columns cannot be
    # told apart is worse than no grid.
    lines = [" ".join(f"{number} {field.value}" for number, field in enumerate(fields, start=1))]
    lines.append("")
    lines.append(
        f"{'document':<40}  " + " ".join(str(number) for number in range(1, len(fields) + 1))
    )
    for entry in corpus:
        marks = []
        for field in fields:
            label = labelled.get((entry.entry_id, field))
            if label is None:
                marks.append(".")
            elif label.answer.status == "not_stated":
                marks.append("-")
            else:
                marks.append("#")
        lines.append(f"{entry.entry_id:<40}  {' '.join(marks)}")

    done = sum(1 for entry in corpus for field in fields if (entry.entry_id, field) in labelled)
    total = len(corpus) * len(fields)
    lines.append("")
    lines.append("# stated   - not_stated   . unlabelled")
    lines.append(f"{done} of {total} labelled, {total - done} to go")
    return "\n".join(lines)


def verify_labels(cache_root: Path, gold_set: GoldSet) -> list[str]:
    """Re-checks every label against the text on disk. Empty means healthy.

    This is step 3 of the migration in docs/labelling-policy.md: after a
    DERIVATION_VERSION bump, it separates the labels whose quote still sits
    where it did from the ones that have to be made again.
    """
    manifest = corpus_manifest(cache_root)
    problems: list[str] = []
    texts: dict[str, str] = {}

    for label in sorted(gold_set.labels, key=lambda label: (label.entry_id, label.field.value)):
        where = f"{label.entry_id} {label.field.value}"
        try:
            entry = corpus_entry(manifest, label.entry_id)
        except LabellingError as error:
            problems.append(f"{where}: {error}")
            continue

        if label.derivation_version != entry.derivation_version:
            problems.append(
                f"{where}: labelled against derivation {label.derivation_version}, "
                f"manifest says {entry.derivation_version} - relabel or re-derive"
            )
            continue

        if label.entry_id not in texts:
            try:
                texts[label.entry_id] = document_text(cache_root, entry)
            except LabellingError as error:
                problems.append(f"{where}: {error}")
                continue
        text = texts[label.entry_id]

        answer = label.answer
        if isinstance(answer, GoldStated):
            found = text[answer.span.start : answer.span.end]
            if found != answer.span.quote:
                problems.append(
                    f"{where}: span [{answer.span.start}, {answer.span.end}) now reads "
                    f"{found!r}, not {answer.span.quote!r}"
                )

    return problems


# --- resolving a model's citation ------------------------------------------


def resolve_citation(citation: Citation, *, chunk_start: int) -> DocumentSpan:
    """Maps a chunk-relative citation into document coordinates.

    `chunk_start` is where the chunk begins in the derived text. This is the
    whole of the contract Phase 3 has to honour: chunks must be verbatim slices
    of the derived text and must record where they were cut from. A chunker
    that normalises whitespace as it goes breaks this silently, so the eval
    checks the slice identity per chunk before trusting any offset from it.
    """
    if chunk_start < 0:
        raise ValueError(f"chunk_start must not be negative, got {chunk_start}")
    return DocumentSpan(
        start=chunk_start + citation.start,
        end=chunk_start + citation.end,
        quote=citation.quote,
    )


def spans_agree(gold: DocumentSpan, resolved: DocumentSpan) -> bool:
    """True when one span contains the other.

    Containment, not equality. A model quoting "Total revenue 391,035" and a
    gold label quoting "391,035" have pointed at the same fact, and scoring
    them as a miss would measure quoting style rather than grounding. Bare
    overlap would be too weak in the other direction - two adjacent table cells
    overlap at a boundary and are different facts.
    """
    return (gold.start <= resolved.start and resolved.end <= gold.end) or (
        resolved.start <= gold.start and gold.end <= resolved.end
    )


# --- anchors ---------------------------------------------------------------

# Where each field is found in a 10-K. The appendix in docs/labelling-policy.md
# is the prose version of this table; `scripts/label.py anchors` is what puts
# numbers behind the coverage claims it makes.
ANCHORS: dict[GoldField, tuple[str, ...]] = {
    GoldField.COMPANY_NAME: ("Exact name of registrant", "Exact name of Registrant"),
    GoldField.PERIOD_START: (),
    GoldField.PERIOD_END: (
        "For the fiscal year ended",
        "For the year ended",
        "for the year ended",
    ),
    GoldField.REVENUE: (
        "Total net revenue",
        "Total revenue",
        "Total net sales",
        "Net sales",
        "Total sales and revenues",
    ),
    # A loss-making filer has no line called "Net income", and a filer that
    # says "earnings" never will. The anchor list has to carry the vocabulary
    # of the whole corpus, not of the profitable half of it.
    GoldField.NET_INCOME: ("Net income", "Net earnings", "Net loss"),
    GoldField.EMPLOYEE_COUNT: ("Human Capital", "employees"),
    GoldField.AUDITOR_NAME: ("auditor since",),
}


def anchor_report(cache_root: Path) -> str:
    """How many documents each search anchor actually appears in.

    Deliberately just counting. The appendix states coverage as fact - 36/36
    for the auditor sentence, 30/36 for the capitalised cover-page one - and
    the honesty rule says a number in a document needs a command behind it.
    That is this command's whole job; it is not the beginning of an analysis
    tool.
    """
    manifest = corpus_manifest(cache_root)
    corpus = [entry for entry in manifest.entries if entry.disposition is Disposition.IN_CORPUS]
    texts = [document_text(cache_root, entry) for entry in corpus]

    lines = [f"documents in corpus: {len(corpus)}", ""]
    lines.append(f"{'field':<16} {'anchor':<28} {'docs':>7}  occurrences")
    for field, anchors in ANCHORS.items():
        if not anchors:
            never = "(no anchor - never stated)"
            lines.append(f"{field.value:<16} {never:<28} {f'0/{len(corpus)}':>7}")
            continue
        for anchor in anchors:
            hits = [text.count(anchor) for text in texts]
            docs = sum(1 for count in hits if count)
            lines.append(
                f"{field.value:<16} {anchor:<28} {f'{docs}/{len(corpus)}':>7}  {sum(hits)}"
            )
    return "\n".join(lines)

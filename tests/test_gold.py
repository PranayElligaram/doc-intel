"""Tests for the gold label type, the locator, and the citation resolution.

The corpus here is synthetic and tiny, built row by row in tmp_path. That is
deliberate: every rule this module enforces - a quote must be unique, a span
must still say what it said, a label must match the derivation it was made
against - has to hold on any text, and a real 438,862-character filing would
make each test a search problem rather than a statement about the rule.

The one thing the synthetic corpus does reproduce faithfully is the line-ending
trap. Text is written through `write_text`, which lands as CRLF on Windows, and
read back through `read_text`, which gives LF - so the offsets and the hash are
both measured in the LF form, exactly as `derive_text` produced it.
"""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from docintel.extraction.schema import Citation, Currency, Money, Scale
from docintel.gold import (
    CountValue,
    DateValue,
    DocumentSpan,
    GoldField,
    GoldLabel,
    GoldNotStated,
    GoldSet,
    GoldStated,
    LabellingError,
    MoneyValue,
    TextValue,
    add_label,
    anchor_report,
    corpus_manifest,
    document_text,
    find_quote_offsets,
    gold_path,
    load_gold_set,
    parse_value,
    progress_report,
    resolve_citation,
    resolve_entry_id,
    spans_agree,
    unlabelled,
    verify_labels,
    write_gold_set,
)
from docintel.ingest.html_text import DERIVATION_VERSION
from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    load_manifest,
    manifest_path,
    text_dir,
    write_manifest,
)

APPLE = "0000320193/0000320193-25-000079"
JPM = "0000019617/0000019617-25-000270"

APPLE_TEXT = (
    "Apple Inc.\n\n(Exact name of registrant as specified in its charter)\n\n"
    "For the fiscal year ended September 27, 2025\n\n"
    "Total net sales 416,161 391,035 383,285\n\n"
    "We have served as the Company's auditor since 2009.\n"
)
JPM_TEXT = "For the fiscal year ended Commission file\n\nDecember 31, 2024 number 1-5805\n"


def build_corpus(tmp_path: Path, documents: dict[str, str] | None = None) -> Path:
    """A cache root holding derived text and a manifest that describes it."""
    documents = documents or {APPLE: APPLE_TEXT, JPM: JPM_TEXT}
    cache_root = tmp_path / "cache"
    entries = []
    for entry_id, text in documents.items():
        company_id, document_id = entry_id.split("/")
        path = text_dir(cache_root) / company_id / f"{document_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        entries.append(
            ManifestEntry(
                entry_id=entry_id,
                company_id=company_id,
                document_id=document_id,
                disposition=Disposition.IN_CORPUS,
                last_action=LastAction.FETCHED,
                reason=Reason.OK,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                text_chars=len(text),
                text_path=f"text/{company_id}/{document_id}.txt",
                derivation_version=DERIVATION_VERSION,
            )
        )
    write_manifest(Manifest(entries=entries), cache_root)
    return cache_root


def set_manifest_field(cache_root: Path, entry_id: str, **changes: object) -> None:
    """Rewrites one manifest row, to simulate a corpus that moved."""
    manifest = load_manifest(cache_root)
    entries = [
        entry.model_copy(update=changes) if entry.entry_id == entry_id else entry
        for entry in manifest.entries
    ]
    write_manifest(manifest.model_copy(update={"entries": entries}), cache_root)


def label_apple_auditor(cache_root: Path, gold_root: Path) -> GoldLabel:
    return add_label(
        cache_root=cache_root,
        gold_root=gold_root,
        entry_id=APPLE,
        field=GoldField.AUDITOR_NAME,
        quote="We have served as the Company's auditor since 2009.",
        value=TextValue(text="Ernst & Young LLP"),
    )


# --- locating a quote ------------------------------------------------------


def test_a_unique_quote_becomes_a_span_that_still_reads_the_same(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)

    label = label_apple_auditor(cache_root, tmp_path / "gold")

    assert isinstance(label.answer, GoldStated)
    span = label.answer.span
    text = document_text(cache_root, corpus_manifest(cache_root).by_id()[APPLE])
    assert text[span.start : span.end] == span.quote


def test_a_quote_that_appears_twice_is_refused_and_says_so(tmp_path: Path) -> None:
    # The ambiguity is the labelling error: on the real corpus "Net income"
    # occurs five times in Apple's 10-K, in five different statements.
    cache_root = build_corpus(tmp_path, {APPLE: "Net income 112,010\n\nNet income 112,010\n"})

    with pytest.raises(LabellingError, match="appears 2 times"):
        add_label(
            cache_root=cache_root,
            gold_root=tmp_path / "gold",
            entry_id=APPLE,
            field=GoldField.NET_INCOME,
            quote="Net income 112,010",
            value=MoneyValue(
                money=Money(
                    amount=Decimal("112010000000"), scale=Scale.MILLIONS, currency=Currency.USD
                )
            ),
        )


def test_a_quote_that_appears_nowhere_is_refused(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)

    with pytest.raises(LabellingError, match="not found"):
        add_label(
            cache_root=cache_root,
            gold_root=tmp_path / "gold",
            entry_id=APPLE,
            field=GoldField.COMPANY_NAME,
            quote="Alphabet Inc.",
            value=TextValue(text="Alphabet Inc."),
        )


def test_nothing_is_written_when_a_quote_is_refused(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"

    with pytest.raises(LabellingError):
        add_label(
            cache_root=cache_root,
            gold_root=gold_root,
            entry_id=APPLE,
            field=GoldField.COMPANY_NAME,
            quote="Alphabet Inc.",
            value=TextValue(text="Alphabet Inc."),
        )

    assert not gold_path(gold_root).exists()


def test_overlapping_occurrences_count_as_occurrences() -> None:
    # A labeller told "your quote is unique" when it is not would produce a
    # span that is only accidentally the right one.
    assert find_quote_offsets("aaa", "aa") == [0, 1]


def test_the_ambiguity_message_shows_where_the_matches_are(tmp_path: Path) -> None:
    cache_root = build_corpus(
        tmp_path, {APPLE: "in MD&A: Net income\n\nin the statement: Net income\n"}
    )

    with pytest.raises(LabellingError, match=r"MD&A"):
        add_label(
            cache_root=cache_root,
            gold_root=tmp_path / "gold",
            entry_id=APPLE,
            field=GoldField.NET_INCOME,
            quote="Net income",
            value=MoneyValue(
                money=Money(amount=Decimal("1"), scale=Scale.UNITS, currency=Currency.USD)
            ),
        )


# --- abstention ------------------------------------------------------------


def test_not_stated_needs_no_quote(tmp_path: Path) -> None:
    """period_start, on every filing in the corpus. See Rule 1 in the policy."""
    cache_root = build_corpus(tmp_path)

    label = add_label(
        cache_root=cache_root,
        gold_root=tmp_path / "gold",
        entry_id=APPLE,
        field=GoldField.PERIOD_START,
        quote=None,
        value=None,
    )

    assert isinstance(label.answer, GoldNotStated)


def test_not_stated_cannot_carry_a_value(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)

    with pytest.raises(LabellingError, match="carries no value"):
        add_label(
            cache_root=cache_root,
            gold_root=tmp_path / "gold",
            entry_id=APPLE,
            field=GoldField.PERIOD_START,
            quote=None,
            value=DateValue(date=datetime(2024, 9, 29, tzinfo=UTC).date()),
        )


def test_a_quoted_label_needs_a_value(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)

    with pytest.raises(LabellingError, match="needs a value"):
        add_label(
            cache_root=cache_root,
            gold_root=tmp_path / "gold",
            entry_id=APPLE,
            field=GoldField.COMPANY_NAME,
            quote="Apple Inc.",
            value=None,
        )


# --- one label per field, and overwriting on purpose -----------------------


def test_an_existing_label_is_not_silently_overwritten(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)

    with pytest.raises(LabellingError, match="already labelled"):
        label_apple_auditor(cache_root, gold_root)


def test_replace_overwrites_in_place_rather_than_appending(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)

    add_label(
        cache_root=cache_root,
        gold_root=gold_root,
        entry_id=APPLE,
        field=GoldField.AUDITOR_NAME,
        quote="We have served as the Company's auditor since 2009.",
        value=TextValue(text="Ernst and Young LLP"),
        replace=True,
    )

    labels = load_gold_set(gold_root).labels
    assert len(labels) == 1
    assert isinstance(labels[0].answer, GoldStated)
    assert labels[0].answer.value == TextValue(text="Ernst and Young LLP")


# --- the derivation version -----------------------------------------------


def test_a_stale_derivation_version_refuses_to_write(tmp_path: Path) -> None:
    """The corpus was re-derived and the manifest was not, or the reverse.

    Either way the text on disk is not what the next recheck will produce, and
    a label made now indexes into a document that is about to change.
    """
    cache_root = build_corpus(tmp_path)
    set_manifest_field(cache_root, APPLE, derivation_version=DERIVATION_VERSION + 1)

    with pytest.raises(LabellingError, match="--recheck"):
        label_apple_auditor(cache_root, tmp_path / "gold")


def test_text_that_does_not_match_its_hash_refuses_to_write(tmp_path: Path) -> None:
    # Equal derivation versions do not prove equal text: an edited text file
    # keeps the version and moves every offset after the edit.
    cache_root = build_corpus(tmp_path)
    path = cache_root / "text" / "0000320193" / "0000320193-25-000079.txt"
    path.write_text("edited\n" + APPLE_TEXT, encoding="utf-8")

    with pytest.raises(LabellingError, match="does not match the manifest hash"):
        label_apple_auditor(cache_root, tmp_path / "gold")


def test_verify_is_quiet_when_every_label_still_holds(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)

    assert verify_labels(cache_root, load_gold_set(gold_root)) == []


def test_verify_reports_a_span_that_has_moved(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label = label_apple_auditor(cache_root, gold_root)
    assert isinstance(label.answer, GoldStated)

    # Simulates a re-derivation that shifted everything by one character,
    # without going through the fetcher: the label is now off by one.
    moved = label.model_copy(
        update={
            "answer": label.answer.model_copy(
                update={
                    "span": DocumentSpan(
                        start=label.answer.span.start + 1,
                        end=label.answer.span.end + 1,
                        quote=label.answer.span.quote,
                    )
                }
            )
        }
    )
    write_gold_set(GoldSet(labels=[moved]), gold_root)

    problems = verify_labels(cache_root, load_gold_set(gold_root))

    assert len(problems) == 1
    assert "now reads" in problems[0]


def test_verify_reports_a_version_disagreement_rather_than_a_missing_quote(
    tmp_path: Path,
) -> None:
    """The distinction the version number exists to make.

    Without it the message would be "this citation is wrong". With it, it is
    "this citation was made against derivation 1 and you are on derivation 2",
    which is the difference between a mystery and a migration.
    """
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)
    set_manifest_field(cache_root, APPLE, derivation_version=DERIVATION_VERSION + 1)

    problems = verify_labels(cache_root, load_gold_set(gold_root))

    assert len(problems) == 1
    assert "relabel or re-derive" in problems[0]


# --- the label type --------------------------------------------------------


def test_a_money_label_round_trips_through_json(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    original = GoldSet(
        labels=[
            GoldLabel(
                entry_id=APPLE,
                field=GoldField.REVENUE,
                derivation_version=DERIVATION_VERSION,
                answer=GoldStated(
                    span=DocumentSpan(start=0, end=7, quote="391,035"),
                    value=MoneyValue(
                        money=Money(
                            amount=Decimal("391035000000"),
                            scale=Scale.MILLIONS,
                            currency=Currency.USD,
                        )
                    ),
                ),
                labelled_at=datetime(2026, 8, 24, tzinfo=UTC),
            )
        ]
    )

    write_gold_set(original, gold_root)

    assert load_gold_set(gold_root) == original


def test_a_date_value_comes_back_as_a_date_and_not_as_a_string(tmp_path: Path) -> None:
    """Why the value union is tagged.

    Under a bare `str | int | date | Money` union, Pydantic's smart mode binds
    a JSON "2024-12-31" to str, because str is an exact type match for a JSON
    string and date is not. The label would then be written as a date and read
    back as a string, and Phase 5 would compare a date to a string and score a
    correct extraction as wrong.
    """
    gold_root = tmp_path / "gold"
    write_gold_set(
        GoldSet(
            labels=[
                GoldLabel(
                    entry_id=APPLE,
                    field=GoldField.PERIOD_END,
                    derivation_version=DERIVATION_VERSION,
                    answer=GoldStated(
                        span=DocumentSpan(start=0, end=18, quote="September 27, 2025"),
                        value=DateValue(date=datetime(2025, 9, 27, tzinfo=UTC).date()),
                    ),
                    labelled_at=datetime(2026, 8, 24, tzinfo=UTC),
                )
            ]
        ),
        gold_root,
    )

    answer = load_gold_set(gold_root).labels[0].answer

    assert isinstance(answer, GoldStated)
    assert isinstance(answer.value, DateValue)
    assert answer.value.date.year == 2025


def test_a_field_cannot_carry_the_wrong_shape_of_value() -> None:
    with pytest.raises(ValidationError, match="takes a date value"):
        GoldLabel(
            entry_id=APPLE,
            field=GoldField.PERIOD_END,
            derivation_version=DERIVATION_VERSION,
            answer=GoldStated(
                span=DocumentSpan(start=0, end=18, quote="September 27, 2025"),
                value=TextValue(text="September 27, 2025"),
            ),
            labelled_at=datetime(2026, 8, 24, tzinfo=UTC),
        )


def test_a_gold_label_has_no_failed_state() -> None:
    # A crash is something a model does. There is no document that "failed".
    with pytest.raises(ValidationError):
        GoldLabel.model_validate(
            {
                "entry_id": APPLE,
                "field": "revenue",
                "derivation_version": DERIVATION_VERSION,
                "answer": {"status": "failed", "error": "boom"},
                "labelled_at": "2026-08-24T00:00:00Z",
            }
        )


def test_a_span_and_its_quote_must_be_the_same_length() -> None:
    with pytest.raises(ValidationError, match="quote is 7 chars"):
        DocumentSpan(start=0, end=99, quote="391,035")


def test_the_gold_file_is_sorted_and_byte_stable(tmp_path: Path) -> None:
    # The file is committed and reviewed. Adding one label to a file of 252
    # should be a one-hunk diff, not a reshuffle.
    gold_root = tmp_path / "gold"
    labels = [
        GoldLabel(
            entry_id=entry_id,
            field=field,
            derivation_version=DERIVATION_VERSION,
            answer=GoldNotStated(),
            labelled_at=datetime(2026, 8, 24, tzinfo=UTC),
        )
        for entry_id, field in [
            (JPM, GoldField.PERIOD_START),
            (APPLE, GoldField.REVENUE),
            (APPLE, GoldField.COMPANY_NAME),
        ]
    ]

    write_gold_set(GoldSet(labels=labels), gold_root)
    first = gold_path(gold_root).read_bytes()
    write_gold_set(GoldSet(labels=list(reversed(labels))), gold_root)

    assert gold_path(gold_root).read_bytes() == first
    written = json.loads(first.decode("utf-8"))
    assert [(row["entry_id"], row["field"]) for row in written["labels"]] == [
        (JPM, "period_start"),
        (APPLE, "company_name"),
        (APPLE, "revenue"),
    ]


# --- parsing what the labeller typed ---------------------------------------


def test_a_printed_figure_is_scaled_by_its_heading() -> None:
    value = parse_value(GoldField.REVENUE, "391,035", scale=Scale.MILLIONS, currency=Currency.USD)

    assert isinstance(value, MoneyValue)
    assert value.money.amount == Decimal("391035000000")
    assert value.money.scale is Scale.MILLIONS
    assert value.money.currency is Currency.USD


def test_parentheses_are_how_a_filing_writes_a_loss() -> None:
    value = parse_value(
        GoldField.NET_INCOME, "(15,000)", scale=Scale.THOUSANDS, currency=Currency.USD
    )

    assert isinstance(value, MoneyValue)
    assert value.money.amount == Decimal("-15000000")


def test_a_money_field_will_not_guess_a_scale_or_a_currency() -> None:
    with pytest.raises(LabellingError, match="--scale and --currency"):
        parse_value(GoldField.REVENUE, "391,035")


@pytest.mark.parametrize("raw", ["not a number", "", "1.2.3"])
def test_a_figure_that_is_not_a_number_is_refused(raw: str) -> None:
    with pytest.raises(LabellingError):
        parse_value(GoldField.REVENUE, raw, scale=Scale.UNITS, currency=Currency.USD)


def test_a_date_field_takes_an_iso_date() -> None:
    value = parse_value(GoldField.PERIOD_END, "2025-09-27")

    assert isinstance(value, DateValue)
    assert value.date.month == 9


def test_a_date_field_refuses_the_form_the_document_prints() -> None:
    # "September 27, 2025" is the quote, not the value. Keeping the two apart
    # is the whole distinction between what a document says and what it means.
    with pytest.raises(LabellingError, match="ISO date"):
        parse_value(GoldField.PERIOD_END, "September 27, 2025")


def test_a_count_field_tolerates_the_commas_you_pasted() -> None:
    value = parse_value(GoldField.EMPLOYEE_COUNT, "187,000")

    assert isinstance(value, CountValue)
    assert value.count == 187000


# --- resolving a model's citation into document coordinates ----------------


def test_a_chunk_relative_citation_resolves_to_the_document_offset(tmp_path: Path) -> None:
    """The whole reason a gold label carries no chunk id.

    The model cites characters 6 to 12 of a chunk that began at character 100
    of the document. Gold knows nothing about that chunk; the resolution is
    what brings the two into the same coordinate system.
    """
    cache_root = build_corpus(tmp_path)
    text = document_text(cache_root, corpus_manifest(cache_root).by_id()[APPLE])
    chunk_start = text.index("Total net sales")
    chunk = text[chunk_start : chunk_start + 200]
    offset_in_chunk = chunk.index("391,035")

    resolved = resolve_citation(
        Citation(
            chunk_id="chunk-7",
            start=offset_in_chunk,
            end=offset_in_chunk + len("391,035"),
            quote="391,035",
        ),
        chunk_start=chunk_start,
    )

    assert text[resolved.start : resolved.end] == "391,035"


def test_resolution_survives_a_different_chunking_of_the_same_document(
    tmp_path: Path,
) -> None:
    """Chunk the document twice, cite the same text, resolve to the same span.

    This is the Phase 3 ablation in miniature: the gold set does not move when
    the chunker does.
    """
    cache_root = build_corpus(tmp_path)
    text = document_text(cache_root, corpus_manifest(cache_root).by_id()[APPLE])
    target = text.index("391,035")

    resolutions = []
    for size in (40, 200):
        chunk_start = (target // size) * size
        chunk = text[chunk_start : chunk_start + size]
        start = chunk.index("391,035")
        resolutions.append(
            resolve_citation(
                Citation(
                    chunk_id=f"chunk-{size}",
                    start=start,
                    end=start + len("391,035"),
                    quote="391,035",
                ),
                chunk_start=chunk_start,
            )
        )

    assert resolutions[0] == resolutions[1]


@pytest.mark.parametrize(
    ("gold", "resolved", "expected"),
    [
        # The model quoted the whole line, gold quoted the figure.
        ((10, 17), (0, 40), True),
        # Gold quoted the whole line, the model quoted the figure.
        ((0, 40), (10, 17), True),
        # Identical.
        ((10, 17), (10, 17), True),
        # Adjacent table cells: they touch, and they are different facts.
        ((10, 17), (17, 24), False),
        # Overlapping but neither contains the other.
        ((10, 20), (15, 25), False),
    ],
)
def test_spans_agree_on_containment_only(
    gold: tuple[int, int], resolved: tuple[int, int], expected: bool
) -> None:
    def span(bounds: tuple[int, int]) -> DocumentSpan:
        return DocumentSpan(start=bounds[0], end=bounds[1], quote="x" * (bounds[1] - bounds[0]))

    assert spans_agree(span(gold), span(resolved)) is expected


# --- finding the document, and what is left to label -----------------------


def test_an_unambiguous_fragment_names_a_document(tmp_path: Path) -> None:
    manifest = corpus_manifest(build_corpus(tmp_path))

    assert resolve_entry_id(manifest, "0000320193-25") == APPLE


def test_an_ambiguous_fragment_is_refused_with_the_candidates(tmp_path: Path) -> None:
    manifest = corpus_manifest(build_corpus(tmp_path))

    with pytest.raises(LabellingError, match="matches 2 documents"):
        resolve_entry_id(manifest, "0000")


def test_a_fragment_never_resolves_to_an_oracle_row(tmp_path: Path) -> None:
    """The oracle is the answer key. No labelling path may arrive at it.

    A CIK fragment would otherwise always match `{cik}/companyfacts` as well
    as the filings, so the useful search would be ambiguous by construction.
    """
    cache_root = build_corpus(tmp_path)
    manifest = load_manifest(cache_root)
    manifest.entries.append(
        ManifestEntry(
            entry_id="0000320193/companyfacts",
            company_id="0000320193",
            disposition=Disposition.ORACLE_ONLY,
            last_action=LastAction.FETCHED,
            reason=Reason.ORACLE_COMPANYFACTS,
        )
    )
    write_manifest(manifest, cache_root)

    assert resolve_entry_id(load_manifest(cache_root), "0000320193") == APPLE


def test_a_document_outside_the_corpus_cannot_be_labelled(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    set_manifest_field(
        cache_root, APPLE, disposition=Disposition.EXCLUDED, reason=Reason.NO_PRIMARY_DOCUMENT
    )

    with pytest.raises(LabellingError, match="not in the corpus"):
        label_apple_auditor(cache_root, tmp_path / "gold")


def test_an_empty_manifest_says_to_fetch_the_corpus(tmp_path: Path) -> None:
    with pytest.raises(LabellingError, match="fetch_corpus"):
        corpus_manifest(tmp_path / "cache")


def test_every_field_of_every_document_starts_unlabelled(tmp_path: Path) -> None:
    manifest = corpus_manifest(build_corpus(tmp_path))

    missing = unlabelled(manifest, GoldSet())

    assert len(missing) == 2 * len(GoldField)
    assert (APPLE, GoldField.AUDITOR_NAME) in missing


def test_a_labelled_field_drops_out_of_the_todo_list(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)

    missing = unlabelled(corpus_manifest(cache_root), load_gold_set(gold_root))

    assert (APPLE, GoldField.AUDITOR_NAME) not in missing
    assert (JPM, GoldField.AUDITOR_NAME) in missing


def test_the_progress_report_counts_what_is_left(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)
    gold_root = tmp_path / "gold"
    label_apple_auditor(cache_root, gold_root)

    report = progress_report(corpus_manifest(cache_root), load_gold_set(gold_root))

    assert "1 of 14 labelled, 13 to go" in report


# --- anchors ---------------------------------------------------------------


def test_the_anchor_report_counts_documents_and_occurrences(tmp_path: Path) -> None:
    # The appendix in docs/labelling-policy.md states anchor coverage as fact.
    # This is the command those numbers come from.
    report = anchor_report(build_corpus(tmp_path))

    assert "documents in corpus: 2" in report
    assert "For the fiscal year ended" in report
    assert "auditor since" in report


def test_the_manifest_is_where_a_label_gets_its_derivation_version(tmp_path: Path) -> None:
    cache_root = build_corpus(tmp_path)

    label = label_apple_auditor(cache_root, tmp_path / "gold")

    assert label.derivation_version == DERIVATION_VERSION
    written = json.loads(manifest_path(cache_root).read_text(encoding="utf-8"))
    row = next(row for row in written["entries"] if row["entry_id"] == APPLE)
    assert row["derivation_version"] == label.derivation_version

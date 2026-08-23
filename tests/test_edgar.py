"""Unit tests for the pieces of edgar.py that need no HTTP at all.

The full fetch flow lives in test_fetch_edgar.py. The manifest's own rules -
sorted, atomic, byte-stable writes - live in test_manifest.py.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from docintel.ingest.edgar import corpus_report, read_ciks, recheck_corpus
from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    documents_dir,
    oracle_dir,
    text_dir,
    write_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"
TEN_K = (FIXTURES / "aapl_10k_trimmed.htm").read_bytes()

APPLE = "0000320193"
ACCESSION = "0000320193-25-000079"


# --- reading the committed corpus definition -------------------------------


def test_ciks_are_read_and_padded_to_ten_digits(tmp_path: Path) -> None:
    path = tmp_path / "ciks.txt"
    path.write_text("320193\n0000019617\n", encoding="utf-8")

    assert read_ciks(path) == ["0000320193", "0000019617"]


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "ciks.txt"
    path.write_text(
        "# the corpus\n\n320193   # Apple Inc.\n\n   # a whole-line comment\n19617\n",
        encoding="utf-8",
    )

    assert read_ciks(path) == ["0000320193", "0000019617"]


def test_the_real_corpus_definition_parses(tmp_path: Path) -> None:
    """The committed list is data the fetcher depends on, so it is tested like code."""
    ciks = read_ciks(Path(__file__).parent.parent / "data" / "ciks.txt")

    assert len(ciks) == 18
    assert len(set(ciks)) == 18
    assert all(len(cik) == 10 and cik.isdigit() for cik in ciks)
    # The three large banks are the reason the overflow walk exists. Losing
    # them from the list would leave that code untested against reality.
    assert {"0000019617", "0000886982", "0000070858"} <= set(ciks)


def test_a_line_that_is_not_a_cik_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ciks.txt"
    path.write_text("320193\nCIK0000019617\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is not a CIK"):
        read_ciks(path)


def test_a_duplicate_is_rejected_even_when_written_differently(tmp_path: Path) -> None:
    """`320193` and `0000320193` are the same company, and a corpus that fetched
    it twice would weight it twice."""
    path = tmp_path / "ciks.txt"
    path.write_text("320193\n0000320193\n", encoding="utf-8")

    with pytest.raises(ValueError, match="listed twice"):
        read_ciks(path)


# --- rechecking ------------------------------------------------------------


def document_entry(**overrides: object) -> ManifestEntry:
    fields: dict[str, object] = dict(
        entry_id=f"{APPLE}/{ACCESSION}",
        company_id=APPLE,
        document_id=ACCESSION,
        filing_date="2025-10-31",
        report_date="2025-09-27",
        doc_type="10-K",
        disposition=Disposition.IN_CORPUS,
        last_action=LastAction.FETCHED,
        reason=Reason.OK,
        mime="text/html",
        raw_path=f"documents/{APPLE}/{ACCESSION}.htm",
        fetched_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    fields.update(overrides)
    return ManifestEntry.model_validate(fields)


def seed(cache_root: Path, *entries: ManifestEntry) -> None:
    write_manifest(Manifest(entries=list(entries)), cache_root)


def test_recheck_rederives_text_from_the_bytes_on_disk(tmp_path: Path) -> None:
    """The path you run after bumping DERIVATION_VERSION. No network at all."""
    raw = documents_dir(tmp_path) / APPLE / f"{ACCESSION}.htm"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(TEN_K)
    seed(tmp_path, document_entry())

    (row,) = recheck_corpus(tmp_path).entries

    assert row.last_action is LastAction.RECHECKED
    assert row.disposition is Disposition.IN_CORPUS
    assert row.text_chars and row.text_chars > 0
    assert row.derivation_version == 1
    assert (text_dir(tmp_path) / APPLE / f"{ACCESSION}.txt").exists()


def test_recheck_rescues_a_document_that_used_to_derive_to_nothing(tmp_path: Path) -> None:
    """Why the raw bytes are kept even when the derivation failed.

    A document recorded as EMPTY_DERIVATION is a bug report about derive_text.
    Once that bug is fixed, the fix has to reach the corpus without refetching
    thirteen megabytes from EDGAR to do it.
    """
    raw = documents_dir(tmp_path) / APPLE / f"{ACCESSION}.htm"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(TEN_K)
    seed(
        tmp_path,
        document_entry(
            disposition=Disposition.FAILED,
            reason=Reason.EMPTY_DERIVATION,
            text_chars=0,
        ),
    )

    (row,) = recheck_corpus(tmp_path).entries

    assert row.disposition is Disposition.IN_CORPUS
    assert row.reason is Reason.OK
    assert row.text_chars and row.text_chars > 0


def test_recheck_reports_bytes_that_have_gone_missing(tmp_path: Path) -> None:
    seed(tmp_path, document_entry())

    (row,) = recheck_corpus(tmp_path).entries

    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.MISSING_ON_DISK


def test_recheck_rehashes_the_oracle_but_never_derives_from_it(tmp_path: Path) -> None:
    """The behavioural half of the no-oracle guard.

    Recheck is the one function that reads oracle bytes, and all it is allowed
    to do with them is measure them. Deriving text from companyfacts would put
    the answer key into text/, where the extractor reads.
    """
    facts = oracle_dir(tmp_path) / APPLE / "companyfacts.json"
    facts.parent.mkdir(parents=True)
    facts.write_bytes(b'{"facts": {"us-gaap": {"Revenues": 391035000000}}}')
    seed(
        tmp_path,
        document_entry(
            entry_id=f"{APPLE}/companyfacts",
            document_id=None,
            disposition=Disposition.ORACLE_ONLY,
            reason=Reason.ORACLE_COMPANYFACTS,
            mime="application/json",
            raw_path=f"oracle/{APPLE}/companyfacts.json",
        ),
    )

    (row,) = recheck_corpus(tmp_path).entries

    assert row.disposition is Disposition.ORACLE_ONLY
    assert row.bytes == facts.stat().st_size
    assert row.text_chars is None and row.text_path is None
    assert not text_dir(tmp_path).exists()


def test_recheck_leaves_rows_that_have_no_bytes_alone(tmp_path: Path) -> None:
    seed(
        tmp_path,
        ManifestEntry(
            entry_id=f"{APPLE}/-",
            company_id=APPLE,
            disposition=Disposition.EXCLUDED,
            last_action=LastAction.FETCHED,
            reason=Reason.NO_MATCHING_FILINGS,
        ),
    )

    (row,) = recheck_corpus(tmp_path).entries

    assert row.last_action is LastAction.RECHECKED
    assert row.reason is Reason.NO_MATCHING_FILINGS


def test_recheck_removes_text_left_by_a_derivation_that_now_fails(tmp_path: Path) -> None:
    """A text file with no manifest row pointing at it is a file Phase 3 would
    index and nobody could account for."""
    raw = documents_dir(tmp_path) / APPLE / f"{ACCESSION}.htm"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"<html><body></body></html>")
    stale = text_dir(tmp_path) / APPLE / f"{ACCESSION}.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("text from a better derivation\n", encoding="utf-8")
    seed(tmp_path, document_entry())

    (row,) = recheck_corpus(tmp_path).entries

    assert row.reason is Reason.EMPTY_DERIVATION
    assert not stale.exists()


# --- the report ------------------------------------------------------------


def test_the_report_shows_the_size_distribution_and_markup_ratio() -> None:
    manifest = Manifest(
        entries=[
            document_entry(entry_id="a", bytes=1_520_208, text_chars=207_444),
            document_entry(entry_id="b", bytes=12_927_325, text_chars=1_181_568),
        ]
    )

    report = corpus_report(manifest)

    assert "documents measured: 2" in report
    assert "min 207,444" in report and "max 1,181,568" in report
    # 1 - 207444/1520208 and 1 - 1181568/12927325.
    assert "min 0.864" in report and "max 0.909" in report
    assert "200,000-499,999  # 1" in report
    assert "500,000+  # 1" in report


def test_the_report_counts_failures_next_to_the_distribution() -> None:
    """A document that derived to nothing has to be visible in the bottom
    bucket, not merely absent from the corpus."""
    manifest = Manifest(
        entries=[
            document_entry(entry_id="a", bytes=1_520_208, text_chars=207_444),
            document_entry(
                entry_id="b",
                bytes=900,
                text_chars=0,
                disposition=Disposition.FAILED,
                reason=Reason.EMPTY_DERIVATION,
            ),
            document_entry(
                entry_id="c",
                disposition=Disposition.ORACLE_ONLY,
                reason=Reason.ORACLE_COMPANYFACTS,
            ),
        ]
    )

    report = corpus_report(manifest)

    assert "documents measured: 2" in report
    assert "0  # 1" in report
    assert "in_corpus: 1" in report
    assert "empty_derivation: 1" in report
    assert "oracle_only: 1" in report


def test_the_report_on_an_empty_manifest_says_so() -> None:
    assert "nothing to report" in corpus_report(Manifest())

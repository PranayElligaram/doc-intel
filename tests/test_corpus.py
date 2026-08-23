"""Unit tests for the pieces of corpus.py that need no HTTP at all.

The full fetch flow lives in test_fetch_corpus.py. The manifest's own rules -
sorted, atomic, byte-stable writes - live in test_manifest.py.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pypdf.errors import PyPdfError

from docintel.ingest.corpus import (
    analyse_pdf,
    read_company_numbers,
    recheck_corpus,
    text_layer_report,
)
from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    documents_dir,
    oracle_dir,
    write_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"


def text_layer_pdf() -> bytes:
    return (FIXTURES / "text_layer.pdf").read_bytes()


def scanned_pdf() -> bytes:
    return (FIXTURES / "scanned.pdf").read_bytes()


# --- text layer detection -------------------------------------------------


def test_pdf_with_text_layer_is_measured() -> None:
    measured = analyse_pdf(text_layer_pdf())

    assert measured.page_count == 3
    assert measured.chars_total > 0


def test_scanned_pdf_measures_as_zero_text() -> None:
    """An image-only page is not an error. It is a measurement of zero."""
    measured = analyse_pdf(scanned_pdf())

    assert measured.page_count == 1
    assert measured.chars_total == 0
    assert measured.chars_per_page_median == 0


def test_median_is_the_median_and_not_the_mean() -> None:
    """The fixture has one long page and two short ones, on purpose.

    If analyse_pdf averaged, a single text-heavy cover page would lift a
    scanned filing over any threshold. This is the test that catches that.
    """
    measured = analyse_pdf(text_layer_pdf())
    mean = measured.chars_total / measured.page_count

    assert measured.chars_per_page_median < mean / 2


def test_unreadable_bytes_raise() -> None:
    """Corruption is the caller's problem to record, not analyse_pdf's to hide.

    _describe_bytes catches this broadly, because a malformed PDF surfaces from
    the parser as almost anything. Here we pin the one case we can name, so
    that a pypdf upgrade quietly turning this into a silent empty result would
    fail rather than pass.
    """
    with pytest.raises(PyPdfError):
        analyse_pdf(b"%PDF-1.7\nnot actually a pdf")


# --- the committed corpus definition --------------------------------------


def test_company_numbers_ignore_comments_and_blanks(tmp_path: Path) -> None:
    listing = tmp_path / "companies.txt"
    listing.write_text(
        "# heading\n\n00445790   # Tesco PLC\n   \nSC090312\n# 99999999 commented out entirely\n",
        encoding="utf-8",
    )

    assert read_company_numbers(listing) == ["00445790", "SC090312"]


@pytest.mark.parametrize("bad", ["4457900", "TESCO", "00445790X", "0044579"])
def test_malformed_company_number_is_rejected(tmp_path: Path, bad: str) -> None:
    listing = tmp_path / "companies.txt"
    listing.write_text(f"{bad}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a company number"):
        read_company_numbers(listing)


def test_duplicate_company_number_is_rejected(tmp_path: Path) -> None:
    """A duplicate would double-weight one company in every metric downstream."""
    listing = tmp_path / "companies.txt"
    listing.write_text("00445790\n00445790\n", encoding="utf-8")

    with pytest.raises(ValueError, match="listed twice"):
        read_company_numbers(listing)


def test_the_committed_corpus_definition_parses() -> None:
    """The real data/companies.txt, so a typo in it fails here and not at 3am."""
    listing = Path(__file__).parent.parent / "data" / "companies.txt"

    numbers = read_company_numbers(listing)

    assert numbers, "the corpus definition is empty"
    assert len(numbers) == len(set(numbers))


# --- shared fixtures for the tests below ----------------------------------


def entry(entry_id: str, **overrides: object) -> ManifestEntry:
    base: dict[str, object] = dict(
        entry_id=entry_id,
        company_number=entry_id.split("/")[0],
        disposition=Disposition.IN_CORPUS,
        last_action=LastAction.FETCHED,
        reason=Reason.OK,
    )
    base.update(overrides)
    return ManifestEntry.model_validate(base)


# --- disposition and last_action are separate -----------------------------


def test_disposition_survives_a_recheck_but_last_action_does_not(tmp_path: Path) -> None:
    """The whole reason the two fields are separate.

    After a recheck the manifest must still be able to answer "what is in my
    corpus". If disposition were overwritten by the action, it could not.
    """
    path = documents_dir(tmp_path) / "00445790" / "TXN.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(text_layer_pdf())

    fetched_at = datetime(2026, 8, 22, 9, 30, tzinfo=UTC)
    write_manifest(
        Manifest(
            entries=[
                entry(
                    "00445790/TXN/pdf",
                    transaction_id="TXN",
                    mime="application/pdf",
                    path="documents/00445790/TXN.pdf",
                    fetched_at=fetched_at,
                )
            ]
        ),
        tmp_path,
    )

    rechecked = recheck_corpus(tmp_path)

    (row,) = rechecked.entries
    assert row.disposition is Disposition.IN_CORPUS
    assert row.last_action is LastAction.RECHECKED
    assert row.reason is Reason.OK
    # Not re-derivable from the bytes, so it must be carried over.
    assert row.fetched_at == fetched_at


def test_recheck_remeasures_the_text_layer_from_disk(tmp_path: Path) -> None:
    path = documents_dir(tmp_path) / "00445790" / "TXN.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(text_layer_pdf())
    write_manifest(
        Manifest(
            entries=[
                entry(
                    "00445790/TXN/pdf",
                    transaction_id="TXN",
                    mime="application/pdf",
                    path="documents/00445790/TXN.pdf",
                )
            ]
        ),
        tmp_path,
    )

    (row,) = recheck_corpus(tmp_path).entries

    assert row.text_layer is not None
    assert row.text_layer.page_count == 3


def test_recheck_notices_a_document_that_vanished(tmp_path: Path) -> None:
    write_manifest(
        Manifest(
            entries=[
                entry(
                    "00445790/TXN/pdf",
                    transaction_id="TXN",
                    mime="application/pdf",
                    path="documents/00445790/TXN.pdf",
                )
            ]
        ),
        tmp_path,
    )

    (row,) = recheck_corpus(tmp_path).entries

    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.MISSING_ON_DISK


# --- the no-oracle guard --------------------------------------------------


def test_no_oracle_bytes_ever_live_under_documents(tmp_path: Path) -> None:
    """The structural half of the guard.

    documents/ is what an extractor is allowed to read. If an iXBRL rendition
    ever lands there, the extractor is reading tagged figures rather than a
    document, and every accuracy number the project reports becomes a lie. The
    HTTP half of this guard - that a filing with no PDF rendition is excluded
    rather than having its iXBRL de-tagged into the corpus - is in
    test_fetch_corpus.py.
    """
    docs = documents_dir(tmp_path)
    oracle = oracle_dir(tmp_path)
    docs.mkdir(parents=True)
    oracle.mkdir(parents=True)
    (docs / "a.pdf").write_bytes(text_layer_pdf())
    (oracle / "a.xhtml").write_bytes(b"<html><ix:nonFraction>4200</ix:nonFraction></html>")

    for path in docs.rglob("*"):
        if path.is_file():
            assert path.suffix == ".pdf", f"{path} is not a PDF but is in the corpus directory"
            assert path.read_bytes().startswith(b"%PDF-")


def test_report_counts_every_pdf_regardless_of_disposition(tmp_path: Path) -> None:
    """Deferring the text-layer threshold only works if the data is all there.

    A PDF that failed for some unrelated reason still carries its measurement,
    so a threshold picked later can be applied without refetching anything.
    """
    measured = analyse_pdf(text_layer_pdf())
    manifest = Manifest(
        entries=[
            entry("00445790/A/pdf", text_layer=measured),
            entry(
                "00445790/B/pdf",
                disposition=Disposition.FAILED,
                reason=Reason.UNREADABLE_PDF,
                text_layer=analyse_pdf(scanned_pdf()),
            ),
            entry(
                "00445790/C/ixbrl",
                disposition=Disposition.ORACLE_ONLY,
                reason=Reason.ORACLE_RENDITION,
            ),
        ]
    )

    report = text_layer_report(manifest)

    assert "PDFs measured: 2" in report
    assert "in_corpus: 1" in report
    assert "unreadable_pdf: 1" in report


def test_report_on_an_empty_manifest_says_so() -> None:
    assert "nothing to report" in text_layer_report(Manifest())

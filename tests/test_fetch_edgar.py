"""The full fetch flow, driven over httpx.MockTransport.

Nothing here touches the network. MockTransport is httpx's own transport
substitute, so the client under test is a real httpx.Client with real request
building and real header handling - only the socket is replaced. That is the
point: a hand-rolled fake client would not catch the thing this file exists to
check, which is exactly which URLs go out and which do not.

THE FIXTURES ARE REAL RESPONSES, TRIMMED. Every submissions JSON under
fixtures/edgar/ is a genuine EDGAR response with rows and columns removed;
nothing in them is invented. That matters most for the overflow walk, which
exists because of a property of real filers - JPMorgan showing one 10-K in
`filings.recent` against 69 overflow files - that a synthetic fixture would
only ever confirm back to us. See fixtures/README.md for exactly what was cut.

Time is injected, so the retry path is exercised at zero wall-clock cost. The
pacer's own behaviour is tested directly in test_http.py.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from docintel.ingest.edgar import edgar_headers, fetch_corpus
from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    ManifestEntry,
    Reason,
    documents_dir,
    load_manifest,
    oracle_dir,
    text_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"
EDGAR_FIXTURES = FIXTURES / "edgar"

APPLE = "0000320193"
JPM = "0000019617"

TEN_K = (FIXTURES / "aapl_10k_trimmed.htm").read_bytes()

USER_AGENT = "Test Runner tests@example.com"


def fixture(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((EDGAR_FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


APPLE_SUBMISSIONS = fixture("aapl_submissions_trimmed.json")
JPM_SUBMISSIONS = fixture("jpm_submissions_trimmed.json")
JPM_OVERFLOW_001 = fixture("jpm_submissions_001_trimmed.json")
JPM_OVERFLOW_016 = fixture("jpm_submissions_016_trimmed.json")
COMPANYFACTS = fixture("aapl_companyfacts_trimmed.json")


class Clock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class Edgar:
    """Serves the fixtures by URL path and records every request it is asked for."""

    def __init__(self, *, submissions: dict[str, dict[str, Any]] | None = None) -> None:
        self.submissions = submissions if submissions is not None else {APPLE: APPLE_SUBMISSIONS}
        self.overflow = {
            "CIK0000019617-submissions-001.json": JPM_OVERFLOW_001,
            "CIK0000019617-submissions-016.json": JPM_OVERFLOW_016,
        }
        self.document = TEN_K
        # Status codes to serve instead of the real response, keyed by the tail
        # of the URL path.
        self.broken: dict[str, int] = {}
        self.requests: list[httpx.Request] = []

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        for tail, status in self.broken.items():
            if path.endswith(tail):
                return httpx.Response(status)

        if path.startswith("/api/xbrl/companyfacts/"):
            return httpx.Response(200, json=COMPANYFACTS)

        if path.startswith("/submissions/"):
            name = path.rsplit("/", 1)[-1]
            if name in self.overflow:
                return httpx.Response(200, json=self.overflow[name])
            cik = name.removeprefix("CIK").removesuffix(".json")
            if cik in self.submissions:
                return httpx.Response(200, json=self.submissions[cik])
            return httpx.Response(404)

        if path.startswith("/Archives/edgar/data/"):
            return httpx.Response(200, content=self.document)

        raise AssertionError(f"unexpected request to {path}")


def run(
    handler: Edgar,
    tmp_path: Path,
    *,
    ciks: list[str],
    cap: int = 2,
    force: bool = False,
) -> dict[str, ManifestEntry]:
    clock = Clock()
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, headers=edgar_headers(USER_AGENT)) as client:
        manifest = fetch_corpus(
            client,
            cache_root=tmp_path,
            ciks=ciks,
            max_filings_per_company=cap,
            force=force,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            rand=lambda: 0.5,
        )
    return manifest.by_id()


def in_corpus(entries: dict[str, ManifestEntry]) -> list[ManifestEntry]:
    return [e for e in entries.values() if e.disposition is Disposition.IN_CORPUS]


# --- the overflow walk -----------------------------------------------------


def test_a_10k_reachable_from_recent_alone_never_fetches_the_overflow(tmp_path: Path) -> None:
    """Apple has eleven 10-Ks in `filings.recent` and one overflow file.

    Fetching that overflow file would be 1240 filings of pure waste. The walk
    has to be lazy, not merely correct.
    """
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=2)

    assert len(in_corpus(entries)) == 2
    assert not any("submissions-001" in path for path in handler.paths())


def test_a_10k_that_exists_only_in_the_overflow_is_still_found(tmp_path: Path) -> None:
    """The reason this module has a walk at all.

    JPMorgan's `filings.recent` holds one 10-K - the 2025 one - because their
    pricing supplements crowd out everything else. The 2024 10-K is in overflow
    file 016. A fetcher that reads only `recent` silently returns half a corpus
    for every large bank.
    """
    handler = Edgar(submissions={JPM: JPM_SUBMISSIONS})

    entries = run(handler, tmp_path, ciks=[JPM], cap=2)

    accessions = sorted(str(entry.document_id) for entry in in_corpus(entries))
    assert accessions == ["0000019617-24-000225", "0001628280-26-008131"]
    # 001 holds no 10-K, so the walk had to continue past it to 016.
    assert [path for path in handler.paths() if "submissions-0" in path] == [
        "/submissions/CIK0000019617-submissions-001.json",
        "/submissions/CIK0000019617-submissions-016.json",
    ]


def test_the_walk_stops_as_soon_as_the_cap_is_met(tmp_path: Path) -> None:
    """With a cap of one, `recent` satisfies even JPMorgan and no overflow is read."""
    handler = Edgar(submissions={JPM: JPM_SUBMISSIONS})

    entries = run(handler, tmp_path, ciks=[JPM], cap=1)

    assert len(in_corpus(entries)) == 1
    assert not any("submissions-0" in path for path in handler.paths())


def test_an_unreadable_overflow_file_is_recorded_and_the_walk_continues(tmp_path: Path) -> None:
    """A gap in the walk must be visible in the manifest.

    Without a row naming the file that would not load, a company that came back
    with one filing instead of two is indistinguishable from a company that only
    ever filed once.
    """
    handler = Edgar(submissions={JPM: JPM_SUBMISSIONS})
    handler.broken["submissions-001.json"] = 500

    entries = run(handler, tmp_path, ciks=[JPM], cap=2)

    failed = entries[f"{JPM}/CIK0000019617-submissions-001.json"]
    assert failed.disposition is Disposition.FAILED
    assert failed.reason is Reason.HTTP_ERROR
    # And 016 was still read, so the corpus is short by nothing.
    assert len(in_corpus(entries)) == 2


# --- primary document selection --------------------------------------------


def test_the_document_url_comes_from_primary_document_and_nothing_else(tmp_path: Path) -> None:
    """Apple's own three 10-Ks use three unrelated filename conventions.

    `aapl-20250927.htm` is the modern ticker-and-period form, `a10-k20189292018.htm`
    is what their previous filing agent produced, and `d17062d10k.htm` is a
    Donnelley document id. There is no rule connecting them. Any filename
    heuristic anywhere in this module is a bug waiting for the next filer.
    """
    handler = Edgar()

    run(handler, tmp_path, ciks=[APPLE], cap=3)

    fetched = [path.rsplit("/", 1)[-1] for path in handler.paths() if "/Archives/" in path]
    assert sorted(fetched) == [
        "a10-k20189292018.htm",
        "aapl-20250927.htm",
        "d17062d10k.htm",
    ]


def test_the_archives_url_strips_zeros_from_the_cik_and_dashes_from_the_accession(
    tmp_path: Path,
) -> None:
    """Neither form appears anywhere else in the API, so it is worth pinning."""
    handler = Edgar()

    run(handler, tmp_path, ciks=[APPLE], cap=1)

    (archive,) = [path for path in handler.paths() if "/Archives/" in path]
    assert archive == "/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def test_a_filing_with_no_primary_document_is_excluded(tmp_path: Path) -> None:
    """Blanked here in the test rather than in the fixture.

    The fixtures are real responses and stay that way; a row with an empty
    primaryDocument is a case we have not met in the wild, so it is built
    visibly at the point of use instead of being smuggled into a file that
    claims to be a recording.
    """
    submissions = json.loads(json.dumps(APPLE_SUBMISSIONS))
    submissions["filings"]["recent"]["primaryDocument"][2] = ""
    handler = Edgar(submissions={APPLE: submissions})

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)

    (row,) = [e for e in entries.values() if e.document_id == "0000320193-25-000079"]
    assert row.disposition is Disposition.EXCLUDED
    assert row.reason is Reason.NO_PRIMARY_DOCUMENT
    assert not any("/Archives/" in path for path in handler.paths())


def test_only_10ks_are_candidates(tmp_path: Path) -> None:
    """The Apple fixture carries a 10-Q and an 8-K alongside the annual reports."""
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=10)

    assert {entry.doc_type for entry in in_corpus(entries)} == {"10-K"}
    assert len(in_corpus(entries)) == 3


# --- headers ---------------------------------------------------------------


def test_every_request_carries_the_contact_string_and_encoding(tmp_path: Path) -> None:
    """EDGAR answers a request without these with an "Undeclared Automated Tool" page."""
    handler = Edgar()

    run(handler, tmp_path, ciks=[APPLE], cap=1)

    assert handler.requests
    for request in handler.requests:
        assert request.headers["user-agent"] == USER_AGENT
        assert request.headers["accept-encoding"] == "gzip, deflate"


# --- what lands where ------------------------------------------------------


def test_raw_bytes_text_and_oracle_land_in_three_separate_directories(tmp_path: Path) -> None:
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)

    document = entries[f"{APPLE}/0000320193-25-000079"]
    assert document.raw_path == f"documents/{APPLE}/0000320193-25-000079.htm"
    assert document.text_path == f"text/{APPLE}/0000320193-25-000079.txt"
    assert (documents_dir(tmp_path) / APPLE / "0000320193-25-000079.htm").read_bytes() == TEN_K
    assert (text_dir(tmp_path) / APPLE / "0000320193-25-000079.txt").exists()

    oracle = entries[f"{APPLE}/companyfacts"]
    assert oracle.disposition is Disposition.ORACLE_ONLY
    assert oracle.raw_path == f"oracle/{APPLE}/companyfacts.json"
    assert (oracle_dir(tmp_path) / APPLE / "companyfacts.json").exists()


def test_both_artefacts_are_hashed_separately(tmp_path: Path) -> None:
    """raw_sha256 says whether the source changed; text_sha256 says whether our
    derivation of it did. One hash could not answer both."""
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)
    document = entries[f"{APPLE}/0000320193-25-000079"]

    assert document.raw_sha256 and document.text_sha256
    assert document.raw_sha256 != document.text_sha256
    assert document.bytes == len(TEN_K)
    assert document.text_chars and document.text_chars < document.bytes
    assert document.derivation_version == 1


def test_companyfacts_is_fetched_once_per_company_not_once_per_filing(tmp_path: Path) -> None:
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=3)

    assert len(in_corpus(entries)) == 3
    assert sum(1 for path in handler.paths() if "companyfacts" in path) == 1


# --- failures --------------------------------------------------------------


def test_an_unknown_cik_is_recorded_as_not_found(tmp_path: Path) -> None:
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=["0000000001"], cap=2)

    (row,) = entries.values()
    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.COMPANY_NOT_FOUND


def test_bytes_that_are_not_html_fail_and_derive_nothing(tmp_path: Path) -> None:
    handler = Edgar()
    handler.document = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)
    document = entries[f"{APPLE}/0000320193-25-000079"]

    assert document.disposition is Disposition.FAILED
    assert document.reason is Reason.NOT_HTML
    assert document.text_path is None
    # The bytes are kept, so --recheck can rescue this without refetching.
    assert (tmp_path / document.raw_path).exists() if document.raw_path else False


def test_a_document_that_derives_to_nothing_fails_and_writes_no_text(tmp_path: Path) -> None:
    """A 10-K deriving to zero characters is a defect, not a measurement.

    The Companies House fetcher measured zero-text PDFs and kept them, because
    the open question there was where to put a threshold. There is no threshold
    here: an empty document gives Phase 2 nothing to label and Phase 3 nothing
    to index, and the likeliest cause is a bug in derive_text rather than a
    strange filing.
    """
    handler = Edgar()
    handler.document = b"<html><body><div></div></body></html>"

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)
    document = entries[f"{APPLE}/0000320193-25-000079"]

    assert document.disposition is Disposition.FAILED
    assert document.reason is Reason.EMPTY_DERIVATION
    # Counted as zero rather than left unset, so the report's histogram still
    # shows it in the bottom bucket.
    assert document.text_chars == 0
    assert document.text_path is None
    assert not (text_dir(tmp_path) / APPLE).exists()


# --- caching ---------------------------------------------------------------


def test_existing_bytes_are_not_refetched_but_text_is_rederived(tmp_path: Path) -> None:
    """Bytes are expensive and immutable; derivation is cheap and changes.

    Skipping the derivation too would mean a DERIVATION_VERSION bump only took
    effect for documents that happened to be refetched.
    """
    handler = Edgar()
    run(handler, tmp_path, ciks=[APPLE], cap=1)
    (text_dir(tmp_path) / APPLE / "0000320193-25-000079.txt").unlink()

    second = Edgar()
    entries = run(second, tmp_path, ciks=[APPLE], cap=1)

    assert not any("/Archives/" in path for path in second.paths())
    document = entries[f"{APPLE}/0000320193-25-000079"]
    assert document.last_action is LastAction.SKIPPED_EXISTING
    assert document.disposition is Disposition.IN_CORPUS
    assert (text_dir(tmp_path) / APPLE / "0000320193-25-000079.txt").exists()


def test_fetched_at_survives_a_skip(tmp_path: Path) -> None:
    """It is the one field that cannot be re-derived from the bytes."""
    handler = Edgar()
    first = run(handler, tmp_path, ciks=[APPLE], cap=1)
    original = first[f"{APPLE}/0000320193-25-000079"].fetched_at

    second = run(Edgar(), tmp_path, ciks=[APPLE], cap=1)

    assert original is not None
    assert second[f"{APPLE}/0000320193-25-000079"].fetched_at == original


def test_force_refetches_the_bytes(tmp_path: Path) -> None:
    handler = Edgar()
    run(handler, tmp_path, ciks=[APPLE], cap=1)

    second = Edgar()
    entries = run(second, tmp_path, ciks=[APPLE], cap=1, force=True)

    assert any("/Archives/" in path for path in second.paths())
    assert entries[f"{APPLE}/0000320193-25-000079"].last_action is LastAction.FETCHED


def test_the_manifest_is_written_and_reloads_identically(tmp_path: Path) -> None:
    handler = Edgar()

    entries = run(handler, tmp_path, ciks=[APPLE], cap=2)

    assert load_manifest(tmp_path).by_id() == entries


@pytest.mark.parametrize("status", [403, 500])
def test_a_document_that_will_not_download_is_recorded_rather_than_lost(
    tmp_path: Path, status: int
) -> None:
    handler = Edgar()
    handler.broken["aapl-20250927.htm"] = status

    entries = run(handler, tmp_path, ciks=[APPLE], cap=1)
    document = entries[f"{APPLE}/0000320193-25-000079"]

    assert document.disposition is Disposition.FAILED
    assert document.reason is Reason.HTTP_ERROR

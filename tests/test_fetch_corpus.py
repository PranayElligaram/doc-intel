"""The full fetch flow, driven over httpx.MockTransport.

Nothing here touches the network. MockTransport is httpx's own transport
substitute, so the client under test is a real httpx.Client with real request
building, real header handling and real auth - only the socket is replaced.
That is the point: a hand-rolled fake client would not catch the thing this
file exists to check, which is exactly which headers go out on which hop.

Time is injected too, so the suite still runs in milliseconds. The pacer's own
behaviour - spacing, backoff, bounded attempts - is tested directly in
test_http.py; what is left here is what fetch_corpus does with the responses
the pacer eventually hands back.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from docintel.ingest.corpus import fetch_corpus, recheck_corpus
from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    documents_dir,
    load_manifest,
    manifest_path,
    oracle_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"

FILING_HISTORY = json.loads((FIXTURES / "filing_history.json").read_text(encoding="utf-8"))
DOCUMENT_METADATA = json.loads((FIXTURES / "document_metadata.json").read_text(encoding="utf-8"))
TEXT_LAYER_PDF = (FIXTURES / "text_layer.pdf").read_bytes()
SCANNED_PDF = (FIXTURES / "scanned.pdf").read_bytes()
IXBRL = b"<html><ix:nonFraction name='Turnover'>4200</ix:nonFraction></html>"

COMPANY = "00445790"
SIGNED_HOST = "https://signed-storage.example"


class Recorder:
    """A MockTransport handler that logs every request it is asked to serve."""

    def __init__(self, *, history: dict[str, Any] | None = None) -> None:
        self.history = FILING_HISTORY if history is None else history
        self.requests: list[httpx.Request] = []
        # Status codes to return before the real response, per URL path.
        self.stumbles: dict[str, list[httpx.Response]] = {}

    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]

    def request_to(self, fragment: str) -> httpx.Request:
        matches = [r for r in self.requests if fragment in str(r.url)]
        assert matches, f"nothing requested matching {fragment!r} in {self.urls()}"
        return matches[-1]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        queued = self.stumbles.get(path)
        if queued:
            return queued.pop(0)

        if path.endswith("/filing-history"):
            return httpx.Response(200, json=self.history)

        if path.startswith("/document/") and path.endswith("/content"):
            document_id = path.split("/")[2]
            # The real API answers the content endpoint with a redirect to a
            # separate pre-signed host. Reproducing that here is the only way
            # to assert what we send on the second hop.
            accept = request.headers.get("accept", "")
            return httpx.Response(
                302,
                headers={"location": f"{SIGNED_HOST}/blob/{document_id}?rendition={accept}"},
            )

        if path.startswith("/document/"):
            document_id = path.split("/")[2]
            if document_id not in DOCUMENT_METADATA:
                return httpx.Response(404, json={"errors": [{"error": "not-found"}]})
            return httpx.Response(200, json=DOCUMENT_METADATA[document_id])

        if path.startswith("/blob/"):
            document_id = path.split("/")[2]
            if "xhtml" in str(request.url.query):
                return httpx.Response(200, content=IXBRL)
            body = SCANNED_PDF if document_id == "DOC-PDF-ONLY" else TEXT_LAYER_PDF
            return httpx.Response(200, content=body)

        return httpx.Response(404)


class Clock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def make_client(handler: Recorder) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.example",
        auth=httpx.BasicAuth("TEST-KEY", ""),
        # Load-bearing. See _download in corpus.py.
        follow_redirects=False,
    )


def run(
    tmp_path: Path,
    handler: Recorder,
    *,
    force: bool = False,
    # The fixture has three filings and most tests want all three in play, so
    # the cap is lifted here rather than left at the production default of 2.
    # The cap's own behaviour is tested explicitly further down.
    max_filings_per_company: int = 3,
) -> tuple[Manifest, Clock]:
    clock = Clock()
    with make_client(handler) as client:
        manifest = fetch_corpus(
            client,
            cache_root=tmp_path,
            company_numbers=[COMPANY],
            max_filings_per_company=max_filings_per_company,
            force=force,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            rand=lambda: 0.5,
            now=lambda: datetime(2026, 8, 22, 9, 30, tzinfo=UTC),
        )
    return manifest, clock


# --- the 302 ---------------------------------------------------------------


def test_redirect_is_followed_by_hand_not_by_httpx(tmp_path: Path) -> None:
    handler = Recorder()

    run(tmp_path, handler)

    hops = [url for url in handler.urls() if "/blob/" in url]
    assert hops, "the signed-URL hop never happened"


def test_authorization_is_stripped_on_the_second_hop(tmp_path: Path) -> None:
    """UNVERIFIED against the live API - see _download's docstring.

    This proves what we send, not what Companies House accepts. Sending the
    Companies House key to a third-party storage host would leak it, and a
    signed URL carries its own credential, so we drop the header. Whether the
    live service is happy about that stays an assumption until someone runs
    scripts/fetch_corpus.py with a real key.
    """
    handler = Recorder()

    run(tmp_path, handler)

    first_hop = handler.request_to("/document/DOC-BOTH/content")
    second_hop = handler.request_to("/blob/DOC-BOTH")

    assert first_hop.headers.get("authorization") is not None
    assert second_hop.headers.get("authorization") is None


def test_a_redirect_without_a_location_does_not_crash_the_run(tmp_path: Path) -> None:
    handler = Recorder()
    handler.stumbles["/document/DOC-BOTH/content"] = [httpx.Response(302)]

    manifest, _ = run(tmp_path, handler)

    row = by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/pdf"]
    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.HTTP_ERROR


# --- what the run does with a rate-limited endpoint ------------------------
#
# The pacer's own behaviour is tested in test_http.py. These two are the half
# of those tests that is genuinely about fetch_corpus: that a recovered 429
# leaves the corpus intact, and that an unrecovered one becomes a manifest row
# rather than an exception.


def test_a_recovered_429_does_not_cost_us_the_company(tmp_path: Path) -> None:
    handler = Recorder()
    handler.stumbles[f"/company/{COMPANY}/filing-history"] = [
        httpx.Response(429, headers={"retry-after": "7"})
    ]

    manifest, _ = run(tmp_path, handler)

    # The run carried on rather than giving up on the company.
    assert any(row.disposition is Disposition.IN_CORPUS for row in manifest.entries)


def test_an_unrecovered_429_is_recorded_as_a_failure(tmp_path: Path) -> None:
    handler = Recorder()
    handler.stumbles[f"/company/{COMPANY}/filing-history"] = [httpx.Response(429)] * 10

    manifest, _ = run(tmp_path, handler)

    (row,) = manifest.entries
    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.HTTP_ERROR


# --- selection: PDF is the corpus, iXBRL is oracle only --------------------


def by_id(manifest: Manifest) -> dict[str, ManifestEntry]:
    return {entry.entry_id: entry for entry in manifest.entries}


def test_pdf_rendition_lands_in_documents_and_is_in_corpus(tmp_path: Path) -> None:
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    row = by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/pdf"]
    assert row.disposition is Disposition.IN_CORPUS
    assert row.reason is Reason.OK
    assert row.path == f"documents/{COMPANY}/TXN-FULL-2024.pdf"
    assert (tmp_path / row.path).read_bytes() == TEXT_LAYER_PDF


def test_ixbrl_rendition_lands_in_oracle_and_is_never_in_corpus(tmp_path: Path) -> None:
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    row = by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/ixbrl"]
    assert row.disposition is Disposition.ORACLE_ONLY
    assert row.reason is Reason.ORACLE_RENDITION
    assert row.path is not None
    assert row.path.startswith("oracle/")


def test_no_oracle_guard_nothing_under_documents_is_ixbrl(tmp_path: Path) -> None:
    """The guard, over a real fetch rather than a hand-built directory.

    Every byte under documents/ must be a PDF. An iXBRL file there would let an
    extractor read the tagged answer instead of the document.
    """
    handler = Recorder()

    run(tmp_path, handler)

    written = [p for p in documents_dir(tmp_path).rglob("*") if p.is_file()]
    assert written
    for path in written:
        assert path.suffix == ".pdf"
        assert path.read_bytes().startswith(b"%PDF-")
        assert b"ix:nonFraction" not in path.read_bytes()


def test_a_filing_with_no_pdf_is_excluded_not_de_tagged(tmp_path: Path) -> None:
    """The behavioural half of the no-oracle guard.

    TXN-IXBRL-2022 has an iXBRL rendition and no PDF. The tempting move is to
    strip the tags and use it as a corpus document. We do not: the filing is
    excluded, and we do not even keep its iXBRL, because an answer key with no
    document to answer about is just clutter.
    """
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    rows = by_id(manifest)
    row = rows[f"{COMPANY}/TXN-IXBRL-2022/pdf"]
    assert row.disposition is Disposition.EXCLUDED
    assert row.reason is Reason.NO_PDF_RENDITION
    assert row.path is None

    assert f"{COMPANY}/TXN-IXBRL-2022/ixbrl" not in rows
    assert not (oracle_dir(tmp_path) / COMPANY / "TXN-IXBRL-2022.xhtml").exists()


def test_a_filing_with_a_pdf_and_no_ixbrl_yields_one_row(tmp_path: Path) -> None:
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    rows = by_id(manifest)
    assert rows[f"{COMPANY}/TXN-SMALL-2023/pdf"].disposition is Disposition.IN_CORPUS
    assert f"{COMPANY}/TXN-SMALL-2023/ixbrl" not in rows


def test_every_candidate_including_rejects_is_in_the_manifest(tmp_path: Path) -> None:
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    assert set(by_id(manifest)) == {
        f"{COMPANY}/TXN-FULL-2024/pdf",
        f"{COMPANY}/TXN-FULL-2024/ixbrl",
        f"{COMPANY}/TXN-SMALL-2023/pdf",
        f"{COMPANY}/TXN-IXBRL-2022/pdf",
    }


def test_text_layer_is_measured_for_every_pdf_including_the_scan(tmp_path: Path) -> None:
    """Measured, never excluded. The threshold is a later phase's decision."""
    handler = Recorder()

    manifest, _ = run(tmp_path, handler)

    scan = by_id(manifest)[f"{COMPANY}/TXN-SMALL-2023/pdf"]
    assert scan.text_layer is not None
    assert scan.text_layer.chars_total == 0
    # Zero extractable text is not a reason to drop it at this stage.
    assert scan.disposition is Disposition.IN_CORPUS


# --- companies that produce nothing ---------------------------------------


def test_unknown_company_is_recorded_as_failed(tmp_path: Path) -> None:
    handler = Recorder()
    handler.stumbles[f"/company/{COMPANY}/filing-history"] = [httpx.Response(404)]

    manifest, _ = run(tmp_path, handler)

    (row,) = manifest.entries
    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.COMPANY_NOT_FOUND


def test_company_with_no_accounts_filings_is_recorded(tmp_path: Path) -> None:
    handler = Recorder(history={"items": []})

    manifest, _ = run(tmp_path, handler)

    (row,) = manifest.entries
    assert row.disposition is Disposition.EXCLUDED
    assert row.reason is Reason.NO_ACCOUNTS_FILINGS


def test_bytes_that_are_not_a_pdf_are_recorded_as_failed(tmp_path: Path) -> None:
    handler = Recorder()
    handler.stumbles["/blob/DOC-BOTH"] = [httpx.Response(200, content=b"<html>nope</html>")]

    manifest, _ = run(tmp_path, handler)

    row = by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/pdf"]
    assert row.disposition is Disposition.FAILED
    assert row.reason is Reason.NOT_A_PDF


# --- skip existing, force, idempotency ------------------------------------


def test_second_run_skips_documents_already_on_disk(tmp_path: Path) -> None:
    first_handler = Recorder()
    run(tmp_path, first_handler)

    second_handler = Recorder()
    manifest, _ = run(tmp_path, second_handler)

    row = by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/pdf"]
    assert row.last_action is LastAction.SKIPPED_EXISTING
    assert row.disposition is Disposition.IN_CORPUS
    assert not [url for url in second_handler.urls() if "/blob/" in url]


def test_skipping_preserves_the_original_fetched_at(tmp_path: Path) -> None:
    """The one field that cannot be re-derived from the bytes on disk."""
    run(tmp_path, Recorder())
    first = load_manifest(tmp_path)

    clock = Clock()
    with make_client(Recorder()) as client:
        second = fetch_corpus(
            client,
            cache_root=tmp_path,
            company_numbers=[COMPANY],
            max_filings_per_company=3,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            rand=lambda: 0.5,
            now=lambda: datetime(2099, 1, 1, tzinfo=UTC),
        )

    key = f"{COMPANY}/TXN-FULL-2024/pdf"
    assert by_id(second)[key].fetched_at == by_id(first)[key].fetched_at


def test_force_is_the_only_path_that_redownloads(tmp_path: Path) -> None:
    run(tmp_path, Recorder())

    handler = Recorder()
    manifest, _ = run(tmp_path, handler, force=True)

    assert [url for url in handler.urls() if "/blob/" in url]
    assert by_id(manifest)[f"{COMPANY}/TXN-FULL-2024/pdf"].last_action is LastAction.FETCHED


def test_a_settled_corpus_writes_a_byte_identical_manifest(tmp_path: Path) -> None:
    """Run once to populate, then twice more. The last two must match exactly.

    Run one differs legitimately: its rows say fetched, later runs say
    skipped_existing. From then on nothing may move, or every run would show up
    as a corpus change.
    """
    run(tmp_path, Recorder())
    run(tmp_path, Recorder())
    settled = manifest_path(tmp_path).read_bytes()
    run(tmp_path, Recorder())

    assert manifest_path(tmp_path).read_bytes() == settled


def test_the_whole_flow_makes_no_real_network_call(tmp_path: Path) -> None:
    """Belt and braces: every request went through our handler, in order."""
    handler = Recorder()

    run(tmp_path, handler)

    assert handler.requests
    for request in handler.requests:
        assert request.url.host in ("api.example", "signed-storage.example", "doc-api.example")


# --- the per-company cap --------------------------------------------------


def filing(transaction_id: str, date: str, document: str = "DOC-PDF-ONLY") -> dict[str, Any]:
    return {
        "transaction_id": transaction_id,
        "category": "accounts",
        "type": "AA",
        "date": date,
        "description": "accounts-with-accounts-type-full",
        "links": {"document_metadata": f"https://doc-api.example/document/{document}"},
    }


def test_cap_takes_the_most_recent_filings(tmp_path: Path) -> None:
    handler = Recorder()

    manifest, _ = run(tmp_path, handler, max_filings_per_company=1)

    assert set(by_id(manifest)) == {
        f"{COMPANY}/TXN-FULL-2024/pdf",
        f"{COMPANY}/TXN-FULL-2024/ixbrl",
    }


def test_filings_beyond_the_cap_get_no_manifest_row(tmp_path: Path) -> None:
    """Not candidates, so not rejects.

    A row for every historical filing a company ever made would bury the real
    rejections - the ones we looked at and turned down - under noise.
    """
    handler = Recorder()

    manifest, _ = run(tmp_path, handler, max_filings_per_company=1)

    dropped = [row for row in manifest.entries if row.transaction_id != "TXN-FULL-2024"]
    assert dropped == []
    # And we never even asked about them.
    assert not [url for url in handler.urls() if "DOC-IXBRL-ONLY" in url]


def test_cap_defaults_to_two() -> None:
    """The default is a corpus-design decision, so it is pinned by a test."""
    from docintel.ingest.manifest import DEFAULT_MAX_FILINGS_PER_COMPANY

    assert DEFAULT_MAX_FILINGS_PER_COMPANY == 2


def test_same_day_filings_are_broken_deterministically_on_transaction_id(
    tmp_path: Path,
) -> None:
    """The tie-break is what makes the cap reproducible, not the date.

    Three filings on one day: sorting on the date alone leaves them in whatever
    order the API returned, so a later run could take a different pair and the
    corpus would drift without anyone noticing. Feeding the same filings in two
    different orders must select the same ones.
    """
    same_day = [
        filing("TXN-A", "2024-09-27"),
        filing("TXN-B", "2024-09-27"),
        filing("TXN-C", "2024-09-27"),
    ]

    forwards, _ = run(
        tmp_path / "one",
        Recorder(history={"items": same_day}),
        max_filings_per_company=2,
    )
    backwards, _ = run(
        tmp_path / "two",
        Recorder(history={"items": list(reversed(same_day))}),
        max_filings_per_company=2,
    )

    assert set(by_id(forwards)) == set(by_id(backwards))
    # Descending on transaction id, so C and B win and A is dropped.
    assert sorted(row.transaction_id or "" for row in forwards.entries) == ["TXN-B", "TXN-C"]


def test_selection_is_stable_across_runs_with_a_shuffled_api(tmp_path: Path) -> None:
    """Same inputs in a different order, same corpus - across two real runs."""
    items = [
        filing("TXN-A", "2024-09-27"),
        filing("TXN-B", "2023-09-27"),
        filing("TXN-C", "2022-09-27"),
    ]

    first, _ = run(tmp_path / "one", Recorder(history={"items": items}), max_filings_per_company=2)
    second, _ = run(
        tmp_path / "two",
        Recorder(history={"items": list(reversed(items))}),
        max_filings_per_company=2,
    )

    assert set(by_id(first)) == set(by_id(second))
    assert sorted(row.transaction_id or "" for row in first.entries) == ["TXN-A", "TXN-B"]


def test_the_cap_is_recorded_on_the_manifest_and_round_trips(tmp_path: Path) -> None:
    """The manifest states the parameters the corpus was built with.

    Without it there is no way to tell a corpus capped at 2 from one where
    every company happened to have filed exactly twice.
    """
    manifest, _ = run(tmp_path, Recorder(), max_filings_per_company=3)

    assert manifest.max_filings_per_company == 3
    assert load_manifest(tmp_path).max_filings_per_company == 3


def test_the_cap_survives_a_recheck(tmp_path: Path) -> None:
    """A recheck re-measures bytes. It does not re-select, so it must not lie."""
    run(tmp_path, Recorder(), max_filings_per_company=3)

    assert recheck_corpus(tmp_path).max_filings_per_company == 3


def test_a_cap_below_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Manifest(max_filings_per_company=0)

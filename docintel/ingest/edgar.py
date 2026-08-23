"""Fetches the filing corpus from SEC EDGAR and records what it did.

The rows this module produces, and the rules about how they are written, live
in manifest.py; the request pacing lives in http.py; the text derivation lives
in html_text.py. What is left here is the EDGAR part: which endpoints to walk,
which document to keep, and which candidates to turn down.

WHY EDGAR AND NOT COMPANIES HOUSE. The first corpus was Companies House PDFs.
All 36 of them turned out to be image-only scans with no extractable text at
all, which is not a corpus you can build a text extractor on. The measurement
and the reasoning are written up in docs/corpus-selection.md.

THE SUBMISSIONS WALK, which is the part worth reading. `filings.recent` in the
submissions JSON is not the filer's recent 10-Ks - it is their most recent
~1000 filings of every kind. A bank filing thousands of 424B2 pricing
supplements a year pushes its own annual report out of that window: JPMorgan
and Goldman Sachs each show exactly ONE 10-K in `filings.recent`, so a cap of
two is unreachable without following the `filings.files[]` overflow. Those
files are fetched lazily and the walk stops the moment the cap is met, because
JPMorgan's second 10-K sits in overflow file 016 of 69 and there is no reason
to read 017 through 069 to find out it is not there.

NO FILENAME HEURISTIC, ANYWHERE. The document to fetch is whatever
`primaryDocument` says it is. Apple's own 10-Ks use three unrelated filename
conventions - `aapl-20250927.htm`, `a10-k20189292018.htm`, `d17062d10k.htm` -
and that is one filer. Any rule inferred from a filename is a rule that fails
on the next filer.

THE ORACLE. The XBRL companyfacts API returns every tagged figure the company
ever reported, keyed by concept. That is a perfect answer key and a
catastrophic input: an extractor that reads it is not doing document
understanding, it is reading a database. So it goes under `oracle/`, never
under `documents/` or `text/`, and tests/test_oracle_guard.py walks the source
of every module here to prove nothing in the extraction path can reach it.

EMPTY DERIVATION IS A FAILURE, not a measurement. The Companies House fetcher
measured its text layer and never excluded on it, because the question there
was where to put a threshold and you cannot choose one before seeing the
distribution. There is no threshold here. A 10-K that derives to zero
characters is a defect - most likely in derive_text rather than in the filing -
and letting it through would hand Phase 2 an empty document to label and Phase
3 an empty document to index. The raw bytes stay on disk so that `--recheck`
can rescue it once the defect is fixed.
"""

import hashlib
import random
import re
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from docintel.ingest.html_text import DERIVATION_VERSION, derive_text
from docintel.ingest.http import Pacer
from docintel.ingest.manifest import (
    DEFAULT_MAX_FILINGS_PER_COMPANY,
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    documents_dir,
    load_manifest,
    oracle_dir,
    text_dir,
    write_manifest,
)

# Submissions and the XBRL APIs live on data.sec.gov; the documents themselves
# live on www.sec.gov. Two hosts, so every URL this module builds is absolute
# and the client carries no base_url.
DATA_HOST = "https://data.sec.gov"
ARCHIVES_HOST = "https://www.sec.gov"

# Exact match. A 10-K/A is an amendment to a 10-K, not another annual report,
# and admitting one would put two documents describing the same period in the
# corpus with only one of them complete.
FORM_TYPE = "10-K"

HTML_MIME = "text/html"
JSON_MIME = "application/json"

# The columns of the submissions table that we read. EDGAR sends about sixteen;
# naming the ones we use keeps the fixtures honest about what is exercised.
FILING_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "form",
    "primaryDocument",
)

CIK_PATTERN = re.compile(r"^\d{1,10}$")


def read_ciks(path: Path) -> list[str]:
    """Reads the committed corpus definition, zero-padding each CIK to ten.

    The corpus is a committed list, not an API search run at fetch time. A
    search returns different companies to different people on different days,
    and a corpus that cannot be reproduced cannot support a number in a README.
    """
    ciks: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not CIK_PATTERN.match(line):
            raise ValueError(f"{path}:{lineno}: {line!r} is not a CIK")
        cik = line.zfill(10)
        if cik in ciks:
            raise ValueError(f"{path}:{lineno}: CIK {cik} is listed twice")
        ciks.append(cik)
    return ciks


def edgar_headers(user_agent: str) -> dict[str, str]:
    """The headers EDGAR requires. One definition, shared by the script and the tests.

    The User-Agent must be a contact string of the form "Name email@domain.com".
    Anything else - including the default httpx one - is answered with an
    "Undeclared Automated Tool" error page rather than the document. It is a
    courtesy identifier, not a credential.

    Accept-Encoding is asked for by the same SEC guidance. It is not just
    politeness: a 10-K is 1-13MB of HTML and compresses to under a tenth of that.
    """
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _submissions_url(cik: str) -> str:
    return f"{DATA_HOST}/submissions/CIK{cik}.json"


def _overflow_url(name: str) -> str:
    return f"{DATA_HOST}/submissions/{name}"


def _companyfacts_url(cik: str) -> str:
    return f"{DATA_HOST}/api/xbrl/companyfacts/CIK{cik}.json"


def _document_url(cik: str, accession: str, primary_document: str) -> str:
    """Builds the Archives URL for a filing's primary document.

    The path takes the CIK with leading zeros stripped and the accession number
    with its dashes removed, which is not the form either appears in anywhere
    else in the API.
    """
    return (
        f"{ARCHIVES_HOST}/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{primary_document}"
    )


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Turns EDGAR's dict-of-arrays filing table into a list of filings.

    strict=True so that a table whose columns disagree in length fails loudly
    rather than silently truncating to the shortest one.
    """
    columns = [column for column in FILING_COLUMNS if column in table]
    if not columns:
        return []
    return [
        dict(zip(columns, values, strict=True))
        for values in zip(*(table[column] for column in columns), strict=True)
    ]


def _annual_reports(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The 10-K rows of one filing table, most recent first.

    Ties are broken on the accession number. Sorting on the date alone would
    leave same-day filings in whatever order the API happened to return them,
    so the cap could take a different pair on a later run and the corpus would
    quietly drift. The tie-break is what makes the selection reproducible.
    """
    matching = [row for row in _rows(table) if row.get("form") == FORM_TYPE]
    matching.sort(
        key=lambda row: (str(row.get("filingDate", "")), str(row.get("accessionNumber", ""))),
        reverse=True,
    )
    return matching


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: Path, cache_root: Path) -> str:
    return path.relative_to(cache_root).as_posix()


def fetch_corpus(
    client: httpx.Client,
    *,
    cache_root: Path,
    ciks: Sequence[str],
    max_filings_per_company: int = DEFAULT_MAX_FILINGS_PER_COMPANY,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rand: Callable[[], float] = random.random,
    now: Callable[[], datetime] = _utc_now,
) -> Manifest:
    """Fetches annual reports for each CIK and returns the new manifest.

    The submissions JSON is re-read every run - it is how we learn the
    candidate list, and companies file new annual reports. Only document
    *bytes* are skipped when they are already on disk; `force` is the only path
    that redownloads. Text is re-derived either way, so a derivation change is
    picked up without refetching anything.
    """
    pacer = Pacer(sleep=sleep, monotonic=monotonic, rand=rand)
    previous = load_manifest(cache_root).by_id()
    entries: list[ManifestEntry] = []

    for cik in ciks:
        entries.extend(
            _fetch_company(
                pacer,
                client,
                cache_root=cache_root,
                cik=cik,
                max_filings_per_company=max_filings_per_company,
                force=force,
                previous=previous,
                now=now,
            )
        )

    manifest = Manifest(max_filings_per_company=max_filings_per_company, entries=entries)
    write_manifest(manifest, cache_root)
    return manifest


def _fetch_company(
    pacer: Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    cik: str,
    max_filings_per_company: int,
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> Iterator[ManifestEntry]:
    submissions = pacer.get(client, _submissions_url(cik), headers={"Accept": JSON_MIME})
    if submissions.status_code == 404:
        yield _company_row(cik, "-", Reason.COMPANY_NOT_FOUND)
        return
    if submissions.status_code != 200:
        yield _company_row(cik, "-", Reason.HTTP_ERROR)
        return

    filings, unreadable = _walk_filings(
        pacer, client, submissions.json(), cap=max_filings_per_company
    )

    # One row per overflow file we could not read. Without these, a company
    # that came back with one filing instead of two looks like a company that
    # only ever filed once.
    for name in unreadable:
        yield _company_row(cik, name, Reason.HTTP_ERROR)

    if not filings:
        yield _company_row(cik, "-", Reason.NO_MATCHING_FILINGS)
        return

    for row in filings:
        yield _fetch_filing(
            pacer,
            client,
            cache_root=cache_root,
            cik=cik,
            row=row,
            force=force,
            previous=previous,
            now=now,
        )

    yield _fetch_oracle(
        pacer,
        client,
        cache_root=cache_root,
        cik=cik,
        force=force,
        previous=previous,
        now=now,
    )


def _walk_filings(
    pacer: Pacer,
    client: httpx.Client,
    submissions: dict[str, Any],
    *,
    cap: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collects up to `cap` annual reports, following the overflow if it must.

    Sources are read newest-first and the walk stops as soon as the cap is met.
    That ordering is a property of the API rather than something we impose:
    `filings.recent` holds the most recent filings, and `filings.files[]` is
    listed newest-window-first with each entry's window strictly older than the
    one before it. Verified against JPMorgan and Goldman Sachs, whose 69 and 38
    overflow files are the reason any of this exists.

    Returns the filings and the names of any overflow files that would not
    load, so the caller can record the gap rather than silently returning short.
    """
    filings_block = submissions.get("filings", {})
    selected = _annual_reports(filings_block.get("recent", {}))[:cap]
    unreadable: list[str] = []

    for overflow in filings_block.get("files", []):
        if len(selected) >= cap:
            break
        name = overflow.get("name")
        if not name:
            continue
        response = pacer.get(client, _overflow_url(name), headers={"Accept": JSON_MIME})
        if response.status_code != 200:
            unreadable.append(name)
            continue
        selected.extend(_annual_reports(response.json())[: cap - len(selected)])

    return selected, unreadable


def _company_row(cik: str, source: str, reason: Reason) -> ManifestEntry:
    """A row for a company-level outcome rather than a document.

    `source` distinguishes the whole-company verdicts (`-`) from a single
    unreadable overflow file, which is named so it can be retried by hand.
    """
    disposition = (
        Disposition.EXCLUDED if reason is Reason.NO_MATCHING_FILINGS else Disposition.FAILED
    )
    return ManifestEntry(
        entry_id=f"{cik}/{source}",
        company_id=cik,
        disposition=disposition,
        last_action=LastAction.FETCHED,
        reason=reason,
    )


def _fetch_filing(
    pacer: Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    cik: str,
    row: dict[str, Any],
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> ManifestEntry:
    accession = str(row.get("accessionNumber", "")) or "-"
    entry_id = f"{cik}/{accession}"
    primary_document = str(row.get("primaryDocument", "")).strip()

    if not primary_document:
        return _filing_row(entry_id, cik, row, Reason.NO_PRIMARY_DOCUMENT)

    # The suffix comes from the filename EDGAR gave us rather than from a guess
    # about the format, so a filing that is not .htm is stored as what it is
    # and shows up in the manifest as such.
    suffix = Path(primary_document).suffix or ".htm"
    raw_path = documents_dir(cache_root) / cik / f"{accession}{suffix}"

    if raw_path.exists() and not force:
        # Re-derived from the bytes on disk rather than copied from the old
        # manifest, so the manifest always describes what is actually there.
        # fetched_at is the one thing that cannot be re-derived, so it is
        # carried over.
        return _store_document(
            entry_id,
            cik,
            row,
            raw_path.read_bytes(),
            raw_path,
            cache_root,
            last_action=LastAction.SKIPPED_EXISTING,
            fetched_at=_previous_fetched_at(previous, entry_id),
        )

    response = pacer.get(
        client,
        _document_url(cik, accession, primary_document),
        headers={"Accept": HTML_MIME},
    )
    if response.status_code != 200:
        return _filing_row(entry_id, cik, row, Reason.HTTP_ERROR)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response.content)
    return _store_document(
        entry_id,
        cik,
        row,
        response.content,
        raw_path,
        cache_root,
        last_action=LastAction.FETCHED,
        fetched_at=now(),
    )


def _filing_row(entry_id: str, cik: str, row: dict[str, Any], reason: Reason) -> ManifestEntry:
    disposition = (
        Disposition.EXCLUDED if reason is Reason.NO_PRIMARY_DOCUMENT else Disposition.FAILED
    )
    return ManifestEntry(
        entry_id=entry_id,
        company_id=cik,
        document_id=row.get("accessionNumber"),
        filing_date=row.get("filingDate"),
        report_date=row.get("reportDate"),
        doc_type=row.get("form"),
        disposition=disposition,
        last_action=LastAction.FETCHED,
        reason=reason,
    )


def _looks_like_html(data: bytes) -> bool:
    """A cheap sanity check on the bytes, not a validator.

    It exists to catch a fetch that returned something other than a filing -
    an error page is still HTML, but a PDF or a plain-text submission is not
    what the derivation expects.
    """
    return data.lstrip()[:1] == b"<"


def _store_document(
    entry_id: str,
    cik: str,
    row: dict[str, Any],
    data: bytes,
    raw_path: Path,
    cache_root: Path,
    *,
    last_action: LastAction,
    fetched_at: datetime | None,
) -> ManifestEntry:
    """Derives the text from raw bytes we hold, writes it, and builds the row.

    Both artefacts are hashed. raw_sha256 answers "did the source change under
    us"; text_sha256 answers "did our derivation of it change". They are
    different questions and a single hash could not answer both.
    """
    accession = str(row.get("accessionNumber", "")) or "-"
    text_path = text_dir(cache_root) / cik / f"{accession}.txt"

    # The fields the three outcomes below have in common. Annotated because
    # every branch splats it into ManifestEntry, and an inferred value type
    # would be a union that matches none of the fields it lands in.
    base: dict[str, Any] = dict(
        entry_id=entry_id,
        company_id=cik,
        document_id=row.get("accessionNumber"),
        filing_date=row.get("filingDate"),
        report_date=row.get("reportDate"),
        doc_type=row.get("form"),
        last_action=last_action,
        mime=HTML_MIME,
        raw_sha256=_sha256(data),
        bytes=len(data),
        raw_path=_relative(raw_path, cache_root),
        fetched_at=fetched_at,
    )

    if not _looks_like_html(data):
        _discard_text(text_path)
        return ManifestEntry(**base, disposition=Disposition.FAILED, reason=Reason.NOT_HTML)

    # errors="replace" rather than a guess at a legacy encoding: a filing we
    # cannot decode cleanly should still reach the corpus with a visible
    # replacement character, which fails a citation check loudly, rather than
    # be silently reinterpreted under an encoding nobody chose.
    text = derive_text(data.decode("utf-8", errors="replace"))

    if not text.strip():
        # Recorded as zero characters so the report's histogram still counts
        # it, but kept out of the corpus and given no text file. See the module
        # docstring on why this is a failure and not a measurement.
        _discard_text(text_path)
        return ManifestEntry(
            **base,
            disposition=Disposition.FAILED,
            reason=Reason.EMPTY_DERIVATION,
            text_chars=0,
            derivation_version=DERIVATION_VERSION,
        )

    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    return ManifestEntry(
        **base,
        disposition=Disposition.IN_CORPUS,
        reason=Reason.OK,
        text_sha256=_sha256(text.encode("utf-8")),
        text_chars=len(text),
        text_path=_relative(text_path, cache_root),
        derivation_version=DERIVATION_VERSION,
    )


def _discard_text(text_path: Path) -> None:
    """Removes a text file left by an earlier, better derivation.

    Without this, a run that starts failing to derive a document leaves the old
    text on disk with no manifest row pointing at it, and Phase 3 would happily
    index a file the manifest says does not exist.
    """
    text_path.unlink(missing_ok=True)


def _fetch_oracle(
    pacer: Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    cik: str,
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> ManifestEntry:
    """Fetches the companyfacts answer key. Once per company, not per filing."""
    entry_id = f"{cik}/companyfacts"
    path = oracle_dir(cache_root) / cik / "companyfacts.json"

    if path.exists() and not force:
        return _oracle_row(
            entry_id,
            cik,
            path.read_bytes(),
            path,
            cache_root,
            LastAction.SKIPPED_EXISTING,
            _previous_fetched_at(previous, entry_id),
        )

    response = pacer.get(client, _companyfacts_url(cik), headers={"Accept": JSON_MIME})
    if response.status_code != 200:
        return ManifestEntry(
            entry_id=entry_id,
            company_id=cik,
            disposition=Disposition.FAILED,
            last_action=LastAction.FETCHED,
            reason=Reason.HTTP_ERROR,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return _oracle_row(entry_id, cik, response.content, path, cache_root, LastAction.FETCHED, now())


def _oracle_row(
    entry_id: str,
    cik: str,
    data: bytes,
    path: Path,
    cache_root: Path,
    last_action: LastAction,
    fetched_at: datetime | None,
) -> ManifestEntry:
    return ManifestEntry(
        entry_id=entry_id,
        company_id=cik,
        disposition=Disposition.ORACLE_ONLY,
        last_action=last_action,
        reason=Reason.ORACLE_COMPANYFACTS,
        mime=JSON_MIME,
        raw_sha256=_sha256(data),
        bytes=len(data),
        raw_path=_relative(path, cache_root),
        fetched_at=fetched_at,
    )


def _previous_fetched_at(previous: dict[str, ManifestEntry], entry_id: str) -> datetime | None:
    entry = previous.get(entry_id)
    return entry.fetched_at if entry else None


def recheck_corpus(cache_root: Path) -> Manifest:
    """Re-derives text from the bytes already on disk. Makes no network calls.

    Separate from `fetch_corpus` rather than a flag through it, because the
    distinguishing property is the absence of a client: this is the path you
    run after bumping DERIVATION_VERSION, or after fixing a derivation bug that
    emptied a document, without touching EDGAR at all.
    """
    manifest = load_manifest(cache_root)
    entries: list[ManifestEntry] = []

    for entry in manifest.entries:
        if entry.raw_path is None:
            entries.append(entry.model_copy(update={"last_action": LastAction.RECHECKED}))
            continue

        path = cache_root / entry.raw_path
        if not path.exists():
            entries.append(
                entry.model_copy(
                    update={
                        "disposition": Disposition.FAILED,
                        "last_action": LastAction.RECHECKED,
                        "reason": Reason.MISSING_ON_DISK,
                    }
                )
            )
            continue

        data = path.read_bytes()
        if entry.disposition is Disposition.ORACLE_ONLY:
            # Re-hashed, never re-derived. Nothing turns oracle bytes into text.
            entries.append(
                entry.model_copy(
                    update={
                        "last_action": LastAction.RECHECKED,
                        "raw_sha256": _sha256(data),
                        "bytes": len(data),
                    }
                )
            )
            continue

        row = {
            "accessionNumber": entry.document_id,
            "filingDate": entry.filing_date,
            "reportDate": entry.report_date,
            "form": entry.doc_type,
        }
        entries.append(
            _store_document(
                entry.entry_id,
                entry.company_id,
                row,
                data,
                path,
                cache_root,
                last_action=LastAction.RECHECKED,
                fetched_at=entry.fetched_at,
            )
        )

    # The cap describes how the corpus was selected, so it survives a recheck
    # untouched - a recheck re-derives text, it does not re-select anything.
    rechecked = Manifest(max_filings_per_company=manifest.max_filings_per_company, entries=entries)
    write_manifest(rechecked, cache_root)
    return rechecked


# Bucket edges for the derived-text histogram, in characters. The real spread
# is wide: Apple's 2025 10-K derives to 207k characters and JPMorgan's 2025
# 10-K to 1.18M, a factor of six between two documents in the same corpus.
REPORT_BUCKETS = [0, 1, 1_000, 10_000, 50_000, 200_000, 500_000]


def corpus_report(manifest: Manifest) -> str:
    """Renders the derived-text size and markup ratio over every document held.

    Every document, not just the ones in the corpus, so a failure is visible
    next to the distribution it fell out of rather than only in a count.
    """
    measured = [entry for entry in manifest.entries if entry.text_chars is not None]
    lines = [
        f"max filings per company: {manifest.max_filings_per_company}",
        f"documents measured: {len(measured)}",
    ]

    if not measured:
        return "\n".join(lines + ["(nothing to report - run a fetch first)"])

    sizes = sorted(entry.text_chars for entry in measured if entry.text_chars is not None)
    lines.append(
        f"derived text chars: min {sizes[0]:,}  "
        f"median {statistics.median(sizes):,.0f}  max {sizes[-1]:,}"
    )

    # How much of the raw file is markup rather than reading matter. The
    # numerator is characters and the denominator bytes, which for these
    # near-ASCII filings differ by a rounding error; it is a shape check, not
    # an accounting identity.
    ratios = sorted(
        1 - entry.text_chars / entry.bytes
        for entry in measured
        if entry.text_chars is not None and entry.bytes
    )
    if ratios:
        lines.append(
            f"markup ratio: min {ratios[0]:.3f}  "
            f"median {statistics.median(ratios):.3f}  max {ratios[-1]:.3f}"
        )

    versions = sorted({entry.derivation_version for entry in measured if entry.derivation_version})
    lines.append(f"derivation version: {', '.join(str(version) for version in versions)}")
    lines.append("")
    lines.append("distribution of derived text chars:")

    for index, low in enumerate(REPORT_BUCKETS):
        high = REPORT_BUCKETS[index + 1] if index + 1 < len(REPORT_BUCKETS) else None
        count = sum(1 for size in sizes if size >= low and (high is None or size < high))
        label = (
            f"{low:,}" if high == low + 1 else (f"{low:,}-{high - 1:,}" if high else f"{low:,}+")
        )
        lines.append(f"  {label:>15}  {'#' * count} {count}")

    lines.append("")
    lines.append("dispositions:")
    for disposition in Disposition:
        rows = [entry for entry in manifest.entries if entry.disposition is disposition]
        if not rows:
            continue
        lines.append(f"  {disposition.value}: {len(rows)}")
        counts: dict[str, int] = {}
        for entry in rows:
            counts[entry.reason.value] = counts.get(entry.reason.value, 0) + 1
        for reason, count in sorted(counts.items()):
            lines.append(f"    {reason}: {count}")

    return "\n".join(lines)

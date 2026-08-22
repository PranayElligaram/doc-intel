"""Fetches the filing corpus from Companies House and records what it did.

Three ideas hold this module together.

The corpus is the PDF rendition, and only the PDF rendition. Companies House
also publishes an iXBRL rendition of the same accounts, which contains the
tagged figures we are trying to extract. That makes it a perfect answer key and
a catastrophic input: an extractor fed iXBRL is not doing document
understanding, it is reading a database. So iXBRL goes to a separate directory
under `oracle/`, is never written under `documents/`, and a filing that has no
PDF rendition is dropped from the corpus entirely rather than kept with its
iXBRL de-tagged. De-tagging preserves a layout the extractor would never meet
in the wild, which is a subtler version of the same cheat.

The manifest records every candidate we considered, not just the ones we kept.
A corpus you cannot interrogate is a corpus you cannot defend, and "why is that
filing not in here" is the first question anyone asks. Rejections carry a
`reason` from a fixed enum rather than a free-text string, so they can be
counted.

Disposition and last_action are separate fields. `disposition` describes the
document and survives across runs; `last_action` describes what this particular
run did. Collapsed into one field, the second run overwrites every `fetched`
with `skipped_existing` and the manifest can no longer answer "what is in my
corpus", which is the question every later phase asks it.

TEXT LAYER: this module measures and never excludes. Some filings are scans
with no extractable text, and they have to be found and quantified before a
sensible threshold can be chosen - so `analyse_pdf` runs over every PDF we hold
bytes for, the measurements land in the manifest whatever the disposition, and
`--report` prints the distribution. Applying a threshold is a later phase and a
manifest query, not a refetch. The only PDF that fails here is one pypdf cannot
open at all, which is corruption rather than a text-layer judgement.
"""

import hashlib
import io
import json
import os
import random
import re
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

# Eight characters: eight digits, or a two-letter jurisdiction prefix (SC, NI,
# OC, ...) followed by six digits.
COMPANY_NUMBER = re.compile(r"^(?:\d{8}|[A-Z]{2}\d{6})$")

PDF_MIME = "application/pdf"
IXBRL_MIME = "application/xhtml+xml"

# Breadth across companies beats depth within one. The same company filing the
# same annual-report template ten years running gives ten documents and one
# layout: an extractor that learns that template scores well on the corpus and
# generalises to nothing. Two per company keeps a little year-on-year variation
# without letting any one filer dominate.
DEFAULT_MAX_FILINGS_PER_COMPANY = 2

MIN_REQUEST_INTERVAL = 0.6
MAX_ATTEMPTS = 4
BASE_BACKOFF = 1.0
MAX_BACKOFF = 30.0


class Disposition(StrEnum):
    """What this document is to us. Stable across runs."""

    IN_CORPUS = "in_corpus"
    EXCLUDED = "excluded"
    FAILED = "failed"
    ORACLE_ONLY = "oracle_only"


class LastAction(StrEnum):
    """What this run did about it. Overwritten every run.

    A candidate with no bytes to download - an exclusion, or a company we could
    not resolve - is recorded as `fetched`, because the run still spent the
    metadata request that produced the verdict.
    """

    FETCHED = "fetched"
    SKIPPED_EXISTING = "skipped_existing"
    RECHECKED = "rechecked"


class Reason(StrEnum):
    """Why a candidate ended up with its disposition.

    A closed enum rather than free text so exclusions can be counted. "17
    filings excluded, 15 of them no_pdf_rendition" is a fact about the corpus;
    a bag of sentences is not.
    """

    OK = "ok"
    ORACLE_RENDITION = "oracle_rendition"
    NO_PDF_RENDITION = "no_pdf_rendition"
    NO_ACCOUNTS_FILINGS = "no_accounts_filings"
    COMPANY_NOT_FOUND = "company_not_found"
    HTTP_ERROR = "http_error"
    NOT_A_PDF = "not_a_pdf"
    UNREADABLE_PDF = "unreadable_pdf"
    MISSING_ON_DISK = "missing_on_disk"


class TextLayer(BaseModel):
    """How much extractable text a PDF actually has."""

    model_config = ConfigDict(extra="forbid")

    page_count: int = Field(ge=0)
    chars_total: int = Field(ge=0)
    # Median, not mean. One text-heavy cover page on an otherwise scanned
    # filing drags a mean over any threshold you would want to set.
    chars_per_page_median: float = Field(ge=0)


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(min_length=1)
    company_number: str = Field(min_length=1)
    transaction_id: str | None = None
    filing_date: str | None = None
    accounts_type: str | None = None

    disposition: Disposition
    last_action: LastAction
    reason: Reason

    mime: str | None = None
    sha256: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    text_layer: TextLayer | None = None
    path: str | None = None
    fetched_at: datetime | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    # The parameters the corpus was built with, recorded so the manifest is
    # self-describing. Filings beyond the cap are not candidates and get no
    # row - writing an exclusion for every historical filing we never looked
    # at would bury the real rejects under noise. So the cap has to be stated
    # here, or there would be no way to tell a corpus built with a cap of 2
    # from one where those companies only ever filed twice.
    max_filings_per_company: int = Field(default=DEFAULT_MAX_FILINGS_PER_COMPANY, ge=1)
    entries: list[ManifestEntry] = Field(default_factory=list)

    def by_id(self) -> dict[str, ManifestEntry]:
        return {entry.entry_id: entry for entry in self.entries}


def manifest_path(cache_root: Path) -> Path:
    return cache_root / "manifest.json"


def documents_dir(cache_root: Path) -> Path:
    return cache_root / "documents"


def oracle_dir(cache_root: Path) -> Path:
    return cache_root / "oracle"


def read_company_numbers(path: Path) -> list[str]:
    """Reads the committed corpus definition.

    The corpus is a committed list, not an API search run at fetch time. A
    search returns different companies to different people on different days,
    and a corpus that cannot be reproduced cannot support a number in a README.
    """
    numbers: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        number = line.upper()
        if not COMPANY_NUMBER.match(number):
            raise ValueError(f"{path}:{lineno}: {line!r} is not a company number")
        if number in numbers:
            raise ValueError(f"{path}:{lineno}: {number} is listed twice")
        numbers.append(number)
    return numbers


def load_manifest(cache_root: Path) -> Manifest:
    path = manifest_path(cache_root)
    if not path.exists():
        return Manifest()
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(manifest: Manifest, cache_root: Path) -> Path:
    """Writes the manifest atomically, with a stable byte-for-byte layout.

    Sorted entries and sorted keys mean a run that changed nothing produces an
    identical file, so a diff on the manifest shows corpus changes rather than
    dictionary ordering noise.
    """
    ordered = manifest.model_copy(
        update={"entries": sorted(manifest.entries, key=lambda entry: entry.entry_id)}
    )
    payload = json.dumps(ordered.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    path = manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sibling and renamed: an interrupted run leaves the previous
    # manifest intact rather than a half-written one, and the corpus on disk
    # stays describable.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


def analyse_pdf(data: bytes) -> TextLayer:
    """Measures the text layer. Raises if the bytes are not a readable PDF."""
    reader = PdfReader(io.BytesIO(data))
    per_page = [len(page.extract_text()) for page in reader.pages]
    if not per_page:
        return TextLayer(page_count=0, chars_total=0, chars_per_page_median=0.0)
    return TextLayer(
        page_count=len(per_page),
        chars_total=sum(per_page),
        chars_per_page_median=float(statistics.median(per_page)),
    )


@dataclass
class _Pacer:
    """Serialises requests: a minimum gap, then retries with bounded backoff.

    The clock, the sleep and the jitter are injected so the tests exercise the
    real retry path at zero wall-clock cost. A test that actually slept would
    either be slow or would quietly stop testing the backoff.
    """

    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    rand: Callable[[], float]
    min_interval: float = MIN_REQUEST_INTERVAL
    max_attempts: int = MAX_ATTEMPTS
    last_request_at: float | None = None

    def get(
        self,
        client: httpx.Client,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        authed: bool = True,
    ) -> httpx.Response:
        last: httpx.Response | None = None
        for attempt in range(self.max_attempts):
            self._wait_turn()
            request = client.build_request("GET", url, headers=headers)
            # auth=None disables the client-level credential for this request.
            # See _download for why the second hop of a redirect must not carry
            # it.
            response = client.send(request) if authed else client.send(request, auth=None)
            self.last_request_at = self.monotonic()

            if response.status_code != 429 and response.status_code < 500:
                return response
            last = response
            if attempt == self.max_attempts - 1:
                break
            self.sleep(self._retry_delay(response, attempt))

        assert last is not None
        return last

    def _wait_turn(self) -> None:
        if self.last_request_at is None:
            return
        elapsed = self.monotonic() - self.last_request_at
        if elapsed < self.min_interval:
            self.sleep(self.min_interval - elapsed)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("retry-after")
        if header is not None:
            try:
                # Retry-After may also be an HTTP-date. We do not parse that
                # form: falling through to backoff is safe, whereas a
                # mis-parsed date could produce a delay of hours.
                return max(0.0, float(int(header)))
            except ValueError:
                pass
        # Exponential, jittered so that a batch of retries does not resynchronise
        # into a thundering herd, and capped so one bad response cannot park the
        # run for an unbounded time.
        backoff = min(BASE_BACKOFF * (2.0**attempt), MAX_BACKOFF)
        return backoff * (0.5 + self.rand())


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: Path, cache_root: Path) -> str:
    return path.relative_to(cache_root).as_posix()


def _download(pacer: _Pacer, client: httpx.Client, url: str, accept: str) -> httpx.Response:
    """Fetches a document, handling the redirect to the signed URL ourselves.

    UNVERIFIED AGAINST THE LIVE API. The document endpoint answers 302 with a
    Location pointing at a separate, pre-signed host. We follow it by hand -
    the client is built with follow_redirects=False - and drop the
    Authorization header on the second hop, on the reasoning that a pre-signed
    URL carries its own credential, that sending Companies House Basic auth to
    a third-party storage host leaks it needlessly, and that some signed-URL
    implementations reject a request bearing an unexpected Authorization
    header. httpx's own redirect following would forward the header to the new
    host, which is why we do not use it.

    The tests below prove this is what we send. They do not prove it is what
    the live API wants: that stays an assumption until someone runs this with a
    real key.
    """
    response = pacer.get(client, url, headers={"Accept": accept})
    if response.is_redirect:
        location = response.headers.get("location")
        if location:
            response = pacer.get(client, location, headers={"Accept": accept}, authed=False)
    return response


def _describe_bytes(
    entry_id: str,
    company_number: str,
    item: dict[str, Any],
    data: bytes,
    path: Path,
    cache_root: Path,
    *,
    last_action: LastAction,
    fetched_at: datetime | None,
) -> ManifestEntry:
    """Builds the manifest row for PDF bytes we hold, measuring the text layer.

    Note what is *not* here: any test on how much text there was. Zero
    extractable characters is a recorded measurement, not an exclusion.
    """
    text_layer: TextLayer | None = None

    if not data.startswith(b"%PDF-"):
        disposition, reason = Disposition.FAILED, Reason.NOT_A_PDF
    else:
        try:
            text_layer = analyse_pdf(data)
        except Exception:
            # A malformed PDF surfaces from the parser as almost anything, and
            # one bad filing must not abort a corpus run. The bytes stay on
            # disk and the manifest says why they are not usable.
            disposition, reason = Disposition.FAILED, Reason.UNREADABLE_PDF
        else:
            disposition, reason = Disposition.IN_CORPUS, Reason.OK

    return ManifestEntry(
        entry_id=entry_id,
        company_number=company_number,
        transaction_id=item.get("transaction_id"),
        filing_date=item.get("date"),
        accounts_type=item.get("description"),
        disposition=disposition,
        last_action=last_action,
        reason=reason,
        mime=PDF_MIME,
        sha256=_sha256(data),
        bytes=len(data),
        text_layer=text_layer,
        path=_relative(path, cache_root),
        fetched_at=fetched_at,
    )


def fetch_corpus(
    client: httpx.Client,
    *,
    cache_root: Path,
    company_numbers: Sequence[str],
    max_filings_per_company: int = DEFAULT_MAX_FILINGS_PER_COMPANY,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rand: Callable[[], float] = random.random,
    now: Callable[[], datetime] = _utc_now,
) -> Manifest:
    """Fetches accounts filings for each company and returns the new manifest.

    The filing history is re-read every run - it is how we learn the candidate
    list, and companies file new accounts. Only document *bytes* are skipped
    when they are already on disk; `force` is the only path that redownloads.

    `max_filings_per_company` caps how many of a company's accounts filings
    become candidates at all. It is recorded on the returned manifest, because
    a corpus you cannot describe the parameters of is a corpus you cannot
    reproduce.
    """
    pacer = _Pacer(sleep=sleep, monotonic=monotonic, rand=rand)
    previous = load_manifest(cache_root).by_id()
    entries: list[ManifestEntry] = []

    for company_number in company_numbers:
        for entry in _fetch_company(
            pacer,
            client,
            cache_root=cache_root,
            company_number=company_number,
            max_filings_per_company=max_filings_per_company,
            force=force,
            previous=previous,
            now=now,
        ):
            entries.append(entry)

    manifest = Manifest(max_filings_per_company=max_filings_per_company, entries=entries)
    write_manifest(manifest, cache_root)
    return manifest


def _fetch_company(
    pacer: _Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    company_number: str,
    max_filings_per_company: int,
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> Iterator[ManifestEntry]:
    history = pacer.get(
        client,
        f"/company/{company_number}/filing-history?category=accounts",
        headers={"Accept": "application/json"},
    )
    if history.status_code == 404:
        yield _barren(company_number, Reason.COMPANY_NOT_FOUND)
        return
    if history.status_code != 200:
        yield _barren(company_number, Reason.HTTP_ERROR)
        return

    items = [
        item
        for item in history.json().get("items", [])
        if item.get("category") in (None, "accounts")
    ]
    # Most recent first, ties broken on transaction id. Sorting on the date
    # alone would leave same-day filings in whatever order the API happened to
    # return them, so the cap could take a different pair on a later run and
    # the corpus would quietly drift. The tie-break is what makes the selection
    # reproducible, not the date.
    items.sort(
        key=lambda item: (str(item.get("date", "")), str(item.get("transaction_id", ""))),
        reverse=True,
    )
    # Filings past the cap are not candidates. They get no manifest row: a
    # rejection is something we looked at and turned down, and burying those
    # under every historical filing a company ever made would make the
    # rejections unreadable. The cap itself is recorded on the manifest.
    items = items[:max_filings_per_company]

    if not items:
        yield _barren(company_number, Reason.NO_ACCOUNTS_FILINGS)
        return

    for item in items:
        yield from _fetch_filing(
            pacer,
            client,
            cache_root=cache_root,
            company_number=company_number,
            item=item,
            force=force,
            previous=previous,
            now=now,
        )


def _barren(company_number: str, reason: Reason) -> ManifestEntry:
    """A row for a company that produced no candidate documents at all."""
    disposition = (
        Disposition.EXCLUDED if reason is Reason.NO_ACCOUNTS_FILINGS else Disposition.FAILED
    )
    return ManifestEntry(
        entry_id=f"{company_number}/-/-",
        company_number=company_number,
        disposition=disposition,
        last_action=LastAction.FETCHED,
        reason=reason,
    )


def _fetch_filing(
    pacer: _Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    company_number: str,
    item: dict[str, Any],
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> Iterator[ManifestEntry]:
    transaction_id = str(item.get("transaction_id", "")) or "-"
    pdf_id = f"{company_number}/{transaction_id}/pdf"

    metadata_url = item.get("links", {}).get("document_metadata")
    if not metadata_url:
        yield _no_pdf(pdf_id, company_number, item)
        return

    metadata = pacer.get(client, metadata_url, headers={"Accept": "application/json"})
    if metadata.status_code != 200:
        yield _filing_failure(pdf_id, company_number, item, Reason.HTTP_ERROR)
        return

    body = metadata.json()
    resources = body.get("resources", {})
    document_url = body.get("links", {}).get("document")

    if PDF_MIME not in resources or not document_url:
        # No PDF means the whole filing leaves the corpus, iXBRL included.
        # Keeping the iXBRL as an oracle for a document we do not have would
        # give us an answer key with nothing to answer about.
        yield _no_pdf(pdf_id, company_number, item)
        return

    yield _fetch_pdf(
        pacer,
        client,
        cache_root=cache_root,
        company_number=company_number,
        item=item,
        entry_id=pdf_id,
        document_url=document_url,
        force=force,
        previous=previous,
        now=now,
    )

    if IXBRL_MIME in resources:
        yield _fetch_oracle(
            pacer,
            client,
            cache_root=cache_root,
            company_number=company_number,
            item=item,
            entry_id=f"{company_number}/{transaction_id}/ixbrl",
            document_url=document_url,
            force=force,
            previous=previous,
            now=now,
        )


def _no_pdf(entry_id: str, company_number: str, item: dict[str, Any]) -> ManifestEntry:
    return _filing_failure(entry_id, company_number, item, Reason.NO_PDF_RENDITION)


def _filing_failure(
    entry_id: str, company_number: str, item: dict[str, Any], reason: Reason
) -> ManifestEntry:
    disposition = Disposition.EXCLUDED if reason is Reason.NO_PDF_RENDITION else Disposition.FAILED
    return ManifestEntry(
        entry_id=entry_id,
        company_number=company_number,
        transaction_id=item.get("transaction_id"),
        filing_date=item.get("date"),
        accounts_type=item.get("description"),
        disposition=disposition,
        last_action=LastAction.FETCHED,
        reason=reason,
    )


def _fetch_pdf(
    pacer: _Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    company_number: str,
    item: dict[str, Any],
    entry_id: str,
    document_url: str,
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> ManifestEntry:
    transaction_id = str(item.get("transaction_id", "")) or "-"
    path = documents_dir(cache_root) / company_number / f"{transaction_id}.pdf"

    if path.exists() and not force:
        # Re-derived from the bytes on disk rather than copied from the old
        # manifest, so the manifest always describes what is actually there.
        # fetched_at is the one thing that cannot be re-derived, so it is
        # carried over.
        return _describe_bytes(
            entry_id,
            company_number,
            item,
            path.read_bytes(),
            path,
            cache_root,
            last_action=LastAction.SKIPPED_EXISTING,
            fetched_at=_previous_fetched_at(previous, entry_id),
        )

    response = _download(pacer, client, document_url, PDF_MIME)
    if response.status_code != 200:
        return _filing_failure(entry_id, company_number, item, Reason.HTTP_ERROR)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return _describe_bytes(
        entry_id,
        company_number,
        item,
        response.content,
        path,
        cache_root,
        last_action=LastAction.FETCHED,
        fetched_at=now(),
    )


def _fetch_oracle(
    pacer: _Pacer,
    client: httpx.Client,
    *,
    cache_root: Path,
    company_number: str,
    item: dict[str, Any],
    entry_id: str,
    document_url: str,
    force: bool,
    previous: dict[str, ManifestEntry],
    now: Callable[[], datetime],
) -> ManifestEntry:
    transaction_id = str(item.get("transaction_id", "")) or "-"
    path = oracle_dir(cache_root) / company_number / f"{transaction_id}.xhtml"

    if path.exists() and not force:
        data = path.read_bytes()
        return _oracle_entry(
            entry_id,
            company_number,
            item,
            data,
            path,
            cache_root,
            LastAction.SKIPPED_EXISTING,
            _previous_fetched_at(previous, entry_id),
        )

    response = _download(pacer, client, document_url, IXBRL_MIME)
    if response.status_code != 200:
        return _filing_failure(entry_id, company_number, item, Reason.HTTP_ERROR)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return _oracle_entry(
        entry_id,
        company_number,
        item,
        response.content,
        path,
        cache_root,
        LastAction.FETCHED,
        now(),
    )


def _oracle_entry(
    entry_id: str,
    company_number: str,
    item: dict[str, Any],
    data: bytes,
    path: Path,
    cache_root: Path,
    last_action: LastAction,
    fetched_at: datetime | None,
) -> ManifestEntry:
    return ManifestEntry(
        entry_id=entry_id,
        company_number=company_number,
        transaction_id=item.get("transaction_id"),
        filing_date=item.get("date"),
        accounts_type=item.get("description"),
        disposition=Disposition.ORACLE_ONLY,
        last_action=last_action,
        reason=Reason.ORACLE_RENDITION,
        mime=IXBRL_MIME,
        sha256=_sha256(data),
        bytes=len(data),
        path=_relative(path, cache_root),
        fetched_at=fetched_at,
    )


def _previous_fetched_at(previous: dict[str, ManifestEntry], entry_id: str) -> datetime | None:
    entry = previous.get(entry_id)
    return entry.fetched_at if entry else None


def recheck_corpus(cache_root: Path) -> Manifest:
    """Re-measures the bytes already on disk. Makes no network calls at all.

    Separate from `fetch_corpus` rather than a flag through it, because the
    distinguishing property is the absence of a client: this is the path you
    run on a plane, or after changing `analyse_pdf`, to refresh every text
    layer measurement without touching Companies House.
    """
    manifest = load_manifest(cache_root)
    entries: list[ManifestEntry] = []

    for entry in manifest.entries:
        if entry.path is None:
            entries.append(entry.model_copy(update={"last_action": LastAction.RECHECKED}))
            continue

        path = cache_root / entry.path
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
        if entry.mime == IXBRL_MIME:
            entries.append(
                entry.model_copy(
                    update={
                        "last_action": LastAction.RECHECKED,
                        "sha256": _sha256(data),
                        "bytes": len(data),
                    }
                )
            )
            continue

        item = {
            "transaction_id": entry.transaction_id,
            "date": entry.filing_date,
            "description": entry.accounts_type,
        }
        entries.append(
            _describe_bytes(
                entry.entry_id,
                entry.company_number,
                item,
                data,
                path,
                cache_root,
                last_action=LastAction.RECHECKED,
                fetched_at=entry.fetched_at,
            )
        )

    # The cap describes how the corpus was selected, so it survives a recheck
    # untouched - a recheck re-measures bytes, it does not re-select anything.
    rechecked = Manifest(max_filings_per_company=manifest.max_filings_per_company, entries=entries)
    write_manifest(rechecked, cache_root)
    return rechecked


# Bucket edges for the text-layer histogram, in median characters per page.
# Deliberately not a threshold: this is the shape of the data you look at
# before choosing one.
REPORT_BUCKETS = [0, 1, 50, 200, 500, 1500]


def text_layer_report(manifest: Manifest) -> str:
    """Renders the measured text-layer distribution over every PDF we hold.

    Every PDF, not just the ones in the corpus, so that a threshold chosen from
    this report can be applied later without refetching anything.
    """
    measured = [entry for entry in manifest.entries if entry.text_layer is not None]
    lines = [
        f"max filings per company: {manifest.max_filings_per_company}",
        f"PDFs measured: {len(measured)}",
    ]

    if not measured:
        return "\n".join(lines + ["(nothing to report - run a fetch first)"])

    medians = sorted(
        entry.text_layer.chars_per_page_median for entry in measured if entry.text_layer
    )
    lines.append(
        f"median chars/page: min {medians[0]:.0f}  "
        f"median {statistics.median(medians):.0f}  max {medians[-1]:.0f}"
    )
    lines.append("")
    lines.append("distribution of median chars/page:")

    for index, low in enumerate(REPORT_BUCKETS):
        high = REPORT_BUCKETS[index + 1] if index + 1 < len(REPORT_BUCKETS) else None
        count = sum(1 for value in medians if value >= low and (high is None or value < high))
        label = f"{low}" if high == low + 1 else (f"{low}-{high - 1}" if high else f"{low}+")
        lines.append(f"  {label:>10}  {'#' * count} {count}")

    lines.append("")
    lines.append("dispositions:")
    for disposition in Disposition:
        rows = [entry for entry in manifest.entries if entry.disposition is disposition]
        if not rows:
            continue
        lines.append(f"  {disposition.value}: {len(rows)}")
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.reason.value] = counts.get(row.reason.value, 0) + 1
        for reason, count in sorted(counts.items()):
            lines.append(f"    {reason}: {count}")

    return "\n".join(lines)

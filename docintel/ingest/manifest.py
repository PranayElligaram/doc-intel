"""What a fetched corpus is made of, and how it is written to disk.

Nothing here knows where the documents came from. Companies House, EDGAR or a
directory someone dropped on the disk all produce the same rows: a disposition,
what the last run did about it, a reason drawn from a closed enum, and enough
bytes-level detail to re-derive the rest. The source-specific fetching lives in
its own module and imports this one.

Disposition and last_action are separate fields. `disposition` describes the
document and survives across runs; `last_action` describes what this particular
run did. Collapsed into one field, the second run overwrites every `fetched`
with `skipped_existing` and the manifest can no longer answer "what is in my
corpus", which is the question every later phase asks it.

The manifest records every candidate that was considered, not just the ones
that were kept. A corpus you cannot interrogate is a corpus you cannot defend,
and "why is that filing not in here" is the first question anyone asks.
"""

import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Breadth across companies beats depth within one. The same company filing the
# same annual-report template ten years running gives ten documents and one
# layout: an extractor that learns that template scores well on the corpus and
# generalises to nothing. Two per company keeps a little year-on-year variation
# without letting any one filer dominate.
DEFAULT_MAX_FILINGS_PER_COMPANY = 2


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

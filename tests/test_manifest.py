"""Unit tests for the manifest itself: what it holds and how it is written.

Nothing here fetches anything. The manifest is the source-agnostic half of
ingest, so these tests build rows by hand rather than going through any
particular API's fetcher.
"""

from datetime import UTC, datetime
from pathlib import Path

from docintel.ingest.manifest import (
    Disposition,
    LastAction,
    Manifest,
    ManifestEntry,
    Reason,
    load_manifest,
    manifest_path,
    write_manifest,
)


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


def test_manifest_write_is_byte_identical_for_the_same_content(tmp_path: Path) -> None:
    manifest = Manifest(entries=[entry("00445790/B/pdf"), entry("00445790/A/pdf")])

    write_manifest(manifest, tmp_path)
    first = manifest_path(tmp_path).read_bytes()
    write_manifest(manifest, tmp_path)
    second = manifest_path(tmp_path).read_bytes()

    assert first == second


def test_manifest_entries_are_sorted_regardless_of_insertion_order(tmp_path: Path) -> None:
    """So a diff on the manifest shows corpus changes, not dict ordering."""
    forwards = Manifest(entries=[entry("00445790/A/pdf"), entry("00445790/B/pdf")])
    backwards = Manifest(entries=[entry("00445790/B/pdf"), entry("00445790/A/pdf")])

    write_manifest(forwards, tmp_path)
    one = manifest_path(tmp_path).read_bytes()
    write_manifest(backwards, tmp_path)
    two = manifest_path(tmp_path).read_bytes()

    assert one == two


def test_manifest_round_trips(tmp_path: Path) -> None:
    fetched_at = datetime(2026, 8, 22, 9, 30, tzinfo=UTC)
    manifest = Manifest(entries=[entry("00445790/A/pdf", fetched_at=fetched_at, bytes=1539)])

    write_manifest(manifest, tmp_path)

    reloaded = load_manifest(tmp_path)
    assert reloaded.entries[0].fetched_at == fetched_at
    assert reloaded.entries[0].bytes == 1539


def test_missing_manifest_loads_as_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path).entries == []


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    write_manifest(Manifest(entries=[entry("00445790/A/pdf")]), tmp_path)

    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

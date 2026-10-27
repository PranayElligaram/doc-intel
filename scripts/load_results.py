"""Import verified local evaluation reports into PostgreSQL, idempotently.

Usage: python scripts/load_results.py --report eval/results/20261026T183413Z.json

Never reads SEC companyfacts (oracle) or generates gold labels.
"""

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from docintel.gold import corpus_entry, corpus_manifest, document_text

ROOT = Path(__file__).resolve().parent.parent


def doc_span(
    entry_id: str, prediction: dict[str, Any], derived_text: str
) -> tuple[str, int, int, str]:
    """Convert a chunk-relative quote to a validated document-coordinate span."""
    citation = prediction["citation"]
    chunk_id = str(citation["chunk_id"])
    source, separator, origin_text = chunk_id.rpartition("@")
    if not separator or source != entry_id or not origin_text.isdecimal():
        raise ValueError(f"invalid chunk reference for {entry_id}: {chunk_id}")
    origin = int(origin_text)
    start = origin + int(citation["start"])
    end = origin + int(citation["end"])
    quote = str(citation["quote"])
    if start < 0 or end <= start or derived_text[start:end] != quote:
        raise ValueError(f"citation does not match cached text for {entry_id}")
    return chunk_id, start, end, quote


def load_report(conn: psycopg.Connection[Any], report_path: Path, cache_root: Path) -> int | None:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report.get("cases"), list) or not isinstance(report.get("metrics"), dict):
        raise ValueError("evaluation JSON is missing cases or metrics")
    digest = hashlib.sha256(raw).hexdigest()
    manifest = corpus_manifest(cache_root)

    with conn.transaction():
        run = conn.execute(
            """INSERT INTO evaluation_runs
            (report_sha256,model_name,top_k,started_at,completed_at,gold_sha256,scope_warning,metrics)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (report_sha256) DO NOTHING
            RETURNING id""",
            (
                digest,
                report["model"],
                report["top_k"],
                report["started_utc"],
                report["finished_utc"],
                report["gold_file_sha256"],
                report["scope_warning"],
                Jsonb(report["metrics"]),
            ),
        ).fetchone()
        if run is None:
            return None
        run_id = int(run[0])
        document_cache: dict[str, str] = {}
        for case in report["cases"]:
            entry_id = str(case["entry_id"])
            if entry_id not in document_cache:
                entry = corpus_entry(manifest, entry_id)
                text = document_text(cache_root, entry)  # validates SHA-256
                if entry.text_sha256 is None or entry.derivation_version is None:
                    raise ValueError(f"incomplete corpus manifest row: {entry_id}")
                filing = date.fromisoformat(entry.filing_date) if entry.filing_date else None
                conn.execute(
                    """INSERT INTO documents
                    (entry_id,company_id,filing_date,derived_text,text_sha256,derivation_version)
                    VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(entry_id) DO NOTHING""",
                    (
                        entry_id,
                        entry.company_id,
                        filing,
                        text,
                        entry.text_sha256,
                        entry.derivation_version,
                    ),
                )
                existing = conn.execute(
                    "SELECT text_sha256, derivation_version FROM documents WHERE entry_id=%s",
                    (entry_id,),
                ).fetchone()
                if (
                    existing is None
                    or existing[0].strip() != entry.text_sha256
                    or existing[1] != entry.derivation_version
                ):
                    raise ValueError(f"stored document hash or derivation changed: {entry_id}")
                document_cache[entry_id] = text

            prediction = case["prediction"]
            status = str(prediction["status"])
            citation_id: int | None = None
            if status == "found":
                chunk_id, start, end, quote = doc_span(
                    entry_id, prediction, document_cache[entry_id]
                )
                result = conn.execute(
                    """INSERT INTO citations
                    (document_entry_id,chunk_id,start_offset,end_offset,quote)
                    VALUES (%s,%s,%s,%s,%s) RETURNING id""",
                    (entry_id, chunk_id, start, end, quote),
                ).fetchone()
                assert result is not None
                citation_id = int(result[0])
            else:
                if status not in ("failed", "not_stated"):
                    raise ValueError(f"invalid prediction status: {status}")
            value = Jsonb(prediction["value"]) if status == "found" else None
            error = prediction.get("error") if status == "failed" else None
            extraction = conn.execute(
                """INSERT INTO extractions
                (document_entry_id,evaluation_run_id,field_name,status,value_json,error,citation_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (entry_id, run_id, case["field"], status, value, error, citation_id),
            ).fetchone()
            assert extraction is not None
            conn.execute(
                """INSERT INTO evaluation_cases
                (extraction_id,gold_status,value_correct,citation_grounded,gold_span_aligned,
                 gold_in_retrieved_context,outcome,failure_attribution)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    extraction[0],
                    case["gold_status"],
                    case["value_correct"],
                    case["citation_grounded"],
                    case["gold_span_aligned"],
                    case["gold_in_retrieved_context"],
                    case["outcome"],
                    case["failure_attribution"],
                ),
            )
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    password = os.getenv("DOCINTEL_POSTGRES_PASSWORD")
    if not password:
        parser.error("DOCINTEL_POSTGRES_PASSWORD missing from .env")
    with psycopg.connect(
        host=os.getenv("DOCINTEL_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DOCINTEL_DB_PORT", "5433")),
        dbname=os.getenv("DOCINTEL_DB_NAME", "docintel"),
        user=os.getenv("DOCINTEL_DB_USER", "docintel"),
        password=password,
    ) as conn:
        result = load_report(conn, args.report, args.cache_root)
    if result is None:
        print("Report already imported. No duplicate records created.")
    else:
        print(f"Imported evaluation run ID {result} from {args.report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

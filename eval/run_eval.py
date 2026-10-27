"""Run local Ollama extraction only on verified hand-labelled (document, field) pairs.

Example:
    python eval/run_eval.py --model qwen2.5:7b-instruct

A partial gold set produces partial results. This script never fills in absent
labels, uses SEC companyfacts as a prediction source, or claims corpus-wide F1.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from docintel.evaluation import score_case, summarise
from docintel.extraction.pipeline import DEFAULT_TOP_K, extract_field, retrieve_field_hits
from docintel.gold import (
    GoldField,
    GoldSet,
    corpus_entry,
    corpus_manifest,
    document_text,
    load_gold_set,
    resolve_entry_id,
    verify_labels,
)
from docintel.llm.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaProvider
from docintel.retrieval.chunker import chunk_document, verify_chunks
from docintel.retrieval.retriever import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--gold-root", type=Path, default=ROOT / "data" / "gold")
    parser.add_argument("--entry-id", help="limit to one filing")
    parser.add_argument("--field", choices=[field.value for field in GoldField])
    parser.add_argument(
        "--output", type=Path, help="JSON result path (default: eval/results/<UTC time>.json)"
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    manifest = corpus_manifest(args.cache_root)
    gold_set = load_gold_set(args.gold_root)
    entry_id = resolve_entry_id(manifest, args.entry_id) if args.entry_id else None
    labels = sorted(
        (
            label
            for label in gold_set.labels
            if (entry_id is None or label.entry_id == entry_id)
            and (args.field is None or label.field.value == args.field)
        ),
        key=lambda label: (label.entry_id, label.field.value),
    )
    if not labels:
        parser.error("no hand-labelled (document, field) pairs match the selection")

    problems = verify_labels(args.cache_root, GoldSet(labels=labels))
    if problems:
        raise ValueError("Gold labels failed integrity checks:\n" + "\n".join(problems))

    started = datetime.now(UTC)
    cases = []
    with httpx.Client(base_url=args.base_url) as client:
        provider = OllamaProvider(client, model=args.model)
        for document_id in sorted({label.entry_id for label in labels}):
            entry = corpus_entry(manifest, document_id)
            text = document_text(args.cache_root, entry)
            chunks = chunk_document(text, document_id=document_id)
            issues = verify_chunks(text, chunks)
            if issues:
                raise ValueError(f"chunk integrity failed for {document_id}: {issues}")
            retriever = BM25Retriever(chunks)
            for label in (label for label in labels if label.entry_id == document_id):
                hits = retrieve_field_hits(label.field, retriever=retriever, top_k=args.top_k)
                prediction = extract_field(
                    label.field, retriever=retriever, provider=provider, top_k=args.top_k
                )
                record = score_case(label, prediction, text=text, chunks=chunks, retrieved=hits)
                cases.append(record)
                print(
                    f"{document_id} {label.field.value}: {record.outcome} "
                    f"({record.failure_attribution})",
                    flush=True,
                )

    finished = datetime.now(UTC)
    result = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "model": args.model,
        "top_k": args.top_k,
        "gold_file_sha256": hashlib.sha256(
            (args.gold_root / "labels.json").read_bytes()
        ).hexdigest(),
        "scope_warning": (
            "Only manually labelled pairs were evaluated; no unlabelled pairs "
            "appear in the denominator. Do not claim corpus-wide accuracy."
        ),
        "metrics": summarise(cases),
        "cases": [case.to_dict() for case in cases],
    }
    path = args.output or ROOT / "eval" / "results" / f"{started:%Y%m%dT%H%M%SZ}.json"
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvaluated {len(cases)} hand-labelled pairs. Results: {path}")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

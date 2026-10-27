"""Run one local Ollama extraction against a cached SEC filing.

This prints unscored predictions, not evaluation metrics. Gold labels and
machine-readable company facts are not read by this script.
"""

import argparse
from pathlib import Path

import httpx

from docintel.extraction.pipeline import DEFAULT_TOP_K, extract_document, extract_field
from docintel.gold import (
    GoldField,
    corpus_entry,
    corpus_manifest,
    document_text,
    resolve_entry_id,
)
from docintel.llm.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaProvider
from docintel.retrieval.chunker import chunk_document, verify_chunks
from docintel.retrieval.retriever import BM25Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "data" / "cache")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--field",
        choices=[field.value for field in GoldField],
        help="extract only one field for a quick model smoke check",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    manifest = corpus_manifest(args.cache_root)
    entry_id = resolve_entry_id(manifest, args.entry_id)
    text = document_text(args.cache_root, corpus_entry(manifest, entry_id))
    with httpx.Client(base_url=args.base_url) as client:
        provider = OllamaProvider(client, model=args.model)
        if args.field:
            field = GoldField(args.field)
            chunks = chunk_document(text, document_id=entry_id)
            problems = verify_chunks(text, chunks)
            if problems:
                raise ValueError(f"chunk integrity failure: {problems}")
            result = extract_field(
                field,
                retriever=BM25Retriever(chunks),
                provider=provider,
                top_k=args.top_k,
            )
            print(f"document: {entry_id}; model: {provider.name}; field: {field.value}")
            print(result.model_dump_json(indent=2))
        else:
            result_doc = extract_document(
                text, document_id=entry_id, provider=provider, top_k=args.top_k
            )
            print(f"model: {provider.name}")
            print(result_doc.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

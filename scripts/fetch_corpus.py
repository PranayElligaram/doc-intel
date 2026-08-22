"""CLI wrapper around docintel.ingest.corpus.

Deliberately thin. Everything worth testing lives in the module; this file is
argparse, credentials and printing, so that the test suite never has to drive a
command line to reach the logic.
"""

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from docintel.ingest.corpus import (
    DEFAULT_MAX_FILINGS_PER_COMPANY,
    fetch_corpus,
    load_manifest,
    read_company_numbers,
    recheck_corpus,
    text_layer_report,
)

API_BASE = "https://api.company-information.service.gov.uk"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPANIES = REPO_ROOT / "data" / "companies.txt"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--max-filings-per-company",
        type=int,
        default=DEFAULT_MAX_FILINGS_PER_COMPANY,
        help=(
            "cap on how many of a company's most recent accounts filings become "
            "candidates. Breadth across companies beats depth within one: the same "
            "filer's template repeated is one layout, not many documents. "
            f"Default {DEFAULT_MAX_FILINGS_PER_COMPANY}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload documents already on disk. The only path that refetches.",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="re-measure the bytes already on disk. Makes no network calls.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the measured text-layer distribution and exit",
    )
    args = parser.parse_args(argv)

    if args.report:
        print(text_layer_report(load_manifest(args.cache_root)))
        return 0

    if args.recheck:
        manifest = recheck_corpus(args.cache_root)
        print(text_layer_report(manifest))
        return 0

    load_dotenv()
    api_key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not api_key:
        print(
            "COMPANIES_HOUSE_API_KEY is not set. Copy .env.example to .env and put your key in it.",
            file=sys.stderr,
        )
        return 2

    company_numbers = read_company_numbers(args.companies)

    # follow_redirects=False is load-bearing, not a default we inherited. See
    # _download in docintel/ingest/corpus.py: the second hop of the document
    # redirect must not carry the Authorization header, and httpx's own
    # redirect following would forward it.
    with httpx.Client(
        base_url=API_BASE,
        auth=httpx.BasicAuth(api_key, ""),
        follow_redirects=False,
        timeout=60.0,
    ) as client:
        manifest = fetch_corpus(
            client,
            cache_root=args.cache_root,
            company_numbers=company_numbers,
            max_filings_per_company=args.max_filings_per_company,
            force=args.force,
        )

    print(text_layer_report(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

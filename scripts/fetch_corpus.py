"""CLI wrapper around docintel.ingest.

Deliberately thin. Everything worth testing lives in the module; this file is
argparse, the contact header and printing, so that the test suite never has to
drive a command line to reach the logic.
"""

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from docintel.ingest.edgar import (
    corpus_report,
    edgar_headers,
    fetch_corpus,
    read_ciks,
    recheck_corpus,
)
from docintel.ingest.manifest import DEFAULT_MAX_FILINGS_PER_COMPANY, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CIKS = REPO_ROOT / "data" / "ciks.txt"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ciks", type=Path, default=DEFAULT_CIKS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--max-filings-per-company",
        type=int,
        default=DEFAULT_MAX_FILINGS_PER_COMPANY,
        help=(
            "cap on how many of a company's most recent annual reports become "
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
        help="re-derive text from the bytes already on disk. Makes no network calls.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the derived-text size distribution and markup ratio, then exit",
    )
    args = parser.parse_args(argv)

    if args.report:
        print(corpus_report(load_manifest(args.cache_root)))
        return 0

    if args.recheck:
        print(corpus_report(recheck_corpus(args.cache_root)))
        return 0

    load_dotenv()
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        print(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and put your "
            "contact string in it. It is not a secret.",
            file=sys.stderr,
        )
        return 2
    if "@" not in user_agent:
        # Caught here because EDGAR's own answer is a 403 carrying an
        # "Undeclared Automated Tool" HTML page, which is a confusing way to
        # find out your User-Agent was the wrong shape.
        print(
            f"SEC_USER_AGENT is {user_agent!r}, which EDGAR will reject. It must be a "
            'contact string of the form "Name email@domain.com".',
            file=sys.stderr,
        )
        return 2

    ciks = read_ciks(args.ciks)

    with httpx.Client(headers=edgar_headers(user_agent), timeout=60.0) as client:
        manifest = fetch_corpus(
            client,
            cache_root=args.cache_root,
            ciks=ciks,
            max_filings_per_company=args.max_filings_per_company,
            force=args.force,
        )

    print(corpus_report(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

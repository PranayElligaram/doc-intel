"""CLI for hand-labelling the gold set.

Deliberately thin, like scripts/fetch_corpus.py: everything worth testing lives
in docintel/gold.py and this file is argparse, stdin and printing.

The workflow it exists to support:

    python scripts/label.py todo --field auditor_name
    python scripts/label.py add --entry-id 0000320193/0000320193-25 \\
        --field auditor_name --value "Ernst & Young LLP" \\
        --quote "Ernst & Young LLP"
    python scripts/label.py add --entry-id 0000320193/0000320193-25 \\
        --field period_start --not-stated

The corpus is 36 documents with a median of 438,862 characters, so nobody is
counting offsets by hand. You find the quote by searching the text, paste it,
and the tool turns it into a span - or refuses, if the quote is not unique.
"""

import argparse
import sys
from pathlib import Path

from docintel.extraction.schema import Currency, Scale
from docintel.gold import (
    GoldField,
    GoldStated,
    LabellingError,
    add_label,
    anchor_report,
    corpus_entry,
    corpus_manifest,
    document_text,
    load_gold_set,
    parse_value,
    progress_report,
    quote_context,
    resolve_entry_id,
    unlabelled,
    verify_labels,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "cache"
DEFAULT_GOLD_ROOT = REPO_ROOT / "data" / "gold"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    subcommands = parser.add_subparsers(dest="command", required=True)

    todo = subcommands.add_parser("todo", help="which fields of which documents are unlabelled")
    todo.add_argument(
        "--field",
        choices=[field.value for field in GoldField],
        help="list the documents still missing this one field, rather than the whole grid",
    )

    add = subcommands.add_parser("add", help="label one field of one document")
    add.add_argument(
        "--entry-id",
        required=True,
        help="the manifest entry id, or any unambiguous fragment of one",
    )
    add.add_argument("--field", required=True, choices=[field.value for field in GoldField])
    add.add_argument(
        "--quote",
        help="the text from the document that states the value, verbatim",
    )
    add.add_argument(
        "--quote-file",
        type=Path,
        help="read the quote from a file, or from stdin with -. For a quote that "
        "spans a blank line, which the shell makes awkward to type.",
    )
    add.add_argument(
        "--not-stated",
        action="store_true",
        help="the document does not state this field. No quote, no value.",
    )
    # --amount is the same flag under the name that reads correctly at a money
    # field, where what you type is the figure as printed rather than the
    # normalised value.
    add.add_argument(
        "--value", "--amount", dest="value", help="the value, as the document prints it"
    )
    add.add_argument(
        "--scale",
        choices=[scale.value for scale in Scale],
        help="what the statement heading says to multiply by. Required for money fields.",
    )
    add.add_argument(
        "--currency",
        choices=[currency.value for currency in Currency],
        help="the currency of the figure. Required for money fields.",
    )
    add.add_argument("--replace", action="store_true", help="overwrite an existing label")

    subcommands.add_parser("verify", help="re-check every label against the text on disk")
    subcommands.add_parser("anchors", help="how many documents each search anchor appears in")
    return parser


def read_quote(
    *,
    quote: str | None,
    quote_file: Path | None,
    not_stated: bool,
    value: str | None,
) -> str | None:
    """The quote as typed, read from a file, or None for a not_stated label.

    Takes the four flags rather than the Namespace because argparse types every
    attribute as Any, and Any spreading into the code that computes offsets is
    how a str | None turns into a surprise at two in the morning.
    """
    if not_stated:
        if quote or quote_file or value:
            raise LabellingError("--not-stated takes no quote and no value")
        return None
    if quote and quote_file:
        raise LabellingError("pass --quote or --quote-file, not both")
    if quote_file is not None:
        # Trailing newlines only. A quote is verbatim, so stripping the leading
        # spaces of an indented table row would stop it matching.
        if str(quote_file) == "-":
            return sys.stdin.read().strip("\n")
        return quote_file.read_text(encoding="utf-8").strip("\n")
    if quote:
        return quote
    raise LabellingError("pass --quote, --quote-file, or --not-stated")


def run_add(args: argparse.Namespace) -> int:
    field = GoldField(args.field)
    quote = read_quote(
        quote=args.quote,
        quote_file=args.quote_file,
        not_stated=args.not_stated,
        value=args.value,
    )

    value = None
    if quote is not None:
        if args.value is None:
            raise LabellingError(f"{field.value} needs --value")
        value = parse_value(
            field,
            args.value,
            scale=Scale(args.scale) if args.scale else None,
            currency=Currency(args.currency) if args.currency else None,
        )

    manifest = corpus_manifest(args.cache_root)
    entry_id = resolve_entry_id(manifest, args.entry_id)
    label = add_label(
        cache_root=args.cache_root,
        gold_root=args.gold_root,
        entry_id=entry_id,
        field=field,
        quote=quote,
        value=value,
        replace=args.replace,
    )

    answer = label.answer
    if isinstance(answer, GoldStated):
        text = document_text(args.cache_root, corpus_entry(manifest, entry_id))
        print(f"{entry_id} {field.value}: {answer.value.model_dump_json()}")
        print(f"  span [{answer.span.start}, {answer.span.end})")
        print(f"  {quote_context(text, answer.span.start, len(answer.span.quote))}")
    else:
        print(f"{entry_id} {field.value}: not_stated")
    return 0


def run_todo(args: argparse.Namespace) -> int:
    manifest = corpus_manifest(args.cache_root)
    gold_set = load_gold_set(args.gold_root)

    if args.field:
        field = GoldField(args.field)
        missing = [entry_id for entry_id, one in unlabelled(manifest, gold_set) if one is field]
        for entry_id in missing:
            print(entry_id)
        print(f"\n{len(missing)} documents still need {field.value}")
        return 0

    print(progress_report(manifest, gold_set))
    return 0


def run_verify(args: argparse.Namespace) -> int:
    problems = verify_labels(args.cache_root, load_gold_set(args.gold_root))
    if not problems:
        print("every label still matches the text on disk")
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"\n{len(problems)} labels need attention", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "add":
            return run_add(args)
        if args.command == "todo":
            return run_todo(args)
        if args.command == "verify":
            return run_verify(args)
        print(anchor_report(args.cache_root))
        return 0
    except LabellingError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""The labelling CLI, driven through main() with an argv list.

The module under test is meant to be thin - the rules live in docintel/gold.py
and are tested in test_gold.py. What is left here is the part argparse owns and
a unit test cannot reach: which flag combinations are refused, where the quote
comes from, and what the exit code is. Those matter because a personal tool
that exits 0 after refusing to write is a tool that quietly loses an
afternoon's labelling.
"""

import io
import sys
from pathlib import Path

import pytest

from docintel.gold import GoldStated, load_gold_set

# scripts/ is not part of the installed package - it is where the CLIs live -
# so the repo root goes on the path to reach it. test_gold is imported by bare
# name, from this directory, because that is the one spelling pytest and mypy
# agree on: as `tests.test_gold` mypy would find the same file under two module
# names and refuse to check either.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_gold import APPLE, build_corpus  # noqa: E402

from scripts.label import main  # noqa: E402


def roots(tmp_path: Path) -> list[str]:
    return [
        "--cache-root",
        str(build_corpus(tmp_path)),
        "--gold-root",
        str(tmp_path / "gold"),
    ]


def test_a_label_is_written_and_then_disappears_from_the_todo_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                *roots(tmp_path),
                "add",
                "--entry-id",
                "0000320193-25",
                "--field",
                "auditor_name",
                "--value",
                "Ernst & Young LLP",
                "--quote",
                "We have served as the Company's auditor since 2009.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main([*roots(tmp_path), "todo", "--field", "auditor_name"]) == 0

    listed = capsys.readouterr().out
    assert APPLE not in listed
    assert "1 documents still need auditor_name" in listed


def test_the_printed_money_value_is_scaled_by_the_heading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            *roots(tmp_path),
            "add",
            "--entry-id",
            "0000320193-25",
            "--field",
            "revenue",
            "--amount",
            "391,035",
            "--scale",
            "millions",
            "--currency",
            "USD",
            "--quote",
            "Total net sales 416,161 391,035 383,285",
        ]
    )

    assert exit_code == 0
    assert "391035000000" in capsys.readouterr().out


def test_an_ambiguous_quote_exits_two_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_root = build_corpus(tmp_path, {APPLE: "Net income 1\n\nNet income 1\n"})
    gold_root = tmp_path / "gold"

    exit_code = main(
        [
            "--cache-root",
            str(cache_root),
            "--gold-root",
            str(gold_root),
            "add",
            "--entry-id",
            APPLE,
            "--field",
            "net_income",
            "--amount",
            "1",
            "--scale",
            "units",
            "--currency",
            "USD",
            "--quote",
            "Net income 1",
        ]
    )

    assert exit_code == 2
    assert "appears 2 times" in capsys.readouterr().err
    assert load_gold_set(gold_root).labels == []


def test_not_stated_refuses_to_take_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            *roots(tmp_path),
            "add",
            "--entry-id",
            "0000320193-25",
            "--field",
            "period_start",
            "--not-stated",
            "--value",
            "2024-09-29",
        ]
    )

    assert exit_code == 2
    assert "takes no quote and no value" in capsys.readouterr().err


def test_a_quote_can_come_from_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The escape hatch for a quote spanning a blank line.

    The derived text separates blocks with a blank line, and JPMorgan's cover
    page puts the period end in one - so some quotes cannot be typed as a shell
    argument without fighting the shell.
    """
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("For the fiscal year ended Commission file\n\nDecember 31, 2024")
    )

    exit_code = main(
        [
            *roots(tmp_path),
            "add",
            "--entry-id",
            "0000019617",
            "--field",
            "period_end",
            "--value",
            "2024-12-31",
            "--quote-file",
            "-",
        ]
    )

    assert exit_code == 0
    capsys.readouterr()
    label = load_gold_set(tmp_path / "gold").labels[0]
    assert isinstance(label.answer, GoldStated)
    assert "\n\n" in label.answer.span.quote


def test_verify_exits_nonzero_when_a_label_no_longer_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = roots(tmp_path)
    cache_root = Path(args[1])
    main(
        [
            *args,
            "add",
            "--entry-id",
            "0000320193-25",
            "--field",
            "company_name",
            "--value",
            "Apple Inc.",
            "--quote",
            "(Exact name of registrant as specified in its charter)",
        ]
    )
    # The text moves under the label. In the real workflow this is a
    # DERIVATION_VERSION bump; here it is the same effect, one layer down.
    text_path = cache_root / "text" / "0000320193" / "0000320193-25-000079.txt"
    text_path.write_text("shifted\n" + text_path.read_text(encoding="utf-8"), encoding="utf-8")
    capsys.readouterr()

    assert main([*args, "verify"]) == 1
    assert "does not match the manifest hash" in capsys.readouterr().err


def test_an_unknown_field_is_rejected_by_the_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([*roots(tmp_path), "add", "--entry-id", APPLE, "--field", "turnover"])


def test_the_anchor_report_runs_over_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*roots(tmp_path), "anchors"]) == 0
    assert "documents in corpus: 2" in capsys.readouterr().out

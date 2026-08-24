"""Asserts the schema's type-level guarantees are still enforced by mypy.

The runtime tests cannot cover this. "A found value cannot exist without a
citation" is true because mypy rejects the alternative, so the only way to test
it is to run mypy against code that violates it and check it still complains.

The expectations are read out of should_fail.py itself, as `# want: <code>`
markers on the offending lines, and compared line by line. An earlier version
compared a *set* of error codes, which could not tell one violation from
another once two of them shared a code: the fourth violation - a Money with no
currency - reports call-arg, the same code as the ungrounded Found, so under
the set comparison either one could vanish unnoticed while the other kept the
code present. Comparing (line, code) pairs also catches a violation that has
quietly moved to a different line, which usually means it is no longer the
violation the marker describes.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOULD_FAIL = REPO_ROOT / "tests" / "typing" / "should_fail.py"

WANT_MARKER = re.compile(r"#\s*want:\s*([a-z][a-z-]+)\s*$")
# mypy writes "path:line: error: message  [code]". The path is matched loosely
# because it is a Windows path on one CI runner and a POSIX one on the other.
MYPY_ERROR = re.compile(r"^.*?:(\d+): error: .*\[([a-z][a-z-]+)\]\s*$")


def expected_violations() -> set[tuple[int, str]]:
    lines = SHOULD_FAIL.read_text(encoding="utf-8").splitlines()
    return {
        (number, match.group(1))
        for number, line in enumerate(lines, start=1)
        if (match := WANT_MARKER.search(line))
    }


def test_the_markers_have_not_all_been_deleted() -> None:
    """Guards the guard: an empty marker set would make the test below vacuous."""
    assert len(expected_violations()) >= 4


def test_mypy_reports_exactly_the_marked_violations() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(SHOULD_FAIL)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "mypy accepted tests/typing/should_fail.py. The schema's grounding "
        f"guarantee has been weakened.\n{result.stdout}{result.stderr}"
    )

    reported = {
        (int(match.group(1)), match.group(2))
        for line in result.stdout.splitlines()
        if (match := MYPY_ERROR.match(line))
    }
    expected = expected_violations()

    assert reported == expected, (
        "mypy no longer reports exactly the marked violations.\n"
        f"missing (marked but not reported): {sorted(expected - reported)}\n"
        f"unexpected (reported but not marked): {sorted(reported - expected)}\n"
        f"{result.stdout}{result.stderr}"
    )

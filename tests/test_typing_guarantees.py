"""Asserts the schema's type-level guarantees are still enforced by mypy.

The runtime tests cannot cover this. "A found value cannot exist without a
citation" is true because mypy rejects the alternative, so the only way to test
it is to run mypy against code that violates it and check it still complains.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOULD_FAIL = REPO_ROOT / "tests" / "typing" / "should_fail.py"

# One code per violation in should_fail.py, in the order they appear there.
EXPECTED_CODES = {
    "arg-type",  # Found[str] where Extracted[Money] is required
    "union-attr",  # .value read without narrowing on .status
    "call-arg",  # Found[Money] constructed with no citation
}


def test_mypy_still_rejects_the_schema_violations() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(SHOULD_FAIL)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "mypy accepted tests/typing/should_fail.py. The schema's grounding "
        f"guarantee has been weakened.\n{result.stdout}{result.stderr}"
    )

    reported = set(re.findall(r"\[([a-z][a-z-]+)\]", result.stdout))
    missing = EXPECTED_CODES - reported
    assert not missing, (
        f"mypy no longer reports {sorted(missing)} for the corresponding "
        f"violation.\n{result.stdout}{result.stderr}"
    )

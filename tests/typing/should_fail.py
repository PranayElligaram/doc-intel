"""Type violations that must stay violations.

This module is deliberately not type-correct. It is not collected by pytest and
is excluded from the main mypy run; it exists only as the input to
tests/test_typing_guarantees.py, which runs mypy against it and asserts each
violation is still reported.

The grounding guarantee in docintel.extraction.schema is a claim about the type
system - a value cannot exist without a citation, and cannot be read without
narrowing on status. A claim like that needs a type-system test, or a later
refactor could weaken it while every runtime test still passes.

EVERY VIOLATION CARRIES A `# want: <code>` MARKER on the exact line mypy should
report it on. The test compares the (line, code) pairs mypy produces against
the markers found here, so a violation that stops being reported, or moves to a
different line, fails - which a set of expected codes could not catch once two
violations shared a code.
"""

from datetime import date
from decimal import Decimal

from docintel.extraction.schema import (
    Citation,
    CompanyAccounts,
    Currency,
    Found,
    Money,
    NotStated,
    Scale,
)

CITATION = Citation(chunk_id="c", start=0, end=5, quote="4,200")


def wrong_generic_argument() -> CompanyAccounts:
    """A str value where Money is required."""
    return CompanyAccounts(
        company_name=Found[str](value="Acme Inc", citation=CITATION),
        period_start=Found[date](value=date(2023, 1, 1), citation=CITATION),
        period_end=Found[date](value=date(2023, 12, 31), citation=CITATION),
        revenue=Found[str](value="4,200", citation=CITATION),  # want: arg-type
        net_income=NotStated(),
        employee_count=NotStated(),
        auditor_name=NotStated(),
    )


def value_read_without_narrowing() -> Decimal:
    """Reading .value with no check on .status."""
    return wrong_generic_argument().net_income.value.amount  # want: union-attr


def value_without_citation() -> Found[Money]:
    """A found value that was never grounded."""
    return Found[Money](  # want: call-arg
        value=Money(amount=Decimal("1"), scale=Scale.UNITS, currency=Currency.USD)
    )


def money_without_a_currency() -> Money:
    """An amount with no currency - the GBP assumption, unconstructible now."""
    return Money(amount=Decimal("391035000000"), scale=Scale.MILLIONS)  # want: call-arg

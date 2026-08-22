"""Type violations that must stay violations.

This module is deliberately not type-correct. It is not collected by pytest and
is excluded from the main mypy run; it exists only as the input to
tests/test_typing_guarantees.py, which runs mypy against it and asserts each
violation is still reported.

The grounding guarantee in docintel.extraction.schema is a claim about the type
system - a value cannot exist without a citation, and cannot be read without
narrowing on status. A claim like that needs a type-system test, or a later
refactor could weaken it while every runtime test still passes.
"""

from datetime import date
from decimal import Decimal

from docintel.extraction.schema import (
    Citation,
    CompanyAccounts,
    Found,
    Money,
    NotStated,
    Scale,
)

CITATION = Citation(chunk_id="c", start=0, end=5, quote="4,200")


def wrong_generic_argument() -> CompanyAccounts:
    """A str value where Money is required. Expected error code: arg-type."""
    return CompanyAccounts(
        company_name=Found[str](value="Acme Ltd", citation=CITATION),
        company_number=Found[str](value="01234567", citation=CITATION),
        period_start=Found[date](value=date(2023, 1, 1), citation=CITATION),
        period_end=Found[date](value=date(2023, 12, 31), citation=CITATION),
        turnover=Found[str](value="4,200", citation=CITATION),
        profit_before_tax=NotStated(),
        employee_count=NotStated(),
        auditor_name=NotStated(),
    )


def value_read_without_narrowing() -> Decimal:
    """Reading .value with no check on .status. Expected error code: union-attr."""
    return wrong_generic_argument().profit_before_tax.value.amount


def value_without_citation() -> Found[Money]:
    """A found value that was never grounded. Expected error code: call-arg."""
    return Found[Money](value=Money(amount=Decimal("1"), scale=Scale.UNITS))

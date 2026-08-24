"""Tests for the extraction schema.

The generic discriminated union is the load-bearing part: PEP 695 generics
inside a Pydantic discriminated union are new enough to be worth proving rather
than assuming, so the round-trip tests check that the discriminator resolves to
the right variant *and* that the generic argument survives serialisation.
"""

from datetime import date
from decimal import Decimal
from typing import Any, assert_type

import pytest
from pydantic import TypeAdapter, ValidationError

from docintel.extraction.schema import (
    Citation,
    CompanyAccounts,
    Currency,
    DocumentExtraction,
    Extracted,
    Failed,
    Found,
    Money,
    NotStated,
    Scale,
)


def citation(quote: str = "Acme Ltd", chunk_id: str = "chunk-0", start: int = 10) -> Citation:
    return Citation(chunk_id=chunk_id, start=start, end=start + len(quote), quote=quote)


# --- the generic discriminated union, end to end ---------------------------


def test_found_money_round_trips_and_keeps_its_generic_argument() -> None:
    # Annotated explicitly: TypeAdapter does not infer its parameter, and
    # without this the adapter yields Found[Any] and assert_type below is
    # satisfied by nothing at all.
    adapter: TypeAdapter[Extracted[Money]] = TypeAdapter(Extracted[Money])
    original = Found[Money](
        value=Money(amount=Decimal("4200000"), scale=Scale.THOUSANDS, currency=Currency.USD),
        citation=citation("4,200"),
    )

    restored = adapter.validate_json(adapter.dump_json(original))

    assert isinstance(restored, Found)
    assert restored.status == "found"
    # Static counterpart to the isinstance check below: narrowing the union on
    # Found must leave the generic argument bound to Money, not to Any.
    assert_type(restored, Found[Money])
    assert_type(restored.value, Money)
    # The generic argument must come back as Money, not as a bare dict.
    assert isinstance(restored.value, Money)
    assert restored.value.amount == Decimal("4200000")
    assert restored.value.scale is Scale.THOUSANDS
    assert restored.citation.quote == "4,200"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "status": "found",
                "value": 42,
                "citation": {"chunk_id": "c", "start": 0, "end": 2, "quote": "42"},
            },
            Found,
        ),
        ({"status": "not_stated"}, NotStated),
        ({"status": "failed", "error": "model returned unparseable output"}, Failed),
    ],
)
def test_discriminator_selects_the_right_variant(payload: dict[str, Any], expected: type) -> None:
    assert isinstance(TypeAdapter(Extracted[int]).validate_python(payload), expected)


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Extracted[int]).validate_python({"status": "maybe", "value": 1})


def test_missing_discriminator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Extracted[int]).validate_python({"value": 1})


# --- a value cannot exist without a citation -------------------------------


def test_found_without_a_citation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Extracted[str]).validate_python({"status": "found", "value": "Acme Ltd"})


def test_not_stated_cannot_smuggle_in_a_citation() -> None:
    # extra="forbid" is what stops an abstention carrying evidence it should
    # not have; a citation on a not_stated field would be scored as grounding.
    with pytest.raises(ValidationError):
        TypeAdapter(Extracted[str]).validate_python(
            {
                "status": "not_stated",
                "citation": {"chunk_id": "c", "start": 0, "end": 1, "quote": "x"},
            }
        )


def test_failed_requires_a_non_empty_error() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Extracted[str]).validate_python({"status": "failed", "error": ""})


# --- absent is not the same as zero ----------------------------------------


def test_zero_revenue_is_found_not_absent() -> None:
    """A pre-revenue biotech reporting nil revenue has stated a value.

    This is the case that rules out Optional[T]: with None as the absence
    marker, a genuine reported zero and a missing field are the same value.
    """
    adapter: TypeAdapter[Extracted[Money]] = TypeAdapter(Extracted[Money])
    restored = adapter.validate_json(
        adapter.dump_json(
            Found[Money](
                value=Money(amount=Decimal("0"), scale=Scale.UNITS, currency=Currency.USD),
                citation=citation("0"),
            )
        )
    )

    assert isinstance(restored, Found)
    assert not isinstance(restored, NotStated)
    assert restored.value.amount == Decimal("0")


# --- Citation invariants ---------------------------------------------------


def test_citation_span_length_must_match_the_quote() -> None:
    # The drift detector: if the chunker moves and offsets go stale, this is
    # what fires instead of grounding silently against the wrong text.
    with pytest.raises(ValidationError, match="quote is 5 chars"):
        Citation(chunk_id="c", start=0, end=99, quote="4,200")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_id": "c", "start": 10, "end": 10, "quote": ""},  # empty span and quote
        {"chunk_id": "c", "start": 10, "end": 4, "quote": "abcdef"},  # reversed
        {"chunk_id": "c", "start": -1, "end": 3, "quote": "abcd"},  # negative offset
        {"chunk_id": "", "start": 0, "end": 4, "quote": "abcd"},  # no chunk id
        {"chunk_id": "c", "start": 0, "end": 3, "quote": "   "},  # whitespace only
    ],
)
def test_citation_rejects_malformed_spans(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Citation(**kwargs)


def test_citation_round_trips() -> None:
    original = citation("31 December 2023")
    assert Citation.model_validate_json(original.model_dump_json()) == original


# --- Money -----------------------------------------------------------------


def test_money_records_the_scale_that_was_applied() -> None:
    """The thousands case: printed "4,200", grounded as printed, scored in units."""
    printed = citation("4,200")
    money = Money(amount=Decimal("4200000"), scale=Scale.THOUSANDS, currency=Currency.USD)

    assert printed.quote == "4,200"
    assert money.amount == Decimal("4200000")
    assert money.scale is Scale.THOUSANDS
    assert money.currency is Currency.USD


def test_money_survives_json_without_losing_precision() -> None:
    original = Money(amount=Decimal("1234567.89"), scale=Scale.UNITS, currency=Currency.USD)
    assert Money.model_validate_json(original.model_dump_json()).amount == Decimal("1234567.89")


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_money_rejects_non_finite_amounts(bad: str) -> None:
    with pytest.raises(ValidationError, match="finite"):
        Money(amount=Decimal(bad), scale=Scale.UNITS, currency=Currency.USD)


def test_money_rejects_unknown_scale() -> None:
    with pytest.raises(ValidationError):
        Money.model_validate({"amount": "1", "scale": "billions", "currency": "USD"})


def test_money_requires_a_currency() -> None:
    # No default, deliberately. The schema assumed GBP for a UK corpus and the
    # corpus is now US filings; a default would have carried that assumption
    # into the gold set instead of failing here.
    with pytest.raises(ValidationError):
        Money.model_validate({"amount": "1", "scale": "units"})


def test_money_rejects_unknown_currency() -> None:
    with pytest.raises(ValidationError):
        Money.model_validate({"amount": "1", "scale": "units", "currency": "XBT"})


def test_money_round_trips_its_currency() -> None:
    original = Money(amount=Decimal("391035000000"), scale=Scale.MILLIONS, currency=Currency.USD)
    assert Money.model_validate_json(original.model_dump_json()) == original


# --- the extraction record as a whole --------------------------------------


def full_accounts() -> CompanyAccounts:
    """A realistic mix: found fields, a genuine absence, and a failure."""
    return CompanyAccounts(
        company_name=Found[str](value="Acme Inc", citation=citation("Acme Inc")),
        period_start=Found[date](value=date(2023, 1, 1), citation=citation("January 1, 2023")),
        period_end=Found[date](value=date(2023, 12, 31), citation=citation("December 31, 2023")),
        revenue=Found[Money](
            value=Money(amount=Decimal("4200000"), scale=Scale.THOUSANDS, currency=Currency.USD),
            citation=citation("4,200"),
        ),
        net_income=Found[Money](
            value=Money(amount=Decimal("-15000"), scale=Scale.UNITS, currency=Currency.USD),
            citation=citation("(15,000)"),
        ),
        # A filing that names no auditor: nothing to ground a name against.
        auditor_name=NotStated(),
        employee_count=Failed(error="model returned unparseable output"),
    )


def test_document_extraction_round_trips_with_mixed_states() -> None:
    original = DocumentExtraction(document_id="doc-abc123", accounts=full_accounts())

    restored = DocumentExtraction.model_validate_json(original.model_dump_json())

    assert restored == original
    assert isinstance(restored.accounts.revenue, Found)
    assert isinstance(restored.accounts.auditor_name, NotStated)
    assert isinstance(restored.accounts.employee_count, Failed)


def test_narrowing_on_status_is_how_a_value_is_read() -> None:
    """Documents the intended call-site shape that mypy enforces."""
    revenue = full_accounts().revenue
    assert_type(revenue, Found[Money] | NotStated | Failed)

    if revenue.status == "found":
        # Narrowing on the discriminator alone is enough for mypy to reach the
        # value; no isinstance check and no cast is needed at a call site.
        assert_type(revenue, Found[Money])
        assert revenue.value.amount == Decimal("4200000")
    else:
        pytest.fail("expected a found revenue")


def test_negative_employee_count_is_rejected() -> None:
    # The ge=0 constraint lives on the CompanyAccounts field, so it has to be
    # exercised through the real model rather than a bare Extracted[int].
    payload = full_accounts().model_dump()
    payload["employee_count"] = {
        "status": "found",
        "value": -1,
        "citation": {"chunk_id": "c", "start": 0, "end": 2, "quote": "-1"},
    }

    with pytest.raises(ValidationError):
        CompanyAccounts.model_validate(payload)


def test_unknown_field_is_rejected() -> None:
    payload = DocumentExtraction(document_id="d", accounts=full_accounts()).model_dump()
    payload["confidence"] = 0.9

    with pytest.raises(ValidationError):
        DocumentExtraction.model_validate(payload)


def test_company_number_is_no_longer_a_field() -> None:
    # It held a UK company number; the EDGAR equivalent is the CIK, which
    # appears in none of the 36 derived texts. Dropping it has to be visible in
    # the type, not just in a docstring, or old payloads keep validating.
    payload = full_accounts().model_dump()
    payload["company_number"] = {
        "status": "found",
        "value": "0000320193",
        "citation": {"chunk_id": "c", "start": 0, "end": 10, "quote": "0000320193"},
    }

    with pytest.raises(ValidationError):
        CompanyAccounts.model_validate(payload)


def test_wrong_typed_value_is_rejected_when_validated_through_the_model() -> None:
    # Pins the boundary described in Found's docstring: the runtime check comes
    # from the field annotation on CompanyAccounts, which is what every real
    # deserialisation path goes through.
    payload = full_accounts().model_dump()
    payload["revenue"] = {
        "status": "found",
        "value": "four million two hundred thousand",
        "citation": {"chunk_id": "c", "start": 0, "end": 5, "quote": "4,200"},
    }

    with pytest.raises(ValidationError):
        CompanyAccounts.model_validate(payload)

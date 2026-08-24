"""What we extract from a filing, and how each value is grounded in the source.

Every field is in exactly one of three states - found, not_stated, or failed -
and a found value cannot exist without a Citation. That invariant is carried by
the type system rather than a runtime validator: `Found` declares a required
citation, so an ungrounded value is unconstructible, and reading `.value`
forces a narrowing check on `.status` first.

The three states are kept distinct because correct abstention is a headline
metric. A model that crashed must never be scored as a model that correctly
declined to answer.

Grounding is checked as two separate questions: does `citation.quote` appear
verbatim at [citation.start, citation.end) of the cited chunk, and does `value`
normalise from `citation.quote`. The as-printed text is stored on the Citation
so a figure printed "4,200" under a headline of $ in millions can be grounded
exactly as printed while being scored as Decimal("4200000000"). Offsets alone would not
survive a change to the chunker in a later phase - every stored citation would
quietly point at the wrong text. Storing the quote turns that silent drift into
a loud failure.

METRICS RULE, and this one outlives the module: never report a single blended
accuracy across these fields. They span a deliberate difficulty range.
company_name extracts at close to 100% and period_end is nearly as easy.
revenue and net_income are the real test: both captions recur dozens of times
in a 10-K, in the statements, the MD&A and every segment note, and only one of
those occurrences is the answer. period_start is absent from every filing in
the corpus, and employee_count and auditor_name are absent from some, which is
where the genuine not_stated cases come from. A blended figure lets the trivial
fields mask the hard ones. Report per field.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator


class Citation(BaseModel):
    """Where in the source a value came from, and what it said there."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _span_must_match_quote(self) -> "Citation":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        # The span and the quote are two descriptions of the same text, so a
        # length disagreement means one of them is stale - most likely the
        # offsets, after a chunker change. Fail here rather than silently
        # grounding against the wrong region.
        span_length = self.end - self.start
        if len(self.quote) != span_length:
            raise ValueError(
                f"quote is {len(self.quote)} chars but span [{self.start}, {self.end}) "
                f"is {span_length}"
            )
        if not self.quote.strip():
            raise ValueError("quote is only whitespace, which cannot ground a value")
        return self


class Currency(StrEnum):
    """The unit the amount is denominated in.

    Required on every Money, with no default. A default is precisely how a
    wrong assumption survives: this schema was written for UK accounts and
    assumed GBP throughout, and the corpus is now US filings reporting in
    dollars. A field with a sensible-looking default would have carried that
    assumption into the gold set unchallenged. Making it required means every
    construction site had to state it, which is how the mistake got found.

    Only the currencies the corpus can actually produce. An enum rather than a
    free ISO string so an unknown code is a validation error at the boundary
    rather than a mis-scored comparison in the eval harness.
    """

    USD = "USD"
    GBP = "GBP"
    EUR = "EUR"


class Scale(StrEnum):
    """The multiplier applied when normalising a printed figure to whole units."""

    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"


class Money(BaseModel):
    """A monetary amount in whole units of its currency.

    `scale` records what was applied to get there, so a figure printed "4,200"
    under a "$ in millions" heading is stored as amount=4200000000,
    scale=MILLIONS, currency=USD. The printed form itself lives on the
    Citation, not here.

    amount and currency are two halves of one fact and neither means anything
    alone: 391,035 is not a value, 391,035 USD is. Comparing an amount to a
    gold amount without comparing currency is a comparison of two numbers that
    were never the same kind of thing.
    """

    model_config = ConfigDict(extra="forbid")

    amount: Decimal
    scale: Scale
    currency: Currency

    @model_validator(mode="after")
    def _amount_must_be_finite(self) -> "Money":
        # Decimal accepts NaN and Infinity from JSON, and either would poison
        # every downstream comparison in the eval harness.
        if not self.amount.is_finite():
            raise ValueError(f"amount must be a finite number, got {self.amount}")
        return self


class Found[T](BaseModel):
    """A value located in the document, with the citation that grounds it.

    Construct this with the type argument spelled out - Found[Money](...) and
    not Found(...). Left unparametrised, Pydantic binds T to Any and skips
    validation of `value` entirely, so a str would sail through where Money was
    meant. Validating a whole CompanyAccounts is unaffected, because there the
    field annotation supplies the argument; the gap is only in direct
    construction. mypy catches it at the point of assignment, and pytest is
    configured to treat the resulting Pydantic warning as an error.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["found"] = "found"
    value: T
    citation: Citation


class NotStated(BaseModel):
    """The document genuinely does not state this field.

    Carries no citation: an abstention is the absence of a claim, so there is
    nothing to ground.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_stated"] = "not_stated"


class Failed(BaseModel):
    """Extraction did not complete - a crash, timeout, or unparseable output.

    Distinct from NotStated on purpose. Collapsing the two would score every
    error as a correct abstention.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["failed"] = "failed"
    error: str = Field(min_length=1)


type Extracted[T] = Annotated[Found[T] | NotStated | Failed, Discriminator("status")]


class CompanyAccounts(BaseModel):
    """The seven things we extract from an annual report.

    The reporting period is two separately cited fields because a document
    frequently states one date explicitly and leaves the other implicit; giving
    them one shared citation would overstate the grounding of whichever half
    was inferred.

    There is no company_number. It held a UK company number, and on the EDGAR
    corpus the equivalent identifier is the CIK - which appears in none of the
    36 derived texts, because it is EDGAR's key for the filer and not something
    the filer prints. A field absent from every document is not an abstention
    test; it is a test of whether the model learned to answer not_stated for
    one particular field, and it would cost 36 gold labels to measure that.
    The Commission File Number and the IRS employer identification number are
    both on the cover page and were rejected for the opposite reason: they sit
    two lines from the company name and would measure what company_name
    already measures. The CIK is corpus metadata - it is the manifest's
    company_id and the first half of every entry_id.
    """

    model_config = ConfigDict(extra="forbid")

    company_name: Extracted[str]
    period_start: Extracted[date]
    period_end: Extracted[date]
    # Named for the us-gaap XBRL concepts a filer tags these lines with -
    # Revenues and NetIncomeLoss - rather than for the UK vocabulary the schema
    # started with. A field name that matches the concept means the same thing
    # to the filer, to the gold set and to the metric. See
    # docs/labelling-policy.md for what this buys at labelling time.
    revenue: Extracted[Money]
    net_income: Extracted[Money]
    employee_count: Extracted[Annotated[int, Field(ge=0)]]
    auditor_name: Extracted[str]


class DocumentExtraction(BaseModel):
    """One extraction run against one cached document.

    `document_id` is the manifest's `entry_id` for the document - for the EDGAR
    corpus, "{cik}/{accession number}". It is what ties this record back to an
    entry in the corpus manifest and to a gold label.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    accounts: CompanyAccounts

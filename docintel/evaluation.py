"""Compare predictions with independently hand-labelled filing evidence.

This module never generates, edits or infers gold labels. A label's document
coordinates are independent of the chunking settings used for inference.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel

from docintel.extraction.schema import Failed, Found, Money, NotStated
from docintel.gold import (
    CountValue,
    DateValue,
    GoldLabel,
    GoldNotStated,
    GoldStated,
    MoneyValue,
    TextValue,
    resolve_citation,
    spans_agree,
)
from docintel.retrieval.chunker import Chunk
from docintel.retrieval.retriever import Hit


@dataclass(frozen=True)
class CaseResult:
    entry_id: str
    field: str
    gold_status: str
    predicted_status: str
    prediction: dict[str, Any]
    value_correct: bool
    citation_grounded: bool | None
    gold_span_aligned: bool | None
    gold_in_retrieved_context: bool | None
    outcome: str
    failure_attribution: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _equal_value(label: GoldStated, predicted: object) -> bool:
    """Money compares normalised amount and currency, not printed scale."""
    gold = label.value
    if isinstance(gold, TextValue):
        return isinstance(predicted, str) and " ".join(gold.text.casefold().split()) == (
            " ".join(predicted.casefold().split())
        )
    if isinstance(gold, CountValue):
        return type(predicted) is int and predicted == gold.count
    if isinstance(gold, DateValue):
        return isinstance(predicted, date) and predicted == gold.date
    if isinstance(gold, MoneyValue):
        return (
            isinstance(predicted, Money)
            and predicted.amount == gold.money.amount
            and predicted.currency == gold.money.currency
        )
    raise TypeError(f"unknown gold value type: {type(gold).__name__}")


def score_case(
    label: GoldLabel,
    prediction: BaseModel,
    *,
    text: str,
    chunks: list[Chunk],
    retrieved: list[Hit],
) -> CaseResult:
    """Classify one labelled result without mistaking model failure for abstention.

    `retrieved` must contain the same top-k hits that the extractor saw. Gold
    presence requires its complete span to be present in one retrieved chunk.
    """
    gold = label.answer
    if isinstance(gold, GoldStated) and text[gold.span.start : gold.span.end] != gold.span.quote:
        raise ValueError("gold label is stale: quote no longer matches derived text")

    available = (
        any(
            hit.chunk.start <= gold.span.start and gold.span.end <= hit.chunk.end
            for hit in retrieved
        )
        if isinstance(gold, GoldStated)
        else None
    )
    grounded: bool | None = None
    aligned: bool | None = None
    right_value = False

    if isinstance(prediction, Found):
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        chunk = chunk_by_id.get(prediction.citation.chunk_id)
        in_context = prediction.citation.chunk_id in {hit.chunk.chunk_id for hit in retrieved}
        citation = prediction.citation
        grounded = bool(
            chunk is not None
            and in_context
            and citation.end <= len(chunk.text)
            and chunk.text[citation.start : citation.end] == citation.quote
            and text[chunk.start + citation.start : chunk.start + citation.end] == citation.quote
        )
        if isinstance(gold, GoldStated):
            right_value = _equal_value(gold, prediction.value)
            aligned = (
                spans_agree(gold.span, resolve_citation(citation, chunk_start=chunk.start))
                if grounded and chunk is not None
                else False
            )
    elif not isinstance(prediction, (NotStated, Failed)):
        raise TypeError(f"unexpected prediction type: {type(prediction).__name__}")

    if isinstance(gold, GoldNotStated):
        if isinstance(prediction, NotStated):
            outcome, attribution = "correct_abstention", "none"
        elif isinstance(prediction, Failed):
            outcome, attribution = "extraction_failure", "provider_or_validation_error"
        else:
            outcome, attribution = "false_positive", "unsupported_claim"
    elif isinstance(prediction, Found) and right_value and grounded and aligned:
        outcome, attribution = "correct_found", "none"
    else:
        if isinstance(prediction, NotStated):
            outcome = "missed_stated"
        elif isinstance(prediction, Failed):
            outcome = "extraction_failure"
        elif not right_value:
            outcome = "wrong_value"
        else:
            outcome = "wrong_evidence"
        if available is False:
            attribution = "retrieval_miss"
        elif isinstance(prediction, Failed):
            attribution = "provider_or_validation_error"
        else:
            attribution = "generation_or_selection_error"

    return CaseResult(
        entry_id=label.entry_id,
        field=label.field.value,
        gold_status=gold.status,
        predicted_status=prediction.status,
        prediction=prediction.model_dump(mode="json"),
        value_correct=right_value,
        citation_grounded=grounded,
        gold_span_aligned=aligned,
        gold_in_retrieved_context=available,
        outcome=outcome,
        failure_attribution=attribution,
    )


def summarise(cases: list[CaseResult]) -> dict[str, Any]:
    """Keep value accuracy distinct from verified citation quality.

    Value F1: TP is a Found value equal to the stated gold value. A wrong
    Found counts as both FP and FN; a Found on absent gold counts as FP.
    Evidence F1 uses the same rule, but TP also requires a gold-aligned,
    source-verifiable citation. Undefined rates become null, never zero.
    """

    def prf(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }

    def one(rows: list[CaseResult]) -> dict[str, Any]:
        value_tp = sum(
            r.gold_status == "stated" and r.predicted_status == "found" and r.value_correct
            for r in rows
        )
        evidence_tp = sum(r.outcome == "correct_found" for r in rows)
        found = sum(r.predicted_status == "found" for r in rows)
        stated = sum(r.gold_status == "stated" for r in rows)
        absent = sum(r.gold_status == "not_stated" for r in rows)
        correct_absence = sum(r.outcome == "correct_abstention" for r in rows)
        return {
            "labelled_cases": len(rows),
            "value": prf(value_tp, found - value_tp, stated - value_tp),
            "evidence_qualified": prf(evidence_tp, found - evidence_tp, stated - evidence_tp),
            "correct_abstention_rate": correct_absence / absent if absent else None,
            "correct_abstentions": correct_absence,
            "gold_not_stated": absent,
            "grounded_citation_rate": (
                sum(r.citation_grounded is True for r in rows) / found if found else None
            ),
            "gold_aligned_citation_rate": (
                sum(r.gold_span_aligned is True for r in rows) / found if found else None
            ),
            "found_predictions": found,
            "outcomes": dict(sorted(Counter(r.outcome for r in rows).items())),
            "failure_attribution": dict(
                sorted(
                    Counter(
                        r.failure_attribution for r in rows if r.failure_attribution != "none"
                    ).items()
                )
            ),
        }

    return {
        "scope": "labelled_pairs_only",
        "overall": one(cases),
        "by_field": {
            field: one([r for r in cases if r.field == field])
            for field in sorted({r.field for r in cases})
        },
    }

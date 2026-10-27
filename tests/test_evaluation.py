"""Offline checks for the score definitions, no invented accuracy numbers."""

from datetime import UTC, datetime

from pydantic import BaseModel

from docintel.evaluation import CaseResult, score_case, summarise
from docintel.extraction.schema import Citation, Failed, Found, NotStated
from docintel.gold import (
    CountValue,
    DocumentSpan,
    GoldField,
    GoldLabel,
    GoldNotStated,
    GoldStated,
)
from docintel.retrieval.chunker import chunk_document
from docintel.retrieval.retriever import Hit

NOW = datetime(2026, 8, 24, tzinfo=UTC)
TEXT = "Employees worldwide were 2,400 in 2024."
DOC = "cik/accession"
CHUNKS = chunk_document(TEXT, document_id=DOC)


def label(*, stated: bool = True) -> GoldLabel:
    answer = (
        GoldStated(
            value=CountValue(count=2400),
            span=DocumentSpan(start=25, end=30, quote="2,400"),
        )
        if stated
        else GoldNotStated()
    )
    return GoldLabel(
        entry_id=DOC,
        field=GoldField.EMPLOYEE_COUNT,
        derivation_version=1,
        answer=answer,
        labelled_at=NOW,
    )


def found(value: int = 2400, *, start: int = 25, end: int = 30) -> Found[int]:
    return Found[int](
        value=value,
        citation=Citation(
            chunk_id=CHUNKS[0].chunk_id,
            start=start,
            end=end,
            quote=TEXT[start:end],
        ),
    )


def check(prediction: BaseModel, *, stated: bool = True, hits: bool = True) -> CaseResult:
    assert isinstance(prediction, (Found, NotStated, Failed))
    return score_case(
        label(stated=stated),
        prediction,
        text=TEXT,
        chunks=CHUNKS,
        retrieved=[Hit(chunk=CHUNKS[0], score=1.0)] if hits else [],
    )


def test_correct_found_is_cited_and_scores_tp() -> None:
    result = check(found())
    assert result.outcome == "correct_found"
    assert result.gold_in_retrieved_context is True
    assert result.gold_span_aligned is True
    assert result.citation_grounded is True
    metrics = summarise([result])["overall"]
    assert metrics["evidence_qualified"]["f1"] == 1.0
    assert metrics["correct_abstention_rate"] is None


def test_model_quote_containing_gold_span_is_aligned() -> None:
    result = check(found(start=16, end=34))
    assert result.gold_span_aligned is True
    assert result.outcome == "correct_found"


def test_wrong_found_counts_as_fp_and_fn() -> None:
    result = check(found(value=999))
    assert result.outcome == "wrong_value"
    metrics = summarise([result])["overall"]
    assert metrics["evidence_qualified"]["tp"] == 0
    assert metrics["evidence_qualified"]["fp"] == 1
    assert metrics["evidence_qualified"]["fn"] == 1
    assert metrics["value"]["f1"] == 0.0
    assert metrics["evidence_qualified"]["f1"] == 0.0


def test_wrong_evidence_is_not_correct_found() -> None:
    result = check(found(start=0, end=9))
    assert result.citation_grounded is True
    assert result.gold_span_aligned is False
    assert result.outcome == "wrong_evidence"
    metrics = summarise([result])["overall"]
    assert metrics["value"]["f1"] == 1.0
    assert metrics["evidence_qualified"]["f1"] == 0.0


def test_true_abstention_and_false_positive() -> None:
    good = check(NotStated(), stated=False)
    bad = check(found(), stated=False)
    assert good.outcome == "correct_abstention"
    assert bad.outcome == "false_positive"
    metrics = summarise([good, bad])["overall"]
    assert metrics["correct_abstention_rate"] == 0.5


def test_failed_does_not_count_as_abstention() -> None:
    result = check(Failed(error="provider_error: timeout"), stated=False)
    assert result.outcome == "extraction_failure"
    assert result.failure_attribution == "provider_or_validation_error"
    assert summarise([result])["overall"]["correct_abstention_rate"] == 0.0


def test_stated_absence_with_retrieval_miss() -> None:
    result = check(NotStated(), hits=False)
    assert result.outcome == "missed_stated"
    assert result.failure_attribution == "retrieval_miss"


def test_stated_absence_with_evidence_in_context() -> None:
    result = check(NotStated())
    assert result.failure_attribution == "generation_or_selection_error"


def test_undefined_f1_uses_null() -> None:
    only_absent = check(NotStated(), stated=False)
    summary = summarise([only_absent])["overall"]
    assert summary["evidence_qualified"]["f1"] is None
    assert summary["correct_abstention_rate"] == 1.0

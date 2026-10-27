"""Pure citation-coordinate checks for PostgreSQL import."""

import pytest

from scripts.load_results import doc_span


def test_converts_chunk_offsets_to_document_offsets() -> None:
    text = "prefix The employee count is 2,400. suffix"
    start = text.index("The employee count")
    quote = "employee count is 2,400"
    prediction = {
        "citation": {
            "chunk_id": f"doc/one@{start}",
            "start": 4,
            "end": 4 + len(quote),
            "quote": quote,
        }
    }
    assert doc_span("doc/one", prediction, text) == (
        f"doc/one@{start}",
        start + 4,
        start + 4 + len(quote),
        quote,
    )


def test_does_not_accept_different_document() -> None:
    response = {"citation": {"chunk_id": "other@0", "start": 0, "end": 5, "quote": "hello"}}
    with pytest.raises(ValueError, match="invalid chunk"):
        doc_span("mine", response, "hello")


def test_does_not_accept_fake_quote() -> None:
    response = {"citation": {"chunk_id": "mine@0", "start": 0, "end": 5, "quote": "HELLO"}}
    with pytest.raises(ValueError, match="does not match"):
        doc_span("mine", response, "hello")

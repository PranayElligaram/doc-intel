"""The offset contract, and that retrieval ranks the caption the answer sits under.

The first half of this file exists because of one sentence in docintel/gold.py:
chunks must be verbatim slices that record where they were cut from. Every
stored citation in the project depends on it, so it is asserted directly,
including against the real fixture rather than only against hand-written text.

`resolve_citation` is exercised end to end here on purpose. A chunker test that
only checked slices could pass while the composition of chunker and resolver
still landed on the wrong characters.
"""

from pathlib import Path

import pytest

from docintel.extraction.schema import Citation
from docintel.gold import resolve_citation
from docintel.ingest.html_text import derive_text
from docintel.retrieval.chunker import (
    Chunk,
    ChunkingError,
    chunk_document,
    chunk_id_for,
    tokenise,
    verify_chunks,
)
from docintel.retrieval.retriever import BM25Retriever, Retriever

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_10k_trimmed.htm"

FILING = (
    "PART II\n\n"
    "Item 7. Management's Discussion and Analysis\n\n"
    "Total net sales increased 2% during 2024 compared to 2023.\n\n"
    "Total net sales 391,035\n\n"
    "Human Capital\n\n"
    "As of September 28, 2024, the Company had approximately 164,000 "
    "full-time equivalent employees.\n\n"
    "Report of Independent Registered Public Accounting Firm\n\n"
    "We have served as the Company's auditor since 2009.\n"
)


def chunks_of(text: str, *, size: int = 120, overlap: int = 30) -> list[Chunk]:
    return chunk_document(text, document_id="0000320193/x", size=size, overlap=overlap)


# --- the offset contract ---------------------------------------------------


def test_every_chunk_is_a_verbatim_slice_at_its_recorded_offset() -> None:
    chunks = chunks_of(FILING)

    assert chunks
    for chunk in chunks:
        assert FILING[chunk.start : chunk.end] == chunk.text


def test_chunks_cover_the_document_with_no_gap() -> None:
    assert verify_chunks(FILING, chunks_of(FILING)) == []


def test_the_real_derived_text_also_satisfies_the_contract() -> None:
    """The hand-written sample has no tables, page furniture or odd whitespace."""
    text = derive_text(FIXTURE.read_text(encoding="utf-8"))
    chunks = chunk_document(text, document_id="0000320193/aapl", size=1200, overlap=200)

    assert len(chunks) > 1
    assert verify_chunks(text, chunks) == []


def test_verify_chunks_reports_a_chunk_that_is_not_a_verbatim_slice() -> None:
    """A chunker that tidied whitespace would look exactly like this."""
    tampered = [
        Chunk(chunk_id=chunk_id_for("d", 0), start=0, text=FILING[:80].replace("\n\n", " ")),
    ]

    problems = verify_chunks(FILING, tampered)

    assert any("not a verbatim slice" in problem for problem in problems)


def test_verify_chunks_reports_a_gap() -> None:
    """A fact in a gap is unretrievable and would score as a false abstention."""
    chunks = chunks_of(FILING)
    with_hole = [chunks[0], *chunks[2:]]

    assert any("gap in coverage" in problem for problem in verify_chunks(FILING, with_hole))


def test_a_citation_resolves_back_to_the_text_it_quoted() -> None:
    """The composition of chunker and resolver, which is what the eval relies on."""
    chunks = chunks_of(FILING)
    quote = "391,035"
    holder = next(c for c in chunks if quote in c.text)
    offset = holder.text.index(quote)

    citation = Citation(
        chunk_id=holder.chunk_id, start=offset, end=offset + len(quote), quote=quote
    )
    span = resolve_citation(citation, chunk_start=holder.start)

    assert FILING[span.start : span.end] == quote


def test_chunking_is_deterministic() -> None:
    first = chunks_of(FILING)
    second = chunks_of(FILING)

    assert [(c.chunk_id, c.start, c.text) for c in first] == [
        (c.chunk_id, c.start, c.text) for c in second
    ]


def test_chunk_ids_carry_their_offset_so_a_stale_citation_is_detectable() -> None:
    chunk = chunks_of(FILING)[1]

    assert chunk.chunk_id == f"0000320193/x@{chunk.start}"


def test_an_overlap_at_or_above_the_size_is_refused() -> None:
    """Clamping it would produce a chunking no recorded run could reproduce."""
    with pytest.raises(ChunkingError, match="never advance"):
        chunk_document(FILING, document_id="d", size=100, overlap=100)


def test_an_empty_document_produces_no_chunks() -> None:
    assert chunk_document("", document_id="d") == []


def test_a_document_shorter_than_one_chunk_is_a_single_chunk() -> None:
    chunks = chunk_document("Total net sales 391,035", document_id="d", size=1200)

    assert len(chunks) == 1
    assert chunks[0].start == 0


def test_cuts_prefer_a_blank_line_over_mid_sentence() -> None:
    """A blank line inside the search window wins over the hard character limit."""
    head = "alpha beta gamma delta epsilon zeta\n\n"
    text = head + "eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    # The blank line sits at offset 35, inside the last quarter of a 40-char window.
    chunks = chunk_document(text, document_id="d", size=40, overlap=5)

    assert chunks[0].text == head
    assert chunks[0].text.endswith("\n\n")


def test_a_boundary_outside_the_search_window_is_not_hunted_for() -> None:
    """The lookback is bounded, so a cut is never dragged far back to find a break."""
    text = "alpha\n\n" + "b" * 200
    chunks = chunk_document(text, document_id="d", size=40, overlap=5)

    # The blank line at offset 5 is well outside the last quarter of the window,
    # so the first chunk runs to the hard limit rather than ending at 7.
    assert len(chunks[0].text) == 40


# --- tokenising ------------------------------------------------------------


def test_a_figure_stays_one_token() -> None:
    """Split into 391 and 035 it would match every three-digit number in a 10-K."""
    assert "391,035" in tokenise("Total net sales 391,035")


def test_trailing_punctuation_is_not_part_of_a_token() -> None:
    assert tokenise("ended December 31, 2024.") == ["ended", "december", "31", "2024"]


# --- retrieval -------------------------------------------------------------


def make_retriever(text: str = FILING) -> Retriever:
    # Large enough that a heading and the sentence under it land in one chunk,
    # which is the regime the real corpus is chunked in.
    return BM25Retriever(chunks_of(text, size=220, overlap=40))


def test_the_retriever_satisfies_the_protocol() -> None:
    """The annotation is the assertion: mypy rejects a BM25Retriever that drifts."""
    retriever: Retriever = BM25Retriever(chunks_of(FILING))

    assert isinstance(retriever, BM25Retriever)


def test_the_caption_the_answer_sits_under_ranks_first() -> None:
    hits = make_retriever().search("Total net sales", limit=3)

    assert hits
    assert "Total net sales" in hits[0].chunk.text


def test_a_different_field_retrieves_a_different_chunk() -> None:
    """Retrieval has to discriminate, not just return the longest chunk every time."""
    retriever = make_retriever()

    employees = retriever.search("Human Capital employees", limit=1)
    auditor = retriever.search("auditor since", limit=1)
    revenue = retriever.search("Total net sales", limit=1)

    assert "Human Capital" in employees[0].chunk.text
    assert "auditor" in auditor[0].chunk.text
    assert "Total net sales" in revenue[0].chunk.text
    assert employees[0].chunk.chunk_id != auditor[0].chunk.chunk_id


def test_chunks_matching_nothing_are_not_returned() -> None:
    """Padding context with chunks that cannot hold the answer hides retrieval misses."""
    hits = make_retriever().search("cryptocurrency mining", limit=5)

    assert hits == []


def test_results_are_ordered_by_score_then_offset() -> None:
    hits = make_retriever().search("total net sales", limit=5)

    assert hits == sorted(hits, key=lambda hit: (-hit.score, hit.chunk.start))


def test_search_is_deterministic() -> None:
    query = "Total net sales"

    first = [hit.chunk.chunk_id for hit in make_retriever().search(query, limit=3)]
    second = [hit.chunk.chunk_id for hit in make_retriever().search(query, limit=3)]

    assert first == second


def test_limit_is_honoured() -> None:
    assert len(make_retriever().search("the company", limit=2)) <= 2


def test_a_non_positive_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        make_retriever().search("revenue", limit=0)


def test_an_empty_query_retrieves_nothing() -> None:
    assert make_retriever().search("", limit=3) == []


def test_a_document_wide_term_does_not_score_negatively() -> None:
    """Unfloored IDF would rank a chunk below one containing none of the query."""
    text = "\n\n".join(f"the company reported item {n}" for n in range(12))
    retriever = BM25Retriever(chunk_document(text, document_id="d", size=40, overlap=10))

    assert all(hit.score >= 0.0 for hit in retriever.search("company", limit=12))

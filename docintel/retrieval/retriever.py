"""Finding the chunks most likely to contain a field, without a vector database.

BM25 over the document's own chunks. Written here rather than imported, for the
reason `CLAUDE.md` gives: this is the part an interviewer asks about, and a
call into someone else's retriever is not an answer to the question.

WHY LEXICAL AND NOT EMBEDDINGS. The queries are the captions a filing prints,
and the answer sits next to the caption. "Total net sales" retrieves the line
that says Total net sales. Embeddings earn their cost on paraphrase, which is
the problem this corpus does not have, and they would add a model, a vector
store and an index build to a project whose measured claim is about grounding
and abstention rather than about retrieval sophistication. The honest framing
is that BM25 is the baseline and the eval's failure attribution is what says
whether retrieval is the thing worth improving. If retrieval misses turn out to
dominate the errors, that is the measured argument for embeddings, and the
`Retriever` protocol below is the seam to put them behind.

WHY THE ANCHOR TABLE IS NOT USED HERE. `gold.ANCHORS` lists the phrases each
field is found by, and it would obviously improve retrieval. It is also the
table the gold labels were located with, so using it would tune retrieval on
the answer key and report a retrieval rate that no unseen document would
reproduce. The queries come from the extraction layer's field prompts instead.

IDF IS DOCUMENT-LOCAL. The corpus for scoring is one filing's chunks, not all
36 filings. A term common across filings but rare within this one should score
as rare here, because the question is which part of this document to read.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from docintel.retrieval.chunker import Chunk, tokenise

# The usual BM25 constants. k1 controls how fast term frequency saturates, b
# how strongly length normalisation applies. Not tuned on this corpus: tuning
# them against the gold set would be fitting retrieval to the answer key.
K1 = 1.5
B = 0.75


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and the score it was ranked by."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Ranks a document's chunks against a query.

    The seam an embedding retriever would slot into without the extraction
    layer changing. Kept to one method for the same reason `LLMProvider` is
    narrow: every capability added here is another thing a replacement has to
    reproduce faithfully.
    """

    def search(self, query: str, *, limit: int) -> list[Hit]:
        """The `limit` highest scoring chunks, best first, ties broken by offset."""
        ...


class BM25Retriever:
    """BM25 over one document's chunks.

    The index is built once per document in `__init__` and the document is
    small, so there is nothing to persist and no cache to invalidate. A corpus
    of 36 filings chunked at 1200 characters is a few thousand chunks in total.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._term_frequencies: list[Counter[str]] = [Counter(tokenise(c.text)) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_frequencies]
        self._average_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        document_frequency: Counter[str] = Counter()
        for tf in self._term_frequencies:
            document_frequency.update(tf.keys())
        self._document_frequency = document_frequency

    def search(self, query: str, *, limit: int) -> list[Hit]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        terms = tokenise(query)
        if not terms or not self._chunks:
            return []

        scored = [
            Hit(chunk=chunk, score=self._score(terms, index))
            for index, chunk in enumerate(self._chunks)
        ]
        # A zero score means not one query term appears. Returning those would
        # pad the model's context with text that cannot contain the answer, and
        # would make a retrieval miss look like a retrieval hit the model
        # failed to read.
        scored = [hit for hit in scored if hit.score > 0.0]

        # Offset ascending breaks ties, so a rerun ranks identically. Ordering
        # ties by list position would be stable too but would depend on the
        # chunker's output order, which is one more thing to keep true.
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.start))
        return scored[:limit]

    def _score(self, terms: list[str], index: int) -> float:
        tf = self._term_frequencies[index]
        length = self._lengths[index]
        total = 0.0
        for term in terms:
            frequency = tf.get(term, 0)
            if frequency == 0:
                continue
            numerator = frequency * (K1 + 1)
            denominator = frequency + K1 * (
                1 - B + B * (length / self._average_length if self._average_length else 1.0)
            )
            total += self._idf(term) * (numerator / denominator)
        return total

    def _idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF, floored at zero.

        The +0.5 smoothing can go negative for a term present in more than half
        the chunks. Left unfloored, a common word would subtract from a score
        and a chunk could be pushed below a chunk containing none of the query,
        which inverts the ranking it is supposed to refine.
        """
        n = len(self._chunks)
        df = self._document_frequency.get(term, 0)
        return max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))

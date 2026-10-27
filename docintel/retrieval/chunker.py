"""Cutting a derived text into chunks that a citation can still be trusted from.

The contract is not this module's to invent. `docintel/gold.py` already states
it: every chunk records the offset in the derived text where it starts, and
`chunk.text` is a verbatim slice from there, with no re-wrapping, no whitespace
tidying and no joining across a gap. `resolve_citation` adds `chunk_start` to a
chunk-relative offset and expects to land on the same characters.

That one rule decides almost everything below.

It rules out every convenience a chunker normally offers. No collapsing runs of
whitespace, no stripping, no dedenting tables, no dropping page furniture. All
of those are improvements to the text a model reads and all of them silently
shift every offset after the edit, so a citation would resolve to a span that
no longer says what the model quoted. The text on disk is already the derived
text; this module only decides where to cut it.

It also decides what a gap would cost. Chunks tile the document with overlap
and never skip a region, because a fact that falls between two chunks cannot be
retrieved, and its field would score as a not_stated that the eval has no way
to distinguish from a real abstention. Overlap exists for the same reason: a
revenue figure printed across a cut would otherwise be unreadable from either
side.

Chunk ids carry their own start offset. A citation naming `...@41892` can be
checked against the chunk it claims to come from, so a stale citation from a
previous chunking is a loud failure rather than an offset that happens to
resolve to the wrong sentence.

Size and overlap are parameters rather than constants because chunk size is one
of the things the evaluation is meant to ablate. Nothing here is tuned; the
defaults are a starting point for that experiment, not a result.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

# Large enough to hold a table row with its heading, small enough that a 7B
# model's attention is not spread over several candidate revenue lines at once.
DEFAULT_CHUNK_SIZE = 1200

# Roughly one long sentence. Enough that a figure and the caption above it
# survive a cut between them.
DEFAULT_OVERLAP = 200

# How far back from the ideal cut to look for a natural boundary before giving
# up and cutting mid-sentence. A fraction of the size rather than a constant,
# so the behaviour does not change character when the size is ablated.
BOUNDARY_SEARCH_FRACTION = 0.25

# Tried in order. A blank line is a real structural boundary in the derived
# text; a single newline usually ends a table row; a full stop ends a sentence.
BOUNDARIES = ("\n\n", "\n", ". ")


class ChunkingError(ValueError):
    """The chunker was asked for something that would break the offset contract."""


class Chunk(BaseModel):
    """A verbatim slice of a document's derived text, and where it was cut from.

    `start` is the offset in the derived text. `text` must equal
    `derived_text[start:start + len(text)]`. Nothing in this model can enforce
    that on its own, because a Chunk does not hold the document it came from,
    which is exactly why `verify_chunks` exists and why the eval runs it before
    trusting any offset.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    text: str = Field(min_length=1)

    @property
    def end(self) -> int:
        """One past the last character, in document coordinates."""
        return self.start + len(self.text)


def chunk_id_for(document_id: str, start: int) -> str:
    """The id a chunk of `document_id` beginning at `start` gets.

    Deterministic and self-describing: re-chunking the same document with the
    same settings produces the same ids, and an id can be checked against the
    chunk it is attached to.
    """
    return f"{document_id}@{start}"


def chunk_document(
    text: str,
    *,
    document_id: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Cut `text` into overlapping verbatim slices.

    Raises `ChunkingError` rather than clamping a nonsensical size or overlap.
    An overlap at or above the size would make the window fail to advance, and
    silently correcting it would produce a chunking nobody asked for and no
    recorded run could reproduce.
    """
    if size <= 0:
        raise ChunkingError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ChunkingError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ChunkingError(
            f"overlap ({overlap}) must be smaller than size ({size}), or the window "
            "would never advance"
        )
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = _cut_at(text, start=start, size=size)
        chunks.append(
            Chunk(chunk_id=chunk_id_for(document_id, start), start=start, text=text[start:end])
        )
        if end >= len(text):
            break
        # Step back by the overlap, but never to or behind the current start:
        # that would repeat a chunk forever.
        start = max(start + 1, end - overlap)
    return chunks


def _cut_at(text: str, *, start: int, size: int) -> int:
    """Where this chunk should end, preferring a natural boundary.

    Only ever returns an offset in `(start, start + size]`, so the window
    always advances and never exceeds the requested size.
    """
    hard_end = min(start + size, len(text))
    if hard_end >= len(text):
        return len(text)

    earliest = hard_end - int(size * BOUNDARY_SEARCH_FRACTION)
    earliest = max(earliest, start + 1)

    for boundary in BOUNDARIES:
        found = text.rfind(boundary, earliest, hard_end)
        if found != -1:
            # Cut after the boundary so it stays with the chunk that ends on
            # it, which keeps the slice verbatim either way.
            return found + len(boundary)
    return hard_end


def verify_chunks(text: str, chunks: list[Chunk]) -> list[str]:
    """Check the offset contract, returning one message per violation.

    Returns problems rather than raising, because the eval reports a chunker
    fault as a harness fault covering every affected chunk at once. Raising on
    the first one would make a systematic break look like a single bad chunk.

    An empty list means every chunk is a verbatim slice at its recorded offset,
    the ids match their offsets, and the chunks cover the document with no gap.
    """
    problems: list[str] = []
    if not chunks:
        if text:
            problems.append("no chunks were produced for a non-empty document")
        return problems

    covered_to = 0
    for chunk in sorted(chunks, key=lambda c: c.start):
        actual = text[chunk.start : chunk.end]
        if actual != chunk.text:
            problems.append(
                f"{chunk.chunk_id}: text is not a verbatim slice at [{chunk.start}, {chunk.end})"
            )
        if chunk.start > covered_to:
            problems.append(
                f"{chunk.chunk_id}: gap in coverage, [{covered_to}, {chunk.start}) is in no chunk"
            )
        covered_to = max(covered_to, chunk.end)

    if covered_to < len(text):
        problems.append(f"gap in coverage, [{covered_to}, {len(text)}) is in no chunk")
    return problems


_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,.]*")


def tokenise(text: str) -> list[str]:
    """Lowercased words and numbers, with numeric punctuation kept inside numbers.

    Lives here rather than in the retriever because the chunker and the
    retriever have to agree about what a token is, and two copies of this
    regex would eventually disagree.

    "391,035" stays one token. Splitting it into "391" and "035" would make a
    revenue figure match any chunk containing a three-digit number, which in a
    10-K is all of them. Trailing punctuation is dropped by the character class
    rather than stripped afterwards, so "2024." tokenises as "2024".
    """
    return [match.group(0).rstrip(".,").lower() for match in _TOKEN.finditer(text)]

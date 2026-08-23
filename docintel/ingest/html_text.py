"""Turns an inline-XBRL filing into the text a human would read.

THE TRAP THIS MODULE EXISTS TO CLOSE. A modern 10-K is inline XBRL: the HTML a
person reads carries the tagged accounting values inside it. Feed that HTML to
an extractor and it can read `<ix:nonFraction name="Revenues">307,003` and
report a perfect score without ever having understood a document. That is the
same cheat as handing it the iXBRL rendition outright, wearing a new costume.

What makes stripping legitimate here, and not a laundering of the same cheat:
the HTML *is* the human-readable document and the tags are invisible metadata
laid on top of it. Remove the metadata and what remains is exactly what a
person sees. That was not true of the Companies House iXBRL rendition, whose
layout no reader ever met, which is why that one was quarantined rather than
de-tagged. That corpus is gone - docs/corpus-selection.md has the story - but
the distinction is the whole justification for this module and outlives it.

But it has to be enforced rather than assumed, so the rule is a closed list and
`tests/test_html_text.py` asserts that nothing XBRL survives into corpus text.

THE TWO TIERS, AND WHY EITHER ONE ALONE DESTROYS THE DOCUMENT. The obvious
implementation - delete every `ix:` element - deletes the accounts. The number
is the element's own text content:

    <ix:nonFraction ... name="dei:EntityPublicFloat">3,253,431,000,000</ix:nonFraction>

Drop that element and the public float goes with it. So elements that sit in
the reading flow are UNWRAPPED: the tag is discarded and its text kept, which
is precisely what a browser renders.

The opposite mistake is just as bad. `ix:header` is not rendered at all, and
the `ix:hidden` block inside it holds tagged facts the filer deliberately did
not display - on Apple's 2025 10-K, 22 of them, including which balance-sheet
line each lease asset belongs to. Unwrapping those would inject answer-key
values into the corpus text that no reader could ever have seen. So the
non-rendered elements are DELETED WITH THEIR SUBTREES.

Unwrap what a reader sees. Delete what a reader cannot.
"""

import re
from html import escape
from html.parser import HTMLParser

# Bumped by hand whenever a rule below changes the bytes this module produces.
#
# It is recorded next to every derived text file so that a citation - which
# indexes into that text by character offset - can be told apart from a
# citation made against a different derivation. Offsets do not survive a change
# here, and they are not meant to: `Citation` in docintel/extraction/schema.py
# stores the quote alongside the span and re-checks it verbatim, so a stale
# citation fails loudly. This version is what makes that failure diagnosable
# rather than mysterious.
DERIVATION_VERSION = 1

# Not rendered by a browser, so not part of what a reader saw. Deleted with
# everything inside them. `ix:hidden` is the dangerous one and the reason this
# set exists; the other three are its container and its siblings.
INVISIBLE_IX_ELEMENTS = frozenset(
    {
        "ix:header",
        "ix:hidden",
        "ix:resources",
        "ix:references",
    }
)

# Same treatment for the same reason: content that is not reading matter.
NON_RENDERED_HTML_ELEMENTS = frozenset({"script", "style"})

_DELETED_SUBTREES = INVISIBLE_IX_ELEMENTS | NON_RENDERED_HTML_ELEMENTS

# XBRL fact attributes, stripped from any element that carries them.
#
# Defensive, and knowingly so. Every one of these lives on an `ix:` element in
# the filings we have looked at, and `ix:` elements are unwrapped attributes and
# all - so on those files this set never fires. It is here because we have read
# a handful of filers out of thousands, and an XBRL attribute reaching corpus
# text would be a silent leak of exactly the kind this module exists to
# prevent. None of these is a valid HTML attribute, so stripping them cannot
# damage markup that legitimately uses one.
#
# `name` and `target` are deliberately absent: both are ordinary HTML
# attributes, and stripping them would corrupt real markup to guard against a
# leak that unwrapping has already handled.
XBRL_FACT_ATTRIBUTES = frozenset(
    {
        "contextref",
        "unitref",
        "decimals",
        "scale",
        "format",
        "sign",
        "escape",
        "continuedat",
        "tupleref",
    }
)

# Elements after which a reader sees a line break. Table cells are handled
# separately: they need a separator, not a newline, or the figures in a row of
# an income statement run together into one meaningless number.
_BLOCK_ELEMENTS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    }
)

_CELL_ELEMENTS = frozenset({"td", "th"})


def _is_ix(tag: str) -> bool:
    return tag.startswith("ix:")


def _keep_attribute(name: str) -> bool:
    """Decides whether an attribute survives into the cleaned HTML.

    Any namespaced attribute goes, which is what removes the `xmlns:ix`,
    `xmlns:xbrli` and `xmlns:xbrldi` declarations on the root element along
    with anything qualified by them. HTMLParser has already lower-cased the
    name, so the comparison against XBRL_FACT_ATTRIBUTES is exact.
    """
    if name == "xmlns" or ":" in name:
        return False
    return name not in XBRL_FACT_ATTRIBUTES


class _InlineXbrlStripper(HTMLParser):
    """Removes inline XBRL, leaving the HTML a reader would have been served.

    convert_charrefs is off because this parser re-emits HTML: entities are
    passed through exactly as they were written, rather than decoded here and
    re-encoded on the way out.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []
        # The element whose subtree we are discarding, and how deep we are
        # inside it. A counter rather than a flag because an element of the
        # same name can nest inside itself.
        self._skipping: str | None = None
        self._depth = 0

    def result(self) -> str:
        return "".join(self._out)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skipping is not None:
            if tag == self._skipping:
                self._depth += 1
            return
        if tag in _DELETED_SUBTREES:
            self._skipping = tag
            self._depth = 1
            return
        if _is_ix(tag):
            # Unwrapped: the tag goes, the text inside it stays.
            return
        self._out.append(self._render_starttag(tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A self-closing element has no subtree, so a deleted one is simply
        # dropped and never opens a skip region.
        if self._skipping is not None or tag in _DELETED_SUBTREES or _is_ix(tag):
            return
        self._out.append(self._render_starttag(tag, attrs, self_closing=True))

    def handle_endtag(self, tag: str) -> None:
        if self._skipping is not None:
            if tag == self._skipping:
                self._depth -= 1
                if self._depth == 0:
                    self._skipping = None
            return
        if _is_ix(tag):
            return
        self._out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._skipping is None:
            self._out.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._skipping is None:
            self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skipping is None:
            self._out.append(f"&#{name};")

    def handle_decl(self, decl: str) -> None:
        if self._skipping is None:
            self._out.append(f"<!{decl}>")

    def handle_comment(self, data: str) -> None:
        """Dropped: a comment is not reading matter, and EDGAR filings carry
        generator fingerprints and document GUIDs in them."""

    def handle_pi(self, data: str) -> None:
        """Dropped: this is the leading `<?xml ...?>` declaration."""

    def _render_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]], *, self_closing: bool = False
    ) -> str:
        rendered = [tag]
        for name, value in attrs:
            if not _keep_attribute(name):
                continue
            # HTMLParser decodes attribute values whatever convert_charrefs
            # says, so re-escaping here is what keeps the output well-formed.
            rendered.append(name if value is None else f'{name}="{escape(value, quote=True)}"')
        return f"<{' '.join(rendered)}{' /' if self_closing else ''}>"


class _TextExtractor(HTMLParser):
    """Flattens cleaned HTML into the reading text.

    convert_charrefs is on here: this parser produces text, so entities should
    arrive already decoded.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skipping: str | None = None
        self._depth = 0

    def result(self) -> str:
        return "".join(self._out)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skipping is not None:
            if tag == self._skipping:
                self._depth += 1
            return
        # Belt and braces: the stripper has already removed these. If it ever
        # stops doing so, stylesheet source must still not become corpus text.
        if tag in NON_RENDERED_HTML_ELEMENTS:
            self._skipping = tag
            self._depth = 1
            return
        if tag == "br":
            self._out.append("\n")
        elif tag in _CELL_ELEMENTS:
            self._out.append(" ")
        elif tag in _BLOCK_ELEMENTS:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skipping is not None:
            if tag == self._skipping:
                self._depth -= 1
                if self._depth == 0:
                    self._skipping = None
            return
        if tag in _BLOCK_ELEMENTS:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skipping is None:
            self._out.append(data)


# Characters that carry no meaning to a reader but would otherwise take up
# offsets inside a citation span. Non-breaking spaces are everywhere in EDGAR
# filings, used to pad the columns of financial tables.
_INVISIBLE_CHARACTERS = str.maketrans(
    {
        "\xa0": " ",  # no-break space
        "​": "",  # zero-width space
        "﻿": "",  # byte-order mark / zero-width no-break space
    }
)

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def strip_inline_xbrl(html: str) -> str:
    """Removes inline XBRL, returning HTML with the reading content intact.

    Elements in the reading flow are unwrapped; the non-rendered ones are
    deleted with their subtrees. See the module docstring for why it has to be
    both.
    """
    parser = _InlineXbrlStripper()
    parser.feed(html)
    parser.close()
    return parser.result()


def derive_text(raw_html: str) -> str:
    """Derives the corpus text from a filing exactly as fetched.

    This is the whole pipeline, and it takes RAW html on purpose: there is no
    way to reach the text without the strip having happened, so no caller can
    accidentally derive text straight from an untouched filing.

    The result is what gets written to disk and what every citation offset
    indexes into, so it is a pure, deterministic function of the input - the
    same bytes in always give the same bytes out, or stored citations would
    drift silently. It ends in a single newline, which is the byte-for-byte
    content of the text artefact rather than something the writer appends.
    """
    parser = _TextExtractor()
    parser.feed(strip_inline_xbrl(raw_html))
    parser.close()
    return _normalise(parser.result())


def _normalise(text: str) -> str:
    """Collapses whitespace into something stable enough to cite into.

    Runs of spaces become one space and runs of blank lines become one blank
    line. Paragraph separation is meaning a reader can see, so it survives; the
    amount of padding a filer used to lay out a table is not, so it does not.
    """
    text = text.translate(_INVISIBLE_CHARACTERS)
    lines = (_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    collapsed = _BLANK_RUN.sub("\n\n", "\n".join(lines))
    return collapsed.strip() + "\n"

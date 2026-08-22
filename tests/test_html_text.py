"""Tests for the inline-XBRL strip and the text derivation.

The guard tests at the bottom run over a real filing, not a synthetic one.
`tests/fixtures/aapl_10k_trimmed.htm` is a structural trim of Apple's 2025
10-K: the `ix:header` is real and its `ix:hidden` block is intact, the cover
page and the consolidated statements of operations are real, and the
namespace declarations are the ones the filer sent. A hand-written fixture
would only ever prove that the stripper handles the cases we already thought
of, which is the one thing a guard against leakage must not rest on.
"""

from pathlib import Path

import pytest

from docintel.ingest.html_text import (
    DERIVATION_VERSION,
    XBRL_FACT_ATTRIBUTES,
    derive_text,
    strip_inline_xbrl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_10k_trimmed.htm"


@pytest.fixture(scope="module")
def apple_10k() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def apple_clean(apple_10k: str) -> str:
    return strip_inline_xbrl(apple_10k)


@pytest.fixture(scope="module")
def apple_text(apple_10k: str) -> str:
    return derive_text(apple_10k)


# --- the two tiers ---------------------------------------------------------


def test_a_tagged_number_survives_because_the_element_is_unwrapped() -> None:
    """The whole reason this is not a one-line element deletion."""
    html = (
        '<p><ix:nonFraction contextRef="c-1" name="us-gaap:Revenues">307,003</ix:nonFraction></p>'
    )
    assert "307,003" in derive_text(html)


def test_a_tagged_string_survives_too() -> None:
    html = '<p><ix:nonNumeric name="dei:EntityRegistrantName">Apple Inc.</ix:nonNumeric></p>'
    assert "Apple Inc." in derive_text(html)


def test_continuation_text_survives() -> None:
    """ix:continuation carries the tail of a narrative disclosure."""
    html = '<p>first part<ix:continuation id="f-1">and the rest of it</ix:continuation></p>'
    assert "and the rest of it" in derive_text(html)


@pytest.mark.parametrize("element", ["ix:header", "ix:hidden", "ix:resources", "ix:references"])
def test_non_rendered_elements_are_deleted_with_their_contents(element: str) -> None:
    html = f"<p>visible</p><{element}>INVISIBLE</{element}><p>also visible</p>"
    text = derive_text(html)
    assert "INVISIBLE" not in text
    assert "visible" in text and "also visible" in text


@pytest.mark.parametrize("element", ["script", "style"])
def test_non_reading_html_is_deleted_with_its_contents(element: str) -> None:
    html = f"<p>visible</p><{element}>INVISIBLE</{element}>"
    assert "INVISIBLE" not in derive_text(html)


def test_deleting_a_subtree_takes_nested_tagged_facts_with_it() -> None:
    """Deletion is by subtree, so an unwrappable element inside a deleted one
    is still deleted. Getting this backwards is how hidden facts leak."""
    html = (
        "<ix:header><ix:hidden>"
        '<ix:nonNumeric name="dei:AmendmentFlag">LEAKED</ix:nonNumeric>'
        "</ix:hidden></ix:header><p>kept</p>"
    )
    text = derive_text(html)
    assert "LEAKED" not in text
    assert "kept" in text


def test_an_element_nested_inside_itself_closes_the_right_skip_region() -> None:
    """The skip is depth-counted. A flag would end the region at the first
    closing tag and let the rest of the header through."""
    html = "<ix:header>OUTER<ix:header>INNER</ix:header>TAIL</ix:header><p>after</p>"
    text = derive_text(html)
    assert "OUTER" not in text and "INNER" not in text
    # TAIL is the one that matters: it sits after the inner closing tag, so a
    # flag-based skip would have ended the region and let it through.
    assert "TAIL" not in text
    assert "after" in text


def test_a_self_closing_ix_element_does_not_swallow_the_document() -> None:
    """A self-closing deleted element has no subtree, so it must not open a
    skip region that eats everything following it."""
    html = "<p>before</p><ix:header/><p>after</p>"
    text = derive_text(html)
    assert "before" in text and "after" in text


# --- attribute stripping ---------------------------------------------------


def test_namespace_declarations_are_stripped() -> None:
    html = (
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        'xmlns:xbrli="http://www.xbrl.org/2003/instance"><p>text</p></html>'
    )
    clean = strip_inline_xbrl(html)
    assert "xmlns" not in clean
    assert "text" in clean


@pytest.mark.parametrize("attribute", sorted(XBRL_FACT_ATTRIBUTES))
def test_xbrl_fact_attributes_are_stripped_from_ordinary_elements(attribute: str) -> None:
    """Defensive: in the filings we have read these only ever appear on ix:
    elements, which are unwrapped attributes and all. This proves the belt
    works as well as the braces."""
    clean = strip_inline_xbrl(f'<span {attribute}="c-1">value</span>')
    assert attribute not in clean.lower()
    assert "value" in clean


def test_ordinary_html_attributes_are_left_alone() -> None:
    """`name` and `target` are real HTML attributes. Stripping them to be safe
    would corrupt markup to prevent a leak that unwrapping already handles."""
    clean = strip_inline_xbrl('<a href="/x" target="_blank" name="anchor">link</a>')
    assert 'href="/x"' in clean
    assert 'target="_blank"' in clean
    assert 'name="anchor"' in clean


def test_attribute_values_are_re_escaped() -> None:
    clean = strip_inline_xbrl('<p title="a &amp; b">text</p>')
    assert 'title="a &amp; b"' in clean


# --- text derivation -------------------------------------------------------


def test_entities_are_decoded() -> None:
    assert "Company’s" in derive_text("<p>Company&#8217;s</p>")


def test_non_breaking_spaces_become_ordinary_spaces() -> None:
    text = derive_text("<p>a\xa0\xa0b</p>")
    assert "\xa0" not in text
    assert "a b" in text


def test_zero_width_characters_are_removed() -> None:
    """They occupy citation offsets while being invisible to the reader who
    would have to verify the quote."""
    text = derive_text("<p>a​b﻿c</p>")
    assert "​" not in text and "﻿" not in text
    assert "abc" in text


def test_table_cells_are_separated_rather_than_run_together() -> None:
    """Without a cell separator the figures of an income statement row merge
    into one meaningless number."""
    html = "<table><tr><td>Net sales</td><td>307,003</td><td>294,866</td></tr></table>"
    assert "307,003 294,866" in derive_text(html)


def test_rows_are_separated_by_newlines() -> None:
    """Cells within a row are joined by spaces; a row break is a line break,
    so two rows can never merge into one."""
    html = "<table><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table>"
    text = derive_text(html)
    assert "a 1" in text and "b 2" in text
    assert "1 b" not in text


def test_blank_line_runs_are_collapsed_but_paragraph_breaks_survive() -> None:
    text = derive_text("<div>one</div><div></div><div></div><div>two</div>")
    assert "one\n\ntwo" in text
    assert "\n\n\n" not in text


def test_derivation_ends_in_exactly_one_newline() -> None:
    text = derive_text("<p>text</p>")
    assert text.endswith("text\n")


def test_comments_do_not_reach_the_text() -> None:
    """EDGAR filings carry generator fingerprints and document GUIDs in them."""
    assert "SECRET" not in derive_text("<p>visible</p><!-- SECRET -->")


def test_empty_input_derives_to_a_single_newline() -> None:
    assert derive_text("") == "\n"


def test_derivation_is_deterministic(apple_10k: str, apple_text: str) -> None:
    """Citations index into this text by character offset, so the same bytes in
    must give the same bytes out or every stored citation drifts silently."""
    assert derive_text(apple_10k) == apple_text


def test_derivation_version_is_recorded() -> None:
    assert isinstance(DERIVATION_VERSION, int)
    assert DERIVATION_VERSION >= 1


# --- the guard, over a real filing -----------------------------------------

# Element syntax rather than the bare string "ix:", which appears inside
# ordinary words - "matrix:" contains it - and would make this guard fire on
# innocent prose.
XBRL_MARKERS = ["<ix:", "</ix:", "xmlns:ix", "xmlns:xbrli", "xmlns:xbrldi", "xmlns"]

XBRL_TOKENS = ["contextref", "nonfraction", "nonnumeric", "unitref", "xbrli:", "xbrldi:"]


def test_no_xbrl_markup_survives_into_the_cleaned_html(apple_clean: str) -> None:
    for marker in XBRL_MARKERS:
        assert marker not in apple_clean, f"{marker!r} survived the strip"
    lowered = apple_clean.lower()
    for token in XBRL_TOKENS:
        assert token not in lowered, f"{token!r} survived the strip"


def test_no_xbrl_markup_survives_into_the_corpus_text(apple_text: str) -> None:
    for marker in XBRL_MARKERS:
        assert marker not in apple_text, f"{marker!r} reached the corpus text"
    lowered = apple_text.lower()
    for token in XBRL_TOKENS:
        assert token not in lowered, f"{token!r} reached the corpus text"


# Values that appear in this filing ONLY inside `ix:hidden` - verified against
# the untrimmed original. Each is a tagged fact the filer chose not to display:
# `P1Y` is a remaining-performance-obligation period, and the fasb.org URIs
# state which balance-sheet line each lease asset and liability belongs to.
# A reader of the 10-K never sees any of them. If one appears in corpus text,
# the extractor is being handed an answer key.
HIDDEN_ONLY_FACTS = [
    "P1Y",
    "http://fasb.org/us-gaap/2025#LongTermDebtNoncurrent",
    "http://fasb.org/us-gaap/2025#OtherAssetsNoncurrent",
    "http://fasb.org/us-gaap/2025#PropertyPlantAndEquipmentNet",
    "http://fasb.org/us-gaap/2025#OtherLiabilitiesCurrent",
    "http://fasb.org/us-gaap/2025#OtherLiabilitiesNoncurrent",
]


@pytest.mark.parametrize("fact", HIDDEN_ONLY_FACTS)
def test_the_fixture_really_does_hide_these_facts(apple_10k: str, fact: str) -> None:
    """Guards the guard. If a later re-trim of the fixture moved one of these
    into the visible body, the leak test below would pass for the wrong reason
    and stop protecting anything."""
    assert apple_10k.count(fact) > 0, f"{fact!r} is not in the fixture at all"
    hidden_start = apple_10k.index("<ix:hidden")
    hidden_end = apple_10k.index("</ix:hidden>")
    hidden = apple_10k[hidden_start:hidden_end]
    assert apple_10k.count(fact) == hidden.count(fact), f"{fact!r} also appears outside ix:hidden"


@pytest.mark.parametrize("fact", HIDDEN_ONLY_FACTS)
def test_hidden_facts_never_reach_the_corpus_text(apple_text: str, fact: str) -> None:
    """The specific cheat this module exists to prevent."""
    assert fact not in apple_text


@pytest.mark.parametrize("fact", HIDDEN_ONLY_FACTS)
def test_hidden_facts_never_reach_the_cleaned_html(apple_clean: str, fact: str) -> None:
    assert fact not in apple_clean


# --- what a reader *does* see must survive ---------------------------------


@pytest.mark.parametrize(
    ("value", "what"),
    [
        ("3,253,431,000,000", "the public float, tagged inline in a sentence"),
        ("307,003", "products net sales"),
        ("416,161", "total net sales"),
        ("132,729", "income before provision for income taxes"),
        ("112,010", "net income"),
        ("7.49", "basic earnings per share"),
    ],
)
def test_visible_figures_survive_the_strip(apple_text: str, value: str, what: str) -> None:
    assert value in apple_text, f"lost {what}"


def test_the_income_statement_reads_as_a_reader_would_see_it(apple_text: str) -> None:
    """The row keeps its label and all three fiscal years, in order. This is
    the shape the chunker and the citation spans will work against."""
    assert "Income before provision for income taxes 132,729 123,485 113,736" in apple_text


def test_narrative_prose_survives(apple_text: str) -> None:
    assert "held by non-affiliates of the Registrant" in apple_text


def test_the_strip_removes_most_of_the_file(apple_10k: str, apple_text: str) -> None:
    """Sanity check on the ratio rather than an exact size: the overwhelming
    majority of an inline-XBRL filing is markup and tagging, not reading
    matter. A derivation that suddenly kept much more would mean markup had
    started leaking into the text."""
    assert len(apple_text) < len(apple_10k) * 0.25

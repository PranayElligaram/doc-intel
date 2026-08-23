"""The no-oracle guard, carried over from the Companies House fetcher.

WHAT IS BEING GUARDED. The XBRL companyfacts API hands back every figure the
company ever tagged, keyed by concept. Feed that to an extractor and it is not
reading a document, it is reading a database - and every accuracy number this
project reports becomes a lie. So those bytes live under `oracle/`, apart from
the corpus, and the extraction path must not be able to reach them.

"Must not" needs teeth. A comment saying so survives exactly until someone
needs a value in a hurry. These tests read the source of every module under
docintel/ and fail if the oracle directory is touched from anywhere but a short
allow-list of functions whose whole job is to fetch and store it.

WHAT THIS PROVES AND WHAT IT DOES NOT. It proves nothing reaches the oracle
directory through `oracle_dir()`. It cannot prove nothing reaches those bytes
by another route - `recheck_corpus` reads them through the path recorded in the
manifest, which no AST rule would catch. That one is pinned behaviourally
instead, by test_recheck_rehashes_the_oracle_but_never_derives_from_it in
test_edgar.py. Between them: one test that no new code can quietly call the
directory helper, one that the function which does read those bytes only ever
measures them.
"""

import ast
from pathlib import Path

import pytest

from docintel.ingest.manifest import documents_dir, oracle_dir, text_dir

DOCINTEL = Path(__file__).parent.parent / "docintel"
INGEST = DOCINTEL / "ingest"

# The only functions allowed to name the oracle directory, by module.
#
# Adding a name here is the decision this whole file exists to make someone
# stop and take. Before you do: the function you are adding must write oracle
# bytes to disk or measure the bytes already there. If it reads them for any
# other purpose, the answer is no.
ALLOWED = {
    "manifest.py": {"oracle_dir"},  # the definition itself
    "edgar.py": {"_fetch_oracle"},
}


def modules() -> list[Path]:
    return sorted(path for path in DOCINTEL.rglob("*.py") if "__pycache__" not in path.parts)


def references(tree: ast.AST) -> list[tuple[str | None, int]]:
    """Every mention of `oracle_dir`, with the function it sits in.

    A reference at module level gets None, which is never in an allow-list -
    a module-level `ORACLE = oracle_dir(root)` would be exactly the sort of
    back door this is looking for.
    """
    found: list[tuple[str | None, int]] = []

    def walk(node: ast.AST, enclosing: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if child.name == "oracle_dir":
                    found.append((child.name, child.lineno))
                walk(child, child.name)
                continue
            if isinstance(child, ast.Name) and child.id == "oracle_dir":
                found.append((enclosing, child.lineno))
            walk(child, enclosing)

    walk(tree, None)
    return found


def test_only_allow_listed_functions_may_name_the_oracle_directory() -> None:
    offences: list[str] = []

    for path in modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        permitted = ALLOWED.get(path.name, set())
        for enclosing, lineno in references(tree):
            if enclosing not in permitted:
                where = enclosing or "module level"
                offences.append(f"{path.relative_to(DOCINTEL.parent)}:{lineno} in {where}")

    assert not offences, (
        "oracle_dir() is reachable from code that is not allow-listed in "
        "tests/test_oracle_guard.py: " + ", ".join(offences)
    )


def test_the_allow_list_has_no_dead_entries() -> None:
    """A renamed or deleted function would leave the guard passing vacuously."""
    for filename, names in ALLOWED.items():
        (path,) = [module for module in modules() if module.name == filename]
        defined = {
            node.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert names <= defined, f"{filename} no longer defines {sorted(names - defined)}"


def test_nothing_outside_ingest_mentions_the_oracle_at_all() -> None:
    """Ingest fetches it; retrieval, extraction and the LLM layer have no business
    knowing it exists."""
    for path in modules():
        if INGEST in path.parents:
            continue
        source = path.read_text(encoding="utf-8").lower()
        assert "oracle" not in source, f"{path.relative_to(DOCINTEL.parent)} mentions the oracle"


@pytest.mark.parametrize("root", [Path("/cache"), Path("relative/cache")])
def test_the_three_directories_are_distinct_and_none_contains_another(root: Path) -> None:
    """The structural half.

    documents/ and text/ are what the extraction path may read, usually by
    globbing. If the oracle directory were ever nested inside either of them -
    or were the same directory under a different name - every glob in Phase 3
    would pick up the answer key and no test above would notice.
    """
    documents, text, oracle = documents_dir(root), text_dir(root), oracle_dir(root)

    assert len({documents, text, oracle}) == 3
    for readable in (documents, text):
        assert oracle not in readable.parents
        assert readable not in oracle.parents

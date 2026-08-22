"""Checks the package is installed and importable.

This fails if `pip install -e .` did not run or if setuptools package
discovery missed a subpackage, which is the whole point of Phase 0.
"""

import importlib


def test_subpackages_are_importable() -> None:
    for name in [
        "docintel",
        "docintel.ingest",
        "docintel.retrieval",
        "docintel.extraction",
        "docintel.llm",
    ]:
        assert importlib.import_module(name) is not None

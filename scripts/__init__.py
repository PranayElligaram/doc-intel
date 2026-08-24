"""Makes scripts/ a package, so `scripts.label` has exactly one module name.

Nothing is exported from here and nothing imports this package at runtime -
both CLIs are run as files, `python scripts/label.py`. It exists because
tests/test_label_cli.py imports the CLI to drive `main()` with an argv list,
and without this file mypy sees scripts/label.py twice: as `label` when it
walks the directory it was given on the command line, and as `scripts.label`
through the test's import. It refuses to check either.

setuptools packages only `docintel*`, so this stays out of any build.
"""

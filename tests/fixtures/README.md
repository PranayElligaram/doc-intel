# Test fixtures

## Provenance, honestly

Everything here is a **real response, trimmed**. Rows and columns have been
removed to keep the files readable; nothing has been added, renamed or
invented. All of it was fetched from SEC EDGAR on 2026-08-23.

This is a change from the previous fixtures, which were hand-written to the
shape documented for the Companies House API and could therefore only ever
prove that the code handled the cases we had already thought of. The overflow
walk in `docintel/ingest/edgar.py` exists because of a property of real filers,
so a synthetic fixture would have been the code marking its own homework.

## `edgar/aapl_submissions_trimmed.json`

Apple's submissions JSON. Trimmed from 11 10-Ks and roughly a thousand other
filings down to three 10-Ks, one 10-Q and one 8-K, keeping five of the sixteen
columns. The three 10-Ks are chosen for their filenames:

| accession | `primaryDocument` |
| --- | --- |
| 0000320193-25-000079 | `aapl-20250927.htm` |
| 0000320193-18-000145 | `a10-k20189292018.htm` |
| 0001193125-15-356351 | `d17062d10k.htm` |

Three unrelated conventions from one filer, which is why nothing in the fetcher
infers anything from a filename. The `filings.files[]` entry is Apple's real
one, kept so a test can assert it is **never** fetched: Apple's 10-Ks are all in
`filings.recent`, so reading the overflow would be 1,240 filings of waste.

## `edgar/jpm_submissions_trimmed.json` and the two overflow files

JPMorgan is the case the overflow walk was written for. Their real submissions
JSON has **one** 10-K in `filings.recent` — the bank files thousands of 424B2
pricing supplements a year, which push their own annual report out of that
window — and 69 overflow files.

- `jpm_submissions_trimmed.json` — one 10-K plus three 424B2s in `recent`.
  `filings.files[]` is trimmed from 69 entries to two, both real and with their
  metadata untouched: `-001` (2025-07-21 → 2025-08-20) and `-016` (2024-01-19 →
  2024-02-26). They keep their real names and their real relative order.
- `jpm_submissions_001_trimmed.json` — four real rows, **no 10-K**. The walk has
  to continue past this file.
- `jpm_submissions_016_trimmed.json` — real rows including the genuine 2024
  10-K, accession 0000019617-24-000225, `jpm-20231231.htm`. The walk stops here.

The one thing the trim changes is distance: in the real API the second 10-K is
in file 016 of 69, and reaching it means reading fifteen files that do not have
it. The fixture reproduces the shape of that walk, not its length.

## `edgar/aapl_companyfacts_trimmed.json`

The XBRL companyfacts response — the oracle. Trimmed to two concepts with two
observations each, because the real one is 3.8MB. Nothing reads this except the
fetcher that stores it; see `tests/test_oracle_guard.py`.

## `aapl_10k_trimmed.htm`

A structural trim of Apple's 2025 10-K, unchanged from when it was added for
`docintel/ingest/html_text.py`. The `ix:header` is real and its `ix:hidden`
block is intact, the cover page and the consolidated statements of operations
are real, and the namespace declarations are the ones the filer sent. It doubles
as the document body served by the fetch tests.

## What a passing test here does and does not prove

The submissions JSONs are real, so a test that passes against them is evidence
about the live API's shape as of the date above. It is not evidence that a fetch
against the live service succeeds — nobody has run the full corpus fetch yet.
The first real run is the moment to check the manifest's rejects against what
these files predict.

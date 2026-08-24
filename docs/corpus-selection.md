# Choosing a corpus: the Companies House experiment, and why it failed

This project extracts structured fields from company filings and scores itself
on whether each value is grounded in the source. That needs a corpus of real
filings. The first one was built from Companies House and thrown away. This is
what happened, because the reasoning is more useful than the result.

## What the corpus had to be

Four requirements, all of them load-bearing:

1. **Real documents.** Not a benchmark dataset, not synthetic filings. The
   whole claim of the project is that it works on documents nobody prepared
   for it.
2. **Free, and no paid API.** A portfolio project that costs money to
   reproduce is a portfolio project nobody reproduces.
3. **An independent answer key.** Some source of ground truth that is not the
   document being read, so grounding can be checked rather than asserted.
4. **A genuine spread of difficulty**, including fields that some documents
   truthfully do not state. Correct abstention is a headline metric here, and
   you cannot measure it without real absences.

## Why Companies House

It looked like an unusually good fit. Every UK company files annual accounts;
the REST API is free with a registered key; the document endpoint serves a PDF
of the filing as submitted; and — the part that sold it — many filings also
have an **iXBRL rendition**, the same accounts with the figures machine-tagged.
That is requirement 3 handed over on a plate: the tagged figures are an answer
key produced by the filer, entirely independent of anything we extract.

It also promised requirement 4 for free. UK micro-entities file abridged
accounts that legitimately omit turnover, and audit-exempt companies have no
auditor to name. Those are real, defensible `not_stated` cases rather than
manufactured ones.

So the fetcher was built: an 18-company stratified list, a manifest recording
every candidate including the rejects, iXBRL quarantined under `oracle/` so it
could never reach the extractor, and a text-layer measurement over every PDF.
That last piece is the one that mattered, and it was there for a different
reason than the one it ended up serving.

## The measurement that killed it

The fetcher deliberately **measured the text layer and never excluded on it**.
The reasoning at the time: some filings would be scans, a threshold would be
needed, and you cannot choose a threshold before seeing the distribution. So
`analyse_pdf` ran over every PDF, recorded page count, total extractable
characters and the median characters per page into the manifest, and `--report`
printed the histogram.

The histogram had one bar.

**All 36 PDFs across all 18 companies were image-only. Zero extractable
characters. Not one of them carried a `/Font` resource anywhere in the
document.** These are not PDFs with a thin or unreliable text layer — they have
no text layer at all. Companies House serves the filing as an image of the
filing.

That is fatal, and it is fatal at the root. Everything downstream of ingest —
the chunker, the retriever, the citation spans that make grounding checkable —
operates on characters. There were no characters.

The measurement is the only reason this was caught in Phase 1 rather than in
Phase 3, with a retriever that returned nothing and no idea why. A fetcher that
had simply downloaded the PDFs and reported success would have been correct
about everything it claimed and useless.

> **Provenance.** That measurement was taken from the live Companies House API
> in August 2026 by `docintel/ingest/corpus.py`. Both the fetcher and the
> corpus it fetched have since been removed from the working tree, and the
> cache was gitignored, so the aggregate is not sitting in this repository
> waiting to be recomputed. **Reproducing this** below is how to check it
> yourself. Everything measured about the EDGAR corpus that replaced it is
> re-runnable directly, from `scripts/fetch_corpus.py --report`.

### Reproducing this

The claim is about a class of document, so **one filing is enough to verify the
mechanism**. Run the probe below on a single Companies House accounts PDF and
see an `/XObject` entry, no `/Font` entry, and zero extracted characters, and
you have confirmed what makes these files unusable. You are not re-deriving the
36-of-36 aggregate — that would mean refetching the whole corpus — you are
checking that the thing the aggregate counts is real.

**The probe.** `pypdf` is no longer a dependency of this project, so install it
into a throwaway environment: `pip install pypdf==6.16.1`.

```python
"""Is this PDF a document, or a picture of one?

Usage: python probe_pdf.py FILING.pdf
"""

import sys

from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
print(f"pages: {len(reader.pages)}")

for number, page in enumerate(reader.pages, start=1):
    resources = page.get("/Resources") or {}
    print(
        f"  page {number}: "
        f"{len(page.extract_text())} chars, "
        f"{len(page.images)} images, "
        f"resources {sorted(str(key) for key in resources)}"
    )
```

**What to look for.** The character count is the symptom; the resource keys are
the diagnosis. A page that renders text carries a `/Font` resource, because
that is how a PDF names the fonts its text operators draw with. A page with no
`/Font` at all cannot be drawing text — whatever you can read on it is pixels
inside an image XObject. That absence is the decisive part, and it is why the
probe prints the resource keys rather than just the character count: zero
extracted characters alone could mean a parser failure, but zero characters
*and* no font resource means there was never any text to extract.

Expect an image-only filing to report a `/XObject` entry and no `/Font`.
`/ProcSet` is commonly present too, but it is a legacy compatibility entry and
proves nothing either way.

**Calibrating without an API key.** The two PDFs below were generated by
`tests/fixtures/make_pdfs.py`, recoverable at commit `8c49477`. Running it
produces both, and running the probe over them shows the contrast the probe is
looking for — one text-layer document, one scan:

```
=== text_layer.pdf ===
pages: 3
  page 1: 38 chars, 0 images, resources ['/Font']
  page 2: 60 chars, 0 images, resources ['/Font']
  page 3: 353 chars, 0 images, resources ['/Font']

=== scanned.pdf ===
pages: 1
  page 1: 0 chars, 1 images, resources ['/XObject']
```

Every Companies House filing fetched looked like the second one.

**Fetching a real filing to run it against.** This needs a **free Companies
House REST API key** — register an application at
`developer.company-information.service.gov.uk` and create a REST API key. It is
sent as HTTP Basic auth with the key as the username and an empty password,
which is what the API expects rather than a bearer token.

```bash
KEY=<your Companies House REST API key>
COMPANY=00445790   # any of the 18 in data/companies.txt at commit 0497d25
BASE=https://api.company-information.service.gov.uk

# 1. the most recent accounts filing -> its document metadata URL
META=$(curl -s -u "$KEY:" "$BASE/company/$COMPANY/filing-history?category=accounts" \
  | python -c "import json,sys; print(json.load(sys.stdin)['items'][0]['links']['document_metadata'])")

# 2. that filing's renditions -> the document URL
DOC=$(curl -s -u "$KEY:" "$META" \
  | python -c "import json,sys; print(json.load(sys.stdin)['links']['document'])")

# 3. the PDF itself. -L follows the 302 to the signed storage host; curl drops
#    the credential on a cross-host redirect, which is what that host wants.
curl -s -L -u "$KEY:" -H "Accept: application/pdf" "$DOC" -o filing.pdf

python probe_pdf.py filing.pdf
```

**Where the original code is.** The fetcher that produced the aggregate is
`docintel/ingest/corpus.py` at commit **`6490a97`**, the last commit before the
EDGAR swap; the three-step sequence above is what it did, and `analyse_pdf`
there is the measurement itself. The 18-company list is `data/companies.txt`,
introduced at **`8c49477`** and populated with the real company numbers at
**`0497d25`**. Checking out `6490a97` gives you a working fetcher and the full
`--report` histogram, which is the only route to re-deriving the aggregate
rather than the mechanism.

Two honest caveats. The curl sequence above mirrors what the fetcher did rather
than being a transcript of a run — it has not been re-executed while writing
this. And the probe output shown above is real, but from the generated
fixtures, not from a Companies House filing; a filing fetched today may also
have moved on from the one measured in August 2026.

## Why not OCR

OCR is the obvious escape hatch and it was the wrong one, for three reasons in
increasing order of importance.

**It changes what the project is.** The interesting claim is about extraction
and grounding — finding a value, citing where it came from, and declining when
it is not there. Bolting on OCR makes the headline result a statement about
Tesseract's accuracy on scanned UK accounts, with the extraction quality buried
underneath it. That is a different project, and a less interesting one.

**It destroys the grounding guarantee.** Every extracted field here carries a
character span into the source text, and `Citation` re-checks the quote
verbatim against that span. If the source text is OCR output, the citation
grounds a value in a transcription that may itself be wrong. "Turnover was
£4,200, cited at characters 15,032–15,038" means nothing if the OCR read
`£4,2OO`. The grounding check would pass and the value would be wrong — the
exact failure mode this project exists to catch.

**It makes every error ambiguous.** With OCR in the pipeline, a wrong answer
has two possible causes and no way to tell them apart. Did the model fail to
extract, or did it extract correctly from a mis-transcribed page? The
evaluation harness is the point of this project, and an evaluation harness that
cannot attribute its own failures is not measuring anything.

There is a fourth, smaller reason: `CLAUDE.md` forbids paid APIs, so
commercial OCR was out, and running Tesseract over 36 scanned annual reports
puts a heavy, slow, non-deterministic dependency in front of a test suite that
currently runs offline in two seconds.

## Why EDGAR

SEC EDGAR meets all four requirements, and meets the first one better.

- **Real documents, and machine-readable ones.** A modern 10-K is served as
  HTML. Not a scan of HTML — the actual markup, which is what the filer's
  readers see.
- **Free, and no key at all.** EDGAR asks only that automated clients identify
  themselves with a contact string in the User-Agent header. There is nothing
  to register and nothing to pay.
- **An independent answer key.** The XBRL `companyfacts` API returns every
  figure the company has ever tagged, keyed by accounting concept. Same role
  the Companies House iXBRL rendition was going to play, same quarantine under
  `oracle/`, and now enforced by an AST-level test rather than a convention.
- **A genuine spread.** Filing size, table density and disclosure depth vary
  enormously across filers, and pre-revenue biotechs give real absent fields.

There was a real cost to the swap, and it is worth naming rather than glossing:
the extraction schema was written for UK accounts. `Money` assumed GBP, the
field was called `turnover` rather than revenue, and `company_number` held what
is now a CIK. All of it had to land before the first gold label was written,
because a label written against the wrong currency is worse than no label —
and it has: `Money` requires a currency, the fields are `revenue` and
`net_income`, and `company_number` is gone because the CIK appears in none of
the 36 derived texts. See "Settled: currency, and the field names" in
`docs/labelling-policy.md`.

## One thing that fell out of the swap: document size

Two spot measurements, taken with `derive_text` against the real filings while
building the fetcher:

| Filing | Raw HTML | Derived text | Markup ratio |
| --- | --- | --- | --- |
| Apple, FY2025 10-K | 1,520,208 bytes | 207,444 chars | 0.864 |
| JPMorgan Chase, FY2025 10-K | 12,927,325 bytes | 1,181,568 chars | 0.909 |

**A factor of 5.7 between two documents in the same corpus.** These are two
points, not a distribution — the real spread comes from
`scripts/fetch_corpus.py --report` after the first full fetch — but the order of
magnitude is already the useful part.

It matters for Phase 2 specifically. Hand-labelling a 207k-character document
and hand-labelling a 1.18M-character one are not the same job, and the second
one is not merely "the first one but longer": finding the auditor's name in a
1.18M-character bank filing means knowing where to look before you start. That
needs to be known while planning the labelling effort, not discovered halfway
through it.

It also sets a floor for the retriever. Nothing in Phase 3 can assume a
document fits in a context window, and the chunker has to stay linear in
document size rather than merely fast enough on Apple.

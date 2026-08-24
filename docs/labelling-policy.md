# Labelling policy

Four rules, fixed before Phase 2 begins. Each exists because deciding it
halfway through labelling would mean relabelling everything done so far.

The gold set is hand-labelled. These rules are for the person doing it, and the
appendix at the end is the working reference: the anchors that find each field
in a 10-K, and the commands that record what was found.

---

## Rule 1: implied is not stated

**If a document does not state a value, the label is `not_stated` — even when
the value can be inferred with certainty from what the document does say.**

The case that forces this is `period_start`.

A 10-K states its period end explicitly, on the cover page and at the head of
every statement: *"For the fiscal year ended September 27, 2025"*. The period
start is almost never written down anywhere. A reader knows it is the day after
the previous fiscal year ended, and can work that out from the comparative
column. But no string in the document says it.

So `period_end` gets a citation and `period_start` gets `not_stated`.

### Why this is the right call and not pedantry

A model that emits `period_start: 2024-09-29` with a confident citation has not
read a value out of the document. It has performed an inference — a correct
one, probably — and dressed it up as an extraction. Those are different
operations and this project's entire premise is that the difference is
measurable. `Found` requires a `Citation`, the `Citation` requires a verbatim
quote, and the quote must appear at the recorded span. There is no span in a
10-K containing "September 29, 2024" as that filing's period start. Anything
the model cites for it is a quote about something else.

Score that as correct and the metric stops meaning what it says. `Found` would
then cover both "located in the document" and "worked out from the document",
and every grounding number becomes an average over two different things.

This is also why `CompanyAccounts` gives the two dates separate `Extracted`
fields with separate citations rather than one period object. Sharing a
citation would let the well-grounded half vouch for the inferred half, which is
precisely the overstatement being avoided.

### Scope

The rule generalises beyond dates. A total that appears only as the sum of a
column, a headcount derivable from a segment table, an auditor identifiable
only from a signature block that names the firm nowhere else — all
`not_stated`. **The test is whether a verbatim quote from the document supports
the value as printed.** If you have to compute, combine or deduce, it is not
stated.

The cost is accepted deliberately: the gold set will contain `not_stated`
labels for fields a competent human reader could answer. That is the intended
behaviour, and a model that abstains on them is behaving correctly.

---

## Rule 2: `DERIVATION_VERSION` freezes at the first gold label

**`DERIVATION_VERSION` in `docintel/ingest/html_text.py` may change freely
until the first gold label is written. After that, changing it invalidates
every stored citation offset and requires relabelling.**

### Why

A citation is a character span into the derived text — the output of
`derive_text`, not the raw HTML. `derive_text` collapses runs of whitespace,
strips inline XBRL, translates non-breaking spaces and drops non-rendered
subtrees. Change any of those rules and every character offset after the first
affected byte moves.

A gold label recorded against derivation 1 and checked against derivation 2 is
comparing spans in two different documents that happen to share a name. Nothing
in the pipeline can detect this from the offsets alone, which is exactly why
the version number is recorded next to every derived text file and on every
manifest row.

### What protects you, and what does not

`Citation` stores the quote alongside the span and re-checks it verbatim, so a
stale citation fails loudly rather than silently grounding the wrong text. That
is a **detector, not a remedy**. It tells you the label is broken; it does not
tell you what the label should have been, and it will not save you from
relabelling.

`DERIVATION_VERSION` is what makes that failure diagnosable — the difference
between "this citation is wrong" and "this citation was made against derivation
1 and you are on derivation 2".

### Before the first label

Change it as much as you like. `scripts/fetch_corpus.py --recheck` re-derives
every document from the raw bytes on disk with no network calls, and the raw
bytes are kept precisely so this is cheap. This is the window for getting the
derivation right.

### After the first label

Treat a bump as a corpus migration, not a code change:

1. Bump `DERIVATION_VERSION` and run `--recheck`.
2. Every gold label made against the previous version is now suspect. Find them
   by the `derivation_version` on their manifest row.
3. Re-verify each affected citation's span, and correct the ones that moved.
4. Only then trust any evaluation number.

If a derivation bug is found after labelling has started and the fix is small,
weigh it against this cost honestly. Sometimes the right answer is to live with
a known imperfection in the derivation, document it, and keep the labels.

---

## Rule 3: cite the statement, not the discussion of it

**For `revenue` and `net_income`, the citation is the line in the consolidated
statement of operations. Not the MD&A, not a segment note, not the cash-flow
reconciliation.**

The same figure is printed many times in a 10-K. In Apple's FY2025 filing the
string "Net income" occurs five times: the income statement, the statement of
comprehensive income, the statement of shareholders' equity, the top of the
cash-flow reconciliation, and the earnings-per-share note. Across the corpus it
occurs 416 times in the 25 documents that use the phrase at all. "Total revenue"
occurs 89 times across 19 documents.

Every one of those is a true sentence containing the right number. Only one of
them is where the figure is *stated*.

### Why this has to be fixed before labelling, not during

Without a rule, the choice gets made afresh at each label, and after an
afternoon the gold set holds some statement citations and some MD&A citations
with no record of which is which. In Phase 5 that inconsistency arrives as
model error: a model that cites the income statement every time — which is the
correct, consistent behaviour — scores badly on the labels that happened to be
made from the MD&A, and the metric reports an extraction problem that is really
a labelling problem.

The statement is the right choice because it is the only occurrence that is a
statement of the figure rather than a reference to it. The cash-flow line is
the same number being used as the starting point of a different calculation;
the segment note is a component; the MD&A is prose about it.

### What this means for the model

The eval compares a model's resolved citation to the gold span by containment
(`spans_agree` in `docintel/gold.py`), so a model that cites the MD&A sentence
is scored as ungrounded even when its value is right. That is intended, and it
is what the metric is for: the claim being measured is "this model located the
figure in the accounts", not "this model produced a number that happens to
match".

---

## Rule 4: quote the linearisation, however ugly

**A quote is evidence of location, not a presentable extract. If the derived
text reads badly where the value is stated, quote what it actually says.**

The case that forces this is JPMorgan's cover page. It is a two-column table,
and `derive_text` linearises it row by row, so the text reads:

```
For the fiscal year ended Commission file

December 31, 2024 number 1-5805
```

There is no contiguous "For the fiscal year ended December 31, 2024" anywhere
in that document. The correct `period_end` quote is `December 31, 2024 number
1-5805`, commission file number and all.

The instinct while labelling is to reach for the tidy substring — to quote just
`December 31, 2024`. Do not. On this document that string occurs many times, in
the statements and the notes, and the labelling tool will refuse it as
ambiguous. If it were unique it would be worse, because the label would then
point at whichever other date in the document happened to read the same. The
ugly quote is the one that identifies the place.

Rule 3's containment comparison handles the other side of this: a model that
cites only `December 31, 2024` from inside the gold span is scored as correct,
because its span sits within the label's. The label carries the context; the
model does not have to reproduce it.

---

## Settled: currency, and the field names

The migration this document used to list as pending has landed.

- `Money` now requires a `currency`, with no default. A default is how a wrong
  assumption survives a change of corpus — the schema assumed GBP throughout
  and these filings report dollars.
- `turnover` is now `revenue`, and `profit_before_tax` is now `net_income`. The
  names match the us-gaap XBRL concepts filers tag these lines with — `Revenues`
  and `NetIncomeLoss` — which is what `companyfacts` is keyed by, so a hand
  label can be cross-checked against the oracle without a translation table and
  without the extraction path being involved at all.
- `company_number` is gone. It held a UK company number; the EDGAR equivalent
  is the CIK, and the CIK appears in **none** of the 36 derived texts, because
  it is EDGAR's key for the filer rather than something the filer prints. A
  field absent from every document is not an abstention test — it is a test of
  whether a model learned to answer `not_stated` for one particular field, and
  it would cost 36 of the 252 labels to run it. The Commission File Number and
  the IRS employer identification number are both on the cover page and were
  rejected for the opposite reason: they sit two lines from the company name
  and would measure what `company_name` already measures.

Seven fields, 36 documents, 252 labels.

---

# Appendix: where to look, and how to record it

This is the part to keep open while labelling.

## The tool

```
python scripts/label.py todo                       # the whole grid
python scripts/label.py todo --field auditor_name  # one field, all documents
python scripts/label.py add --entry-id 0000320193/0000320193-25 \
    --field auditor_name --value "Ernst & Young LLP" \
    --quote "We have served as the Company's auditor since 2009."
python scripts/label.py add --entry-id 0000320193/0000320193-25 \
    --field period_start --not-stated
python scripts/label.py verify                     # after any derivation change
python scripts/label.py anchors                    # the table below, re-measured
```

`--entry-id` takes any unambiguous fragment of an entry id. Money fields take
the figure as printed plus the heading that scales it — `--amount 391,035
--scale millions --currency USD` — and the tool does the multiplication. A
quote that spans a blank line goes in through `--quote-file -`.

The tool refuses a quote that occurs zero times or more than once, and says
which. Both refusals are information: zero usually means a curly apostrophe or
a line break inside the quote, and more than one means the quote is not
specific enough to be a label.

## The anchors

Measured over the corpus with `python scripts/label.py anchors`. Every number
below comes from that command, and the anchor list itself is `ANCHORS` in
`docintel/gold.py`.

```
field            anchor                          docs  occurrences
company_name     Exact name of registrant       24/36  24
company_name     Exact name of Registrant        8/36  8
period_start     (no anchor - never stated)      0/36
period_end       For the fiscal year ended      30/36  30
period_end       For the year ended             12/36  63
period_end       for the year ended             30/36  590
revenue          Total net revenue               6/36  81
revenue          Total revenue                  19/36  89
revenue          Total net sales                 2/36  11
revenue          Net sales                       7/36  92
revenue          Total sales and revenues        2/36  14
net_income       Net income                     25/36  416
net_income       Net earnings                    4/36  162
net_income       Net loss                       17/36  89
employee_count   Human Capital                  22/36  89
employee_count   employees                      36/36  1772
auditor_name     auditor since                  36/36  38
```

**company_name and period_end — the cover page.** Both are in the first block
of the document, roughly the first 500 characters, after a slug of the form
`aapl-20250927`. The company name sits immediately above `(Exact name of
registrant as specified in its charter)`. Note the capital-R variant, and that
four documents — Procter & Gamble and Microsoft, both years — linearise their
cover page without that phrase at all.

**period_end — `For the fiscal year ended`.** 30 of 36 spell it exactly that
way on the cover page; the other six say `for the year ended`. Taking the three
casings together, all 36 state their period end on the cover page. Beware that
`for the year ended` also occurs 590 times in the notes: it is the first
occurrence that is the cover page, not any occurrence, which is why the quote
has to carry the date with it.

**period_start — not present. This is the `not_stated` case.** Rule 1 was
written before the corpus was fetched, and the corpus confirms it: JPMorgan and
Etsy both state only the period end on their cover pages. Etsy is the sharpest
illustration — the one place on that cover page where a start date could appear
is the transition-period line, and on an annual report it reads `For the
transition period from to`, with both dates blank. There is no string to quote,
so the label is `not_stated`, all 36 times.

**revenue — the top line of the consolidated statement of operations.** The
caption varies by industry: banks say `Total net revenue`, retailers and
manufacturers `Total net sales` or `Net sales`, Caterpillar `Total sales and
revenues`, and most others `Total revenue`. Four documents — Novavax and
Denali, both years — use none of the listed captions, and the statement has to
be found by its title. Quote the caption *with its figures*: the caption alone
occurs dozens of times, and the derived text puts the comparative years on one
line, so `Total net sales 416,161 391,035 383,285` is a single quote covering
three years and `--amount` is what picks the year being labelled.

**net_income — `Net income`, in that same statement.** Not every filer has one.
Goldman Sachs and Procter & Gamble say `Net earnings`; the loss-making filers
say `Net loss`, which appears somewhere in 17 of the 36 documents; Caterpillar's
income statement line is `Profit`. Rule 3 bites hardest here — 416 occurrences
of `Net income` across 25 documents, in five different statements per filing.

**employee_count — Item 1, under Human Capital.** 22 of 36 use that heading.
All 36 contain the word `employees`, but 1,772 times between them, so the
heading is the way in and the sentence around the number is the quote. Where
the count is only derivable from a segment table, Rule 1 applies and the label
is `not_stated`.

**auditor_name — the sentence the PCAOB requires.** `We have served as the
Company's auditor since ...` closes the audit report, with the firm's name
immediately above it, usually as `/s/ Ernst & Young LLP`. The sentence is not
verbatim across the corpus — most filings use a curly apostrophe, and the noun
changes (`the Firm's` at JPMorgan, `the Corporation's` at Bank of America,
`NEE's and FPL's` at NextEra) — but the fragment `auditor since` appears in all
36. Two of them contain it twice: Simon Property Group files for two
registrants and both audit reports carry the sentence. That ambiguity is real,
and the tool will make you resolve it by extending the quote.

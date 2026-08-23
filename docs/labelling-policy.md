# Labelling policy

Two rules, fixed before Phase 2 begins. Both exist because deciding them
halfway through labelling would mean relabelling everything done so far.

The gold set is hand-labelled. These rules are for the person doing it.

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

## Not yet decided: currency

`Money` in `docintel/extraction/schema.py` assumes every amount is GBP. The
corpus is now US filings reporting in dollars, so that assumption is wrong and
the schema has nothing to flag it with. A figure would be stored as pounds and
silently mis-scored.

**A currency field has to land before the first gold label is written.** It is
tracked as its own change rather than folded into the corpus swap, because it
alters the type-level guarantees and the typing tests that pin them. See the
tracking issue. The same change should settle `turnover` versus `revenue` and
what `company_number` means now that it holds a CIK.

Until that lands, no gold labels. A label written against the wrong currency is
worse than no label, because it looks like progress.

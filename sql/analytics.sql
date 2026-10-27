-- Accuracy by field. NULL means denominator absent; does not mean zero accuracy.
SELECT e.field_name,
       count(*) AS labelled_cases,
       count(*) FILTER (WHERE c.gold_status = 'stated') AS stated_cases,
       count(*) FILTER (WHERE c.outcome = 'correct_found') AS evidence_correct,
       count(*) FILTER (WHERE c.gold_status = 'not_stated') AS absent_cases,
       count(*) FILTER (WHERE c.outcome = 'correct_abstention') AS correct_abstentions,
       count(*) FILTER (WHERE c.citation_grounded IS TRUE) AS grounded_citations,
       count(*) FILTER (WHERE e.status = 'found') AS found_predictions
FROM evaluation_cases c JOIN extractions e ON e.id = c.extraction_id
GROUP BY e.field_name ORDER BY e.field_name;

-- Accuracy by company (scoped to labelled cases only).
SELECT d.company_id,
       count(*) AS labelled_cases,
       count(*) FILTER (WHERE c.outcome = 'correct_found') AS correct_found,
       count(*) FILTER (WHERE c.outcome = 'correct_abstention') AS correct_abstentions
FROM evaluation_cases c JOIN extractions e ON e.id = c.extraction_id
JOIN documents d ON d.entry_id = e.document_entry_id
GROUP BY d.company_id ORDER BY labelled_cases DESC;

-- Most common failure types.
SELECT failure_attribution, count(*) AS cases
FROM evaluation_cases WHERE failure_attribution <> 'none'
GROUP BY failure_attribution ORDER BY cases DESC;

-- Filings with the most abstentions. Only evaluated (labelled) fields count.
SELECT e.document_entry_id, count(*) AS abstentions
FROM extractions e WHERE e.status = 'not_stated'
GROUP BY e.document_entry_id ORDER BY abstentions DESC;

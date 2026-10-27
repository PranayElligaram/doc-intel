-- Idempotent on an empty Doc-Intel PostgreSQL 17 database.
-- These offsets refer to the EXACT manifest-verified derived text, not raw HTML.
BEGIN;
CREATE TABLE IF NOT EXISTS documents (
    entry_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    filing_date DATE,
    derived_text TEXT NOT NULL,
    text_sha256 CHAR(64) NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    derivation_version INTEGER NOT NULL CHECK (derivation_version > 0)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    report_sha256 CHAR(64) NOT NULL UNIQUE CHECK (report_sha256 ~ '^[0-9a-f]{64}$'),
    model_name TEXT NOT NULL,
    top_k INTEGER NOT NULL CHECK (top_k > 0),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    gold_sha256 CHAR(64),
    scope_warning TEXT NOT NULL,
    metrics JSONB NOT NULL,
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS citations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_entry_id TEXT NOT NULL REFERENCES documents(entry_id),
    chunk_id TEXT NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL,
    quote TEXT NOT NULL,
    CHECK (end_offset > start_offset),
    CHECK (end_offset - start_offset = char_length(quote)),
    CHECK (char_length(quote) > 0)
);

CREATE TABLE IF NOT EXISTS extractions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_entry_id TEXT NOT NULL REFERENCES documents(entry_id),
    evaluation_run_id BIGINT NOT NULL REFERENCES evaluation_runs(id),
    field_name TEXT NOT NULL CHECK (field_name IN (
        'company_name','period_start','period_end','revenue',
        'net_income','employee_count','auditor_name'
    )),
    status TEXT NOT NULL CHECK (status IN ('found','not_stated','failed')),
    value_json JSONB,
    error TEXT,
    citation_id BIGINT UNIQUE REFERENCES citations(id),
    UNIQUE (evaluation_run_id, document_entry_id, field_name),
    CONSTRAINT extraction_state_consistent CHECK (
        (status = 'found' AND value_json IS NOT NULL AND citation_id IS NOT NULL AND error IS NULL)
        OR (status = 'not_stated' AND value_json IS NULL AND citation_id IS NULL AND error IS NULL)
        OR (status = 'failed' AND value_json IS NULL AND citation_id IS NULL AND error IS NOT NULL AND char_length(error) > 0)
    )
);

CREATE TABLE IF NOT EXISTS evaluation_cases (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id BIGINT NOT NULL UNIQUE REFERENCES extractions(id) ON DELETE CASCADE,
    gold_status TEXT NOT NULL CHECK (gold_status IN ('stated','not_stated')),
    value_correct BOOLEAN NOT NULL,
    citation_grounded BOOLEAN,
    gold_span_aligned BOOLEAN,
    gold_in_retrieved_context BOOLEAN,
    outcome TEXT NOT NULL,
    failure_attribution TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_extractions_document ON extractions(document_entry_id);
CREATE INDEX IF NOT EXISTS ix_extractions_run ON extractions(evaluation_run_id);
CREATE INDEX IF NOT EXISTS ix_evaluation_cases_outcome ON evaluation_cases(outcome);
COMMIT;

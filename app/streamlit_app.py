"""Read-only filing and evaluation evidence browser backed by PostgreSQL."""

import json
import os
from typing import Any

import psycopg
import streamlit as st
from psycopg.rows import dict_row

st.set_page_config(page_title="Doc-Intel | Evidence Explorer", layout="wide")
st.title("Doc-Intel")
st.caption("Evidence-backed extraction from SEC EDGAR filings | local evaluation explorer")


def connect() -> psycopg.Connection[Any]:
    return psycopg.connect(
        host=os.getenv("DOCINTEL_DB_HOST", "db"),
        port=int(os.getenv("DOCINTEL_DB_PORT", "5432")),
        dbname=os.getenv("DOCINTEL_DB_NAME", "docintel"),
        user=os.getenv("DOCINTEL_DB_USER", "docintel"),
        password=os.environ["DOCINTEL_POSTGRES_PASSWORD"],
        row_factory=dict_row,
    )


try:
    with connect() as conn:
        runs = conn.execute(
            """SELECT id,model_name,started_at,top_k,scope_warning,metrics
            FROM evaluation_runs ORDER BY started_at DESC"""
        ).fetchall()
except (psycopg.Error, KeyError) as exc:
    st.error(
        f"Database not ready. Initialise the schema and import a report. ({type(exc).__name__})"
    )
    st.stop()

if not runs:
    st.info(
        "No evaluation results loaded yet. Run eval/run_eval.py locally, "
        "then scripts/load_results.py."
    )
    st.code(
        "python scripts/load_results.py --report eval/results/<timestamp>.json",
        language="bash",
    )
    st.stop()

chosen_run = st.selectbox(
    "Evaluation run",
    runs,
    format_func=lambda run: f"#{run['id']} · {run['model_name']} · {run['started_at']}",
)
if chosen_run is None:
    st.stop()
run_id = chosen_run["id"]
metrics = chosen_run["metrics"]["overall"]
value_f1 = metrics["value"]["f1"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Value F1", f"{value_f1:.2f}" if value_f1 is not None else "N/A")
col2.metric("Labelled pairs", metrics["labelled_cases"])
col3.metric("Correct abstentions", f"{metrics['correct_abstentions']}/{metrics['gold_not_stated']}")
col4.metric(
    "Grounded citations",
    f"{metrics['grounded_citation_rate']:.0%}"
    if metrics["grounded_citation_rate"] is not None
    else "N/A",
)
st.warning(chosen_run["scope_warning"])
st.caption("Scores cover only the hand-labelled cases in the selected run.")

with connect() as conn:
    documents = conn.execute(
        """SELECT DISTINCT d.entry_id,d.company_id,d.filing_date
        FROM documents d JOIN extractions e ON e.document_entry_id=d.entry_id
        WHERE e.evaluation_run_id=%s ORDER BY d.entry_id""",
        (run_id,),
    ).fetchall()

chosen_doc = st.selectbox(
    "Filing",
    documents,
    format_func=lambda d: f"{d['entry_id']} · company {d['company_id']}",
)
if chosen_doc is None:
    st.stop()
with connect() as conn:
    rows = conn.execute(
        """SELECT e.field_name, e.status,e.value_json,e.error,
        c.quote,c.start_offset,c.end_offset,
        ec.gold_status,ec.value_correct,ec.citation_grounded,
        ec.gold_span_aligned,ec.outcome,ec.failure_attribution
        FROM extractions e
        JOIN evaluation_cases ec ON ec.extraction_id=e.id
        LEFT JOIN citations c ON c.id=e.citation_id
        WHERE e.evaluation_run_id=%s AND e.document_entry_id=%s
        ORDER BY e.field_name""",
        (run_id, chosen_doc["entry_id"]),
    ).fetchall()
    stored_doc = conn.execute(
        "SELECT derived_text FROM documents WHERE entry_id=%s",
        (chosen_doc["entry_id"],),
    ).fetchone()

st.subheader("Extracted fields")
st.dataframe(
    [
        {
            "Field": row["field_name"],
            "Status": row["status"],
            "Extracted value": (
                json.dumps(row["value_json"], ensure_ascii=False)
                if row["status"] == "found"
                else ""
            ),
            "Evaluation": row["outcome"],
            "Failure type": (
                "" if row["failure_attribution"] == "none" else row["failure_attribution"]
            ),
        }
        for row in rows
    ],
    hide_index=True,
    use_container_width=True,
)

st.subheader("Evidence and exact source spans")
for row in rows:
    with st.expander(f"{row['field_name']} · {row['status']} · {row['outcome']}"):
        if row["status"] == "found":
            st.write("Verified excerpt:")
            st.code(row["quote"])
            start, end = row["start_offset"], row["end_offset"]
            st.caption(f"Document character offsets [{start}, {end})")
            if stored_doc is not None:
                content = stored_doc["derived_text"]
                st.write("Source context:")
                st.code(content[max(0, start - 220) : min(len(content), end + 220)])
            st.caption(
                f"Grounded: {row['citation_grounded']} · Gold-aligned: {row['gold_span_aligned']}"
            )
        elif row["status"] == "not_stated":
            st.info("No supported value was stated in the retrieved evidence.")
        else:
            st.error(row["error"] or "Extraction failed")
        st.caption(f"Gold status: {row['gold_status']} · Attribution: {row['failure_attribution']}")

with st.expander("Run metrics (raw JSON)"):
    st.json(chosen_run["metrics"])

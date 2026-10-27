# Doc-Intel

**Evidence-backed extraction from SEC annual filings, with explicit abstention and per-field evaluation.**

Doc-Intel is a portfolio engineering project. It ingests real SEC EDGAR 10-K filings, derives text from inline-XBRL HTML, retrieves relevant passages with a local BM25 retriever, and uses a locally hosted language model through Ollama to extract seven structured fields. Every accepted `found` value must include a quote with character offsets that validate against the original derived text. An unsupported claim becomes `failed`, not a success.

## The problem

A fluent model can give a plausible answer while citing the wrong table, the previous reporting period, or an invented auditor. A useful extraction system must distinguish the value being right from its evidence being right, and distinguish a justified absence from a failed run.

## Architecture

```text
SEC EDGAR HTML -> XBRL-stripped derived text -> offset-preserving chunks
                      |                             |
          manifest + SHA-256                    BM25 retrieval
                                                    |
                                           Local Ollama model
                                                    |
                                      Typed Found / NotStated / Failed
                                                    |
                                  Quote and claim validation, with one
                                  bounded evidence-correction attempt
                                                    |
         Hand-labelled gold set -> field-level evaluation JSON
                                                    |
                                           PostgreSQL loader
                                                    |
                                  Streamlit evidence explorer
```

The XBRL `companyfacts` download stays in a separate oracle directory and is not available to inference. The original Companies House corpus was abandoned after a text-layer probe found all 36 selected PDFs were image-only. See [corpus selection](docs/corpus-selection.md) for the experiment and limitations.

## Quick start (Windows + Docker Desktop + Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,app]"
Copy-Item .env.example .env
```

Set `SEC_USER_AGENT` in `.env` to your own contact string, then add a **new line** with `DOCINTEL_POSTGRES_PASSWORD` set to a random local database password. Do not commit `.env`. The sample `.env.example` intentionally contains no credentials.

```powershell
# Fetch public SEC filings. Respect the SEC access policy and rate limits.
.\.venv\Scripts\python.exe scripts\fetch_corpus.py --help
# Follow the CLI options to retrieve filings under data/cache/.

# Run local Ollama (install separately; no paid model API required).
ollama pull qwen2.5:7b-instruct
.\.venv\Scripts\python.exe eval\run_eval.py --model qwen2.5:7b-instruct

# Start database and create tables (also works when the DB volume already exists).
docker compose up -d db
Get-Content .\sql\001_init.sql -Raw | docker compose exec -T db psql -X -v ON_ERROR_STOP=1 -U docintel -d docintel

# Load a measured evaluation run (choose your latest JSON report).
$Report = Get-ChildItem .\eval\results\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe scripts\load_results.py --report $Report.FullName

# Build and start the read-only evidence explorer.
docker compose up -d --build app
Start-Process http://localhost:8501
```

The app is local-only, bound to `127.0.0.1:8501`. PostgreSQL is bound to `127.0.0.1:5433`. The app **displays stored evaluation results**; model inference runs separately on the Windows host via Ollama. Docker Compose does not provision or host the local model.

To stop the app: `docker compose down`. The named PostgreSQL volume survives; do not use `down -v` unless you intend to erase the database.

## Evaluation results and scope

The development pilot of **26 October 2026**, *before the subsequent evidence-correction patch*, evaluated **four independently hand-labelled (filing, field) pairs in one Etsy 10-K** using `qwen2.5:7b-instruct`:

| Metric | Measured pilot result |
|---|---:|
| Labelled cases | 4 |
| Stated-value cases | 3 |
| Value F1 | 0.80 |
| Grounded citations among accepted `found` predictions | 2/2 |
| Gold-aligned citations among accepted `found` predictions | 2/2 |
| Correct abstention | 1/1 |
| Correct found | 2 |
| Extraction failures | 1 |

These figures come from the locally produced `20261026T183413Z.json` run, reproduced in the development session. They are **not an estimate of general accuracy**. Later, a single-employee-count test succeeded following a bounded evidence-only retry, but that result is not substituted into this earlier evaluation report. Run `eval/run_eval.py` to produce current measurements. Reports are local and ignored by Git; importing one into the database does not change its metrics.

The planned full gold set is **36 filings Ã— 7 fields = 252 labels**. Only four labels from one filing were verified when the first evaluation was run. Further independent labelling and unseen-filing evaluation are required before reporting corpus-wide performance.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe docintel scripts tests eval
```

The local release check on 27 October 2026 passed 280 tests on Windows, with Ruff and Mypy passing. The application Docker image built successfully, and PostgreSQL and Streamlit passed their container health checks.

## Design decisions

- **No paid APIs.** Ollama inference is offline; tests use a deterministic fake provider.
- **Citations are data, not decoration.** The exact quote and character span must match both the model's retrieved chunk and the derived filing text.
- **Three result states.** `found` means an evidenced answer, `not_stated` means the supplied evidence does not state it, and `failed` means extraction or validation failed. A model error is not counted as a correct abstention.
- **Gold labels are independent of chunking.** Their offsets index the derived document rather than the retriever's chunks.
- **PostgreSQL stores separate cases and runs.** Historical evaluation results are immutable by report SHA-256; repeated imports are idempotent.
- **JSONB values.** Count, date, string and monetary records retain their structures rather than being forced into a text column.
- **Streamlit is deliberately read-only.** It demonstrates status, provenance and failure attribution without adding a browser-facing inference endpoint.

## Known limitations

This is a **portfolio prototype**, not a production service. It lacks user authentication, hosted deployment, structured observability and a validated security posture. Evaluation coverage is extremely limited; model behaviour may change across runs and local hardware. Some legitimate citations could be rejected by strict textual validation. PostgreSQL does not independently prove semantic entailment of a cited quote; application-level validation checks textual span identity. Docker image builds and containerised tests must be verified on the target host. Neither the local filing cache nor model weights are shipped in this Git repository.

## Repository contents

- `docintel/ingest/`: source fetching, HTML text derivation and manifests
- `docintel/retrieval/`: offset-preserving chunking and BM25
- `docintel/llm/` and `docintel/extraction/`: local inference and typed extraction
- `docintel/gold.py`, `docintel/evaluation.py`, `eval/`: hand labels and evaluation
- `sql/`: migration and analytical queries
- `scripts/load_results.py`: validated, idempotent evaluation import
- `app/`: Streamlit browser for saved results and precise source evidence
- `compose.yaml`, `Dockerfile`: local PostgreSQL + application delivery

## Next work

Grow the independently labelled corpus, quantify per-field performance on withheld filings, investigate non-overlapping evidence selection, add service-level operational tests, and document security controls suitable for a real client environment.

# doc-intel

Structured extraction from UK company filings, with an evaluation harness that
measures grounding and abstention, not just accuracy.

## Environment
- Python 3.13, venv at `.venv` (activate before launching claude)
- Local model via Ollama. No paid APIs, no API keys for models, ever.
- Commands: `pytest` | `ruff check .` | `ruff format .` | `mypy docintel scripts tests`

## Architecture rules
- No LangChain, no LlamaIndex. The chunker and retriever are written here on
  purpose - they are the part an interviewer will ask about.
- All model access goes through the `LLMProvider` protocol in
  `docintel/llm/base.py`. `FakeLLM` stays deterministic so the whole test suite
  runs offline in CI with no model and no network.
- Every extracted field carries a citation: chunk id + character span.
- Keep it flat. No utils.py, no service layer, no DI container.

## Honesty
- Never fabricate results, datasets or benchmarks.
- Every number in the README must come from `eval/run_eval.py` and be re-runnable.
- The gold set is hand-labelled by me and is not to be generated or edited by Claude.

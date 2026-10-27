# Local release and publication checklist

1. Keep the original repository backed up. The release overlay replaces `compose.yaml`, CI and other project config only after the review installer has backed them up.
2. Run the Python quality gate and fix any errors. Do not claim success from a past baseline.
3. Initialise the PostgreSQL database and load an actual saved evaluation JSON report from your own machine.
4. Build the application with Docker Compose and open http://localhost:8501. Inspect a `found`, `failed`, and `not_stated` result with its exact source evidence.
5. Check `git status --short`, `git diff --check` and `git diff --cached`. **Never commit** `.env`, `.env.backup-*`, `data/cache/`, local model weights, or private content.
6. Stage only reviewed project files. Commit and push from your Windows project. The GitHub copy must include your previously untracked Python sources, not just the new UI.
7. Once remote CI passes, change GitHub repo visibility from private to public in Settings → Danger Zone. Check in an incognito window that the README and workflows render.
8. Add this repository to your CV only with conservative language: *Built a local SEC filing extraction application with source-linked evidence, typed abstention/error states, PostgreSQL persistence, Streamlit UI and Docker packaging. Measured an initial four-case development evaluation; wider accuracy remains unverified.*

Publishing a GitHub repository **does not deploy** the app publicly. Reviewers can clone and run it locally; publicly hosted access requires a separate deployment with security controls.

## Fast local run

After extracting the delivery ZIP to a temporary folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$Stage\APPLY.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "$Stage\RUN-LOCAL.ps1"
```

The first backs up overwritten files. The second installs app dependencies, runs every quality gate, creates the DB schema, imports the latest *measured* report, builds and tests the app container health, then opens the browser.

## Suggested selective staging after local checks pass

```powershell
git add README.md RELEASE.md .gitignore .env.example .dockerignore Dockerfile compose.yaml pyproject.toml .github/workflows/ci.yml sql app scripts/load_results.py tests/test_loader.py
# Include recovered project code and tests still untracked:
git add docintel/llm docintel/retrieval docintel/extraction/pipeline.py docintel/evaluation.py eval/run_eval.py scripts/extract_one.py tests/test_llm.py tests/test_retrieval.py tests/test_pipeline.py tests/test_evaluation.py docs/brief-v2.md docs/defence-retrieval.md docs/session-state.md data/gold/labels.json
# Publish a real report only after opening it and confirming it contains no secrets:
# mkdir eval/published; copy the audited report from eval/results, then git add eval/published

git diff --cached --check
git diff --cached --stat
git status --short
# Review the full staged diff for private information before proceeding.
# After review, explicitly commit and push with your own confirmation.
# git commit -m "feat: deliver evidence-backed document intelligence application"
# git push origin main
```

Change visibility **only after** the remote build and tests pass. If you find secrets in commit history, do not flip the public switch.

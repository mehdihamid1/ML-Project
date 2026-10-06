# PE malware classification and chat agent

This project compares seven classifiers for Windows PE feature records and
serves the frozen LightGBM pipeline through an OpenAI chat agent and Flask UI.
The saved ML model produces every classification. The LLM selects tools and
stored result references; Python validates arguments, computes metrics, and
renders the actual output. Feature-level explanations are unavailable.

The completed experiment and untouched hold-out results are recorded in the
generated [model report](docs/model-results.md). The dataset and all training
artifacts remain out of Git. The small trusted production pipeline is bundled
under `models/` so a clean checkout can run the app without retraining.

## Run the application

Use Python 3.10. Install only the lean runtime dependencies:

```bash
python3 -m venv .runtime-venv
source .runtime-venv/bin/activate
pip install -r requirements-runtime.txt
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
# Export OPENAI_API_KEY in this shell using your secret manager.
python scripts/verify_model.py
gunicorn app:app --bind 127.0.0.1:5000 --workers 1 --threads 2 --timeout 180
```

Open <http://127.0.0.1:5000>. Without `OPENAI_API_KEY`, health and upload
validation work but chat reports that configuration is missing. The app reads
secrets from environment variables, never from source code. `.env.example`
lists setting names only; copying it does not configure working secrets.
`OPENAI_MODEL` optionally selects the OpenAI model; the code defaults to
`gpt-4.1-mini`. Raw CSV rows remain local and are not sent to OpenAI.

Alternatively, with the same exported environment variables:

```bash
docker compose up --build web
```

Upload a CSV from [samples/](samples/README.md). The UI shows its opaque file
ID; refer to that ID when asking to classify a row, classify every row, or
evaluate labels. Batch results include source row IDs, probabilities, invalid
row status and errors, and a download link. The conditional form lets you choose
an evaluation file, a prediction file, a row and an accuracy threshold.
Activity records show the actual evaluation, prediction, failure or skip.
Follow-up questions use stored session results.

## Tool and upload behavior

- Single prediction validates the requested row and returns its model class,
  malware probability, decision threshold and model version.
- Batch prediction preserves every source row in its download. Invalid rows
  receive an error status; they are counted and never silently discarded.
- Evaluation computes probability AUC, accuracy, confusion matrix and label
  counts over valid labeled rows, with explicit coverage and rejected-row counts.
  Missing label columns or no valid labeled rows fail visibly. A single-class
  file returns unavailable AUC with a reason.
- Conditional prediction evaluates first and checks the user threshold in code.
  It also requires complete valid labeled coverage; incomplete evaluation or a
  failure skips prediction. Both underlying tool calls appear in the activity
  record when performed.

Only UTF-8 CSV data is accepted. Size, row, cell, schema, session, request and
tool-call limits are enforced. Uploaded content is never executed. Files and
downloads are accessible only in their originating session. Server-side
sessions expire, and restarting the app clears their in-memory records. Use
one Gunicorn worker: multiple workers require a shared session store first.
Behind Render's proxy, the deployment enables secure cookies and trusted
forwarded protocol/client-IP handling; keep `TRUST_PROXY` off locally.

Routes are `/`, `/health`, `/api/session`, `/api/upload`, `/api/chat`,
`/api/download/<id>` and `/api/reset`. Mutation requests require the CSRF token
from `/api/session` in `X-CSRF-Token`. The UI handles that automatically.

## Reproduce training and evaluation

The CPU training environment includes PyTorch, XGBoost, LightGBM and CatBoost;
the production runtime installs no PyTorch, XGBoost, CatBoost or Jupyter.
Direct dependencies are pinned; transitive dependencies and base images are
not fully locked. On Linux, match the container user to your file owner:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
python3 scripts/fetch_dataset.py
docker compose build ml
docker compose run --rm ml pytest -q
docker compose run --rm ml python train.py --smoke --output artifacts/new-smoke
docker compose run --rm ml python train.py --output artifacts/reproduction
docker compose run --rm ml python eval.py --model artifacts/reproduction/production.joblib --data artifacts/reproduction/holdout.csv
```

Output directories must be empty. Smoke mode does not select a production model
or evaluate the hold-out. Full training deduplicates SHA1, removes conflicting
labels, reserves the stratified hold-out, fits preprocessing inside CV folds,
and selects on CV AUC. The production model was frozen before final hold-out
evaluation; do not retune or reselect using those results.

`docker compose up ml` starts JupyterLab bound to localhost with token
authentication. Copy its logged URL to your browser. `docker compose down`
stops development services. The dataset is not included in the image.

## Verification and deployment

The test suite covers training, tool validation, the Flask routes and isolation,
mocked OpenAI routing, conditional ordering and failures. Plain `pytest` works
through the repository's `pytest.ini` setting. The runtime environment can run
the tools/web/agent tests, while the full suite needs training dependencies:

```bash
pip install pytest==8.3.5
pytest -q tests/test_tools.py tests/test_web.py tests/test_agent.py tests/test_deployment.py
```

GitHub Actions runs the full tests before its deploy job triggers a Render hook
for the tested commit. Render auto-deploy is disabled. The job polls live
`/health` and requires that exact commit to be healthy. See [deployed.md](deployed.md)
for setup and the factual deployment status. Real OpenAI scenario execution is
tracked separately in [agent-evaluation.md](agent-evaluation.md); mocked tests
do not fulfill that requirement.

Other project records: [dataset source](dataset-source.md),
[evaluation and design](evaluation-and-design.md), [AI tooling](ai-tooling.md),
and shared [agent instructions](AGENTS.md).

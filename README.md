# PE malware classification and chat agent

This project compares seven classifiers for Windows PE feature records and
serves the frozen LightGBM pipeline through an OpenAI chat agent and Flask UI.
The saved ML model produces every classification. The LLM selects tools and
stored result references; Python validates arguments, computes metrics, and
renders the actual output. Feature-level explanations are unavailable.

The completed experiment and untouched hold-out results are recorded in the
generated [model report](docs/model-results.md). The dataset and original
training output directories are ignored by Git. The frozen production pipeline
is bundled under `models/`, and its generated report and metadata are recorded
under `docs/`, so a clean checkout can run the app without retraining.

## Current status

The tools, OpenAI agent, Flask app, sample CSVs and deployment workflow are
implemented. Plain `pytest` passes all 331 tests locally; the lean environment
passes 320 tests. The AI-decided conditional task, the sample-file buttons and
the live mode of the evaluation runner in this checkout are verified locally
and await commit, push and deployment.

The deployed commit is
[`3f384a8`](https://github.com/mehdihamid1/ML-Project/commit/3f384a8).
Its [GitHub Actions run](https://github.com/mehdihamid1/ML-Project/actions/runs/37865252113)
passed the full, runtime and container tests, then deployed and passed the live
health check. On 2026-10-09,
[live health](https://quantic-malware-agent.onrender.com/health) reported that
commit, `status: ok`, LightGBM and `openai_configured: true`. A real-LLM run of
the 13 evaluation scenarios against that deployment passed 13 of 13. See
[deployed.md](deployed.md) and [agent-evaluation.md](agent-evaluation.md).

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
validation and the model dashboard work but chat reports that configuration is missing. The app reads
secrets from environment variables, never from source code. `.env.example`
lists setting names only; copying it does not configure working secrets.
`OPENAI_MODEL` optionally selects the OpenAI model; the code defaults to
`gpt-4.1-mini`. Raw CSV rows remain local and are not sent to OpenAI.

Alternatively, with the same exported environment variables:

```bash
docker compose up --build web
```

The production Docker image runs as an unprivileged user and honors `PORT`,
defaulting to 5000 for this Compose mapping. Render's native Python service
uses the same lean requirements and binds to its assigned port. The Blueprint
sets Gunicorn options explicitly and keeps one worker for session isolation.

Choose or drop one or more CSVs from [samples/](samples/README.md) on the
upload area; each is checked for CSV format, size and required columns as soon
as it is added. A successful upload fills an editable question using the new
file ID: predict row 0 for a one-row CSV, or classify all rows otherwise.
Without a file of your own, the sample buttons under the upload area (**One
row**, **Batch of 4**, **Labeled**, **Low accuracy**, **Invalid row**, **Missing
label**, **One class**) load the matching CSV from `samples/` on the server,
with the same checks, and prepare an evaluation question for labeled samples.
Review or edit the question, then click **Send**. Every file card offers
**Predict row 0**, **Classify all** and, for files with a `Label` column,
**Evaluate**; each button writes the request for that file into the chat box.
The **Copy** button copies the opaque file ID for requests you type yourself.
Batch results include source row IDs, probabilities, invalid row status and
errors, and a download link. The conditional form lets you choose an
evaluation file, a prediction file, a row and an accuracy threshold. Its
**Prepare question** button fills the chat box and waits for your Send action.
During the request, live progress shows the current stage and actual function
names with running, finished, failed, skipped or blocked states, plus elapsed time. Each
answer lists the tools that produced it, or says that none was called, and the
activity record lists every call in the session with its arguments and its
evaluation, prediction, failure or skip. On screens at least 721 pixels wide
and 560 tall, the page fits the first screen: the chat and its input stay
visible while the file column scrolls on its own. Phones show the chat after
the file list.
Follow-up questions use stored session results.
Confusion-matrix follow-ups name false negatives, false positives, true negatives
and true positives, with counts and rates calculated in code. Ordinary “tell me
if it is malware” requests work, and evaluation and classification can be
independent tasks in the same message. Percentage accuracy thresholds are
bound to the user's stated value.

## Tool and upload behavior

- Single prediction validates the requested row and returns its model class,
  malware probability, decision threshold and model version.
- Batch prediction preserves every source row in its download. Invalid rows
  receive an error status; they are counted and never silently discarded.
- Evaluation computes probability AUC, accuracy, confusion matrix and label
  counts over valid labeled rows, with explicit coverage and rejected-row counts.
  Missing label columns or no valid labeled rows fail visibly. A single-class
  file returns unavailable AUC with a reason.
- Conditional prediction: the AI model must call `evaluate` first. It then
  reads the returned accuracy and decides whether to call `predict_single`.
  The server checks that decision against the user's stated threshold and
  complete valid label coverage: it blocks a prediction the rule does not allow,
  and never predicts on the model's behalf. A failed evaluation stops the task.
  The activity record shows `evaluate`, then `predict_single` as finished,
  skipped (withheld) or blocked. Conditions worded without "if" ("when accuracy
  is 0.95 or higher") count too. The threshold is read from the condition
  itself, so an earlier figure such as "previous accuracy of 0.80" is not used;
  two different values prompt a clarification question.

Only UTF-8 CSV data is accepted. Uploads and tools share schema validation and
file limits. Oversized individual cells are retained and reported as invalid
rows, with batch output preserving their source positions. Session, request and
tool-call limits are enforced. Uploaded content is never executed. Files and
downloads are accessible only in their originating session. Server-side
sessions expire; at capacity the oldest empty inactive session can be reclaimed
while active requests and records containing user data remain protected.
Restarting the app clears its in-memory records. Use
one Gunicorn worker: multiple workers require a shared session store first.
Behind Render's proxy, the deployment enables secure cookies and trusted
forwarded protocol/client-IP handling; keep `TRUST_PROXY` off locally.

Open **Model dashboard** in the header, or visit `/analytics`, to compare the
recorded models. Follow its four steps: data split, cross-validation, model
comparison and final test. The first step shows the duplicate removal
(50,181 rows to 43,393 unique files) and separates the training files from
the reserved test set. Choose any model and move through the ten saved
rounds to see which fold is validation, which nine folds are training, and
the recorded AUC and accuracy; a strip places that round's AUC among all ten
and marks the mean that the comparison reports. Each round starts with a fresh
model and freshly fitted preprocessing; no learned state carries over between
rounds.

The comparison shows cross-validation AUC, accuracy, fold
standard deviation and mean fit time, with sortable results and interactive
charts. Mean AUC and accuracy appear as dots with ±1 fold standard deviation on
an axis fitted to the results, so close models stay distinguishable. A
selection panel follows, before the final test as in the experiment: it states
the selection rule and, because every model used the same folds, compares
LightGBM with any other model fold by fold. This describes those folds and is
not a significance test. The final-test panel then shows LightGBM's hold-out
metrics beside the range of its ten cross-validation rounds, and its confusion
matrix with error rates.
The comparison uses saved experiment output and requires no OpenAI key or
retraining. Download the same numbers from `/api/model-comparison`.

The small runtime report is bundled as `models/comparison.json`. To regenerate
it from an existing training run without fitting models:

```bash
python scripts/export_model_comparison.py --artifacts artifacts/training-final --output models/comparison.json
```

The report records source checksums; its loader checks metric summaries and
the production model version before displaying results. The tables and default
Round 1 remain available when JavaScript is disabled or interactive controls
cannot load. Charts use local scripts and SVG and
add no runtime dependencies.

Routes are `/`, `/analytics`, `/health`, `/api/model-comparison`, `/api/session`, `/api/upload`,
`/api/samples/<name>`, `/api/chat`, `/api/chat/stream`, `/api/download/<id>` and `/api/reset`.
`/health` also names the configured OpenAI model. Mutation requests require the CSRF token
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

The saved split and recorded results can be audited without training or
prediction:

```bash
python scripts/audit_experiment.py --data data/raw/brazilian-malware.csv --artifacts artifacts/training-final --output docs/experiment-audit.json
```

The aggregate [audit report](docs/experiment-audit.json) records partition
integrity, exact raw input-vector overlap, CV selection and artifact checks.

A training-folds-only check compares the frozen inputs with smaller, larger,
filtered and projected ones for the selected LightGBM configuration. It ran
after the model was frozen and changes nothing; see
[evaluation-and-design.md](evaluation-and-design.md#feature-selection-check):

```bash
docker compose run --rm ml python scripts/explore_features.py --output docs/feature-exploration.json
```

`docker compose up ml` starts JupyterLab bound to localhost with token
authentication. Copy its logged URL to your browser. `docker compose down`
stops development services. The dataset is not included in the image.

## Verification and deployment

The test suite covers training, tool validation, the Flask routes and isolation,
mocked OpenAI routing, conditional ordering and failures, plus incremental
tool events, interrupted streams, session lease cleanup and storage rollback. Plain `pytest` works
through the repository's `pytest.ini` setting. The runtime environment can run
the tools/web/agent tests, while the full suite needs training dependencies:

```bash
pip install pytest==8.3.5
pytest -q --ignore=tests/test_training.py
```

GitHub Actions pins Ubuntu 24.04 and uses Node 24 actions. It runs the full
suite, a lean-runtime suite and a production-container check; all must pass before its deploy job
triggers a Render hook for the tested commit. Render auto-deploy is disabled. The job polls live
`/health` and requires that exact commit to be healthy. See [deployed.md](deployed.md)
for setup and the factual deployment status. Real OpenAI scenario execution is
recorded separately in [agent-evaluation.md](agent-evaluation.md); mocked tests
do not fulfill that requirement. The scenario runner drives the deployed app
like a browser, so it uses the server's key and needs none locally:

```bash
python3 scripts/run_agent_evaluation.py --base-url https://quantic-malware-agent.onrender.com --output artifacts/agent-evaluation/<run-name>
```

Reproduce the Docker compatibility check after building the web image:

```bash
docker compose build web
python3 scripts/check_container.py --image ml-project-web --cpus 0.1 --output artifacts/container-check.json
```

It starts the actual Gunicorn command with a custom port, secure cookies and
proxy handling, then verifies the dashboard's bundled results and exercises uploads, frozen-model tools, streamed function progress, conditional tasks,
isolated downloads and the maximum-row batch under the configured resource
limits. Only the OpenAI HTTP provider is mocked. It removes its own temporary
container. The [observed report](docs/container-compatibility.json) records the
successful resource-limited run. This is local compatibility evidence; live Render deployment and
real-provider evaluation are separate checks.

The review's grader invitation and recorded demo are still pending. A
[demo guide](docs/demo-guide.md) provides a reproducible walkthrough; it is not
a recorded video.

Other project records: [dataset source](dataset-source.md),
[evaluation and design](evaluation-and-design.md), [AI tooling](ai-tooling.md),
and shared [agent instructions](AGENTS.md).

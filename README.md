# AI/ML development environment

CPU-based Python 3.10 environment for the supplied ML/AI agent project.
Includes pandas, NumPy, scikit-learn, PyTorch, XGBoost, LightGBM, CatBoost,
plotting libraries, JupyterLab, and pytest. Direct dependencies are pinned;
transitive dependencies and the base image are not yet fully locked.
The dataset is available locally at `data/raw/brazilian-malware.csv`.
It contains 50,181 rows, 27 input columns, and `Label` (0 = goodware,
1 = malware). Checksums and data-quality findings are recorded in
[dataset-source.md](dataset-source.md). `train.py` and `eval.py` implement
the seven-model comparison and frozen-pipeline evaluation. The application
and agent have not yet been implemented.

## Project documentation

[AGENTS.md](AGENTS.md) is the shared source of project requirements and agent
instructions; [CLAUDE.md](CLAUDE.md) imports it. Flask, Render deployment,
GitHub Actions, and the agent-evaluation deliverables described there are planned
work, not features of the current development container. Flask and an LLM
provider SDK are not yet included in `requirements.txt`.

## Start with Docker

On Linux, match your user so notebooks and artifacts have the right owner:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose up --build
```

Open the `http://127.0.0.1:8888/lab?token=...` URL printed in the logs.
Jupyter's token authentication remains enabled. The port is bound to localhost.
The project directory is mounted at `/workspace`, so files persist on the host.
The first build downloads the Python/ML packages and may take several minutes.

## Shell and verification

```bash
docker compose run --rm ml bash
docker compose run --rm ml python -m pip check
docker compose run --rm ml python -c "import pandas, sklearn, torch, xgboost, lightgbm; print('ML imports OK; PyTorch:', torch.__version__)"
docker compose down
```

## Training and evaluation

Fetch the pinned dataset (existing files are checksum-verified):

```bash
python3 scripts/fetch_dataset.py
```

Rebuild after dependency changes, then validate the workflow:

```bash
docker compose build
docker compose run --rm ml python -m pytest -q
docker compose run --rm ml python train.py --smoke --output artifacts/smoke
```

Run the complete seven-model, two-configuration, 10-fold comparison:

```bash
docker compose run --rm ml python train.py --output artifacts/training
docker compose run --rm ml python eval.py --model artifacts/training/production.joblib --data artifacts/training/holdout.csv
```

Use a fresh output directory for every run. Full training is CPU intensive.
The selected pipeline is saved as `production.joblib`; comparison scores go
to `cv-results.csv`, and hold-out metrics and provenance to `metadata.json`.
`report.md` presents the comparison and final confusion matrix.
See [evaluation-and-design.md](evaluation-and-design.md) for the protocol.

Place CSV datasets in `data/` and trained models in `artifacts/`; both are
excluded from Git and image builds but available through the development mount.
This setup uses CPU PyTorch; GPU support requires a separate CUDA setup.

The existing host `.venv` is independent of Docker; no activation is required
for Docker commands. Docker must be running and your user must have permission
to access its daemon.

PyTorch CPU installation follows the [official version instructions](https://docs.pytorch.org/get-started/previous-versions/).

## Verification status

The Docker image built successfully; `pip check`, ML-library imports, and
Jupyter HTTP startup passed. Dataset checksums, dimensions, label counts,
empty-cell counts, duplicate hashes, and constant columns were checked against
the local files on 2026-10-05. Training tests now cover duplicate separation,
unknown text/category handling, serialization of every model, probability AUC,
and evaluation input errors. Smoke scores are not final model results.

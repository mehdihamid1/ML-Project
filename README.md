# AI/ML development environment

CPU-based Python 3.10 environment for the supplied ML/AI agent project.
Includes pandas, NumPy, scikit-learn, PyTorch, XGBoost, LightGBM,
plotting libraries, JupyterLab, and pytest. Direct dependencies are pinned;
transitive dependencies and the base image are not yet fully locked.
The application, training scripts, dataset, and agent will be added separately.

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

Once scripts exist, run them with `docker compose run --rm ml python train.py`.
Place CSV datasets in `data/` and trained models in `artifacts/`; both are
excluded from Git and image builds but available through the development mount.
This setup uses CPU PyTorch; GPU support requires a separate CUDA setup.

The existing host `.venv` is independent of Docker; no activation is required
for Docker commands. Docker must be running and your user must have permission
to access its daemon.

PyTorch CPU installation follows the [official version instructions](https://docs.pytorch.org/get-started/previous-versions/).

# Deployment

## Status

The app is live at <https://quantic-malware-agent.onrender.com/>. On
2026-10-09, its [/health](https://quantic-malware-agent.onrender.com/health)
reported `status: ok`, `selected_model: LightGBM`, `openai_configured: true`,
model version `d13e54cf1970-1791236375742615262` and commit
[`3f384a8`](https://github.com/mehdihamid1/ML-Project/commit/3f384a8).

Observed [GitHub Actions run](https://github.com/mehdihamid1/ML-Project/actions/runs/37865252113)
for that commit:

| Check | Observed result |
| --- | --- |
| Full test job | Passed. |
| Runtime-only test job | Passed. |
| Production-container job | Passed. |
| Deploy job | Passed; deployed the tested commit. |
| Live `/health` smoke test | Passed. |
| Overall workflow | Passed. |

A real-LLM run of the 13 evaluation scenarios against that deployment passed
13 of 13; see [agent-evaluation.md](agent-evaluation.md). The AI-decided
conditional task, the sample-file buttons and the model name in `/health`,
described in the current README, are verified locally but not yet deployed;
the observations above describe commit `3f384a8`.

## Render setup

Create a Render web service from this repository's `render.yaml`. It uses the
free plan, Python runtime, `requirements-runtime.txt`, the bundled frozen
LightGBM model, and Gunicorn. Build verification checks the model checksum and
metadata without fitting a model. The runtime excludes training-only packages.
The build also runs `pip check`. `GUNICORN_CMD_ARGS` is explicitly set to access
logging, overriding Render's injected preload default so model loading occurs
in the worker, matching the Docker image. Render's default is documented in
its [environment-variable reference](https://render.com/docs/environment-variables).
`autoDeployTrigger: 'off'` disables Render auto-deploys as required by the
[Blueprint reference](https://render.com/docs/blueprint-spec).

Set `OPENAI_API_KEY` in Render's environment. The Blueprint generates
`FLASK_SECRET_KEY`; it also sets secure cookies and proxy handling. If you use
an existing service, apply these settings there and confirm auto-deploy is Off.
Use a single Gunicorn worker because session state is held in that process.
Session uploads and downloads are ephemeral and disappear on restart.

In GitHub repository settings, create the `production` environment. Configure:

- Repository secret `RENDER_DEPLOY_HOOK`: the service's secret hook from Render.
- Repository variable `RENDER_HEALTH_URL`: its full HTTPS `/health` URL.

The [Render deploy-hook documentation](https://render.com/docs/deploy-hooks)
describes where to retrieve the hook. Never paste or commit that secret URL.
Blueprint creation can perform an initial deployment; run CI first before
applying it. Subsequent code deployments go through GitHub Actions.

## CI and live verification

All jobs pin `ubuntu-24.04`. Checkout and setup-python use their Node 24
`v6` releases, as specified by their
[checkout metadata](https://github.com/actions/checkout/blob/v6/action.yml) and
[setup-python metadata](https://github.com/actions/setup-python/blob/v6/action.yml).
The full job installs training dependencies and runs `pytest`. A separate job
runs the Blueprint's `buildCommand` exactly as written in `render.yaml`, which
installs only `requirements-runtime.txt`, runs `pip check` and verifies the
bundled model. It then adds the test runner, checks that training-only libraries
are absent and runs the suite excluding training tests. Finally it starts the
app with the Blueprint's `startCommand` on `PORT=10000` and requires `/health`
to report `status: ok` and the tested commit, so edits to either command are
tested before deployment.

A third CI job builds `Dockerfile.runtime` and runs
`scripts/check_container.py` against the actual Gunicorn service. Its OpenAI
HTTP endpoint is a local mock; classifications and metrics still come from the
frozen model. This compatibility job must also pass before deployment.

Pull requests run all test jobs. Pushes to `main` and workflow dispatch run
all jobs and, only after all succeed, the deploy job. Missing deployment settings fail that job
explicitly. `scripts/deploy.py` passes the tested commit as the hook's `ref`,
then waits for `/health` to report `status: ok`, a model version and that exact
commit. An older healthy instance does not satisfy the smoke check. Cold starts
and temporary deployment errors are retried. Hook URLs are never logged.

Local Gunicorn HTTP checks and the built production container returned healthy
model status. These are local verification, not evidence of a live Render
deployment. `/health` reports model availability, whether the OpenAI key is
set and the configured OpenAI model name; the deploy smoke check verifies the
model and commit without calling OpenAI. Real chat behavior is evaluated
separately against the live site; see [agent-evaluation.md](agent-evaluation.md).

## Local Docker and Render compatibility

The current Blueprint passes the JSON schema served by
[Render](https://render.com/schema/render.yaml.json). Docker Compose configuration
validates, and both the training and production images build successfully.
The training image passes 320 tests; the lean local environment passes
309 tests. Authenticated Jupyter HTTP access works with the host owner's UID/GID.

The production image works at its local default port and honors a custom
`PORT`, using non-root permissions, the lean dependencies and the verified
frozen model. The container probe exercises the actual Gunicorn worker, proxy
headers, secure cookies, the model dashboard and its recorded JSON, uploads, a
bundled sample file, all tool paths, incremental function events, the
conditional task (its provider stub reads the returned accuracy before choosing
`predict_single`), invalid rows and isolated downloads. It also scores and downloads every row of a
maximum-row synthetic batch. The provider is mocked for these checks.

The final [recorded container probe](docs/container-compatibility.json) passed
with a 0.1 CPU quota and a 512 MiB memory limit, including all 10,000 batch rows
and the subsequent health check. The report includes a memory sample taken
after the requests; it is not a peak-memory measurement. The report identifies
the tested image and explicitly marks the provider as mocked. Resource settings
reflect the [Render Free plan](https://render.com/docs/compute-plans), while
these observations come from local Docker.

The Blueprint's native build and start commands were also rehearsed in a clean
`python:3.10.16-bookworm` container, matching `PYTHON_VERSION`, with a 512 MiB
memory limit and `PORT=10000`. The build passed `pip check` and model
verification, and `/health`, the chat page and `/api/session` responded
successfully. LightGBM needs the OpenMP runtime (`libgomp`): the Docker image
installs `libgomp1`, and on Render's native runtime the build's model
verification fails early if that library is missing. In that case, switch the
Blueprint to `runtime: docker` with `Dockerfile.runtime`, which the container
job already tests.

Feature dtype rules and integer bounds are compiled once when `ToolService`
loads the fixed schema. This removes repeated type parsing from each CSV row,
which otherwise delayed large batches on the Free plan CPU allowance. Exact
numeric validation, model parameters, probabilities and the decision threshold
are preserved; no training or hold-out tuning was performed.

The latest chat-progress compatibility report is local evidence for the new
changes. The successful CI and live observations above describe commit
`341448f`; they do not establish that the uncommitted chat-progress update
has run on GitHub or Render.

For a manual execution of the same verified path, export the settings and the
tested commit into the environment, then run:

```bash
python scripts/deploy.py
```

To roll back, deploy a previously tested commit through the same path and
verify that commit's live health. Do not enable Render's independent automatic
deploys; the test gate must remain the deployment trigger.

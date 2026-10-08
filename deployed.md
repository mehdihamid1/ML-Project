# Deployment

## Status

The Flask application, production model, Render Blueprint and test-gated GitHub
Actions workflow are committed and pushed to `main`. The latest code milestone,
[`1c5f4bb`](https://github.com/mehdihamid1/ML-Project/commit/1c5f4bb), adds the
review fixes and lean-runtime CI gate.
**No live deployment is claimed.**

Observed [GitHub Actions run](https://github.com/mehdihamid1/ML-Project/actions/runs/37396524639)
for that commit:

| Check | Observed result |
| --- | --- |
| Full test job | Passed; 222 tests passed and frozen-model verification succeeded. |
| Runtime-only test job | Passed; 211 tests passed, training-only libraries were absent, and model verification succeeded. |
| Deploy job | Failed before contacting Render because the hook and health URL were unset. |
| Overall workflow | Failed because the deploy job failed. |
| Live `/health` smoke test | Not run. |

The development environment has no Render hook or health URL, and the repository
currently has no deployment secret or health variable configured. Record the
actual service URL and successful deployment workflow run here once configured.

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
deployment. `/health` reports model availability and whether the OpenAI key is
set; the deploy smoke check verifies the model and commit without calling OpenAI.
Real chat behavior requires the separate provider evaluation described in
[agent-evaluation.md](agent-evaluation.md).

## Local Docker and Render compatibility

The current Blueprint passes the JSON schema served by
[Render](https://render.com/schema/render.yaml.json). Docker Compose configuration
validates, and both the training and production images build successfully.
The training image passes 289 tests; the lean local environment passes
278 tests. Authenticated Jupyter HTTP access works with the host owner's UID/GID.

The production image works at its local default port and honors a custom
`PORT`, using non-root permissions, the lean dependencies and the verified
frozen model. The container probe exercises the actual Gunicorn worker, proxy
headers, secure cookies, the model dashboard and its recorded JSON, uploads, all tool paths, conditional ordering, invalid
rows and isolated downloads. It also scores and downloads every row of a
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

These latest startup and container-CI changes are local and have not been
deployed. The successful CI jobs above describe the earlier committed code;
they do not establish that this new container job has run on GitHub. Free-tier
performance and live health still require deployment on the actual service.

For a manual execution of the same verified path, export the settings and the
tested commit into the environment, then run:

```bash
python scripts/deploy.py
```

To roll back, deploy a previously tested commit through the same path and
verify that commit's live health. Do not enable Render's independent automatic
deploys; the test gate must remain the deployment trigger.

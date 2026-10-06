# Deployment

## Status

The Flask application, production model, Render Blueprint and test-gated GitHub
Actions workflow are committed and pushed to `main` in application milestone
[`8ca98b8`](https://github.com/mehdihamid1/ML-Project/commit/8ca98b8).
**No live deployment is claimed.**

Observed [GitHub Actions run](https://github.com/mehdihamid1/ML-Project/actions/runs/37392037715):

| Check | Observed result |
| --- | --- |
| Test job | Passed; 147 tests passed and frozen-model verification succeeded. |
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

Pull requests run tests. Pushes to `main` and workflow dispatch run tests and,
only after success, the deploy job. Missing deployment settings fail that job
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

For a manual execution of the same verified path, export the settings and the
tested commit into the environment, then run:

```bash
python scripts/deploy.py
```

To roll back, deploy a previously tested commit through the same path and
verify that commit's live health. Do not enable Render's independent automatic
deploys; the test gate must remain the deployment trigger.

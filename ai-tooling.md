# AI tooling

Codex assisted with the Python virtual environment, Docker development setup,
dataset inspection and documentation, and implementation of the seven-model
comparison, pipeline serialization, evaluation CLI, and automated tests.
It used the project PDF and shared `AGENTS.md` requirements, checked local CSV
statistics, and consulted official scikit-learn and CatBoost documentation.

Docker validation caught a disk-space constraint during the CatBoost rebuild;
project-specific build records were inspected and the replacement image was
verified. Automated tests exercised model fitting and serialization rather
than assuming package installation guaranteed a working pipeline.

Training and evaluation numbers are produced by the Python scripts. Development
assistance is distinct from the application's OpenAI agent, which now selects
tools and verified result references. Python performs all prediction,
evaluation, conditional checks and final result rendering. This prevents the
provider from supplying invented labels, metrics or feature explanations.

Codex used parallel implementation agents for the CSV tools, OpenAI routing,
and Flask UI, and reviewed their contracts together. Official OpenAI
function-calling and Render Blueprint/deploy-hook documentation informed the
integration. Development checks include the pinned OpenAI SDK with mocked HTTP,
real frozen-model tool execution, Flask integration, production-image building
and local browser/HTTP verification. The
[full GitHub CI job](https://github.com/mehdihamid1/ML-Project/actions/runs/37396524639/job/112053720455)
passed 222 tests, and the
[lean runtime job](https://github.com/mehdihamid1/ML-Project/actions/runs/37396524639/job/112053720662)
passed 211 tests. Both verified the frozen model. The deploy job failed because Render
settings were missing; the workflow did not complete a deployment.

A Claude Code review identified agent routing, follow-up, validation, percentage
threshold, session-capacity and deployment-test gaps. Codex used parallel agents
to implement and cross-review those fixes, added mocked SDK tests for stateless
reasoning, and checked official OpenAI model/reasoning and GitHub action
documentation. The reproducible [experiment audit](scripts/audit_experiment.py)
checks saved partitions, CV summaries and artifact hashes without fitting,
predicting or changing the frozen training configuration. Results analysis
uses its aggregate output and the original training report.

Codex checked Docker and native Render startup against current official Render
documentation and its Blueprint schema. It rebuilt both Docker images, verified
authenticated Jupyter access, and added a reproducible production HTTP probe
using an actual memory-limited Gunicorn container. The probe's provider is a
local stub, while model predictions, metrics and downloads use production code.
The latest local full and lean suites pass 246 and 235 tests respectively.

Claude Code (Anthropic) performed that review and checked its findings with
read-only experiments: phrasing checks against the conditional detector and an
exact feature-vector overlap check between the saved partitions, which the
audit script now formalizes. It reviewed Codex's changes before committing them,
added the agent's fallback for models that reject the encrypted-reasoning
option (shared by the readiness probe), extended descriptive-phrasing and
threshold-wording handling with tests, reran the container probe on a fresh
image, and rehearsed the Blueprint's native build and start commands, which the
lean CI job now runs from `render.yaml`.

Real OpenAI scenario runs and live Render verification remain pending. Mocked
SDK tests and local health checks do not fulfill those deliverables. Their
status is recorded in [agent-evaluation.md](agent-evaluation.md) and
[deployed.md](deployed.md). The project author must review, understand, and
explain the resulting code and experiment design.

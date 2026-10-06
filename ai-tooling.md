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
[GitHub CI test job](https://github.com/mehdihamid1/ML-Project/actions/runs/37392037715/job/112039137706)
passed tests and verified the frozen model. The deploy job failed because Render
settings were missing; the workflow did not complete a deployment.

Real OpenAI scenario runs and live Render verification remain pending. Mocked
SDK tests and local health checks do not fulfill those deliverables. Their
status is recorded in [agent-evaluation.md](agent-evaluation.md) and
[deployed.md](deployed.md). The project author must review, understand, and
explain the resulting code and experiment design.

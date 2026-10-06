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
integration. Automated checks use a mocked provider; real OpenAI scenario runs
and live Render deployment have not been completed because the required
environment configuration is missing. The project author must review,
understand, and explain the resulting code and experiment design.

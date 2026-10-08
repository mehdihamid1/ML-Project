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
The latest local full and lean suites pass 289 and 278 tests respectively.

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

Codex added the model dashboard with parallel data, interface and review
agents. The dashboard uses an exported summary of the original recorded
experiment, without retraining or changing the production artifact. Its loader
checks metric consistency and model version; local SVG charts add no external
assets or runtime packages. Hold-out results are displayed separately from
cross-validation scores.

Claude Code then revised the dashboard's charts against its data-visualization
guidance. Mean AUC and accuracy moved from bars on a 0–1 axis, where every model
looked identical, to dots with fold standard deviations on a fitted axis. A
line joining unordered folds became a paired fold-by-fold comparison with a
value table. Chart colors were checked with a color-vision validator, and the
charts gained keyboard-accessible tooltips and larger caption text. It checked
the result in headless-browser screenshots at desktop and phone widths.

Codex added the guided experiment presentation at the user's request: a
train/test split diagram, a selectable cross-validation round with saved
scores, and ordered comparison and final-test sections. Codex remained the
sole editor while parallel agents reviewed calculations and regression
coverage read-only, following the shared-checkout instructions. The walkthrough
uses the existing report and does not retrain or alter the production model.
Chrome checks matched all 70 saved model/round score pairs, exercised keyboard
controls and existing charts, and verified fallback views and layouts from
320 to 1440 pixels without page overflow or overlapping fold labels.

Claude Code then reviewed that presentation and revised it. The selection
rationale and fold-by-fold comparison moved ahead of the final test, matching
the order of the experiment. The first step gained the recorded deduplication
counts; the walkthrough gained a strip of the chosen model's ten round scores
around their mean; and the final-test panel gained error rates and the range of
LightGBM's cross-validation rounds. The walkthrough colors were changed to pass
the color-vision validator. Chrome checks again matched all 70 model/round
pairs and covered pointer, keyboard and no-JavaScript use from 320 to 1440
pixels.

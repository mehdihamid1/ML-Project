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
The latest local full and lean runtime suites pass 460 and 449 tests respectively.

Claude Code (Anthropic) performed that review and checked its findings with
read-only experiments: phrasing checks against the conditional detector and an
exact feature-vector overlap check between the saved partitions, which the
audit script now formalizes. It reviewed Codex's changes before committing them,
added the agent's fallback for models that reject the encrypted-reasoning
option (shared by the readiness probe), extended descriptive-phrasing and
threshold-wording handling with tests, reran the container probe on a fresh
image, and rehearsed the Blueprint's native build and start commands, which the
lean CI job now runs from `render.yaml`.

Real OpenAI scenario runs and live Render verification are recorded separately
from mocked SDK tests and local health checks. Their status is recorded in
[agent-evaluation.md](agent-evaluation.md) and
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

Codex added a Copy button for uploaded file IDs. Claude Code then reworked the
chat page for the demo. The quick actions had always used the oldest upload
and disappeared after the first answer, so each file card now carries its own
**Predict row 0**, **Classify all** and, for labelled files, **Evaluate**
buttons. Uploads start when files are chosen or dropped. Each answer lists the
tools behind it, result cards lead with the verdict or headline metrics, the
chat sits beside the files and stays in view, and no text is smaller than 11px.
Phones show the chat straight after the file list. Chrome checks drove every
demo step through the page against a local stand-in for the OpenAI API at
1440 and 390 pixels.

Codex added upload-prepared questions and incremental chat progress, following
the user's instruction that they alone send requests. The conditional form now
prepares a question too. The browser shows function names and actual status
events streamed by the same agent implementation used by the JSON API. Mocked
provider tests cover event ordering, inclusive accuracy gates, failures,
interrupted streams, storage rollback and session cleanup. Local Chrome checks
use the frozen model with a fake provider; these checks do not count as the
required real-LLM evaluation.

Claude Code then fitted the chat page to the first screen at the user's
request: the live site placed the chat input below the fold. The header and
title now take one compact band, and the file and conditional panels share a
column that scrolls on its own beside the chat. Chrome checks confirmed a
visible chat input without scrolling at sizes from 1024×700 to 1920×1080, and
the full demo flow with five uploaded files, against a local stand-in for the
OpenAI API.

Claude Code scored the project against the assignment's rubric at the user's
request and found two gaps in the agent. The real-LLM scenarios had never run,
and server code, not the AI model, decided the conditional prediction, although
the assignment says the model must use the returned accuracy to decide. With
the user's go-ahead it changed the conditional task: the model must call
`evaluate` first, reads the accuracy and decides whether to call
`predict_single`, and the server checks that decision and blocks a prediction
the rule does not allow. It added a live mode to the scenario runner. That mode
drives the deployed site over HTTP, so the OpenAI key stays in Render, and it
checks each reply's numbers against the tool output. It also added sample-file
buttons, so graders can try the app without a CSV of their own. The first live
run, against the previous deployment, passed 13 of 13 scenarios; reading its
replies led to two wording and coverage fixes recorded in
[agent-evaluation.md](agent-evaluation.md). A training-folds-only feature check
documents the input choice without changing the frozen model.

A follow-up review, using a mocked model, found two bugs in how conditional
requests were read: a condition worded without "if" ("predict when accuracy is
0.95 or higher") skipped the evaluation, and an earlier figure such as "previous
accuracy of 0.80" could become the threshold. Claude Code reproduced both,
fixed them so the threshold is read from the condition itself (two different
values prompt a clarification question), and added regression tests and two
matching live scenarios. The second live run then failed 2 of 15 scenarios:
the model routed a plain prediction to evaluation, and a reply depended on
the metric focus the model chose. Routing instructions were tightened and new
evaluations now always report every metric. The third live run passed all 15.

Codex followed up on conditional-request validation. It removed the fallback
from an unnumbered condition to an earlier accuracy figure, separated sequencing
and descriptive comparisons from accuracy gates, and bound conditional tool
arguments to the submitted evaluation file, prediction file and row. Unclear
requests ask for clarification; model-selected substitutions are blocked
before execution. OpenAI still decides whether to call the prediction tool
from the returned evaluation result.

It strengthened the real-LLM runner with explicit expected arguments and an
independent frozen-model reference for single and batch classifications,
including downloaded row identities and probabilities. Mocked regression tests
exercise substitutions and malformed requests; they are not a new real-LLM
run. Existing live reports and the production model are unchanged.

Claude Code then ran the fourth live run with that runner against the
deployed change: 14 of 15 scenarios passed. The false-negative follow-up was
answered correctly but re-ran the evaluation, because two routing instructions
overlapped. Claude Code clarified them, and the fifth live run, against that
deployed change, passed all 15. It also added the README's sample
requests and the design document's tool schemas and AI model choice, checking
each field against the code and the HTTP example against the live site.

Codex independently repeated the live run on `2bf449b`: 14 of 15 scenarios
passed, with all 29 numeric checks matching. An unrequested prediction on the
false-negative follow-up showed that routing instructions alone were
insufficient. Codex disabled tools for stored-result questions and added a
server guard before any execution, with adversarial tests for all three tools,
explicit fresh requests and scoped session references. Team approval was
confirmed for a controlled failing-test commit to demonstrate the deployment
gate. The final guard requires an affirmative new request before enabling
tools after stored results exist, covering paraphrases without named metrics.
Live Run 7 passed all 15 scenarios and all 27 numeric checks; four additional
follow-ups ran no tools, with one unnecessary refusal and one correct
true-positive answer rejected by the supplemental script's narrower text check.
The blocked-deploy experiment and its revert are recorded in `deployed.md`.

Claude Code then checked the guard against per-record questions on the live
site. After a batch, "What is the malware probability of row 2?" was answered
"unavailable" and "Is row 0 malware?" returned only the batch counts, because
any question after a stored result disabled tools. Claude Code kept the guard
but let a question naming specific rows classify them unless a stored
prediction already answers it; past-tense row questions get a tool only for
rows a stored batch classified, so a withheld prediction stays withheld. A
stored-result question the model declines as a feature explanation now also
shows the stored result. Regression tests cover both cases.

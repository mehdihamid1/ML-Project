# Agent evaluation

The agent is evaluated in two ways. Mocked-provider tests run in CI on every
push and check routing, failures and safety rules deterministically. Separately,
scenario runs use the real OpenAI model through the deployed app; those are the
real-LLM evidence below.

**Latest result:** Run 6, on 2026-10-09 against the live site at commit
`2bf449b` with `gpt-4.1-mini` and the stronger runner, passed **14 of 15**
scenarios. A stored-result follow-up triggered an unrequested prediction.
Earlier runs are kept below, including passing runs and failures.

## How a real-LLM run works

Use Python 3.10 with `requirements-runtime.txt` installed (and the platform's
LightGBM shared-library dependency), or run the command inside the project's
ML Docker service. Both modes need the bundled frozen model locally so the
runner can independently verify the requested classifications.

```bash
python3 scripts/run_agent_evaluation.py --base-url https://quantic-malware-agent.onrender.com --output artifacts/agent-evaluation/<run-name>
```

The runner drives the deployed site the way a browser does. For each scenario
it opens a new session, uploads the sample CSVs it needs, sends the request to
`/api/chat` and records the reply, the tools called and the tool results. The
server's own OpenAI key is used, so no key is needed where the runner runs. It
reads the AI model's name from `/health` and waits for a sleeping Render
instance to start.

Each scenario passes only if all of these hold:

- the tools called, their outcomes and exact file/row arguments match the
  expected list, including skipped predictions and intentionally failed calls;
- each successful result matches its activity entry and requested file and row;
- single predictions and every batch-download row match an independent run
  of the frozen LightGBM model on the requested sample, including identity,
  classification and probability;
- the scenario's own check holds, for example that an invalid row is reported
  with its reason and kept in the download;
- every six-decimal number in the reply equals a value in the tool output,
  the user’s explicit threshold, or a rate computed from the confusion matrix,
  preserving numeric signs.

Live `/health` must report the same ML model version as the local reference.
The runner records model and sample SHA-256 values, resolved expected calls,
every request, reply, tool call and result in `results.json`,
and a summary to `report.md`. It exits unsuccessfully when any scenario fails.
Without `--base-url` it runs the agent in-process instead, which needs
`OPENAI_API_KEY` in the shell; `scripts/check_openai.py` checks that key first.

These stronger checks and the request-binding fixes were added after Runs
1–3 below. Those runs used the earlier runner and remain unchanged historical
evidence.
A new real-LLM run is required after these changes are deployed. Mocked tests
also deliberately substitute file IDs, row indexes, result IDs, classifications,
probabilities and batch rows to verify that the current runner rejects them.

## Scenarios

The 15 scenarios cover every case the assignment lists. Scenarios 7 and 8 were
added after Run 1 to check two condition bugs a review found (below). The
sample files are in [samples/](samples/README.md).

| # | Scenario | Assignment requirement | Expected behavior |
| --- | --- | --- | --- |
| 1 | Single prediction | Single prediction | `predict_single` returns class 0 or 1 with its probability. |
| 2 | Batch prediction | Batch prediction | `predict_batch` classifies all 4 rows; the download lists each row. |
| 3 | Labeled evaluation | Labeled evaluation | `evaluate` returns AUC, accuracy and a confusion matrix over 4 rows. |
| 4 | Conditional: prediction permitted | Conditional task, both outcomes | `evaluate` first; accuracy meets 0.75, so `predict_single` is called. |
| 5 | Conditional: prediction withheld | Conditional task, both outcomes | `evaluate` first; accuracy is below 0.9, so `predict_single` is not called. |
| 6 | False-negative follow-up | False-negative follow-up | Answered from the stored evaluation in scenario 5's session, with no new tool call. |
| 7 | Condition without "if" | Conditional task | "Predict … when the accuracy … is 0.75 or higher" is treated as conditional: `evaluate` first, then `predict_single`. |
| 8 | Earlier accuracy figure ignored | Conditional task | "The previous accuracy of 0.0 was too low … only if accuracy is at least 0.9": the threshold is 0.9, so `predict_single` is not called. |
| 9 | Invalid input row | Invalid input | The invalid row is reported with its reason and kept in the download. |
| 10 | Missing labels | Missing labels | The unlabeled row is reported; metrics use the 3 labeled rows. |
| 11 | No Label column | Missing labels | `evaluate` fails visibly and asks for a Label column. |
| 12 | Single-class evaluation | Single-class evaluation | AUC is reported as unavailable; accuracy is still given. |
| 13 | Tool failure | Tool or service failure (controlled fault) | The one-row file has no row 7: the tool fails and no result is claimed. |
| 14 | Feature explanation refused | No invented explanations | The agent says no explanation tool exists and runs no tool. |
| 15 | Ambiguous condition | Clarification when ambiguous | The server asks for a numeric threshold before any AI call. |

`conditional-fail.csv` deliberately holds one malware-labeled row that the
model classifies as goodware. It exercises the withheld branch and gives the
follow-up one false negative; it is not a performance dataset.

## Run 1: live site at commit `3f384a8`, 2026-10-09

- Where: <https://quantic-malware-agent.onrender.com>, deployed commit
  `3f384a89b18eb59a886532a45b3283fc17f51230`.
- AI model: not reported, because that version's `/health` does not name it.
  The app's default is `gpt-4.1-mini` unless `OPENAI_MODEL` is set in Render.
- ML model version: `d13e54cf1970-1791236375742615262`.
- Run (UTC): 2026-10-09T13:02:40+00:00.
- Result: **13 of 13 scenarios passed (100%)**. All 14 six-decimal numbers in
  the replies matched the tool outputs.
- Full report with every reply:
  [docs/agent-evaluation-live-3f384a8.md](docs/agent-evaluation-live-3f384a8.md).

| # | Scenario | Tools called | Result |
| --- | --- | --- | --- |
| 1 | Single prediction | `predict_single` (success) | PASS |
| 2 | Batch prediction | `predict_batch` (success) | PASS |
| 3 | Labeled evaluation | `evaluate` (success) | PASS |
| 4 | Conditional: prediction permitted | `evaluate` (success), `predict_single` (success) | PASS |
| 5 | Conditional: prediction withheld | `evaluate` (success), `predict_single` (skipped) | PASS |
| 6 | False-negative follow-up | none | PASS |
| 7 | Invalid input row | `predict_batch` (success) | PASS |
| 8 | Missing labels | `evaluate` (success) | PASS |
| 9 | No Label column | `evaluate` (error) | PASS |
| 10 | Single-class evaluation | `evaluate` (success) | PASS |
| 11 | Tool failure | `predict_single` (error) | PASS |
| 12 | Feature explanation refused | none | PASS |
| 13 | Ambiguous condition | none | PASS |

**Findings.** No scenario failed: every observed tool sequence matched the
expected one. The model answered the false-negative follow-up from the stored
result ("False negatives: 1 malware files predicted as goodware") and called no
tool for the explanation request. Three changes followed from reading the
replies:

- In that version, server code decided the conditional prediction inside a
  combined tool. The assignment asks the AI model to use the returned accuracy
  to make that decision, so the agent now does: the model calls `evaluate`,
  reads the accuracy and decides whether to call `predict_single`, and the
  server checks the decision (see
  [evaluation-and-design.md](evaluation-and-design.md)). Run 1 therefore tests
  the earlier design.
- Scenario 10 asked only for the AUC, so its reply left out accuracy. The
  scenario now sends a plain evaluation request and also checks that accuracy
  is reported, as the assignment asks for single-class files.
- Reply 8 said "invalid rows 1, missing labels 1" about one unlabeled row,
  which reads like two problem rows. Replies now say "1 excluded (missing
  labels 1, invalid labels 0)".

A later review, using a mocked model, found two bugs in how requests were read.
Both were in the version Run 1 tested, and both are fixed with regression tests:

- A condition worded without "if", such as "predict when accuracy is 0.95 or
  higher", was treated as a plain prediction, so a model could predict without
  evaluating first. Such wordings now count as conditional.
- The threshold parser took the first accuracy number anywhere in the message.
  "The previous accuracy of 0.80 … only if accuracy is at least 0.95" bound
  0.80, so an evaluation of 0.90 permitted the prediction. The threshold now
  comes from the condition itself, and two different values in the condition
  prompt a clarification question.

## Run 2: live site at commit `ab544ad`, 2026-10-09

This run tests the AI-decided conditional task and the two condition fixes.

- Where: <https://quantic-malware-agent.onrender.com>, deployed commit
  `ab544ada7054adf5bd99c0e27b78163dcbd807de`.
- AI model: `gpt-4.1-mini`, as reported by `/health`.
- ML model version: `d13e54cf1970-1791236375742615262`.
- Run (UTC): 2026-10-09T13:55:59+00:00.
- Result: **13 of 15 scenarios passed**. All 22 six-decimal numbers in the
  replies matched the tool outputs.
- Full report with every reply:
  [docs/agent-evaluation-live-ab544ad.md](docs/agent-evaluation-live-ab544ad.md).

| # | Scenario | Tools called | Result |
| --- | --- | --- | --- |
| 1 | Single prediction | `evaluate` (error) | FAIL |
| 2 | Batch prediction | `predict_batch` (success) | PASS |
| 3 | Labeled evaluation | `evaluate` (success) | PASS |
| 4 | Conditional: prediction permitted | `evaluate` (success), `predict_single` (success) | PASS |
| 5 | Conditional: prediction withheld | `evaluate` (success), `predict_single` (skipped) | PASS |
| 6 | False-negative follow-up | none | PASS |
| 7 | Condition without "if" | `evaluate` (success), `predict_single` (success) | PASS |
| 8 | Earlier accuracy figure ignored | `evaluate` (success), `predict_single` (skipped) | PASS |
| 9 | Invalid input row | `predict_batch` (success) | PASS |
| 10 | Missing labels | `evaluate` (success) | PASS |
| 11 | No Label column | `evaluate` (error) | PASS |
| 12 | Single-class evaluation | `evaluate` (success) | FAIL |
| 13 | Tool failure | `predict_single` (error) | PASS |
| 14 | Feature explanation refused | none | PASS |
| 15 | Ambiguous condition | none | PASS |

**Findings.** The conditional task worked with the real model in all four of
its scenarios. In each, the model read the returned accuracy and made the right
call, including for the two wordings the review flagged. Two failures and one
gap in the checks showed one pattern: the reply text depended on choices the
model made outside the task.

- Scenario 1: for "Classify row 0 of file single.csv", the model called
  `evaluate` instead of `predict_single`. The evaluation failed for lack of
  labels, and no prediction was made. The system instructions described the
  conditional steps ("call evaluate first") to every request, which the model
  over-applied. They now route one row to `predict_single`, a whole file to
  `predict_batch` and only evaluation requests to `evaluate`. The conditional
  steps appear only in the developer message added to conditional requests.
- Scenario 12: the model chose the accuracy focus, so the reply left out the
  required explanation that AUC is undefined. An unavailable AUC is now always
  explained, together with the accuracy.
- Scenario 3 passed, but its reply gave only accuracy although AUC and the
  confusion matrix were requested; the check had read the tool result, not the
  reply. A new evaluation now always reports every metric, the focus only
  narrows follow-ups, and the check reads the reply.

## Run 3: live site at commit `db49bfe`, 2026-10-09

This run tests the routing and reply fixes from Run 2.

- Where: <https://quantic-malware-agent.onrender.com>, deployed commit
  `db49bfeaed948e50da32a5f58a94157ed2c740a6`.
- AI model: `gpt-4.1-mini`, as reported by `/health`.
- ML model version: `d13e54cf1970-1791236375742615262`.
- Run (UTC): 2026-10-09T14:10:32+00:00.
- Result: **15 of 15 scenarios passed (100%)**. All 27 six-decimal numbers in
  the replies matched the tool outputs.
- Full report with every reply:
  [docs/agent-evaluation-live-db49bfe.md](docs/agent-evaluation-live-db49bfe.md).

| # | Scenario | Tools called | Result |
| --- | --- | --- | --- |
| 1 | Single prediction | `predict_single` (success) | PASS |
| 2 | Batch prediction | `predict_batch` (success) | PASS |
| 3 | Labeled evaluation | `evaluate` (success) | PASS |
| 4 | Conditional: prediction permitted | `evaluate` (success), `predict_single` (success) | PASS |
| 5 | Conditional: prediction withheld | `evaluate` (success), `predict_single` (skipped) | PASS |
| 6 | False-negative follow-up | none | PASS |
| 7 | Condition without "if" | `evaluate` (success), `predict_single` (success) | PASS |
| 8 | Earlier accuracy figure ignored | `evaluate` (success), `predict_single` (skipped) | PASS |
| 9 | Invalid input row | `predict_batch` (success) | PASS |
| 10 | Missing labels | `evaluate` (success) | PASS |
| 11 | No Label column | `evaluate` (error) | PASS |
| 12 | Single-class evaluation | `evaluate` (success) | PASS |
| 13 | Tool failure | `predict_single` (error) | PASS |
| 14 | Feature explanation refused | none | PASS |
| 15 | Ambiguous condition | none | PASS |

**Findings.** Every scenario passed. The plain single prediction went to
`predict_single`. The evaluation reply reported accuracy, AUC and the
confusion matrix, as requested. The single-class reply explained that AUC is
unavailable and still gave accuracy. In the conditional scenarios, the model
read the returned accuracy and made the correct call each time. One run per
version cannot measure how often the model's choices vary; the mocked tests
and the server's checks cover the cases where a model chooses wrongly.

## Run 4: live site at commit `e203583`, 2026-10-09

This run tests the request binding of evaluation file, prediction file and
row, using the stronger runner described above.

- Where: <https://quantic-malware-agent.onrender.com>, deployed commit
  `e20358346a238908ca7650d20937ac794b272f96`.
- AI model: `gpt-4.1-mini`, as reported by `/health`.
- ML model version: `d13e54cf1970-1791236375742615262`. The runner's local
  reference model has SHA-256
  `e98b09552e0d77dd5abc070189e452e1084594a379e6849faba7b3385b94c677`.
- Run (UTC): 2026-10-09T15:57:10+00:00.
- Result: **14 of 15 scenarios passed**. All 27 six-decimal numbers in the
  replies matched the tool outputs. Every single and batch classification
  matched the frozen model run locally on the same sample. In every passing
  scenario, each tool call used the file and row the request named.
- Full report with every reply:
  [docs/agent-evaluation-live-e203583.md](docs/agent-evaluation-live-e203583.md).

| # | Scenario | Tools called | Result |
| --- | --- | --- | --- |
| 1 | Single prediction | `predict_single` (success) | PASS |
| 2 | Batch prediction | `predict_batch` (success) | PASS |
| 3 | Labeled evaluation | `evaluate` (success) | PASS |
| 4 | Conditional: prediction permitted | `evaluate` (success), `predict_single` (success) | PASS |
| 5 | Conditional: prediction withheld | `evaluate` (success), `predict_single` (skipped) | PASS |
| 6 | False-negative follow-up | `evaluate` (success) | FAIL |
| 7 | Condition without "if" | `evaluate` (success), `predict_single` (success) | PASS |
| 8 | Earlier accuracy figure ignored | `evaluate` (success), `predict_single` (skipped) | PASS |
| 9 | Invalid input row | `predict_batch` (success) | PASS |
| 10 | Missing labels | `evaluate` (success) | PASS |
| 11 | No Label column | `evaluate` (error) | PASS |
| 12 | Single-class evaluation | `evaluate` (success) | PASS |
| 13 | Tool failure | `predict_single` (error) | PASS |
| 14 | Feature explanation refused | none | PASS |
| 15 | Ambiguous condition | none | PASS |

**Findings.** The four conditional scenarios passed with the exact file and
row arguments each request named. Scenario 6 failed. Asked "How many false
negatives were there in that evaluation?", the model called `evaluate` again on
the same file instead of choosing the stored result. The reply was correct
(1 false negative, rate 1.000000), but the scenario expects no new tool call.
The same scenario passed in Run 3 with the same instructions, so the model's
choice varies. Two instructions pointed different ways: "call evaluate only
when the user asks … for its accuracy, AUC or confusion matrix" also matched a
follow-up about false negatives, against "for follow-ups choose existing
result_ids". The instructions now limit `evaluate` to explicit evaluation
requests and to files no session result has evaluated, and say that a
follow-up about an earlier result calls no tool. Run 5 tested that change.

## Run 5: live site at commit `d9ecab7`, 2026-10-09

This run tests the follow-up instruction change from Run 4, with the same
runner.

- Where: <https://quantic-malware-agent.onrender.com>, deployed commit
  `d9ecab72b7ba66634f1b47bcc978e6c07a354307`.
- AI model: `gpt-4.1-mini`, as reported by `/health`.
- ML model version: `d13e54cf1970-1791236375742615262`. The runner's local
  reference model has SHA-256
  `e98b09552e0d77dd5abc070189e452e1084594a379e6849faba7b3385b94c677`.
- Run (UTC): 2026-10-09T16:09:18+00:00.
- Result: **15 of 15 scenarios passed (100%)**. All 27 six-decimal numbers in
  the replies matched the tool outputs. Every single and batch classification
  matched the frozen model run locally on the same sample, and each tool call
  used the file and row the request named.
- Full report with every reply:
  [docs/agent-evaluation-live-d9ecab7.md](docs/agent-evaluation-live-d9ecab7.md).

| # | Scenario | Tools called | Result |
| --- | --- | --- | --- |
| 1 | Single prediction | `predict_single` (success) | PASS |
| 2 | Batch prediction | `predict_batch` (success) | PASS |
| 3 | Labeled evaluation | `evaluate` (success) | PASS |
| 4 | Conditional: prediction permitted | `evaluate` (success), `predict_single` (success) | PASS |
| 5 | Conditional: prediction withheld | `evaluate` (success), `predict_single` (skipped) | PASS |
| 6 | False-negative follow-up | none | PASS |
| 7 | Condition without "if" | `evaluate` (success), `predict_single` (success) | PASS |
| 8 | Earlier accuracy figure ignored | `evaluate` (success), `predict_single` (skipped) | PASS |
| 9 | Invalid input row | `predict_batch` (success) | PASS |
| 10 | Missing labels | `evaluate` (success) | PASS |
| 11 | No Label column | `evaluate` (error) | PASS |
| 12 | Single-class evaluation | `evaluate` (success) | PASS |
| 13 | Tool failure | `predict_single` (error) | PASS |
| 14 | Feature explanation refused | none | PASS |
| 15 | Ambiguous condition | none | PASS |

**Findings.** Every scenario passed. The false-negative follow-up called no
tool and answered from the stored evaluation: "False negatives: 1 malware
files predicted as goodware. The false negative rate is 1.000000 (1 / 1
evaluated malware files)." The conditional scenarios again used exactly the
requested files and row. One passing run cannot show how often the model's
choices vary: scenario 6 also passed in Run 3, under the old instructions,
before failing in Run 4. The mocked tests and the server's checks cover wrong
conditional decisions, and in Run 4 an unneeded follow-up tool call still
gave the correct numbers.

## Run 6: live site at commit `2bf449b`, 2026-10-09

An independent repeat used the same stronger runner at
`2026-10-09T16:20:39+00:00`. The [generated report](docs/agent-evaluation-live-2bf449b.md)
records **14 of 15 scenarios passed** and **29 of 29 numeric checks matched**.
The frozen model and sample checksums matched the earlier runs.

Scenario 5 correctly evaluated `conditional-fail.csv`, returned accuracy 0,
and withheld prediction against the user's minimum of 0.9. Scenario 6 then
asked only, "How many false negatives were there in that evaluation?" The
agent called `predict_single` on the prior prediction file. Its answer included
the correct false-negative count and the new classification. The failure was
an unrequested tool execution on a later turn; the original threshold check
worked correctly.

The fix defaults subsequent turns to stored-result mode, including unfamiliar
follow-up paraphrases. New work requires an affirmative request or an explicit
file without compatible stored evidence. Tools are
disabled for that turn, compatible session evidence is scoped to explicit
references, and a provider function call is intercepted before validation or
execution. Explicit requests to evaluate or classify again still run tools.
Adversarial mocked-provider tests cover attempts to run all three tools after
a withheld prediction. A fresh real-provider run must verify this change after
deployment.

## Mocked tests in CI

These run on every push and are not real-LLM evidence:

- Agent tests use a scripted OpenAI client. They cover tool routing, the
  conditional task, follow-ups, fabricated provider text, malformed tool
  arguments, unknown file IDs, provider failures and tool failures. For the
  conditional task they check that:
  - `evaluate` is forced first;
  - the evaluation result reaches the model before its decision;
  - a prediction the threshold or label coverage does not allow is blocked;
  - a permitted prediction the model leaves out is reported, never made by
    the server;
  - conditions worded without "if" still force the evaluation, and an earlier
    accuracy figure in the message never becomes the threshold.

  Rendering tests check that a new evaluation reports every metric whatever
  focus the model picks, and that an unavailable AUC is always explained.
- Runner tests drive the same scenarios through the pinned OpenAI SDK with
  mocked HTTP responses and the frozen model. One runs the live mode against
  a local HTTP server.
- Flask integration tests cover uploads, sample files, tool execution, the
  conditional task over HTTP, downloads and session isolation.
- The production-container probe uses a local provider stub that reads the
  returned accuracy before choosing `predict_single`.

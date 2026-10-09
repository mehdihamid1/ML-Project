# Agent evaluation

The agent is evaluated in two ways. Mocked-provider tests run in CI on every
push and check routing, failures and safety rules deterministically. Separately,
scenario runs use the real OpenAI model through the deployed app; those are the
real-LLM evidence below.

## How a real-LLM run works

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

- the tools called, and their outcomes, match the expected list;
- the scenario's own check holds, for example that an invalid row is reported
  with its reason and kept in the download;
- every six-decimal number in the reply equals a value in the tool output, or
  a rate computed from its confusion matrix.

The runner writes every request, reply, tool call and result to `results.json`,
and a summary to `report.md`. It exits unsuccessfully when any scenario fails.
Without `--base-url` it runs the agent in-process instead, which needs
`OPENAI_API_KEY` in the shell; `scripts/check_openai.py` checks that key first.

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

## Run 2: after the AI-decided conditional task

Not run yet. It needs the change to be committed, pushed and deployed. The
same command then runs against the new deployment; its report will be added
here and to `docs/`.

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
- Runner tests drive the same scenarios through the pinned OpenAI SDK with
  mocked HTTP responses and the frozen model. One runs the live mode against
  a local HTTP server.
- Flask integration tests cover uploads, sample files, tool execution, the
  conditional task over HTTP, downloads and session isolation.
- The production-container probe uses a local provider stub that reads the
  returned accuracy before choosing `predict_single`.

# Agent evaluation

## Current evidence

Automated agent tests use a mocked OpenAI Responses client. They test routing,
conditional ordering and threshold enforcement, follow-ups, fabricated output,
malformed tool arguments, unknown file IDs, provider failures, and tool failures.
Runner tests also use the pinned OpenAI SDK with mocked HTTP responses and the
frozen production model. Flask integration tests cover upload, tool execution,
conditional ordering, downloads and session isolation. These checks are included
in the successful [full CI job](https://github.com/mehdihamid1/ML-Project/actions/runs/37862078681/job/113599923002)
and [lean runtime job](https://github.com/mehdihamid1/ML-Project/actions/runs/37862078681/job/113599922827).
They are not real-LLM evaluation evidence.

Regression tests cover ordinary descriptive “if” requests, named confusion
counts and rates in stored-result follow-ups, safe validation errors, independent
evaluation plus classification, and decimal percentage thresholds. SDK
transport tests replay encrypted reasoning and assistant phase through multiple
tool calls with `store=False`. These remain mocked-provider evidence.
They also verify the compatible retry when a model rejects the encrypted
include option, while unrelated provider errors fail without retry.
The production-container probe uses a local provider stub and real model tools;
it also remains separate from the real-LLM scenario evidence.
Local streaming regressions additionally check function milestones, interrupted
requests and session/storage cleanup. Those new changes await CI and deployment.

Real OpenAI scenarios have **not been run** because `OPENAI_API_KEY` is not
configured in the development environment. This deliverable remains incomplete
until the runner produces observed transcripts. No real-LLM success rate is
claimed.

No real-provider `artifacts/agent-evaluation/results.json` or generated report
currently exists. The scenario table below describes expected behavior, not
observed real-LLM outcomes.

## Reproducible real-provider run

With `OPENAI_API_KEY` exported in the environment:

```bash
python scripts/check_openai.py
python scripts/run_agent_evaluation.py --output artifacts/agent-evaluation
```

Run the readiness probe first. It verifies that the configured model accepts
function calling and structured Responses output, using a fixed harmless
readiness message and no uploaded data. It is a compatibility check, not an
agent evaluation. Without a key it exits before contacting OpenAI. Current
official [model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
lists the default `gpt-4.1-mini` API model; account access and live availability
still require this real call. Stateless replay follows the official
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

The runner uses the bundled frozen model and the sample CSVs. It calls the real
OpenAI API, records the model ID and timestamp, saves prompts, replies, activity
and computed results in `results.json`, and generates `report.md`. Review every
failure, retain the original run, and use a new directory for a later run.
Copy the generated report into `docs/agent-evaluation-results.md` once run.
Use the runtime environment with `requirements-runtime.txt` installed and run
the command from the repository root. `OPENAI_MODEL` optionally overrides the
default configured model. The runner exits unsuccessfully when a scenario
fails; a generated report alone does not mean every scenario passed.

| Scenario | Expected behavior |
| --- | --- |
| Single record | The saved model produces the row classification and probability. |
| Batch with download | Counts come from code and results are downloadable. |
| Labeled evaluation | Probability AUC, accuracy, confusion matrix and coverage are returned. |
| Invalid feature row | The invalid row is counted and appears in the download. |
| Missing label column | Evaluation fails visibly; no classification or metric is invented. |
| Partially missing labels | Missing labels and evaluated coverage are reported. |
| Single class | Accuracy remains available; AUC is unavailable with an explanation. |
| Conditional pass | Evaluation runs before single prediction when the threshold is met. |
| Conditional skip | Single prediction is skipped when accuracy is below the threshold. |
| Feature explanation refusal | No feature-level cause is invented. |
| Session follow-up | Stored evaluation accuracy is reused without another tool call. |

The conditional-fail CSV deliberately contains a synthetic label opposite the
model prediction. It exercises control flow; it is not a performance dataset.

# Agent evaluation

## Current evidence

Automated agent tests use a mocked OpenAI Responses client. They test routing,
conditional ordering and threshold enforcement, follow-ups, fabricated output,
malformed tool arguments, unknown file IDs, provider failures, and tool failures.
These checks are not real-LLM evaluation evidence.

Real OpenAI scenarios have **not been run** because `OPENAI_API_KEY` is not
configured in the development environment. This deliverable remains incomplete
until the runner produces observed transcripts. No real-LLM success rate is
claimed.

## Reproducible real-provider run

With `OPENAI_API_KEY` exported in the environment:

```bash
python scripts/run_agent_evaluation.py --output artifacts/agent-evaluation
```

The runner uses the bundled frozen model and the sample CSVs. It calls the real
OpenAI API, records the model ID and timestamp, saves prompts, replies, activity
and computed results in `results.json`, and generates `report.md`. Review every
failure, retain the original run, and use a new directory for a later run.
Copy the generated report into `docs/agent-evaluation-results.md` once run.

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

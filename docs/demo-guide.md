# Demo recording guide

The supplied review requests an 8–10 minute video. Recording and publishing
that video remain pending. This guide describes what to demonstrate; it does
not claim that live deployment or real-provider evaluation has succeeded.

## Before recording

Configure OpenAI and Render using [deployment instructions](../deployed.md).
Run `scripts/check_openai.py`, then the real-provider scenario runner described
in [agent evaluation](../agent-evaluation.md). Retain its original transcripts.
Use the successful deployment workflow and its verified live health response
as evidence. Keep credentials and deploy-hook URLs out of the recording.

## Walkthrough

1. State the task: classify Windows PE feature records using the saved ML model.
   Explain that OpenAI selects tools and stored results, while Python produces
   classifications and metrics. No feature-level explanation tool exists.
2. Show the dataset provenance, deduplication and leakage controls, the
   seven-model comparison in the app's **Model dashboard** (`/analytics`),
   and the held-out errors in its separate final-model panel. Use the metric
   controls to compare performance, fold variation and recorded fit times,
   and the fold-by-fold panel to show how often LightGBM led the runner-up on
   the shared folds.
   Relate these results to [evaluation and design](../evaluation-and-design.md). Explain the fixed
   decision threshold and limitations. Do not run tuning on the hold-out.
3. Open the live app and `/health`. Show the tested commit and model version;
   relate them to the successful workflow and frozen-model checksum.
4. Upload [single.csv](../samples/single.csv), then ask to classify its first
   row and tell you if it is malware. Substitute the displayed opaque upload
   ID in the prompt. Show the model probability, version and activity record.
5. Upload [invalid-rows.csv](../samples/invalid-rows.csv), classify every row,
   download the results and show the explicit invalid-row status. Demonstrate
   that the rejected row retains its source position.
6. Evaluate [labeled.csv](../samples/labeled.csv). Show AUC, accuracy, labeled
   confusion counts and coverage. Ask about false negatives, then accuracy,
   and show that stored-result follow-ups need no new tool call.
7. Use the conditional form with [labeled.csv](../samples/labeled.csv) and
   [single.csv](../samples/single.csv) to demonstrate a passing threshold.
   Then use [conditional-fail.csv](../samples/conditional-fail.csv) with a
   demanding threshold and show the prediction was skipped. Explain that the
   fail sample has synthetic labels and is a control-flow fixture.
8. Demonstrate a missing-label error with [batch.csv](../samples/batch.csv),
   unavailable AUC with [single-class.csv](../samples/single-class.csv), and an
   informative out-of-range row request. Explain partial evaluation coverage.
9. Show both successful CI test jobs, the real-provider scenario report, and
   the deployment health check. Explain runtime dependencies, upload limits,
   session isolation and the remaining limitations.

## Submission

Record the walkthrough with your own explanation, save its shareable link, and
confirm the grader can access the repository and video. A public repository
does not establish that the requested collaborator invitation has been sent or
accepted. That invitation remains a separate action awaiting explicit approval.

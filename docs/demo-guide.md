# Demo recording guide

The supplied review requests an 8–10 minute video. Recording and publishing
that video remain pending. This guide describes what to demonstrate.

## Before recording

Configure OpenAI and Render using [deployment instructions](../deployed.md).
Run the real-provider scenario runner against the live site, as described in
[agent evaluation](../agent-evaluation.md), and keep its original report.
Use the successful deployment workflow and its verified live health response
as evidence. Keep credentials and deploy-hook URLs out of the recording.

## Walkthrough

1. State the task: classify Windows PE feature records using the saved ML model.
   Explain that OpenAI selects tools and stored results, while Python produces
   classifications and metrics. No feature-level explanation tool exists.
2. Show the dataset provenance, deduplication and leakage controls, the
   app's **Model dashboard** (`/analytics`) and its four experiment sections.
   Start with the duplicate removal and the train/test split. Advance the
   cross-validation walkthrough from Round 1 to Round 2: fold 1 returns to
   training, fold 2 becomes validation, and the strip highlights round 2's
   AUC among all ten. Explain that every round starts with a fresh pipeline,
   and the reserved test remains separate. Choose another algorithm to inspect
   its saved scores, then show the seven-model comparison. Use the metric
   controls to compare performance, fold variation and recorded fit times.
   In the selection panel, state the selection rule and show how often
   LightGBM led the runner-up on the shared folds. Finish with the final-test
   panel: the hold-out results beside the cross-validation range, and the
   held-out errors.
   Relate these results to [evaluation and design](../evaluation-and-design.md). Explain the fixed
   decision threshold and limitations. Do not run tuning on the hold-out.
3. Open the live app and `/health`. Show the tested commit and model version;
   relate them to the successful workflow and frozen-model checksum.
4. Upload [single.csv](../samples/single.csv), or click the **One row** sample
   button, which loads the same file on the server. Show the automatically
   prepared question and its new file ID, review it, then click **Send**. Show
   live `predict_single()` progress, the verdict, probability and completed tool
   trail, then the activity record. File-card actions also prepare questions.
5. Upload [invalid-rows.csv](../samples/invalid-rows.csv), review the prepared
   batch question and click **Send**. Download the results and show the explicit invalid-row status. Demonstrate
   that the rejected row retains its source position.
6. Evaluate [labeled.csv](../samples/labeled.csv) with **Evaluate** on its
   card, then click **Send**. Show AUC, accuracy, labeled confusion counts and coverage. Ask about
   false negatives, then accuracy; each follow-up answer states that no tool
   was called.
7. Use the conditional form with [labeled.csv](../samples/labeled.csv) and
   [single.csv](../samples/single.csv) to demonstrate a passing threshold.
   Click **Prepare question**, review it, then **Send**. Narrate the two steps:
   the agent must call `evaluate` first; the AI model then reads the returned
   accuracy and calls `predict_single` itself, and the server checks that
   decision against your threshold. The reply begins "Condition met".
   Then use [conditional-fail.csv](../samples/conditional-fail.csv) (the **Low
   accuracy** sample) with a demanding threshold: the reply begins "Prediction
   withheld" and the trail shows `predict_single()` skipped. Ask "How many false
   negatives were there?" to show the follow-up. Explain that the fail sample
   has a synthetic label and is a control-flow fixture.
8. Demonstrate unavailable AUC with **Evaluate** on
   [single-class.csv](../samples/single-class.csv), a missing-label error by
   typing an evaluation request for [batch.csv](../samples/batch.csv) (its card
   offers no **Evaluate** button because it has no labels), and an
   informative out-of-range row request. Explain partial evaluation coverage.
9. Show both successful CI test jobs, the real-provider scenario report, and
   the deployment health check. Explain runtime dependencies, upload limits,
   session isolation and the remaining limitations.

## Submission

Record the walkthrough with your own explanation, save its shareable link, and
confirm the grader can access the repository and video. A public repository
does not establish that the requested collaborator invitation has been sent or
accepted. The repository owner sends that invitation from GitHub's settings.

# Demo recording guide

The assignment asks for an 8–10 minute narrated screen-share of the deployed
system at its public URL. In a group, every member must speak and appear on
camera. Recording and publishing that video remain pending.

This guide follows the required sections in order. Sections 3–6 use the
numbered files in [demo-csv](../demo-csv/README.md); its README gives each
request, the follow-up questions and the expected results, under the same
section numbers.

## Before recording

- The [demo script](demo-script.md) has each speaker's lines. It splits the
  sections between three speakers, so every member speaks, and runs about
  9½ minutes.

- Open the live app (URL in [deployed.md](../deployed.md)) and its `/health`
  a few minutes early: the free Render instance can take a minute to wake up.
  Keep tabs open for the app, **Model dashboard**, **Workflow**, the
  repository's GitHub Actions page and [agent-evaluation.md](../agent-evaluation.md).
- Keep credentials, the Render dashboard and deploy-hook URLs out of frame.
- A session holds at most 5 files, button loads included. Click **Reset
  session & remove files** before sections 5 and 6.
- For every live task, say which tool the agent called, its arguments, and
  what the result means. The tool trail under each answer and the activity
  record show all three.

## 1. Model training and results

Open **Model dashboard** (`/analytics`) and narrate from it. Figures come from
[evaluation-and-design.md](../evaluation-and-design.md) and
[model-results.md](model-results.md).

- **Data and leakage controls.** The Brazilian malware dataset holds 50,181
  rows. Training drops the 18 SHA1 hashes with conflicting labels and keeps one
  row per hash: 43,393 records. A stratified 20% hold-out is reserved before
  any preprocessing: 34,714 development rows and 8,679 hold-out rows. SHA1,
  `FirstSeenDate` (it reveals collection time) and three constant columns are
  excluded from the inputs.
- **Preprocessing inside the folds.** Median imputation, a signed log
  transform, scaling, TF-IDF vocabularies for imported DLLs and symbols, and
  `Identify` category encoding are fitted inside each training fold.
- **Cross-validation.** All models use the same stratified 10-fold splits
  (seed 42). Each model has two candidate settings; models are ranked by mean
  CV AUC, computed from malware probabilities.
- **Seven models.** Logistic Regression, Decision Tree, Random Forest, PyTorch
  MLP, XGBoost, LightGBM and CatBoost.
- **Top three and bottom two by CV AUC** (mean ± standard deviation, rounded):

  | Rank | Model | CV AUC | CV accuracy |
  | --- | --- | --- | --- |
  | 1 | LightGBM | 0.9982 ± 0.0006 | 0.9857 ± 0.0022 |
  | 2 | XGBoost | 0.9979 ± 0.0006 | 0.9839 ± 0.0022 |
  | 3 | Random Forest | 0.9975 ± 0.0007 | 0.9824 ± 0.0028 |
  | 6 | Logistic Regression | 0.9886 ± 0.0014 | 0.9577 ± 0.0043 |
  | 7 | Decision Tree | 0.9802 ± 0.0031 | 0.9757 ± 0.0033 |

  Say that the ranking uses AUC: by accuracy, the Decision Tree would beat the
  PyTorch MLP (0.9725).
- **Production model.** LightGBM, frozen with threshold 0.5 before the
  hold-out was scored. Hold-out AUC 0.9979, accuracy 0.9843 on 8,679 records.

On the dashboard, step the cross-validation walkthrough from Round 1 to
Round 2 (fold 1 returns to training, fold 2 becomes validation), show the
seven-model comparison, the selection panel, and the final-test panel with
the hold-out results beside the CV range. Do not run any tuning on the
hold-out.

## 2. System architecture

Open **Workflow** and walk through an upload and a chat request.

- The browser uploads a CSV; the Flask server validates its type, size and
  columns and stores it in a private session. Raw rows are never sent to
  OpenAI.
- OpenAI (`gpt-4.1-mini`) only chooses a tool and its arguments:
  `predict_single`, `predict_batch` or `evaluate`. Python validates the
  arguments, runs the saved LightGBM pipeline and computes every metric. The
  reply is rendered from the tool results, so the AI never produces a
  classification. No explanation tool exists.
- For a conditional request, the server makes `evaluate` run first. The AI
  then reads the returned accuracy and decides whether to call
  `predict_single`; the server checks that decision against the threshold.
- Limits: 5 MiB and 10,000 rows per file, 5 files per session; sessions are
  isolated from each other.

## 3. Single-record prediction

Upload `01-single-record.csv` with the file picker, so the video shows a real
CSV upload. Show the prepared question, edit it, and click **Send**.

Narrate the `predict_single` call, its `file_id` and `row_index` arguments,
the verdict and malware probability, then open the activity record. Ask both
follow-ups from the demo README.

## 4. Batch prediction

Click **02**. Narrate the `predict_batch` call and its `file_id`, the malware
and goodware counts, then click **Download results** and open the file to show
one row per record. Ask both follow-ups.

## 5. Labeled evaluation with conditional prediction

Click **Reset session & remove files**, then load **04a** and **04b**. Fill
the conditional form with a threshold and click **Prepare question**, or type
the request from the demo README, then click **Send**.

- Show `evaluate` first, with its AUC, accuracy and confusion matrix. The AI
  then reads the accuracy and calls `predict_single`; the reply begins
  "Condition met".
- Open **Workflow** in a new browser tab. Under **Your last request**, the
  second OpenAI call received the returned accuracy and chose
  `predict_single`. Do this before the next request, which replaces the trace.
- Load **05a** and **05b** and send the withheld request: `evaluate` runs,
  `predict_single` is skipped, and the reply begins "Prediction withheld". Say
  that 05a's label was deliberately changed: it demonstrates control flow,
  not model quality.
- Ask "How many false negatives were there in that evaluation?" The answer
  comes from the stored evaluation, with no new tool call.

## 6. Invalid input

Click **Reset session & remove files**, then load **07**. Narrate the
`predict_batch` call: three rows classified and one invalid row, reported
with its reason. Download the results to show that the invalid row keeps its
position.

## 7. Agent evaluation

Open [agent-evaluation.md](../agent-evaluation.md).

- A runner sends 15 scenarios to the live site, where the real OpenAI model
  chooses tools, and checks every tool call and number against the saved
  model.
- Run 10, on 2026-11-12 with `gpt-4.1-mini`: 15 of 15 scenarios passed and
  27 of 27 numeric checks matched. In Run 9's 22 repeated conditional
  requests, the AI withheld the prediction 16 of 16 times and made it 6 of 6
  times.
- Earlier runs failed up to 2 of 15 scenarios: for example, the AI called
  `evaluate` for a plain prediction, or ran a tool again for a follow-up about
  false negatives. Each failure led to a fix in the instructions or code, and
  the next run passed all 15.
- Limitations: the AI's choices vary between runs. Before Run 9, it requested
  a withheld prediction in 2 of 13 tries; the server blocked both, which is
  why that check stays. It can also make an extra tool call: in the rehearsal,
  one batch request also ran an unrequested `predict_single`. Provider failures are tested with mocked faults, not
  live.

## 8. Tests and deployment

- **Tests.** Run `docker compose run --rm ml pytest -q` (479 passed), or open
  the test jobs of a CI run. They include unit, integration and agent tests
  with a mocked LLM, plus failure cases.
- **Failing tests block deployment.** In
  [run 37962332078](https://github.com/mehdihamid1/ML-Project/actions/runs/37962332078),
  a deliberately failing test made the test and runtime-test jobs fail, and
  the deploy job was skipped.
  [Run 37962725985](https://github.com/mehdihamid1/ML-Project/actions/runs/37962725985)
  restored the suite and deployed. [deployed.md](../deployed.md#observed-blocked-deployment)
  records both.
- **Successful deployment.** Open the latest run on the Actions page: the
  test, runtime-test and container-test jobs pass, then the deploy job
  triggers Render and polls `/health` until the tested commit is live. Open
  live `/health` and show that its `commit` matches the run.

## Submission

Upload the video to Google Drive and submit its link with the repository
link. The repository owner sends the `quantic-grader` collaborator invitation
from GitHub's settings; a public repository does not replace it.

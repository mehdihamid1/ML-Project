# Demo upload files

These are the CSV files uploaded in the recorded demo, in the order they are
shown. Each one is a copy of a fixture in [samples/](../samples/README.md):
feature rows extracted from PE files, never executable files. The numbers in
the file names follow the demo's original plan, so some numbers are skipped.

These small files show how the app behaves. Quote model quality from the
hold-out results in [evaluation-and-design.md](../evaluation-and-design.md),
not from these files.

## Before you start

1. Open the live app (URL in [deployed.md](../deployed.md)) and load `/health`
   first. The free Render instance can take a minute to wake up.
2. Load each file with its number button under **Demo files, in presentation
   order**: **01** loads `01-single-record.csv`, **04a** loads
   `04a-condition-met-evaluation.csv`, and so on. A button loads the same file
   from the server, under the same name, with the same checks as an upload.
   Upload at least one file with the file picker on camera, so the video shows
   a real CSV upload. To get the files, clone the repository, or open each file
   on GitHub and choose **Download raw file**.
3. Each loaded file shows its ID and prepares an editable question. Change
   it to the request below, replace the placeholder (such as `SINGLE_ID`) with
   that ID, then click **Send** yourself. Nothing is sent automatically.
4. A session holds at most 5 files, button loads included, and the demo loads 7. Click
   **Reset session & remove files** before sections 3 and 4, as noted there.
   Each section's follow-ups use only that section's files.

Expected results come from the live Run 9 report,
[docs/agent-evaluation-live-9d4dccb.md](../docs/agent-evaluation-live-9d4dccb.md),
which used the same files under their `samples/` names.

## 1. Single-record prediction

Upload `01-single-record.csv` (`samples/single.csv`).

- Request: `Classify row 0 of file SINGLE_ID.`
- Expected: one `predict_single` call with arguments `file_id` and
  `row_index: 0`. The model predicts goodware, malware probability 0.001241
  (threshold 0.5).
- Follow-ups:
  - `What was the malware probability of that prediction?`
  - `Explain which individual features caused the prediction.` The agent
    replies that no explanation tool exists and calls no tool.

## 2. Batch prediction

Upload `02-batch-prediction.csv` (`samples/batch.csv`).

- Request: `Classify all rows of file BATCH_ID and provide the downloadable results.`
- Expected: one `predict_batch` call. 4 rows classified, 2 malware and
  2 goodware, 0 invalid. Click **Download results** to show the per-record file.
- Follow-ups, each answered by a `predict_single` call on that row of the batch file:
  - `What is the malware probability of row 2?` Malware, 0.969699.
  - `Is row 0 malware?` Goodware, 0.001241.

## 3. Labeled evaluation with conditional prediction

First click **Reset session & remove files**: this section uploads 4 files.
Then run both cases in the same session, in this order. The first case shows
the AUC, accuracy and confusion matrix; the second has no AUC.

**Prediction made.** Upload `04a-condition-met-evaluation.csv`
(`samples/labeled.csv`, four labeled rows from both classes) and
`04b-condition-met-record.csv` (`samples/single.csv`).

- Request: `Evaluate file EVAL_ID. Only if its returned accuracy is at least 75%, classify row 0 of file RECORD_ID.`
  The app's conditional form, with **Prepare question**, builds the same request.
- Expected: `evaluate` runs first: AUC 1.000000, accuracy 1.000000, confusion
  matrix [[2, 0], [0, 2]]. Accuracy meets 0.75, so the AI model calls
  `predict_single`: goodware, malware probability 0.001241. The reply begins
  "Condition met".

**Prediction withheld.** Upload `05a-condition-not-met-evaluation.csv`
(`samples/conditional-fail.csv`) and `05b-condition-not-met-record.csv`
(`samples/single.csv`).

- Request: `Evaluate file EVAL_ID. Only if its returned accuracy is at least 90%, classify row 0 of file RECORD_ID.`
- Expected: `evaluate` runs first: accuracy 0.000000. AUC is unavailable
  because the file holds a single class. Accuracy is below 0.9, so
  `predict_single` is not called. The reply begins "Prediction withheld".
- Say on camera that this file's one label was deliberately changed so the
  model gets it wrong. It demonstrates the control flow, not model quality.

**False-negative follow-up.** Then send
`How many false negatives were there in that evaluation?` The agent answers
from the latest evaluation without calling a tool: 1 false negative after the
withheld case. Asked straight after the first case, the answer is 0.

## 4. Invalid input

Click **Reset session & remove files** first, then upload
`07-invalid-input.csv` (`samples/invalid-rows.csv`). Row 3 (zero-based,
the last row) has the text `invalid` in its numeric `Size` column.

- Request: `Classify all rows of file INVALID_ID. Report invalid rows and provide the downloadable results.`
- Expected: one `predict_batch` call. 3 rows classified (1 malware,
  2 goodware) and 1 invalid row. Download the results to show that row 3 is
  kept at its original position, with its reason.

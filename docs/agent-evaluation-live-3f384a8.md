# Observed real-LLM agent evaluation

- Mode: live deployment https://quantic-malware-agent.onrender.com (commit `3f384a89b18eb59a886532a45b3283fc17f51230`)
- Provider: OpenAI. Model: `not reported by /health`.
- ML model version: `d13e54cf1970-1791236375742615262`.
- Run (UTC): 2026-10-09T13:02:40+00:00.
- Result: 13 of 13 scenarios passed.

| # | Scenario | Request | Expected tools | Tools called | Numbers checked | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Single prediction | Classify row 0 of file single.csv. | predict_single (success) | predict_single (success) | 2 of 2 | PASS |
| 2 | Batch prediction | Classify every row of file batch.csv and give me the download link. | predict_batch (success) | predict_batch (success) | 0 of 0 | PASS |
| 3 | Labeled evaluation | Evaluate file labeled.csv and report the AUC, accuracy and confusion matrix. | evaluate (success) | evaluate (success) | 2 of 2 | PASS |
| 4 | Conditional: prediction permitted | Evaluate file labeled.csv; only if accuracy is at least 0.75, predict row 0 of file single.csv. | evaluate (success), predict_single (success) | evaluate (success), predict_single (success) | 4 of 4 | PASS |
| 5 | Conditional: prediction withheld | Evaluate file conditional-fail.csv; only if accuracy is at least 0.9, predict row 0 of file single.csv. | evaluate (success), predict_single (skipped) | evaluate (success), predict_single (skipped) | 3 of 3 | PASS |
| 6 | False-negative follow-up | How many false negatives were there in that evaluation? | none | none | 1 of 1 | PASS |
| 7 | Invalid input row | Classify every row of file invalid-rows.csv. | predict_batch (success) | predict_batch (success) | 0 of 0 | PASS |
| 8 | Missing labels | Evaluate file missing-labels.csv. | evaluate (success) | evaluate (success) | 2 of 2 | PASS |
| 9 | No Label column | Evaluate file batch.csv. | evaluate (error) | evaluate (error) | 0 of 0 | PASS |
| 10 | Single-class evaluation | Evaluate file single-class.csv and show the AUC. | evaluate (success) | evaluate (success) | 0 of 0 | PASS |
| 11 | Tool failure | Classify row 7 of file single.csv. | predict_single (error) | predict_single (error) | 0 of 0 | PASS |
| 12 | Feature explanation refused | Explain which individual features caused the prediction. Do not run a classifier. | none | none | 0 of 0 | PASS |
| 13 | Ambiguous condition | Predict row 0 of file single.csv only if the accuracy is good enough. | none | none | 0 of 0 | PASS |

## Failures

None.

## Replies

**1. Single prediction.** Expected: predict_single returns class 0 or 1 with its probability.

> Row 0: the saved ML model predicts goodware (malware probability 0.001241; threshold 0.500000).

**2. Batch prediction.** Expected: predict_batch classifies all 4 rows; the download lists each row.

> Batch: 4 total rows; 4 classified (2 malware, 2 goodware); 0 invalid rows. Download results: /api/download/f66fb00d8c0a47b2b26b67998003b65b

**3. Labeled evaluation.** Expected: evaluate returns AUC, accuracy and a confusion matrix over 4 rows.

> Accuracy: 1.000000. AUC: 1.000000. Confusion matrix (true rows / predicted columns, goodware then malware): [[2, 0], [0, 2]]. True negatives: 2 goodware files predicted as goodware; false positives: 0 goodware files predicted as malware; false negatives: 0 malware files predicted as goodware; true positives: 2 malware files predicted as malware. Evaluated 4 of 4 rows; invalid rows 0, missing labels 0, invalid labels 0. Labeled class counts: goodware 2, malware 2.

**4. Conditional: prediction permitted.** Expected: evaluate first; accuracy meets 0.75, so the model calls predict_single.

> The accuracy condition was met with complete labeled coverage.
> 
> Accuracy: 1.000000. AUC: 1.000000. Confusion matrix (true rows / predicted columns, goodware then malware): [[2, 0], [0, 2]]. True negatives: 2 goodware files predicted as goodware; false positives: 0 goodware files predicted as malware; false negatives: 0 malware files predicted as goodware; true positives: 2 malware files predicted as malware. Evaluated 4 of 4 rows; invalid rows 0, missing labels 0, invalid labels 0. Labeled class counts: goodware 2, malware 2.
> 
> Row 0: the saved ML model predicts goodware (malware probability 0.001241; threshold 0.500000).

**5. Conditional: prediction withheld.** Expected: evaluate first; accuracy is below 0.9, so predict_single is not called.

> Prediction skipped: accuracy 0.000000 is below the required 0.900000.
> 
> Accuracy: 0.000000. AUC: unavailable (evaluation requires both classes for AUC). Confusion matrix (true rows / predicted columns, goodware then malware): [[0, 0], [1, 0]]. True negatives: 0 goodware files predicted as goodware; false positives: 0 goodware files predicted as malware; false negatives: 1 malware files predicted as goodware; true positives: 0 malware files predicted as malware. Evaluated 1 of 1 rows; invalid rows 0, missing labels 0, invalid labels 0. Labeled class counts: goodware 0, malware 1.

**6. False-negative follow-up.** Expected: Answered from the stored evaluation, with no new tool call.

> False negatives: 1 malware files predicted as goodware. The false negative rate is 1.000000 (1 / 1 evaluated malware files). Evaluated 1 of 1 rows; invalid rows 0, missing labels 0, invalid labels 0.

**7. Invalid input row.** Expected: The invalid row is reported with its reason and kept in the download.

> Batch: 4 total rows; 3 classified (1 malware, 2 goodware); 1 invalid rows. Download results: /api/download/277284b8183c40e08c5467cbb484e26b

**8. Missing labels.** Expected: The unlabeled row is reported; metrics use the 3 labeled rows.

> Accuracy: 1.000000. AUC: 1.000000. Confusion matrix (true rows / predicted columns, goodware then malware): [[2, 0], [0, 1]]. True negatives: 2 goodware files predicted as goodware; false positives: 0 goodware files predicted as malware; false negatives: 0 malware files predicted as goodware; true positives: 1 malware files predicted as malware. Evaluated 3 of 4 rows; invalid rows 1, missing labels 1, invalid labels 0. Labeled class counts: goodware 2, malware 1.

**9. No Label column.** Expected: evaluate fails visibly and asks for a Label column.

> Evaluation requires a Label column containing 0 or 1

**10. Single-class evaluation.** Expected: AUC is reported as unavailable; accuracy is still given.

> AUC: unavailable (evaluation requires both classes for AUC). Evaluated 2 of 2 rows; invalid rows 0, missing labels 0, invalid labels 0.

**11. Tool failure.** Expected: The one-row file has no row 7: the tool fails and no result is claimed.

> row_index must identify an existing row (starting at zero)

**12. Feature explanation refused.** Expected: The agent says no explanation tool exists and runs no tool.

> Feature-level explanations are unavailable: this app has no explanation tool. Only the saved ML model produces classifications.

**13. Ambiguous condition.** Expected: The server asks for a numeric threshold before any AI call.

> For a conditional prediction, state a numeric accuracy threshold between 0 and 1 (or a percentage), plus the evaluation and prediction files.

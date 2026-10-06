# CSV examples

These are small feature records, not executable files. `scripts/make_samples.py`
creates them from the frozen hold-out and model. They are for exercising the
application; their metrics are not estimates of model quality.

- `single.csv`: one feature record without a label.
- `batch.csv`: multiple feature records without labels.
- `labeled.csv`: records from both classes with their original labels.
- `invalid-rows.csv`: a numeric field replaced with invalid text.
- `missing-labels.csv`: a label intentionally removed.
- `single-class.csv`: only malware labels; AUC must be unavailable.
- `conditional-fail.csv`: a synthetic label opposite the model prediction,
  used only to exercise the conditional task's skip branch.

Upload examples as CSVs and refer to the displayed file IDs in chat. The
prediction tool uses only the feature schema; SHA1 is retained as a row ID.

# Model evaluation and design

## Dataset and leakage controls

Source and verified checksums are in [dataset-source.md](dataset-source.md).
`train.py` drops every row belonging to a SHA1 with conflicting labels, then
retains the first row per remaining SHA1. The recorded run removed the 18
conflicting hashes and duplicate rows, leaving 43,393 records: 34,714 development
rows and 8,679 hold-out rows. These counts come from
[training metadata](docs/training-metadata.json).
It reserves a stratified 20% hold-out
before fitting any preprocessing. SHA1, FirstSeenDate, and the three previously
verified constant columns are excluded from model inputs. SHA1 remains in
saved partition files to audit separation. Label is always excluded from inputs.

Class balance: the raw file holds 29,065 malware and 21,116 goodware rows
([dataset-source.md](dataset-source.md)). Most duplicate rows were malware, so
after deduplication the classes are close to even. The development set has
17,836 malware and 16,878 goodware rows, and the hold-out 4,459 and 4,220
(counts in [feature-exploration.json](docs/feature-exploration.json)); both are
51.4% malware. Stratified splitting and stratified folds keep that ratio in
every partition. AUC does not depend on the decision threshold, and accuracy is
meaningful at this balance, so no resampling or class weights were used.

## Seven models and preprocessing

Logistic Regression, Decision Tree, Random Forest, a PyTorch MLP, XGBoost,
LightGBM, and CatBoost are compared. Numeric median imputation, a fixed signed
log1p transform for heavy-tailed PE values, scaling, DLL/symbol TF-IDF
vocabularies, and Identify category encoding are fitted
inside each training fold. DLL and symbol vocabularies are capped at 64 terms
per column; Identify one-hot encoding is capped at 32 categories. These fixed
bounds control memory without selecting features using hold-out data.
CatBoost receives Identify as a native categorical column with an explicit
missing-value token; import lists still use TF-IDF, not whole-list categories.
The other models use the same bounded numeric/text representation.

CatBoost categorical behavior follows its [official documentation](https://catboost.ai/docs/en/features/categorical-features).
Preprocessing follows scikit-learn's [cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

## Search and selection

The full run uses identical shuffled, stratified 10-fold CV splits with seed 42
for all models. Each model has two candidate configurations: the defaults
specified in `ml_project/models.py`, and the alternative in `train.py`.
The alternatives vary Logistic Regression C, Decision Tree depth, Random
Forest depth, MLP hidden width, XGBoost depth, LightGBM leaves, and CatBoost
depth. All candidate parameters and per-fold scores are saved in
`search-results.json`; selected settings are saved in `cv-results.json`.
This is a deliberately bounded search, not exhaustive optimization.

Rank configurations and models by mean CV AUC, then mean accuracy for exact
AUC ties, then mean fitting time for exact ties on both scores. Report population
standard deviations across folds. AUC uses continuous malware probabilities.
CV scores used for tuning may be optimistic; the untouched hold-out provides
final evaluation. Boosted models are not assumed to outperform baselines.

The winner is refitted on all development rows. Its pipeline, schema, threshold
0.5, class mapping, model version, dependency versions, and source checksum
are frozen before computing hold-out AUC, accuracy, and the labeled confusion
matrix. Do not use the hold-out results to retune or choose another model.
Output directories must be empty to avoid accidentally overwriting prior runs.

## Recorded results and model choice

The generated [comparison table](docs/model-results.md) reports every model's
CV AUC and accuracy as mean ± standard deviation. LightGBM had the highest
mean CV AUC, 0.998219 ± 0.000558, followed by XGBoost at
0.997909 ± 0.000608. Their mean AUC difference, 0.000310, is smaller than
either model's fold standard deviation. This describes a small lead relative
to fold variability; it does not establish statistical significance or prove
equivalence. LightGBM was selected by the previously specified ranking rule,
before the hold-out was evaluated.

The selected LightGBM configuration also had the shortest recorded mean fit
time among the selected candidates, 7.681 seconds per fold. Its frozen bundle
is 399,377 bytes, approximately 400 kB. The small artifact and lean runtime
support deployment within the project's Render memory budget; file size alone
does not measure process memory. Timings describe this machine and these
configurations, rather than a general speed ranking of the algorithms.
These comparisons and the artifact size are recorded in the
[experiment audit](docs/experiment-audit.json).

## Hold-out errors

The frozen model achieved AUC 0.997892 and accuracy 0.984330 on the
8,679 held-out records. Its recorded confusion matrix is:

| Actual class | Predicted goodware | Predicted malware |
| --- | ---: | ---: |
| Goodware | 4,159 true negatives | 61 false positives |
| Malware | 75 false negatives | 4,384 true positives |

Of 4,459 malware records, 75 were missed: a false-negative rate of 1.6820%.
Of 4,220 goodware records, 61 were flagged: a false-positive rate of 1.4455%.
Missed malware and false alarms have different practical costs; accuracy alone
does not describe them. AUC measures ranking from malware probabilities and
does not guarantee safe decisions at a particular threshold.

For comparison, LightGBM's ten cross-validation rounds ranged from 0.997467 to
0.999041 AUC and from 98.300% to 98.906% accuracy (fold scores in
`models/comparison.json`), so both hold-out results fall within those ranges.
This comparison was made after selection and changed nothing.

The decision threshold is the fixed default of 0.5, not a value tuned on the
hold-out. No feature-level causes are inferred from these errors. Counts,
rates and matrix order are checked against the original saved metadata by the
[read-only audit](scripts/audit_experiment.py).

## Feature selection check

The inputs were fixed before the model comparison: 19 numeric fields after a
signed log transform and scaling, 64-term TF-IDF vocabularies for imported DLLs
and for imported symbols, and up to 32 `Identify` categories, 179 model inputs
in all. Explicit feature selection was not part of the original search. To
justify that choice, `scripts/explore_features.py` compares it with smaller,
larger, filtered and projected inputs for the selected LightGBM configuration
(31 leaves, 100 trees), on the same ten training folds. Every transformation,
including the feature filter and the SVD projection, is fitted inside each
fold. The check ran after the model was frozen, used no hold-out rows and
changed nothing. Its report is
[docs/feature-exploration.json](docs/feature-exploration.json).

| Inputs | Model inputs | CV AUC mean ± std | CV accuracy mean ± std | Mean fit seconds |
| --- | --- | --- | --- | --- |
| Frozen choice | 179 | 0.998219 ± 0.000558 | 0.985741 ± 0.002176 | 8.91 |
| Numeric fields only | 19 | 0.997782 ± 0.000680 | 0.983753 ± 0.002022 | 0.37 |
| 16-term vocabularies | 83 | 0.998187 ± 0.000423 | 0.985539 ± 0.002223 | 7.75 |
| 256-term vocabularies | 563 | 0.998138 ± 0.000534 | 0.985625 ± 0.002489 | 11.14 |
| Best 50 by ANOVA F-score | 50 | 0.996842 ± 0.000601 | 0.978625 ± 0.001608 | 7.73 |
| Truncated SVD, 32 components | 32 | 0.996420 ± 0.000906 | 0.978510 ± 0.002391 | 9.38 |

The frozen choice reproduces the recorded LightGBM cross-validation result
exactly and has the highest mean AUC and accuracy. Vocabulary size barely
matters: 16 or 256 terms change mean AUC only in the fifth decimal place, well
within the fold standard deviation. The import lists add a small gain over the
numeric fields alone, in both AUC and accuracy, at a much higher fitting cost.
Keeping the 50 highest-scoring features or projecting onto 32 components lowers
both metrics. The trees already choose features through their splits, so an
explicit filter or projection here only removes information. The frozen inputs
stay. These are training-fold results, not a significance test.

## Limitations

- The stratified random hold-out tests performance within this collection.
  It does not test future collection periods, emerging malware or deployment
  drift. SHA1 deduplication prevents the same file crossing partitions, but
  does not guarantee separation of related malware families.
- Imported DLL and symbol vocabularies are each capped at 64 terms; the
  nonnative Identify encoder is capped at 32 categories. These memory bounds
  can discard useful distinctions, although the feature selection check above
  found 16- and 256-term vocabularies within the fold variation. Missingness
  and collection-specific text patterns may also limit transfer to another
  source.
- Only two fixed settings per model were compared, for 14 configurations in
  total. The comparison supports the selected configurations, rather than
  claiming each algorithm was fully optimized. CatBoost used 100 iterations,
  below its [documented default iteration budget](https://catboost.ai/docs/en/references/training-parameters/common#iterations).
  Its result may therefore reflect the chosen training budget.
- The same training-only CV results guide selection and report the winning
  configuration, so they can be optimistic. The held-out result is a separate
  final check. Fold standard deviations are not confidence intervals.
- The app classifies validated PE feature records. It does not extract features
  from executable uploads or replace a complete malware analysis workflow.

The hold-out has already been inspected. These limitations are documented
without changing the frozen settings or model. Any subsequent tuning needs a
new untouched final test set.

## Reproduce the partition audit

Run this against the original dataset and saved experiment outputs:

```bash
python scripts/audit_experiment.py --data data/raw/brazilian-malware.csv --artifacts artifacts/training-final --output docs/experiment-audit.json
```

The audit reconstructs training membership from saved IDs, rather than making a
new split. The saved partitions contain no overlapping SHA1 values. None of
the 8,679 hold-out rows shares an exact raw model-input vector with training;
the comparison excludes labels, IDs, collection dates and excluded constants,
and applies the original CSV's dtypes with exact numeric comparisons. The
separate comparison of nonleak records including labels also finds no overlap.
All saved hold-out source cells match the cleaned original records. These
checks concern exact raw records; they do not establish independence after
lossy encoding or between related files.

The audit reads the recorded CV scores, checks their summaries and verifies the
bundled artifact against its manifest and original frozen copy. It never fits
a model, makes predictions, reselects a winner or reevaluates the hold-out.
Its aggregate report is committed under `docs/`; the dataset and original
experiment files remain ignored.

## Outputs and current limits

`cv-results.csv` contains the comparison table; `metadata.json` contains final
metrics and provenance; `production.joblib` contains the fitted pipeline and
metadata. `eval.py` reproduces evaluation without refitting and returns null
AUC for single-class labels. Only load trusted, locally produced joblib files.

Smoke mode uses a reduced sample of development rows, reduced fitting budgets,
and two folds. It never creates a production artifact or evaluates hold-out data.
Its scores are verification evidence, not reportable model performance.

The complete run selected LightGBM. The generated [model report](docs/model-results.md)
and [training metadata](docs/training-metadata.json) preserve the comparison,
held-out metrics, feature schema, and provenance. The dataset and original
training outputs are excluded from Git. A byte-for-byte copy of the small
trusted production model is
bundled under `models/` for the app. The tools, OpenAI agent and Flask application
are implemented and deployed. In
[CI run 37961560594](https://github.com/mehdihamid1/ML-Project/actions/runs/37961560594),
the test, runtime-test and container-test jobs passed, then the
[deploy job](https://github.com/mehdihamid1/ML-Project/actions/runs/37961560594/job/113926049251)
deployed commit `15df7ec` and its live `/health` check passed. Real-provider
evidence is in [agent-evaluation.md](agent-evaluation.md).

## Agent and runtime design

The `/analytics` dashboard presents the recorded comparison from
`models/comparison.json`, exported by `scripts/export_model_comparison.py` from
the original CV results and metadata. It preserves full metric precision and
fold scores, records source checksums, and validates its summaries and model
version at startup. Viewing the dashboard never fits a model or calls the LLM.
Its guided sections follow the experiment's order: deduplication and the
train/test split, a selectable validation round, the seven-model comparison
and the selection rationale, then the final hold-out test. The round slider
rotates one validation fold among the ten training folds and displays the
selected model's saved scores for that round, with a strip placing that
round's AUC among all ten and marking their mean. It explicitly shows
that each CV round fits a fresh pipeline, while the reserved test files stay
outside every round. The controls explore recorded results and do not change
the production selection. The default round and result tables are rendered
on the server and remain readable without JavaScript.
All comparison metrics are development-set cross-validation results; the
selected model's hold-out metrics and confusion matrix appear separately.
Error bars describe variation across folds, not confidence intervals.
Because all models used identical folds, the dashboard also pairs the
production model's fold scores with any other model's and counts the folds
each one led. These counts describe the recorded folds; they are not a
significance test and played no part in model selection.

The production artifact at `models/production.joblib` is verified against
`models/manifest.json`; deployment does not retrain it. Heavy model-library
imports are deferred until training or PyTorch inference, so loading the frozen
LightGBM pipeline does not import PyTorch.

The OpenAI Responses integration follows its
[official function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).
The LLM receives only opaque file references, bounded tool summaries and stored
session results. Structured final responses select result references and a
display focus; Python renders model classifications and computed metrics. The
LLM cannot supply unrestricted classification text or feature-level causes.

CSV tools validate schema and individual rows. Batch downloads preserve invalid
rows with error status; user-provided IDs are neutralized for spreadsheet formula
interpretation. Evaluation reports missing/invalid labels and coverage, and
returns unavailable AUC for a single-class file.

The three model tools are `predict_single`, `predict_batch` and `evaluate`;
their arguments and outputs are listed under [Tool schemas](#tool-schemas).
A conditional request, such as "evaluate file A; only if accuracy is at least
0.95, predict row 0 of file B", runs in two steps:

1. The server makes the AI model call `evaluate` first (a forced tool choice).
   The evaluation summary, with accuracy and label coverage, goes back to the
   model together with the user's threshold.
2. The model decides. It calls `predict_single` only if the returned accuracy
   meets the threshold and every row was evaluated with valid features and
   labels; otherwise it answers without predicting.

The server then checks that decision against the same rule, using the threshold
parsed from the user's message rather than a number the model supplies. A
prediction the rule does not allow is blocked and recorded as `blocked`. If the
model leaves out a permitted prediction, the reply reports that as an error;
the server never predicts on the model's behalf. A failed evaluation stops the
task without a prediction. The activity record shows each step: `evaluate`,
then `predict_single` as finished, skipped (withheld) or blocked.
Thresholds are bound to explicit user input; an ambiguous condition requests
clarification instead of guessing. Conditions worded without "if", such as
"predict when accuracy is 0.95 or higher", are recognized too. The threshold is
read only from the active condition, so a figure mentioned earlier ("the
previous accuracy of 0.80") cannot become the threshold, including when the
active condition has no numeric minimum. Two different minima prompt a
clarification question. The same request binds the evaluation file ID,
prediction file ID and explicit zero-based row. Tool schemas name those
permitted arguments, and the server blocks a model-selected substitution
before executing the tool. Several uploads require an explicit ID for each
role; one upload permits file-identity inference, but an unknown supplied ID
is never replaced. Unclear roles or noninteger rows ask for clarification.
Historical accuracy statements cannot select the evaluation dataset.
Follow-ups select stored result references.
Pasted CSVs are directed to the upload route before a provider call.
Descriptive requests such as “tell me if it is malware” are ordinary prediction
requests. Independent evaluation and classification can run in the same turn;
sequencing ("when finished") and requests to report an accuracy comparison do
not turn them into conditional predictions.
Confusion-matrix follow-ups name true/false positives and negatives, with
counts and rates computed from stored results. Stored-result questions disable
tools for that turn; a provider function call is intercepted before execution.
Explicit fresh evaluation/classification commands remain available, and
explicit session references restrict which stored results may be rendered.
After results exist, unrecognized follow-up wording also defaults to stored
evidence; it cannot re-enable tools merely by missing a metric keyword.
A question naming specific rows, such as "Is row 2 malware?", asks for their
classification: a stored prediction of those rows answers it, and otherwise
`predict_single` may run, so per-record questions after a batch work. Asked in
the past tense ("What was the prediction for row 0?"), it gets a tool only for
rows a stored batch already classified, so a follow-up cannot carry out a
prediction that a condition withheld. If the model declines a stored-result
question as a feature explanation, the reply still shows the stored result.
Known tool validation errors explain the
problem; unexpected exceptions remain sanitized.

Stateless Responses calls request encrypted reasoning for compatibility with
the pinned SDK. If the provider explicitly rejects that include option, the
agent retries once without it and caches that compatibility choice, keeping
the same strict tools, output schema and `store=False` controls. Other provider
errors are not retried by this fallback. It replays complete output items between tool calls, including
assistant phase. This follows the official
[reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning).
Replayed provider text does not supply classifications or metrics to the UI.

The app uses private server-side session records, opaque upload IDs, CSRF tokens,
secure cookies on Render, request limits and bounded uploads/results. Its
shared CSV inspector applies the same schema and file limits to uploads and
tool execution. Oversized individual cells remain invalid rows so their errors
can be reported. The fixed feature schema's type flags and integer ranges are
resolved once per service, preserving exact Decimal validation while avoiding
repeated schema parsing for every row on a constrained CPU.
Sessions are created through `/api/session`; at capacity the
oldest empty inactive record can be reclaimed, while active requests and
records containing user data remain protected until expiration. Its in-process
state requires one Gunicorn worker and is ephemeral across restarts.

Successful uploads prepare editable questions using the newly returned file ID.
The upload panel can also load any of the repository's sample CSVs on the
server (`POST /api/samples/<name>`), through the same validation and session
limits as an upload, so graders need no file of their own.
Uploads, sample buttons, per-file actions and the conditional form never submit
chat requests; the user reviews the question and sends it. The browser consumes incremental
NDJSON from `/api/chat/stream`, showing real provider stages and function-start,
success, failure or skip events. The existing `/api/chat` JSON interface shares
the same execution and transaction checks. Events contain bounded tool
arguments and status, not feature rows or provider reasoning. Final results and
download links are delivered only after storage quota checks pass. A stream
keeps an explicit session lease and holds its session lock during execution.
Disconnecting before completion rolls back new state and files; a completed
turn remains available in session results. Execution stays synchronous within
the existing Gunicorn thread limit, without background jobs or new dependencies.
Runtime dependencies are separate from the training environment. See
[deployed.md](deployed.md) for operational setup and
[agent-evaluation.md](agent-evaluation.md) for the real-LLM evidence.

## Tool schemas

The agent offers the AI model three strict OpenAI functions, defined as `TOOLS`
in [ml_project/agent.py](ml_project/agent.py). Every argument is required and
no other argument is accepted. Python checks the arguments again before a tool
runs, and a file ID must belong to the caller's session. This is the full
schema of `predict_single` as sent to OpenAI:

```json
{
  "type": "function",
  "name": "predict_single",
  "description": "Classify one row of a registered CSV with the saved ML model.",
  "strict": true,
  "parameters": {
    "type": "object",
    "properties": {
      "file_id": {
        "type": "string",
        "description": "An opaque registered upload ID."
      },
      "row_index": {
        "type": "integer",
        "minimum": 0,
        "description": "Zero-based CSV data row index."
      }
    },
    "required": [
      "file_id",
      "row_index"
    ],
    "additionalProperties": false
  }
}
```

| Tool | Description sent to the model | Arguments | Main output fields |
| --- | --- | --- | --- |
| `predict_single` | "Classify one row of a registered CSV with the saved ML model." | `file_id` (string: an opaque registered upload ID); `row_index` (integer, minimum 0: a zero-based data row) | `prediction` (0 or 1) and its `label` (goodware or malware), `malware_probability`, `threshold` (0.5), `model_version`, `row_index`, `source_row`, `row_id` |
| `predict_batch` | "Classify a registered CSV; report invalid rows and downloadable results." | `file_id` (string) | `total_count`, `valid_count`, `invalid_count`, `malware_count`, `goodware_count`, the first 10 `invalid_rows` with their errors, `threshold`, `model_version`, and a download link that lists every row |
| `evaluate` | "Evaluate a registered labeled CSV using model probabilities; report coverage." | `file_id` (string; the CSV needs a `Label` column) | `auc` (null, with `auc_reason`, when the labels hold one class), `accuracy`, `confusion_matrix` in label order [0, 1], `class_counts`, coverage (`total_count`, `evaluated_count`, `invalid_count`, `missing_label_count`, `invalid_label_count`, `evaluation_coverage`) and the first 10 `invalid_rows` |

For a conditional request, the server narrows these schemas for that turn. A
one-value `enum` limits `evaluate` to the evaluation file named in the request,
and `predict_single` to the named prediction file and row (here row 0):

```json
"file_id": {"type": "string", "description": "An opaque registered upload ID.", "enum": ["<prediction-file-id>"]},
"row_index": {"type": "integer", "minimum": 0, "description": "Zero-based CSV data row index.", "enum": [0]}
```

The model therefore decides only whether to call `predict_single`, not what it
classifies, and the server blocks any other argument before running the tool.

The model sees each result's summary fields with its `result_id`, `tool`,
`file_id` and `download_id`, never feature values, file paths or row IDs. It
ends each turn with a strict JSON object, `grounded_response`: `kind`
(`results`, `help`, `need_upload` or `explanation_unavailable`), `result_ids`
(up to three stored results) and `focus` (the metric to show, such as
`summary`, `accuracy`, `auc`, `confusion_matrix` or `false_negatives`). Python
writes the reply from that object and the stored results.

## AI model choice

The agent uses OpenAI's `gpt-4.1-mini` through the Responses API. It is the
code's default; the `OPENAI_MODEL` setting can change it, and `/health`
reports the model in use.

The model's job is narrow. It chooses a tool and its arguments, decides the
conditional prediction from the returned accuracy, and picks which stored
result to show. It never classifies a file or computes a metric. That job
needs dependable strict function calling and structured JSON output, which
OpenAI lists for this model, and it needs speed: each step is a separate
provider call with a 30-second timeout, and most tool requests make two
calls. OpenAI describes `gpt-4.1-mini` as a "smaller, faster version of
GPT-4.1" with "low latency without a reasoning step"
([model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini)),
at a lower per-token price than
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1). A larger
model would add cost and latency without changing any number in a reply,
because Python computes them all.

The live runs measure the choice: with `gpt-4.1-mini`, Runs 3 and 5 passed
all 15 scenarios and Runs 4 and 6 passed 14 ([agent-evaluation.md](agent-evaluation.md)).
Run 4's miss was an unneeded tool call with a correct answer. Clearer
instructions passed Run 5, but the independent Run 6 again made an unrequested
tool call, this time a prediction after an earlier withheld prediction. The
server now enforces stored-result follow-ups with tools disabled. Run 7 passed
all 15 standard scenarios; four extra paraphrases ran no tools, with one
unnecessary refusal of an evaluation-summary question. Whatever the
model, the server checks its conditional decision and blocks a disallowed
prediction. The agent also accepts reasoning models through `OPENAI_MODEL`,
replaying their encrypted reasoning between tool calls, but only
`gpt-4.1-mini` has been evaluated live.

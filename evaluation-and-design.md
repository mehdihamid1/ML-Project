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

## Limitations

- The stratified random hold-out tests performance within this collection.
  It does not test future collection periods, emerging malware or deployment
  drift. SHA1 deduplication prevents the same file crossing partitions, but
  does not guarantee separation of related malware families.
- Imported DLL and symbol vocabularies are each capped at 64 terms; the
  nonnative Identify encoder is capped at 32 categories. These memory bounds
  can discard useful distinctions. Missingness and collection-specific text
  patterns may also limit transfer to another source.
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
are implemented. Live deployment and real-provider evaluation remain pending
environment configuration. The
[full CI job](https://github.com/mehdihamid1/ML-Project/actions/runs/37396524639/job/112053720455)
and [lean runtime job](https://github.com/mehdihamid1/ML-Project/actions/runs/37396524639/job/112053720662)
passed tests and model verification; the subsequent deploy job failed on missing
Render settings. This is not a completed deployment.

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
returns unavailable AUC for a single-class file. The conditional orchestration
calls evaluation first, validates a finite user threshold, and predicts only
when returned accuracy meets it and all rows have valid labels and features.
Incomplete coverage and tool failures fail closed. The activity record shows
the underlying calls and skipped predictions.

The three model tools are `predict_single`, `predict_batch` and `evaluate`.
The agent's `evaluate_then_predict` function orchestrates the conditional task
by calling the evaluation and single-prediction tools in order. Thresholds are
bound to explicit user input; an ambiguous condition requests clarification
instead of guessing. Follow-ups select stored result references. Pasted CSVs
are directed to the upload route before a provider call.
Descriptive requests such as “tell me if it is malware” are ordinary prediction
requests. Independent evaluation and classification can run in the same turn.
Confusion-matrix follow-ups name true/false positives and negatives, with
rates computed from stored counts. Known tool validation errors explain the
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
Runtime dependencies are separate from the training environment. See
[deployed.md](deployed.md) for operational setup and
[agent-evaluation.md](agent-evaluation.md) for the pending real-LLM evidence.

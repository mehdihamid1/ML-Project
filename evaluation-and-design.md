# Model evaluation and design

## Dataset and leakage controls

Source and verified checksums are in [dataset-source.md](dataset-source.md).
`train.py` drops every row belonging to a SHA1 with conflicting labels, then
retains the first row per remaining SHA1. It reserves a stratified 20% hold-out
before fitting any preprocessing. SHA1, FirstSeenDate, and the three previously
verified constant columns are excluded from model inputs. SHA1 remains in
saved partition files to audit separation. Label is always excluded from inputs.

## Seven models and preprocessing

Logistic Regression, Decision Tree, Random Forest, a PyTorch MLP, XGBoost,
LightGBM, and CatBoost are compared. Numeric median imputation, a fixed signed log1p transform for heavy-tailed PE
values, and scaling,
DLL/symbol TF-IDF vocabularies, and Identify category encoding are fitted
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
held-out metrics, feature schema, and provenance without committing the dataset
in those reports. A byte-for-byte copy of the small trusted production model is
bundled under `models/` for the app. The tools, OpenAI agent and Flask application
are implemented. Live deployment and real-provider evaluation remain pending
environment configuration.

## Agent and runtime design

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

The app uses private server-side session records, opaque upload IDs, CSRF tokens,
secure cookies on Render, request limits and bounded uploads/results. Its
in-process state requires one Gunicorn worker and is ephemeral across restarts.
Runtime dependencies are separate from the training environment. See
[deployed.md](deployed.md) for operational setup and
[agent-evaluation.md](agent-evaluation.md) for the pending real-LLM evidence.

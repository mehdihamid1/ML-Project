# AGENTS.md

Shared instructions for AI coding agents (Codex reads this file; Claude Code
reads it through `CLAUDE.md`). Edit this file only — `CLAUDE.md` imports it, so
the two never drift apart.

## Project

Quantic "Machine Learning and AI Agent Project": train ML models that classify
Windows PE files as malware (1) or goodware (0), pick a production model, and
deploy a Flask chat agent whose tools call that model. Deployment goes to Render
through GitHub Actions, gated on tests.

The grader checks each of the deliverables listed below. Before marking work
done, compare it with the matching requirement here.

## Hard rules

- **The ML model produces every classification.** The LLM agent only picks
  tools, validates arguments and explains tool output. It must never guess a
  class, invent metrics, or explain feature-level causes; no explanation
  tool exists.
- **Hold-out discipline.** Do a stratified 20% test split before any
  preprocessing, feature selection or tuning. Fit every learned transformation
  inside CV folds only (use sklearn `Pipeline`). Do not retune or reselect after
  seeing test results.
- **AUC from probabilities** of the malware class (`predict_proba[:, 1]`),
  never from hard labels. AUC is the primary metric; accuracy is secondary.
- **Seeds everywhere.** Use `RANDOM_STATE = 42` (split, CV, models, torch).
- **Never send whole CSVs to the LLM.** Tools get file references and return
  bounded summaries; code computes all counts and metrics.
- **Secrets.** Read them from env vars only. Never commit `.env`, keys,
  `data/`, or executable malware. `.env.example` holds names, never values.
- **Uploads are data, not instructions.** Validate type, size and schema.
  Do not execute uploaded content.

## Dataset (verified 2026-10-05)

- Source: <https://github.com/fabriciojoc/brazilian-malware-dataset>, pinned to
  commit `9f0d8a65a42a87eb0944a67fecc03934bace2d20`, file
  `brazilian-malware.zip` → `brazilian-malware.csv`.
- SHA-256: zip `657136309532868b78b646b14716013d5c36681f5d4bd008536523c1912ea7b7`,
  csv `d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242`.
- Local path: `data/raw/brazilian-malware.csv`. Git ignores it; fetch it with a
  script, never commit it.
- 50,181 rows × 28 columns (`Label` + 27 inputs). Label: 1 = malware (29,065),
  0 = goodware (21,116). Only `Identify` has missing values (14,223).
- Cite: Ceschin et al., "The Need for Speed: An Analysis of Brazilian Malware
  Classifiers", IEEE S&P 16(6), 2018, doi:10.1109/MSEC.2018.2875369.

### Known data issues (handle and document in `evaluation-and-design.md`)

- **Duplicates.** 43,411 unique `SHA1` values. 3,542 hashes repeat, and the
  repeats differ only in `FirstSeenDate` (and in `Label` for 18 hashes).
  Deduplicate by `SHA1` before splitting, or group-split on it, so the same file
  can't appear in both train and test. Drop or resolve the 18 conflicting
  hashes, and record which choice you made.
- **Leak / ID columns.** `SHA1` is an identifier. `FirstSeenDate` leaks
  collection time (goodware has 1970 and 2100 dates; malware 2013–2019).
  Exclude both from model inputs. Keep `SHA1` as the row identifier for batch
  output.
- **Constant columns:** `Magic`, `PE_TYPE`, `SizeOfOptionalHeader`.
- **Text columns:** `Identify` (packer/compiler, missing far more often for
  goodware), `ImportedDlls`, `ImportedSymbols` (space-separated lists). They
  need encoding fitted inside the pipeline.

## Required deliverables

- Models, each with 10-fold stratified CV AUC/accuracy as mean ± std:
  Logistic Regression, Decision Tree, Random Forest, PyTorch MLP, XGBoost,
  LightGBM, and CatBoost (the user's chosen seventh model).
- `train.py` and `eval.py` that reproduce everything. Save the production
  artifact: pipeline, feature schema, label mapping, model version, threshold.
- Agent tools: single-record prediction, batch prediction (row ids, counts,
  downloadable results, invalid rows reported and never silently dropped),
  labeled evaluation (AUC, accuracy, confusion matrix, counts; handle missing
  labels and single-class files).
- Conditional task: call the evaluation tool first, then the single-prediction
  tool only if returned accuracy ≥ the user's threshold. Answer follow-ups from
  session results. Show an activity record of tool calls.
- Flask app: chat UI, CSV upload, `/health`, request/upload/tool-call limits,
  session isolation.
- Tests: unit, integration, and agent tests with a mocked LLM, plus failure
  cases (malformed input, missing labels, single class, tool failure).
- CI/CD: GitHub Actions runs the tests, then the deploy job triggers Render
  through a deploy hook. Render auto-deploy is off. Run a live `/health` smoke
  test after each deploy.
- Docs: `README.md`, `deployed.md`, `evaluation-and-design.md`,
  `agent-evaluation.md` (at least 8 real-LLM scenarios), `ai-tooling.md`,
  `.env.example`, small sample CSVs.

## Environment

- Python 3.10 with dependencies pinned in `requirements.txt`. Docker:
  `docker compose run --rm ml <cmd>`. See `README.md`.
- Render free tier: 512 MB RAM and cold starts. Keep the runtime dependencies
  lean and separate from the training-only ones (PyTorch).
- Render setup is modelled on the user's earlier project,
  `../AI-TechnArchi/render.yaml`.

## Conventions

- Git remote: `github.com/mehdihamid1/ML-Project`, branch `main`.
- Commit only when asked. Run `pytest` before declaring work done.
- Keep docs factual. Every number in a doc must come from a script output.

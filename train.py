"""Compare seven models on training-only CV, then freeze and test the winner."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, make_scorer
from ml_project.data import prepare_data, split_data, feature_frame
from ml_project.models import build_models
from ml_project.report import write_report


def metrics(y, probabilities, threshold=0.5):
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {'sample_count': len(y), 'auc': float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
            'accuracy': float(accuracy_score(y, predicted)), 'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            'class_mapping': {'0': 'goodware', '1': 'malware'}, 'confusion_matrix_order': [0, 1]}


def run(dataset, output, smoke=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; use a new directory to preserve frozen results')
    output.mkdir(parents=True, exist_ok=True)
    frame, data_stats = prepare_data(pd.read_csv(dataset))
    training, holdout = split_data(frame)
    # Smoke mode samples only development data and never evaluates the hold-out.
    if smoke:
        training, _ = train_test_split(training, train_size=min(400, len(training)-2), stratify=training.Label, random_state=42)
    X, y = feature_frame(training), training.Label
    folds = 2 if smoke else 10
    if y.value_counts().min() < folds:
        raise ValueError('Not enough records per class for stratified CV')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = []
    models = build_models(quick=smoke)
    alternatives = {
        'Logistic Regression': {'model__C': 0.1},
        'Decision Tree': {'model__max_depth': 20},
        'Random Forest': {'model__max_depth': 20},
        'PyTorch MLP': {'model__hidden': 64},
        'XGBoost': {'model__max_depth': 4},
        'LightGBM': {'model__num_leaves': 15},
        'CatBoost': {'model__depth': 4},
    }
    selected_pipelines = {}
    candidates_report = []
    for name, pipeline in models.items():
        candidates = [pipeline] if smoke else [pipeline, clone(pipeline).set_params(**alternatives[name])]
        candidate_rows = []
        for index, candidate in enumerate(candidates):
            print(f'Comparing {name}, configuration {index + 1} ({folds} folds)', flush=True)
            scores = cross_validate(candidate, X, y, cv=cv, scoring={'auc': make_scorer(roc_auc_score, response_method='predict_proba'), 'accuracy': 'accuracy'}, error_score='raise', n_jobs=1)
            row = {'model': name, 'configuration': index, 'auc_mean': float(np.mean(scores['test_auc'])), 'auc_std': float(np.std(scores['test_auc'])),
                   'accuracy_mean': float(np.mean(scores['test_accuracy'])), 'accuracy_std': float(np.std(scores['test_accuracy'])),
                   'fit_seconds_mean': float(np.mean(scores['fit_time'])),
                   'fold_auc': scores['test_auc'].tolist(), 'fold_accuracy': scores['test_accuracy'].tolist(),
                   'parameters': candidate.named_steps['model'].get_params()}
            candidate_rows.append(row)
            candidates_report.append(row)
            (output / 'search-results.json').write_text(json.dumps(candidates_report, indent=2))
        best = sorted(candidate_rows, key=lambda r: (-r['auc_mean'], -r['accuracy_mean'], r['fit_seconds_mean']))[0]
        selected_pipelines[name] = candidates[best['configuration']]
        results.append(best)
        (output / 'cv-results.json').write_text(json.dumps(results, indent=2))
    ranked = sorted(results, key=lambda r: (-r['auc_mean'], -r['accuracy_mean'], r['fit_seconds_mean']))
    pd.DataFrame([{k: r[k] for k in ['model', 'auc_mean', 'auc_std', 'accuracy_mean', 'accuracy_std', 'fit_seconds_mean']} for r in ranked]).to_csv(output / 'cv-results.csv', index=False)
    metadata = {'smoke': smoke, 'folds': folds, 'random_state': 42, 'threshold': 0.5, 'data': data_stats,
                'training_rows': len(training), 'holdout_rows': len(holdout), 'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                'selection_rule': 'CV AUC descending, then accuracy descending, then fit time ascending',
                'search_scope': 'Two fixed configurations per model on identical training-only CV folds; one reduced configuration in smoke mode',
                'dependencies': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost', 'lightgbm', 'catboost']}}
    if not smoke:
        winner = ranked[0]['model']
        pipeline = selected_pipelines[winner].fit(X, y)
        metadata.update({'selected_model': winner, 'model_version': f'{metadata["dataset_sha256"][:12]}-{time.time_ns()}',
                         'features': [{'name': c, 'dtype': str(X[c].dtype), 'allow_missing': c == 'Identify'} for c in X.columns],
                         'class_mapping': {'0': 'goodware', '1': 'malware'}})
        joblib.dump({'pipeline': pipeline, 'metadata': metadata}, output / 'production.joblib')
        holdout.to_csv(output / 'holdout.csv', index=False)
        pd.DataFrame({'SHA1': training.SHA1, 'partition': 'training'}).to_csv(output / 'training-ids.csv', index=False)
        metadata['holdout_metrics'] = metrics(holdout.Label, pipeline.predict_proba(feature_frame(holdout))[:, 1])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_report(output)
    print(pd.read_csv(output / 'cv-results.csv').to_string(index=False), flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/brazilian-malware.csv')
    parser.add_argument('--output', default='artifacts/training')
    parser.add_argument('--smoke', action='store_true', help='Reduced development-only check; no production artifact or hold-out metrics')
    args = parser.parse_args()
    run(args.data, args.output, args.smoke)

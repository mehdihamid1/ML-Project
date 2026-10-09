"""Check feature selection and dimensionality reduction for the selected model, on training folds only.

This runs after the production model was frozen and changes nothing. It scores the
selected LightGBM configuration on the same ten training folds as train.py, with its
inputs reduced or enlarged, so the input choice can be compared with alternatives.
The hold-out rows are only counted (for the class balance); no model sees them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml_project.data import feature_frame, prepare_data, split_data
from ml_project.models import TEXT, build_models, signed_log1p


class NumericOnly(TransformerMixin, BaseEstimator):
    """The numeric PE fields alone: the DLL, symbol and Identify text columns are dropped."""
    def fit(self, X, y=None):
        self.columns_ = [column for column in X.columns if column not in TEXT]
        self.pipeline_ = Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('log', FunctionTransformer(signed_log1p)),
            ('scale', StandardScaler()),
        ]).fit(X[self.columns_])
        return self

    def transform(self, X):
        values = self.pipeline_.transform(X[self.columns_]).astype(np.float32)
        return pd.DataFrame(values, index=X.index, columns=[f'feature_{i}' for i in range(values.shape[1])])


def variants(selected):
    model = selected.named_steps['model']
    encoder = selected.named_steps['features']
    return [
        ('Production inputs', 'Numeric fields, 64-term DLL and symbol vocabularies, Identify categories (the frozen choice)',
         clone(selected)),
        ('Numeric fields only', 'Drops the DLL, symbol and Identify text columns',
         Pipeline([('features', NumericOnly()), ('model', clone(model))])),
        ('Smaller vocabularies', '16 terms per DLL and symbol list instead of 64',
         clone(selected).set_params(features__max_features=16)),
        ('Larger vocabularies', '256 terms per DLL and symbol list instead of 64',
         clone(selected).set_params(features__max_features=256)),
        ('Univariate selection', 'Keeps the 50 encoded features with the highest ANOVA F-score, fitted in each fold',
         Pipeline([('features', clone(encoder)), ('select', SelectKBest(f_classif, k=50)), ('model', clone(model))])),
        ('Truncated SVD', 'Projects the encoded features onto 32 components, fitted in each fold',
         Pipeline([('features', clone(encoder)), ('reduce', TruncatedSVD(n_components=32, random_state=42)), ('model', clone(model))])),
    ]


def run(dataset, output):
    frame, _ = prepare_data(pd.read_csv(dataset))
    training, holdout = split_data(frame)
    X, y = feature_frame(training), training.Label
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    selected = build_models()['LightGBM']
    rows = []
    for name, description, pipeline in variants(selected):
        print(f'Scoring {name} (10 folds)', flush=True)
        # The number of columns the model receives, from inputs fitted on all development rows.
        width = clone(pipeline)[:-1].fit(X, y).transform(X.iloc[:5]).shape[1]
        scores = cross_validate(pipeline, X, y, cv=cv, error_score='raise', n_jobs=1, scoring={
            'auc': make_scorer(roc_auc_score, response_method='predict_proba'), 'accuracy': 'accuracy'})
        rows.append({'variant': name, 'description': description, 'model_input_width': int(width),
                     'auc_mean': float(np.mean(scores['test_auc'])), 'auc_std': float(np.std(scores['test_auc'])),
                     'accuracy_mean': float(np.mean(scores['test_accuracy'])), 'accuracy_std': float(np.std(scores['test_accuracy'])),
                     'fit_seconds_mean': float(np.mean(scores['fit_time'])), 'fold_auc': scores['test_auc'].tolist()})
    report = {
        'purpose': 'Training-folds-only check of the input choice for the frozen LightGBM configuration; run after selection, changed nothing.',
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'folds': 10, 'random_state': 42, 'development_rows': len(training),
        'model': 'LightGBM', 'model_parameters': {k: v for k, v in selected.named_steps['model'].get_params().items()
                                                  if k in ('n_estimators', 'num_leaves', 'learning_rate', 'random_state')},
        'class_counts': {'development': {str(k): int(v) for k, v in training.Label.value_counts().sort_index().items()},
                         'holdout': {str(k): int(v) for k, v in holdout.Label.value_counts().sort_index().items()}},
        'variants': rows,
    }
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(rows)[['variant', 'model_input_width', 'auc_mean', 'auc_std', 'accuracy_mean', 'accuracy_std', 'fit_seconds_mean']].to_string(index=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw/brazilian-malware.csv')
    parser.add_argument('--output', default='docs/feature-exploration.json')
    arguments = parser.parse_args()
    run(arguments.data, arguments.output)

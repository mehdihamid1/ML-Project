"""Evaluate a trusted local production artifact without refitting."""
import argparse
import json
import joblib
import pandas as pd
from train import metrics


def evaluate(artifact, csv_path):
    bundle = joblib.load(artifact)  # Only load artifacts produced by this project.
    frame = pd.read_csv(csv_path)
    if 'Label' not in frame or frame.Label.isna().any() or not set(frame.Label.unique()).issubset({0, 1}):
        raise ValueError('Evaluation requires Label values 0 or 1')
    if frame.empty:
        raise ValueError('Evaluation CSV is empty')
    names = [f['name'] for f in bundle['metadata']['features']]
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    result = metrics(frame.Label, bundle['pipeline'].predict_proba(frame[names])[:, 1], bundle['metadata']['threshold'])
    result['model_version'] = bundle['metadata']['model_version']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts/training/production.joblib')
    parser.add_argument('--data', default='artifacts/training/holdout.csv')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.data), indent=2))

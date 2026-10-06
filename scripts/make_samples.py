"""Create tiny CSV examples from the frozen hold-out, never executable PE files."""
import json
from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_samples():
    bundle = joblib.load('models/production.joblib')
    frame = pd.read_csv('artifacts/training-final/holdout.csv', nrows=200)
    names = [feature['name'] for feature in bundle['metadata']['features']]
    parts = []
    for label in [0, 1]:
        subset = frame[frame.Label == label]
        parts.append(subset.loc[subset.ImportedSymbols.str.len().sort_values().index[:2]])
    small = pd.concat(parts)[['SHA1', *names, 'Label']].copy()
    directory = Path('samples')
    directory.mkdir(exist_ok=True)
    small.to_csv(directory / 'labeled.csv', index=False)
    small.drop(columns='Label').to_csv(directory / 'batch.csv', index=False)
    small.drop(columns='Label').iloc[:1].to_csv(directory / 'single.csv', index=False)
    bad = small.drop(columns='Label').copy()
    bad['Size'] = bad['Size'].astype(object)
    bad.loc[bad.index[-1], 'Size'] = 'invalid'
    bad.to_csv(directory / 'invalid-rows.csv', index=False)
    missing = small.copy()
    missing.loc[missing.index[-1], 'Label'] = None
    missing.to_csv(directory / 'missing-labels.csv', index=False)
    small[small.Label == 1].to_csv(directory / 'single-class.csv', index=False)
    flipped = small.iloc[:1].copy()
    flipped['Label'] = 1 - bundle['pipeline'].predict(flipped[names])
    flipped.to_csv(directory / 'conditional-fail.csv', index=False)
    counts = {path.name: len(pd.read_csv(path)) for path in directory.glob('*.csv')}
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    make_samples()

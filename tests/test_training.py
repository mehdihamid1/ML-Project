import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from ml_project.data import prepare_data, split_data, feature_frame
from ml_project.models import build_models
from train import metrics
from eval import evaluate


def sample():
    return pd.DataFrame({'SHA1': [f'h{i}' for i in range(60)], 'FirstSeenDate': ['1970'] * 60,
                         'Label': [0, 1] * 30, 'Size': np.arange(60) + 1,
                         'Identify': ['compiler', None] * 30,
                         'ImportedDlls': ['kernel32.dll user32.dll'] * 60,
                         'ImportedSymbols': ['ReadFile WriteFile'] * 60})


def test_duplicate_and_conflict_control():
    frame = sample()
    frame = pd.concat([frame, frame.iloc[[0]], frame.iloc[[1]].assign(Label=0)])
    clean, stats = prepare_data(frame)
    assert clean.SHA1.is_unique
    assert 'h1' not in set(clean.SHA1)
    assert stats['conflicting_hashes_removed'] == 1
    training, holdout = split_data(clean)
    assert set(training.SHA1).isdisjoint(holdout.SHA1)
    assert set(training.Label) == set(holdout.Label) == {0, 1}
    assert not {'SHA1', 'FirstSeenDate', 'Label'} & set(feature_frame(clean).columns)


@pytest.mark.parametrize('name', list(build_models(quick=True)))
def test_models_handle_unknown_categories_and_roundtrip(name, tmp_path):
    frame = sample()
    pipeline = clone(build_models(quick=True)[name])
    pipeline.fit(feature_frame(frame), frame.Label)
    unseen = feature_frame(frame.iloc[:2]).copy()
    unseen['Identify'] = 'never-seen-compiler'
    unseen['ImportedSymbols'] = 'NeverSeenSymbol'
    p = pipeline.predict_proba(unseen)
    assert p.shape == (2, 2)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    path = tmp_path / 'pipeline.joblib'
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(unseen), p)
    encoder = pipeline.named_steps['features']
    assert 'neverseensymbol' not in encoder.vectors_['ImportedSymbols'].vocabulary_


def test_auc_uses_continuous_scores_and_single_class():
    result = metrics([0, 1], [0.1, 0.4])
    assert result['auc'] == 1
    assert result['accuracy'] == 0.5
    result = metrics([1, 1], [0.6, 0.2])
    assert result['auc'] is None
    assert result['confusion_matrix'] == [[0, 0], [1, 1]]


def test_evaluation_requires_labels_and_features(tmp_path):
    frame = sample()
    pipeline = build_models(quick=True)['Decision Tree'].fit(feature_frame(frame), frame.Label)
    artifact = tmp_path / 'artifact.joblib'
    joblib.dump({'pipeline': pipeline, 'metadata': {'features': [{'name': c} for c in feature_frame(frame)], 'threshold': 0.5, 'model_version': 'test'}}, artifact)
    csv = tmp_path / 'input.csv'
    frame.drop(columns='Label').to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Label'):
        evaluate(artifact, csv)
    frame.drop(columns='Size').to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Missing features'):
        evaluate(artifact, csv)
    frame[frame.Label == 1].to_csv(csv, index=False)
    assert evaluate(artifact, csv)['auc'] is None


def test_smoke_compares_seven_without_using_holdout(tmp_path):
    from train import run
    path = tmp_path / 'data.csv'
    sample().to_csv(path, index=False)
    output = tmp_path / 'results'
    result = run(path, output, smoke=True)
    assert result['smoke'] is True
    assert 'holdout_metrics' not in result
    assert not (output / 'production.joblib').exists()
    assert len(pd.read_csv(output / 'cv-results.csv')) == 7
    with pytest.raises(ValueError, match='empty'):
        run(path, output, smoke=True)

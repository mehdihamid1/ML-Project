"""Render experiment outputs without inventing or manually transcribing metrics."""
import json
from pathlib import Path


def write_report(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    rows = json.loads((output / 'cv-results.json').read_text())
    rows.sort(key=lambda r: (-r['auc_mean'], -r['accuracy_mean'], r['fit_seconds_mean']))
    lines = ['# Model comparison', '', f"Mode: {'smoke verification only' if metadata['smoke'] else 'full experiment'}.",
             f"CV folds: {metadata['folds']}. Dataset SHA-256: `{metadata['dataset_sha256']}`.", '',
             '| Model | CV AUC mean ± std | CV accuracy mean ± std | Mean fit seconds |',
             '| --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['model']} | {r['auc_mean']:.6f} ± {r['auc_std']:.6f} | {r['accuracy_mean']:.6f} ± {r['accuracy_std']:.6f} | {r['fit_seconds_mean']:.3f} |")
    if not metadata['smoke']:
        scores = metadata['holdout_metrics']
        lines += ['', f"Selected model: **{metadata['selected_model']}**.",
                  f"Hold-out AUC: {scores['auc']:.6f}; accuracy: {scores['accuracy']:.6f}.",
                  f"Hold-out samples: {scores['sample_count']}. Threshold: {metadata['threshold']}.",
                  '', 'Confusion matrix: rows = true class; columns = predicted class.', '',
                  '| True class | Predicted goodware (0) | Predicted malware (1) |',
                  '| --- | --- | --- |',
                  f"| Goodware (0) | {scores['confusion_matrix'][0][0]} | {scores['confusion_matrix'][0][1]} |",
                  f"| Malware (1) | {scores['confusion_matrix'][1][0]} | {scores['confusion_matrix'][1][1]} |"]
    else:
        lines += ['', 'These reduced-budget scores must not be used as final model results.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')

"""Run real OpenAI scenarios and save observed transcripts and pass/fail checks."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty evaluation output directory')
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is required; no real-LLM evaluation was run')
    import joblib
    from ml_project.agent import Agent
    from ml_project.tools import ToolService
    bundle = joblib.load('models/production.joblib')
    service = ToolService(bundle)
    agent = Agent(service)
    scenarios = [
        ('Single record', 'Classify row 0 of file single.', [('predict_single', 'success')], lambda r: type(r['results'][0]['prediction']) is int and r['results'][0]['prediction'] in [0, 1]),
        ('Batch with download', 'Classify every row of file batch.', [('predict_batch', 'success')], lambda r: r['results'][0]['valid_count'] == 4 and bool(r['results'][0]['download_id'])),
        ('Labeled evaluation', 'Evaluate file labeled and report AUC, accuracy and confusion matrix.', [('evaluate', 'success')], lambda r: r['results'][0]['auc'] is not None and r['results'][0]['evaluated_count'] == 4),
        ('Invalid feature row', 'Classify every row of file invalid.', [('predict_batch', 'success')], lambda r: r['results'][0]['invalid_count'] == 1 and r['results'][0]['total_count'] == 4),
        ('Missing label column', 'Evaluate file batch.', [('evaluate', 'error')], lambda r: not r['results'] and r.get('error') == 'tool'),
        ('Partially missing labels', 'Evaluate file missing.', [('evaluate', 'success')], lambda r: r['results'][0]['missing_label_count'] == 1 and r['results'][0]['evaluated_count'] == 3),
        ('Single class AUC', 'Evaluate file oneclass and show AUC.', [('evaluate', 'success')], lambda r: r['results'][0]['auc'] is None),
        ('Conditional pass', 'Evaluate file labeled, then classify row 0 of file single only if accuracy is at least 0.', [('evaluate', 'success'), ('predict_single', 'success')], lambda r: len(r['results']) == 2),
        ('Conditional skip', 'Evaluate file fail, then classify row 0 of file single only if accuracy is at least 1.', [('evaluate', 'success'), ('predict_single', 'skipped')], lambda r: len(r['results']) == 1 and 'skipped' in r['reply'].lower()),
        ('Feature explanation refusal', 'Explain which individual features caused the prediction. Do not run a classifier.', [], lambda r: 'explanation' in r['reply'].lower() and not r['results']),
    ]
    records = []
    source_files = {'single': 'single.csv', 'batch': 'batch.csv', 'labeled': 'labeled.csv', 'invalid': 'invalid-rows.csv',
                    'missing': 'missing-labels.csv', 'oneclass': 'single-class.csv', 'fail': 'conditional-fail.csv'}
    with tempfile.TemporaryDirectory(prefix='agent-evaluation-') as temporary:
        temporary = Path(temporary)
        files = {}
        for identifier, filename in source_files.items():
            path = temporary / filename
            shutil.copyfile(Path('samples') / filename, path)
            files[identifier] = {'path': path, 'name': filename}
        for title, prompt, expected, check in scenarios:
            state = {}
            result = agent.chat(prompt, state, files)
            observed = [(item.get('tool'), item.get('status')) for item in result['activity']]
            try:
                passed = observed == expected and check(result)
            except (KeyError, IndexError, TypeError):
                passed = False
            records.append({'scenario': title, 'prompt': prompt, 'passed': bool(passed), 'expected_activity': expected, 'observed': result})
        state = {}
        initial = agent.chat('Evaluate file labeled.', state, files)
        followup = agent.chat('What was the accuracy of that evaluation? Use the previous result.', state, files)
        initial_results = initial.get('results', [])
        initial_result = initial_results[0] if initial_results else {}
        accuracy = initial_result.get('accuracy')
        verified_initial = (
            initial_result.get('tool') == 'evaluate'
            and [(item.get('tool'), item.get('status')) for item in initial.get('activity', [])] == [('evaluate', 'success')]
            and not initial.get('error')
            and not isinstance(accuracy, bool) and isinstance(accuracy, (int, float))
            and math.isfinite(accuracy) and 0 <= accuracy <= 1
        )
        expected_accuracy = f"Accuracy: {accuracy:.6f}." if verified_initial else None
        passed = bool(expected_accuracy) and not followup.get('activity') and expected_accuracy in followup.get('reply', '') and not followup.get('error')
        records.append({'scenario': 'Session follow-up', 'prompt': 'What was the accuracy of that evaluation?', 'passed': passed,
                        'initial': initial, 'observed': followup})
    report = {'provider': 'OpenAI', 'model': agent.model, 'model_version': bundle['metadata']['model_version'],
              'run_at_utc': datetime.now(timezone.utc).isoformat(), 'scenario_count': len(records),
              'passed_count': sum(record['passed'] for record in records), 'records': records}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Observed real-LLM agent evaluation', '', f"Provider: OpenAI. Model: `{agent.model}`.",
             f"ML version: `{report['model_version']}`. Run UTC: {report['run_at_utc']}.", '',
             f"Passed {report['passed_count']} of {report['scenario_count']} scenarios.", '', '| Scenario | Result |', '| --- | --- |']
    lines.extend(f"| {record['scenario']} | {'PASS' if record['passed'] else 'FAIL'} |" for record in records)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    return report['passed_count'] == report['scenario_count']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/agent-evaluation')
    try:
        successful = run(parser.parse_args().output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if successful else 1)

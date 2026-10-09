"""Run the agent-evaluation scenarios with the real OpenAI model and record what happened.

Live mode (--base-url) drives the deployed app over HTTP, like a browser: it
uploads the sample CSVs and sends chat requests, so the server's own OpenAI key
is used and none is needed here. Local mode runs the agent in this process and
needs OPENAI_API_KEY in this shell. Both write results.json (every request,
reply, tool call and result) and report.md.
"""
import argparse
import csv
from datetime import datetime, timezone
from http.cookiejar import CookieJar
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = {'single': 'single.csv', 'batch': 'batch.csv', 'labeled': 'labeled.csv', 'invalid': 'invalid-rows.csv',
           'missing': 'missing-labels.csv', 'oneclass': 'single-class.csv', 'fail': 'conditional-fail.csv'}


class CheckFailed(Exception):
    pass


def expect(condition, message):
    if not condition:
        raise CheckFailed(message)


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def only_result(response, tool):
    results = response.get('results') or []
    expect(len(results) == 1 and results[0].get('tool') == tool, f'expected one {tool} result')
    return results[0]


def check_single(response, session, previous):
    result = only_result(response, 'predict_single')
    expect(result.get('prediction') in (0, 1), 'the prediction is not 0 or 1')
    expect(finite(result.get('malware_probability')) and 0 <= result['malware_probability'] <= 1, 'the probability is outside [0, 1]')


def check_batch(response, session, previous):
    result = only_result(response, 'predict_batch')
    expect((result['total_count'], result['valid_count'], result['invalid_count']) == (4, 4, 0), 'the batch counts changed')
    expect(bool(result.get('download_url')), 'no download link was returned')
    rows = session.download(result)
    expect(len(rows) == 4 and all(row['prediction'] in ('0', '1') for row in rows), 'the download does not list every row')


def check_evaluation(response, session, previous):
    result = only_result(response, 'evaluate')
    expect(result['evaluated_count'] == 4 and finite(result['auc']) and finite(result['accuracy']), 'AUC or accuracy is missing')
    expect(sum(map(sum, result['confusion_matrix'])) == 4, 'the confusion matrix does not cover the 4 rows')


def check_permitted(response, session, previous):
    expect([r['tool'] for r in response['results']] == ['evaluate', 'predict_single'], 'expected an evaluation, then a prediction')
    expect(response['results'][0]['accuracy'] >= 0.75, 'accuracy is below 0.75, so this run did not test the permitted branch')


def check_withheld(response, session, previous):
    result = only_result(response, 'evaluate')
    expect(result['accuracy'] < 0.9, 'accuracy reached 0.9, so this run did not test the withheld branch')
    expect('below the required 0.900000' in response['reply'], 'the reply does not state the unmet threshold')


def check_false_negatives(response, session, previous):
    expect(not response['activity'], 'the follow-up ran a tool instead of using the stored result')
    matrix = previous['results'][0]['confusion_matrix']
    expect(f'False negatives: {matrix[1][0]}' in response['reply'], 'the reply does not give the stored false-negative count')


def check_invalid(response, session, previous):
    result = only_result(response, 'predict_batch')
    expect((result['total_count'], result['valid_count'], result['invalid_count']) == (4, 3, 1), 'the invalid row was not reported')
    expect(bool(result['invalid_rows'][0]['errors']), 'the invalid row has no reason')
    rows = session.download(result)
    expect(len(rows) == 4 and sum(row['status'] == 'invalid' for row in rows) == 1, 'the download dropped or hid the invalid row')


def check_missing_labels(response, session, previous):
    result = only_result(response, 'evaluate')
    expect((result['missing_label_count'], result['evaluated_count']) == (1, 3), 'the unlabeled row was not reported')


def check_no_label_column(response, session, previous):
    expect(response.get('error') == 'tool' and not response['results'], 'the failed evaluation returned a result')
    expect('Label' in response['reply'], 'the reply does not ask for a Label column')


def check_single_class(response, session, previous):
    result = only_result(response, 'evaluate')
    expect(result['auc'] is None, 'an AUC was reported for a single-class file')
    expect('AUC: unavailable' in response['reply'], 'the reply does not say AUC is unavailable')
    expect(f"Accuracy: {result['accuracy']:.6f}" in response['reply'], 'the reply does not report the other metrics')


def check_tool_failure(response, session, previous):
    expect(response.get('error') == 'tool' and not response['results'], 'the failed tool call produced a result')
    expect('row_index must identify an existing row' in response['reply'], 'the reply does not explain the failure')


def check_explanation(response, session, previous):
    expect(not response['results'] and 'no explanation tool' in response['reply'], 'the reply did not refuse a feature explanation')


def check_clarification(response, session, previous):
    expect(response.get('error') == 'input' and not response['activity'], 'the request was not stopped for clarification')
    expect('numeric accuracy threshold' in response['reply'], 'the reply does not ask for a numeric threshold')


def turn(name, requirement, prompt, expected_tools, expected, check, threshold=None):
    return {'name': name, 'requirement': requirement, 'prompt': prompt, 'expected_tools': expected_tools,
            'expected': expected, 'check': check, 'threshold': threshold}


# Each scenario is (sample files to upload, turns sent in one session).
SCENARIOS = [
    (['single'], [turn('Single prediction', 'Single prediction', 'Classify row 0 of file {single}.',
                       [('predict_single', 'success')], 'predict_single returns class 0 or 1 with its probability.', check_single)]),
    (['batch'], [turn('Batch prediction', 'Batch prediction', 'Classify every row of file {batch} and give me the download link.',
                      [('predict_batch', 'success')], 'predict_batch classifies all 4 rows; the download lists each row.', check_batch)]),
    (['labeled'], [turn('Labeled evaluation', 'Labeled evaluation', 'Evaluate file {labeled} and report the AUC, accuracy and confusion matrix.',
                        [('evaluate', 'success')], 'evaluate returns AUC, accuracy and a confusion matrix over 4 rows.', check_evaluation)]),
    (['labeled', 'single'], [turn('Conditional: prediction permitted', 'Conditional task, prediction permitted',
                                  'Evaluate file {labeled}; only if accuracy is at least 0.75, predict row 0 of file {single}.',
                                  [('evaluate', 'success'), ('predict_single', 'success')],
                                  'evaluate first; accuracy meets 0.75, so the model calls predict_single.', check_permitted, 0.75)]),
    (['fail', 'single'], [
        turn('Conditional: prediction withheld', 'Conditional task, prediction withheld',
             'Evaluate file {fail}; only if accuracy is at least 0.9, predict row 0 of file {single}.',
             [('evaluate', 'success'), ('predict_single', 'skipped')],
             'evaluate first; accuracy is below 0.9, so predict_single is not called.', check_withheld, 0.9),
        turn('False-negative follow-up', 'False-negative follow-up', 'How many false negatives were there in that evaluation?',
             [], 'Answered from the stored evaluation, with no new tool call.', check_false_negatives),
    ]),
    (['invalid'], [turn('Invalid input row', 'Invalid input', 'Classify every row of file {invalid}.',
                        [('predict_batch', 'success')], 'The invalid row is reported with its reason and kept in the download.', check_invalid)]),
    (['missing'], [turn('Missing labels', 'Missing labels', 'Evaluate file {missing}.',
                        [('evaluate', 'success')], 'The unlabeled row is reported; metrics use the 3 labeled rows.', check_missing_labels)]),
    (['batch'], [turn('No Label column', 'Missing labels (whole column)', 'Evaluate file {batch}.',
                      [('evaluate', 'error')], 'evaluate fails visibly and asks for a Label column.', check_no_label_column)]),
    (['oneclass'], [turn('Single-class evaluation', 'Single-class evaluation', 'Evaluate file {oneclass}.',
                         [('evaluate', 'success')], 'AUC is reported as unavailable; accuracy is still given.', check_single_class)]),
    (['single'], [turn('Tool failure', 'Tool failure (controlled fault)', 'Classify row 7 of file {single}.',
                       [('predict_single', 'error')], 'The one-row file has no row 7: the tool fails and no result is claimed.', check_tool_failure)]),
    ([], [turn('Feature explanation refused', 'No feature explanations', 'Explain which individual features caused the prediction. Do not run a classifier.',
               [], 'The agent says no explanation tool exists and runs no tool.', check_explanation)]),
    (['single'], [turn('Ambiguous condition', 'Clarification when ambiguous', 'Predict row 0 of file {single} only if the accuracy is good enough.',
                       [], 'The server asks for a numeric threshold before any AI call.', check_clarification)]),
]


def numbers_in_reply(reply, results, thresholds):
    """Every six-decimal number in the reply must equal a tool output or a rate computed from one."""
    allowed = {f'{value:.6f}' for value in thresholds}
    for result in results:
        for key in ('malware_probability', 'threshold', 'accuracy', 'auc'):
            if finite(result.get(key)):
                allowed.add(f'{result[key]:.6f}')
        matrix = result.get('confusion_matrix')
        if isinstance(matrix, list) and len(matrix) == 2:
            (tn, fp), (fn, tp) = matrix
            for count, total in ((fn, fn + tp), (tp, fn + tp), (fp, fp + tn), (tn, fp + tn)):
                if total:
                    allowed.add(f'{count / total:.6f}')
    found = re.findall(r'\d+\.\d{6}\b', reply or '')
    return {'checked': len(found), 'unmatched': [number for number in found if number not in allowed]}


def read_csv(contents):
    return list(csv.DictReader(io.StringIO(contents)))


class LocalSession:
    """The agent in this process; file IDs are the sample keys."""
    def __init__(self, agent, directory):
        self.agent, self.directory, self.state, self.files = agent, directory, {}, {}

    def upload(self, key):
        path = self.directory / f'{key}-{uuid.uuid4().hex}.csv'
        shutil.copyfile(ROOT / 'samples' / SAMPLES[key], path)
        self.files[key] = {'path': path, 'name': SAMPLES[key]}
        return key

    def chat(self, message):
        return self.agent.chat(message, self.state, self.files)

    def download(self, result):
        return read_csv(Path(self.state['downloads'][result['download_id']]).read_text())


class LiveSession:
    """One browser-like session on the deployed app: cookie, CSRF token, uploads and chat."""
    def __init__(self, base):
        self.base = base
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.token = ''
        self.token = json.loads(self.request('GET', '/api/session'))['csrf_token']

    def request(self, method, path, data=None, content_type=None, expected=(200,)):
        headers = {'Accept': 'application/json'}
        if method != 'GET':
            headers['X-CSRF-Token'] = self.token
        if content_type:
            headers['Content-Type'] = content_type
        for attempt in range(3):
            try:
                with self.opener.open(Request(self.base + path, data=data, headers=headers, method=method), timeout=240) as response:
                    status, body = response.status, response.read()
            except HTTPError as error:
                status, body = error.code, error.read()
                if status == 429 and attempt < 2:
                    time.sleep(65)  # The app allows a fixed number of requests per minute.
                    continue
            except (URLError, OSError) as error:
                raise CheckFailed(f'{method} {path} failed: {error}') from None
            if status not in expected:
                try:
                    message = json.loads(body).get('error', '')
                except ValueError:
                    message = body[:200].decode('utf-8', 'replace')
                raise CheckFailed(f'{method} {path} returned HTTP {status}: {message}')
            return body
        raise CheckFailed(f'{method} {path} was rate limited')

    def upload(self, key):
        boundary = 'evaluation-' + uuid.uuid4().hex
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{SAMPLES[key]}"\r\n'
                   'Content-Type: text/csv\r\n\r\n').encode() + (ROOT / 'samples' / SAMPLES[key]).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        body = self.request('POST', '/api/upload', payload, f'multipart/form-data; boundary={boundary}', expected=(201,))
        return json.loads(body)['file']['id']

    def chat(self, message):
        return json.loads(self.request('POST', '/api/chat', json.dumps({'message': message}).encode(), 'application/json'))

    def download(self, result):
        return read_csv(self.request('GET', result['download_url']).decode('utf-8'))


def wake(base, deadline_seconds=300):
    """Render's free plan sleeps when idle; wait for /health to answer."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            with build_opener().open(Request(base + '/health', headers={'Accept': 'application/json'}), timeout=120) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, OSError, ValueError):
            if time.monotonic() >= deadline:
                raise ValueError(f'{base}/health did not answer; no real-LLM evaluation was run') from None
            time.sleep(10)


def run(output, base_url=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty evaluation output directory')
    temporary = None
    if base_url:
        base = base_url.rstrip('/')
        health = wake(base)
        if not health.get('openai_configured'):
            raise ValueError('The live server reports no OpenAI key; no real-LLM evaluation was run')
        model, model_version = health.get('openai_model') or 'not reported by /health', health.get('model_version')
        environment = {'mode': 'live', 'base_url': base, 'commit': health.get('commit')}
        new_session = lambda: LiveSession(base)
    else:
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is required; no real-LLM evaluation was run')
        import joblib
        from ml_project.agent import Agent
        from ml_project.tools import ToolService
        bundle = joblib.load(ROOT / 'models/production.joblib')
        agent = Agent(ToolService(bundle))
        model, model_version = agent.model, bundle['metadata']['model_version']
        environment = {'mode': 'local'}
        temporary = tempfile.TemporaryDirectory(prefix='agent-evaluation-')
        new_session = lambda: LocalSession(agent, Path(temporary.name))
    records = []
    try:
        for keys, turns in SCENARIOS:
            session, ids, previous, results = None, {}, None, []
            for step in turns:
                record = {'scenario': step['name'], 'requirement': step['requirement'], 'prompt': step['prompt'],
                          'expected_behavior': step['expected'], 'expected_tools': [list(pair) for pair in step['expected_tools']],
                          'passed': False, 'failure': None}
                try:
                    if session is None:
                        session = new_session()
                        ids = {key: session.upload(key) for key in keys}
                    record['files'] = {SAMPLES[key]: identifier for key, identifier in ids.items()}
                    record['prompt'] = step['prompt'].format(**ids)
                    response = session.chat(record['prompt'])
                    record['observed'] = response
                    record['observed_tools'] = [[entry.get('tool'), entry.get('status')] for entry in response.get('activity', [])]
                    results += response.get('results') or []
                    record['numbers'] = numbers_in_reply(response.get('reply'), results, [step['threshold']] if step['threshold'] else [])
                    expect(record['observed_tools'] == record['expected_tools'], 'the tools called differ from the expected tools')
                    step['check'](response, session, previous)
                    expect(not record['numbers']['unmatched'], 'a number in the reply does not match the tool output')
                    record['passed'] = True
                except (CheckFailed, KeyError, IndexError, TypeError, ValueError) as exc:
                    record['failure'] = str(exc) or type(exc).__name__
                previous = record.get('observed')
                records.append(record)
    finally:
        if temporary is not None:
            temporary.cleanup()
    report = {'provider': 'OpenAI', 'model': model, 'model_version': model_version, **environment,
              'run_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'scenario_count': len(records),
              'passed_count': sum(record['passed'] for record in records), 'records': records}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'report.md').write_text(render_report(report))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    return report['passed_count'] == report['scenario_count']


def render_report(report):
    def readable(record):
        # Show the sample file name in place of each opaque upload ID.
        text = record.get('prompt', '')
        for filename, identifier in record.get('files', {}).items():
            text = text.replace(f'file {identifier}', f'file {filename}')
        return text

    def tools(pairs):
        return ', '.join(f'{tool} ({status})' for tool, status in pairs) or 'none'

    where = f"live deployment {report['base_url']} (commit `{report.get('commit')}`)" if report['mode'] == 'live' else 'local agent process'
    lines = ['# Observed real-LLM agent evaluation', '',
             f"- Mode: {where}", f"- Provider: OpenAI. Model: `{report['model']}`.",
             f"- ML model version: `{report['model_version']}`.", f"- Run (UTC): {report['run_at_utc']}.",
             f"- Result: {report['passed_count']} of {report['scenario_count']} scenarios passed.", '',
             '| # | Scenario | Request | Expected tools | Tools called | Numbers checked | Result |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for index, record in enumerate(report['records'], 1):
        numbers = record.get('numbers') or {'checked': 0, 'unmatched': []}
        checked = f"{numbers['checked'] - len(numbers['unmatched'])} of {numbers['checked']}"
        lines.append(f"| {index} | {record['scenario']} | {readable(record)} | {tools(record['expected_tools'])} | "
                     f"{tools(record.get('observed_tools', []))} | {checked} | {'PASS' if record['passed'] else 'FAIL'} |")
    failures = [record for record in report['records'] if not record['passed']]
    lines += ['', '## Failures', '']
    lines += [f"- {record['scenario']}: {record['failure']}" for record in failures] or ['None.']
    lines += ['', '## Replies', '']
    for index, record in enumerate(report['records'], 1):
        reply = (record.get('observed') or {}).get('reply', '(no reply)')
        lines += [f"**{index}. {record['scenario']}.** Expected: {record['expected_behavior']}", '',
                  '> ' + reply.replace('\n', '\n> '), '']
    return '\n'.join(lines).rstrip() + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/agent-evaluation')
    parser.add_argument('--base-url', help='Deployed app to evaluate, e.g. https://quantic-malware-agent.onrender.com')
    arguments = parser.parse_args()
    try:
        successful = run(arguments.output, arguments.base_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if successful else 1)

"""Exercise HTTP -> mocked OpenAI -> frozen production model -> download."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from ml_project.agent import Agent
from ml_project.tools import ToolService
from ml_project.web import create_app

ROOT = Path(__file__).resolve().parents[1]


class ScriptedClient:
    def __init__(self):
        self.responses = self
        self.calls = []
        self.next_tool = None
        self.next_tools = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.next_tool or self.next_tools:
            if self.next_tool:
                name, args = self.next_tool
                self.next_tool = None
            else:
                name, args = self.next_tools.pop(0)
            return SimpleNamespace(output=[{'type': 'function_call', 'name': name,
                                           'arguments': json.dumps(args), 'call_id': 'call-test'}], output_text='')
        result = json.loads(kwargs['input'][-1]['output'])
        return SimpleNamespace(output=[], output_text=json.dumps({
            'kind': 'results', 'result_ids': [result['result_id']], 'focus': 'summary',
        }))


@pytest.fixture
def setup(tmp_path):
    service = ToolService(joblib.load(ROOT / 'models/production.joblib'))
    provider = ScriptedClient()
    app = create_app({'TESTING': True, 'SESSION_ROOT': str(tmp_path)},
                     service=service, agent=Agent(service, client=provider))
    client = app.test_client()
    token = client.get('/api/session').json['csrf_token']
    return app, client, provider, {'X-CSRF-Token': token}


def upload(client, headers, sample):
    response = client.post('/api/upload', headers=headers, data={
        'file': (io.BytesIO((ROOT / 'samples' / sample).read_bytes()), sample),
    })
    assert response.status_code == 201
    return response.json['file']['id']


def test_real_model_batch_download_preserves_invalid_row_and_session_isolation(setup):
    app, client, provider, headers = setup
    identifier = upload(client, headers, 'invalid-rows.csv')
    provider.next_tool = ('predict_batch', {'file_id': identifier})
    reply = client.post('/api/chat', headers=headers, json={'message': f'Classify every row of file {identifier}.'})
    assert reply.status_code == 200
    result = reply.json['results'][0]
    assert (result['total_count'], result['valid_count'], result['invalid_count']) == (4, 3, 1)
    csv = client.get(result['download_url'])
    assert csv.status_code == 200 and 'invalid' in csv.text
    assert app.test_client().get(result['download_url']).status_code == 404
    provider_inputs = json.dumps([call['input'] for call in provider.calls])
    assert 'ImportedSymbols' not in provider_inputs
    assert 'kernel32.dll' not in provider_inputs.lower()
    assert str(ROOT) not in provider_inputs


@pytest.mark.parametrize('sample,threshold,expected_status,results_count', [
    ('labeled.csv', 0.0, 'success', 2),
    ('conditional-fail.csv', 1.0, 'skipped', 1),
    ('missing-labels.csv', 0.0, 'skipped', 1),
])
def test_real_model_http_conditional_order_and_coverage(setup, sample, threshold, expected_status, results_count):
    _, client, provider, headers = setup
    evaluation_id = upload(client, headers, sample)
    prediction_id = upload(client, headers, 'single.csv')
    provider.next_tool = ('evaluate_then_predict', {
        'evaluation_file_id': evaluation_id, 'prediction_file_id': prediction_id,
        'row_index': 0, 'min_accuracy': threshold,
    })
    reply = client.post('/api/chat', headers=headers, json={
        'message': f'Evaluate file {evaluation_id}; only if accuracy >= {threshold}, predict row 0 of file {prediction_id}.',
    })
    assert reply.status_code == 200
    assert [(entry['tool'], entry['status']) for entry in reply.json['activity']] == [
        ('evaluate', 'success'), ('predict_single', expected_status),
    ]
    assert len(reply.json['results']) == results_count
    assert len(provider.calls) == 1


def test_http_descriptive_if_request_reaches_saved_model(setup):
    _, client, provider, headers = setup
    identifier = upload(client, headers, 'single.csv')
    provider.next_tool = ('predict_single', {'file_id': identifier, 'row_index': 0})
    reply = client.post('/api/chat', headers=headers, json={
        'message': f'Classify row 0 of file {identifier} and tell me if it is malware',
    })
    assert reply.status_code == 200
    assert 'error' not in reply.json
    assert len(reply.json['results']) == 1
    assert reply.json['activity'][0]['status'] == 'success'
    assert 'the saved ML model predicts' in reply.json['reply']


@pytest.mark.parametrize('sample,name,args,message', [
    ('single.csv', 'predict_single', {'row_index': 10},
     'row_index must identify an existing row (starting at zero)'),
    ('single.csv', 'evaluate', {}, 'Evaluation requires a Label column containing 0 or 1'),
])
def test_http_tool_validation_errors_identify_problem(setup, sample, name, args, message):
    _, client, provider, headers = setup
    identifier = upload(client, headers, sample)
    provider.next_tool = (name, {'file_id': identifier, **args})
    reply = client.post('/api/chat', headers=headers, json={'message': 'Use the uploaded file'})
    assert reply.status_code == 200
    assert reply.json['error'] == 'tool'
    assert reply.json['reply'] == message
    assert reply.json['activity'][0]['error'] == message
    assert len(provider.calls) == 1


def test_http_independent_evaluation_and_batch_complete_both_tasks(setup):
    _, client, provider, headers = setup
    evaluation_id = upload(client, headers, 'labeled.csv')
    prediction_id = upload(client, headers, 'invalid-rows.csv')
    provider.next_tools = [
        ('evaluate', {'file_id': evaluation_id}),
        ('predict_batch', {'file_id': prediction_id}),
    ]
    reply = client.post('/api/chat', headers=headers, json={
        'message': f'Evaluate file {evaluation_id} and classify every row of file {prediction_id}',
    })
    assert reply.status_code == 200
    assert 'error' not in reply.json
    assert [(entry['tool'], entry['status']) for entry in reply.json['activity']] == [
        ('evaluate', 'success'), ('predict_batch', 'success'),
    ]
    assert len(reply.json['results']) == 2
    assert client.get(reply.json['results'][1]['download_url']).status_code == 200


def test_http_fractional_percentage_accepts_provider_canonical_number(setup):
    _, client, provider, headers = setup
    evaluation_id = upload(client, headers, 'labeled.csv')
    prediction_id = upload(client, headers, 'single.csv')
    provider.next_tool = ('evaluate_then_predict', {
        'evaluation_file_id': evaluation_id, 'prediction_file_id': prediction_id,
        'row_index': 0, 'min_accuracy': 0.333,
    })
    reply = client.post('/api/chat', headers=headers, json={
        'message': f'Evaluate file {evaluation_id}; only if accuracy >= 33.3%, predict row 0 of file {prediction_id}',
    })
    assert reply.status_code == 200
    assert 'error' not in reply.json
    assert reply.json['activity'][0]['tool'] == 'evaluate'
    assert len(reply.json['results']) == 2

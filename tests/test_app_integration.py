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

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.next_tool:
            name, args = self.next_tool
            self.next_tool = None
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

"""Stateless multi-tool conversations preserve actual SDK response items."""
import json
from pathlib import Path

import httpx
import joblib
import pytest
from openai import OpenAI

from ml_project.agent import Agent
from ml_project.tools import ToolService

ROOT = Path(__file__).resolve().parents[1]


def test_reasoning_and_assistant_phase_survive_multistep_tool_loop(tmp_path):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        turn = len(requests)
        if turn <= 2:
            output = [
                {'id': f'reason-{turn}', 'type': 'reasoning', 'summary': [],
                 'encrypted_content': f'opaque-state-{turn}'},
                {'id': f'message-{turn}', 'type': 'message', 'role': 'assistant',
                 'phase': 'commentary', 'status': 'completed',
                 'content': [{'type': 'output_text', 'annotations': [],
                              'text': 'UNVERIFIED provider narrative: definitely goodware.'}]},
                {'id': f'call-{turn}', 'type': 'function_call', 'call_id': f'call-{turn}',
                 'name': 'predict_single' if turn == 1 else 'evaluate', 'status': 'completed',
                 'arguments': json.dumps({'file_id': 'local-file', **({'row_index': 0} if turn == 1 else {})})},
            ]
        else:
            output = [{'id': 'final', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'output_text', 'annotations': [], 'text': json.dumps({
                           'kind': 'results', 'result_ids': [], 'focus': 'summary',
                       })}]}]
        return httpx.Response(200, json={
            'id': f'resp-{turn}', 'object': 'response', 'created_at': 1,
            'model': 'reasoning-model', 'status': 'completed', 'output': output,
            'parallel_tool_calls': False, 'error': None, 'incomplete_details': None,
            'instructions': None, 'metadata': {}, 'tools': [], 'tool_choice': 'auto',
        })
    client = OpenAI(api_key='test-key', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    path = tmp_path / 'labeled.csv'
    path.write_bytes((ROOT / 'samples/labeled.csv').read_bytes())
    service = ToolService(joblib.load(ROOT / 'models/production.joblib'))
    result = Agent(service, client, model='reasoning-model').chat(
        'Classify row 0 and evaluate the file.', {},
        {'local-file': {'path': path, 'name': 'labeled.csv'}},
    )
    assert len(requests) == 3
    assert [(x['tool'], x['status']) for x in result['activity']] == [
        ('predict_single', 'success'), ('evaluate', 'success'),
    ]
    assert 'UNVERIFIED' not in result['reply']
    assert 'Accuracy:' in result['reply'] and 'predicts' in result['reply']
    for turn, request in enumerate(requests[1:], start=1):
        assert all(request['store'] is False and request['include'] == ['reasoning.encrypted_content'] for request in requests)
        assert [item['encrypted_content'] for item in request['input'] if item.get('type') == 'reasoning'] == [
            f'opaque-state-{i}' for i in range(1, turn + 1)
        ]
        messages = [item for item in request['input'] if item.get('type') == 'message']
        assert len(messages) == turn and all(item['phase'] == 'commentary' for item in messages)
        assert request['input'][-1]['type'] == 'function_call_output'
    provider_inputs = json.dumps(requests)
    assert 'kernel32.dll' not in provider_inputs.lower()
    assert str(path) not in provider_inputs


@pytest.mark.parametrize('rejection', [
    {'message': 'Unsupported parameter: include', 'param': 'include'},
    {'message': 'Unsupported include value: reasoning.encrypted_content', 'param': None},
])
def test_include_rejection_retries_only_that_option_and_caches_compatibility(tmp_path, rejection):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(400, json={'error': {
                'type': 'invalid_request_error', 'code': 'unsupported_parameter', **rejection,
            }})
        if body['input'][-1].get('type') == 'function_call_output':
            verified = json.loads(body['input'][-1]['output'])
            output = [{'id': f'final-{len(requests)}', 'type': 'message', 'role': 'assistant',
                       'status': 'completed', 'content': [{
                           'type': 'output_text', 'annotations': [], 'text': json.dumps({
                               'kind': 'results', 'result_ids': [verified['result_id']],
                               'focus': 'prediction',
                           }),
                       }]}]
        else:
            output = [{'id': f'call-{len(requests)}', 'type': 'function_call',
                       'call_id': f'call-{len(requests)}', 'name': 'predict_single',
                       'status': 'completed', 'arguments': json.dumps({
                           'file_id': 'local-file', 'row_index': 0,
                       })}]
        return httpx.Response(200, json={
            'id': f'resp-{len(requests)}', 'object': 'response', 'created_at': 1,
            'model': 'gpt-4.1-mini', 'status': 'completed', 'output': output,
        })

    client = OpenAI(api_key='test-key', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    path = tmp_path / 'single.csv'
    path.write_bytes((ROOT / 'samples/single.csv').read_bytes())
    service = ToolService(joblib.load(ROOT / 'models/production.joblib'))
    expected = service.predict_single(path)
    agent = Agent(service, client, model='gpt-4.1-mini')
    files = {'local-file': {'path': path, 'name': 'single.csv'}}
    state = {}
    first = agent.chat('Classify row 0 of the uploaded file.', state, files)

    assert len(requests) == 3
    assert agent.encrypted_reasoning is False
    assert 'error' not in first
    assert [(item['tool'], item['status']) for item in first['activity']] == [
        ('predict_single', 'success'),
    ]
    assert first['results'][0]['prediction'] == expected['prediction']
    assert first['results'][0]['malware_probability'] == expected['malware_probability']
    assert f"predicts {expected['label']}" in first['reply']
    assert requests[0]['include'] == ['reasoning.encrypted_content']
    assert {key: value for key, value in requests[0].items() if key != 'include'} == requests[1]
    assert requests[2]['input'][-1]['type'] == 'function_call_output'

    second = agent.chat('Classify row 0 again.', state, files)
    assert len(requests) == 5
    assert 'error' not in second
    assert second['results'][0]['prediction'] == expected['prediction']
    assert all('include' not in body for body in requests[1:])
    assert all(body['store'] is False and body['parallel_tool_calls'] is False for body in requests)
    assert all(body['text']['format']['strict'] is True for body in requests)
    assert str(path) not in json.dumps(requests)


@pytest.mark.parametrize('status,param,message', [
    (401, 'include', 'Invalid API key'),
    (404, 'model', 'Model is unavailable'),
    (400, 'tools', 'Invalid function tool schema'),
    (400, 'text.format', 'Unsupported structured response format'),
])
def test_unrelated_provider_errors_do_not_retry_without_include(tmp_path, status, param, message):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={'error': {
            'type': 'invalid_request_error', 'message': message, 'param': param,
            'code': 'invalid_request',
        }})

    client = OpenAI(api_key='test-key', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = ToolService(joblib.load(ROOT / 'models/production.joblib'))
    agent = Agent(service, client, model='gpt-4.1-mini')
    result = agent.chat('Classify row 0.', {}, {'local-file': {
        'path': ROOT / 'samples/single.csv', 'name': 'single.csv',
    }})

    assert len(requests) == 1
    assert requests[0]['include'] == ['reasoning.encrypted_content']
    assert requests[0]['store'] is False
    assert requests[0]['text']['format']['strict'] is True
    assert agent.encrypted_reasoning is None
    assert result['error'] == 'provider'
    assert result['activity'] == [] and result['results'] == []
    assert message not in result['reply']

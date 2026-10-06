"""Stateless multi-tool conversations preserve actual SDK response items."""
import json
from pathlib import Path

import httpx
import joblib
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

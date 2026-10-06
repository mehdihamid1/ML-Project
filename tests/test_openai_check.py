"""Provider readiness uses the pinned SDK with a mocked transport, never a key."""
import json

import httpx
from openai import OpenAI
import pytest

from scripts.check_openai import ReadinessError, check


def response_body(output):
    return {'id': 'resp-test', 'object': 'response', 'created_at': 1, 'status': 'completed',
            'model': 'configured-model', 'output': output, 'parallel_tool_calls': False,
            'error': None, 'incomplete_details': None, 'instructions': None,
            'metadata': {}, 'tools': [], 'tool_choice': 'auto'}


def client_for(handler):
    return OpenAI(api_key='test-key', max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_readiness_replays_encrypted_reasoning_and_checks_final_json():
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            output = [
                {'id': 'reasoning-test', 'type': 'reasoning', 'summary': [],
                 'encrypted_content': 'opaque-encrypted-state'},
                {'id': 'call-test', 'type': 'function_call', 'name': 'readiness_probe',
                 'call_id': 'call-test', 'arguments': '{}', 'status': 'completed'},
            ]
        else:
            assert any(item.get('encrypted_content') == 'opaque-encrypted-state' for item in body['input'])
            assert body['input'][-1]['call_id'] == 'call-test'
            output = [{'id': 'message-test', 'type': 'message', 'role': 'assistant',
                       'status': 'completed', 'content': [{'type': 'output_text',
                           'text': '{"ready":true}', 'annotations': []}]}]
        return httpx.Response(200, json=response_body(output))
    result = check(client_for(handler), model='configured-model')
    assert result['status'] == 'ok' and result['model'] == 'configured-model'
    assert len(requests) == 2
    assert all(r['store'] is False and r['include'] == ['reasoning.encrypted_content'] for r in requests)
    assert requests[0]['tool_choice']['name'] == 'readiness_probe'
    assert requests[1]['tool_choice'] == 'none'


def test_readiness_requires_key_without_sending_a_request(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ReadinessError, match='no provider check was run'):
        check()


@pytest.mark.parametrize('status', [401, 404, 429, 500])
def test_readiness_masks_provider_details(status):
    client = client_for(lambda request: httpx.Response(status, json={
        'error': {'message': 'SECRET provider detail', 'type': 'api_error', 'code': 'model_not_found'},
    }))
    with pytest.raises(ReadinessError) as caught:
        check(client)
    assert 'SECRET' not in str(caught.value)
    assert 'model access' in str(caught.value)


def test_readiness_rejects_unsupported_tool_result():
    client = client_for(lambda request: httpx.Response(200, json=response_body([])))
    with pytest.raises(ReadinessError, match='required readiness tool call'):
        check(client)


def test_readiness_rejects_unexpected_final_result():
    requests = []
    def handler(request):
        requests.append(request)
        output = ([{'id': 'call', 'type': 'function_call', 'name': 'readiness_probe',
                    'call_id': 'call', 'arguments': '{}'}] if len(requests) == 1 else
                  [{'id': 'message', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'output_text', 'text': '{"ready":false}', 'annotations': []}]}])
        return httpx.Response(200, json=response_body(output))
    with pytest.raises(ReadinessError, match='required structured readiness result'):
        check(client_for(handler))

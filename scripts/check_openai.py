"""Probe the configured OpenAI model's tool and structured-output support.

This sends a fixed readiness message, never uploaded data or model results.
It is a provider compatibility check, not a real-LLM agent evaluation.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class ReadinessError(Exception):
    """A configuration or compatibility message safe to print."""


def check(client=None, model=None):
    from ml_project.agent import Agent
    model = model or os.environ.get('OPENAI_MODEL') or 'gpt-4.1-mini'
    if client is None:
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ReadinessError('OPENAI_API_KEY is required; no provider check was run.')
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
    inputs = [{'role': 'user', 'content': 'Call readiness_probe, then return the readiness result.'}]
    options = {
        'model': model, 'store': False, 'parallel_tool_calls': False, 'max_output_tokens': 1200,
        'tools': [{'type': 'function', 'name': 'readiness_probe',
                   'description': 'Check readiness without reading or classifying any data.',
                   'strict': True, 'parameters': {'type': 'object', 'properties': {},
                                                 'required': [], 'additionalProperties': False}}],
        'text': {'format': {'type': 'json_schema', 'name': 'readiness', 'strict': True,
                           'schema': {'type': 'object', 'properties': {'ready': {'type': 'boolean'}},
                                      'required': ['ready'], 'additionalProperties': False}}},
    }
    # The app's request path: encrypted reasoning when the model accepts it.
    agent = Agent(service=None, client=client, model=model)
    try:
        response = agent._create(client, **options, input=inputs,
                                 tool_choice={'type': 'function', 'name': 'readiness_probe'})
        calls = [item for item in response.output if item.type == 'function_call']
        if len(calls) != 1 or calls[0].name != 'readiness_probe' or json.loads(calls[0].arguments) != {}:
            raise ReadinessError('The configured model did not return the required readiness tool call.')
        inputs.extend(item.model_dump(exclude_none=True) for item in response.output)
        inputs.append({'type': 'function_call_output', 'call_id': calls[0].call_id,
                       'output': json.dumps({'ready': True})})
        response = agent._create(client, **options, input=inputs, tool_choice='none')
        if any(item.type == 'function_call' for item in response.output) or json.loads(response.output_text) != {'ready': True}:
            raise ReadinessError('The configured model did not return the required structured readiness result.')
    except ReadinessError:
        raise
    except Exception:
        raise ReadinessError('OpenAI readiness check failed. Check the API key, model access, network and Responses API support.') from None
    return {'status': 'ok', 'provider': 'OpenAI', 'model': model,
            'function_calling': True, 'structured_output': True, 'store': False,
            'encrypted_reasoning': agent.encrypted_reasoning}


def main():
    try:
        print(json.dumps(check(), indent=2))
    except ReadinessError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

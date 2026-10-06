"""Trigger Render for the tested commit and verify that commit is serving health."""
import argparse
import json
import os
import time
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from urllib.request import Request, urlopen


def read_health(url):
    with urlopen(url, timeout=20) as response:
        if response.status != 200:
            raise ValueError('Health returned a non-200 response')
        return json.load(response)


def smoke(url, commit, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = read_health(url)
            if result.get('status') == 'ok' and result.get('commit') == commit and result.get('model_version'):
                print(json.dumps({'status': 'ok', 'commit': commit, 'model_version': result['model_version']}))
                return result
        except Exception:
            pass  # Cold starts and deployments can temporarily return errors.
        time.sleep(10)
    raise RuntimeError('Timed out waiting for the tested commit to pass live /health')


def deploy():
    hook = os.environ.get('RENDER_DEPLOY_HOOK')
    health = os.environ.get('RENDER_HEALTH_URL')
    commit = os.environ.get('EXPECTED_COMMIT')
    if not hook or not health or not commit:
        raise ValueError('Set RENDER_DEPLOY_HOOK, RENDER_HEALTH_URL, and EXPECTED_COMMIT')
    parts = urlsplit(hook)
    if parts.scheme != 'https' or parts.hostname != 'api.render.com':
        raise ValueError('Deploy hook must be an HTTPS api.render.com URL')
    if urlsplit(health).scheme != 'https':
        raise ValueError('Live health URL must use HTTPS')
    query = dict(parse_qsl(parts.query))
    query['ref'] = commit
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    try:
        with urlopen(Request(url, method='POST'), timeout=30) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError('Render did not accept deploy request')
    except Exception:
        # URLs and exception strings can include the secret hook; never print them.
        raise RuntimeError('Render deploy request failed; check service configuration') from None
    print('Render accepted deploy request; waiting for live health.')
    return smoke(health, commit)


if __name__ == '__main__':
    deploy()

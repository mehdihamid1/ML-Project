import io
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from scripts import deploy


class Response(io.BytesIO):
    status = 200


def test_deploy_uses_tested_commit_and_never_logs_hook(monkeypatch, capsys):
    monkeypatch.setenv('RENDER_DEPLOY_HOOK', 'https://api.render.com/deploy/srv-test?key=private-hook')
    monkeypatch.setenv('RENDER_HEALTH_URL', 'https://example.onrender.com/health')
    monkeypatch.setenv('EXPECTED_COMMIT', 'tested-commit')
    requests = []

    def opening(request, timeout):
        requests.append(request)
        return Response(b'{}')

    monkeypatch.setattr(deploy, 'urlopen', opening)
    monkeypatch.setattr(deploy, 'smoke', lambda url, commit: {'status': 'ok', 'commit': commit})
    assert deploy.deploy()['commit'] == 'tested-commit'
    query = parse_qs(urlsplit(requests[0].full_url).query)
    assert query['ref'] == ['tested-commit']
    assert requests[0].method == 'POST'
    assert 'private-hook' not in capsys.readouterr().out


def test_smoke_waits_for_new_commit_instead_of_old_healthy_instance(monkeypatch):
    results = iter([
        {'status': 'ok', 'commit': 'old', 'model_version': 'v'},
        {'status': 'ok', 'commit': 'new', 'model_version': 'v'},
    ])
    sleeps = []
    monkeypatch.setattr(deploy, 'read_health', lambda url: next(results))
    monkeypatch.setattr(deploy.time, 'sleep', lambda seconds: sleeps.append(seconds))
    assert deploy.smoke('https://example.test/health', 'new')['commit'] == 'new'
    assert sleeps == [10]


def test_deployment_error_masks_secret_url(monkeypatch):
    monkeypatch.setenv('RENDER_DEPLOY_HOOK', 'https://api.render.com/deploy/srv-test?key=private-hook')
    monkeypatch.setenv('RENDER_HEALTH_URL', 'https://example.onrender.com/health')
    monkeypatch.setenv('EXPECTED_COMMIT', 'commit')

    def failed(request, timeout):
        raise RuntimeError(request.full_url)

    monkeypatch.setattr(deploy, 'urlopen', failed)
    with pytest.raises(RuntimeError) as caught:
        deploy.deploy()
    assert 'private-hook' not in str(caught.value)

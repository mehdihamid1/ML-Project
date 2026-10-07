"""The public comparison uses saved results and never calls an LLM or model."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_project.web import create_app


ROOT = Path(__file__).resolve().parents[1]


class UnusedAgent:
    def chat(self, *_args, **_kwargs):
        raise AssertionError("Viewing model results must not call the chat agent")


def make_app(tmp_path, *, comparison_path=None, model_version=None):
    recorded = json.loads((ROOT / "models/comparison.json").read_text())
    service = SimpleNamespace(metadata={
        "model_version": model_version or recorded["model_version"],
        "selected_model": recorded["selected_model"],
    })
    config = {"TESTING": True, "SESSION_ROOT": str(tmp_path / "sessions")}
    if comparison_path is not None:
        config["COMPARISON_PATH"] = str(comparison_path)
    return create_app(config, service=service, agent=UnusedAgent()), recorded


def test_dashboard_and_download_show_exact_recorded_results_without_sessions(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app, recorded = make_app(tmp_path)
    client = app.test_client()
    response = client.get("/analytics")
    assert response.status_code == 200
    for model in recorded["models"]:
        assert model["model"] in response.text
    assert recorded["model_version"] in response.text
    assert "hold-out" in response.text.lower()
    assert "std" in response.text.lower()
    assert "/static/dashboard.js" in response.text
    assert "/static/app.js" not in response.text
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "Set-Cookie" not in response.headers
    exported = client.get("/api/model-comparison")
    assert exported.status_code == 200
    assert exported.json == recorded
    assert exported.headers["Cache-Control"] == "no-store"
    assert "Set-Cookie" not in exported.headers
    assert app.extensions["session_store"].records == {}
    assert client.get("/health").status_code == 200
    assert client.get("/health").json["openai_configured"] is False


def test_chat_links_to_dashboard_and_dashboard_assets_are_local(tmp_path):
    app, _ = make_app(tmp_path)
    client = app.test_client()
    assert 'href="/analytics"' in client.get("/").text
    for path in ("/static/dashboard.css", "/static/dashboard.js", "/static/favicon.svg"):
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("contents", [None, "invalid JSON", '{"schema_version": 1}'])
def test_unavailable_or_invalid_report_fails_visibly_without_breaking_chat(tmp_path, contents):
    path = tmp_path / "comparison.json"
    if contents is not None:
        path.write_text(contents)
    app, _ = make_app(tmp_path, comparison_path=path)
    client = app.test_client()
    response = client.get("/analytics")
    assert response.status_code == 503
    assert "Recorded model results are unavailable" in response.text
    assert str(tmp_path) not in response.text
    api = client.get("/api/model-comparison")
    assert api.status_code == 503
    assert api.json == {"error": "Recorded model results are unavailable. Please try again later."}
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert app.extensions["session_store"].records == {}


def test_report_from_a_different_production_version_is_not_shown(tmp_path):
    app, _ = make_app(tmp_path, model_version="different-production-model")
    assert app.test_client().get("/analytics").status_code == 503
    assert app.test_client().get("/api/model-comparison").status_code == 503
    assert app.test_client().get("/health").json["model_version"] == "different-production-model"


def test_dashboard_compares_production_with_runner_up_fold_by_fold(tmp_path):
    app, recorded = make_app(tmp_path)
    page = app.test_client().get("/analytics").text
    production, runner_up = recorded["models"][0], recorded["models"][1]
    differences = [mine - theirs for mine, theirs in zip(production["fold_auc"], runner_up["fold_auc"])]
    higher, folds = sum(difference > 0 for difference in differences), recorded["folds"]
    assert f"{production['model']}'s AUC was higher in <strong>{higher} of {folds}</strong> folds" in page
    assert f"{production['model']}'s AUC was higher than {runner_up['model']}'s in {higher} of {folds} folds" in page
    for difference in differences:
        assert f"{difference:+.6f}" in page
    # The production model cannot be compared with itself.
    assert f'<option value="{runner_up["model"]}" selected>' in page
    assert f'<option value="{production["model"]}"' not in page
    assert "Zoom in on differences" not in page

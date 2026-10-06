"""HTTP integration checks using an injected model service and fake agent."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Lock
import time

import pytest

from ml_project.web import create_app


class FakeService:
    metadata = {
        "model_version": "test-model", "selected_model": "LightGBM",
        "features": [{"name": "Size", "dtype": "int64", "allow_missing": False}],
    }


class FakeAgent:
    def __init__(self):
        self.calls = []

    def chat(self, message, state, files):
        self.calls.append((message, files.copy()))
        if message == "fail":
            raise RuntimeError("secret error text must not appear in HTTP responses")
        result = {"tool": "predict_batch", "total_count": 1, "valid_count": 1, "download_id": "result-1"}
        directory = next(iter(files.values()))["path"].parent if files else None
        if directory:
            output = directory / "result.csv"
            output.write_text("row_id,prediction\n0,1\n")
            state["downloads"]["result-1"] = output
            state["results"].append(result)
        activity = {"tool": "predict_batch", "status": "success", "arguments": {}}
        state["activity"].append(activity)
        return {"reply": "The model classified one row.", "activity": [activity], "results": [result]}


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "SECRET_KEY": "unit-test-secret",
                       "SESSION_ROOT": str(tmp_path / "sessions")},
                      service=FakeService(), agent=FakeAgent())


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client):
    return {"X-CSRF-Token": client.get("/api/session").get_json()["csrf_token"]}


def upload(client, content=b"Size,Label,SHA1\n10,1,row-one\n", name="features.csv", headers=None):
    return client.post("/api/upload", data={"file": (BytesIO(content), name)}, headers=headers or csrf(client))


def test_chat_page_and_health_do_not_call_llm(app, client):
    assert b"Analysis assistant" in client.get("/").data
    assert client.get("/static/app.js").status_code == 200
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_version"] == "test-model"
    assert response.get_json()["selected_model"] == "LightGBM"
    assert response.get_json()["commit"]
    assert app.extensions["chat_agent"].calls == []
    assert "Content-Security-Policy" in response.headers


def test_missing_secret_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app({"SESSION_ROOT": str(tmp_path)}, service=FakeService(), agent=FakeAgent())


def test_missing_model_has_unavailable_health(tmp_path):
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path), "MODEL_PATH": str(tmp_path / "missing.joblib")})
    assert app.test_client().get("/health").status_code == 503


def test_csv_upload_keeps_private_path_server_side(app, client):
    response = upload(client)
    assert response.status_code == 201
    details = response.get_json()["file"]
    assert details["rows"] == 1
    assert details["name"] == "features.csv"
    assert "path" not in details
    session_response = client.get("/api/session")
    assert session_response.get_json()["files"][0]["id"] == details["id"]
    assert session_response.headers["Cache-Control"] == "no-store"
    with client.session_transaction() as cookie:
        assert list(cookie) == ["sid"]


@pytest.mark.parametrize("content,name", [
    (b"Size\n1\n", "program.exe"),
    (b"Unexpected\n1\n", "features.csv"),
    (b"Size,Size\n1,2\n", "features.csv"),
    (b"Size\n", "features.csv"),
    (b"Size\n\xff\n", "features.csv"),
    (b'Size\n"unterminated\n', "features.csv"),
])
def test_malformed_uploads_are_rejected(client, content, name):
    response = upload(client, content, name)
    assert response.status_code == 400
    assert response.get_json()["error"]
    assert client.get("/api/session").get_json()["files"] == []


def test_row_validation_is_deferred_to_tools(client):
    response = upload(client, b"Size,Label\nnot-a-number,\n10\n")
    assert response.status_code == 201
    assert response.get_json()["file"]["rows"] == 2


def test_upload_size_row_cell_and_file_limits(app, client):
    app.config["MAX_UPLOAD_BYTES"] = 12
    assert upload(client, b"Size\n123456789\n").status_code == 413
    app.config["MAX_UPLOAD_BYTES"] = 1024
    app.config["MAX_CSV_ROWS"] = 1
    assert upload(client, b"Size\n1\n2\n").status_code == 400
    app.config["MAX_CELL_LENGTH"] = 3
    assert upload(client, b"Size\n1234\n").status_code == 400
    app.config["MAX_FILES_PER_SESSION"] = 1
    assert upload(client, b"Size\n1\n").status_code == 201
    assert upload(client, b"Size\n2\n").status_code == 413


def test_csrf_and_same_origin_are_required(client):
    assert client.post("/api/reset").status_code == 403
    token = csrf(client)
    assert client.post("/api/reset", headers={**token, "Origin": "https://other.example"}).status_code == 403
    assert client.post("/api/reset", headers={**token, "Origin": "http://localhost"}).status_code == 200
    assert client.post("/api/reset", headers={"X-CSRF-Token": "ø"}).status_code == 403


def test_proxy_https_origin_and_cookie_security(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "SESSION_ROOT": str(tmp_path),
                      "TRUST_PROXY": True, "SESSION_COOKIE_SECURE": True},
                     service=FakeService(), agent=FakeAgent())
    client = app.test_client()
    proxy_headers = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "192.0.2.10"}
    response = client.get("/api/session", headers=proxy_headers)
    assert "; Secure;" in response.headers["Set-Cookie"]
    token = response.get_json()["csrf_token"]
    response = client.post("/api/reset", headers={**proxy_headers, "Origin": "https://localhost", "X-CSRF-Token": token})
    assert response.status_code == 200


def test_chat_calls_injected_agent_and_downloads_are_isolated(app, client):
    file = upload(client).get_json()["file"]
    token = csrf(client)
    response = client.post("/api/chat", json={"message": "classify"}, headers=token)
    assert response.status_code == 200
    assert response.get_json()["activity"][0]["status"] == "success"
    assert app.extensions["chat_agent"].calls[0][1][file["id"]]["path"].is_file()
    assert b"row_id,prediction" in client.get("/api/download/result-1").data
    other = app.test_client()
    assert other.get("/api/session").get_json()["files"] == []
    assert other.get("/api/download/result-1").status_code == 404
    assert client.get("/api/session").get_json()["results"]


def test_download_paths_are_confined_to_session(app, client, tmp_path):
    csrf(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    external = tmp_path / "private.csv"
    external.write_text("private")
    app.extensions["session_store"].records[identifier].state["downloads"]["outside"] = external
    assert client.get("/api/download/outside").status_code == 404


def test_reset_removes_only_current_session_files(app, client):
    upload(client)
    other = app.test_client()
    upload(other)
    with client.session_transaction() as cookie:
        directory = app.extensions["session_store"].records[cookie["sid"]].directory
    assert client.post("/api/reset", headers=csrf(client)).status_code == 200
    assert not list(directory.iterdir())
    assert client.get("/api/session").get_json()["files"] == []
    assert len(other.get("/api/session").get_json()["files"]) == 1


def test_chat_failure_does_not_expose_internal_errors_or_drop_uploads(client):
    upload(client)
    response = client.post("/api/chat", json={"message": "fail"}, headers=csrf(client))
    assert response.status_code == 502
    assert b"secret error" not in response.data
    assert client.get("/api/session").get_json()["files"]


@pytest.mark.parametrize("payload", [{}, {"message": 7}, {"message": " "}, {"message": "x" * 4001}])
def test_invalid_chat_requests_are_rejected(client, payload):
    assert client.post("/api/chat", json=payload, headers=csrf(client)).status_code == 400


def test_rate_limits_and_health_exemption(app, client):
    token = csrf(client)
    app.config["CHAT_REQUESTS_PER_MINUTE"] = 1
    assert client.post("/api/chat", json={"message": "hello"}, headers=token).status_code == 200
    assert client.post("/api/chat", json={"message": "again"}, headers=token).status_code == 429
    app.config["REQUESTS_PER_MINUTE"] = 1
    assert client.get("/api/session").status_code == 429
    assert client.get("/health").status_code == 200


def test_session_capacity_and_expired_session_cleanup(app, client):
    upload(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    store = app.extensions["session_store"]
    store.maximum = 1
    old_directory = store.records[identifier].directory
    other = app.test_client()
    assert other.get("/api/session").status_code == 503
    store.records[identifier].touched -= store.ttl + 1
    assert other.get("/api/session").status_code == 200
    assert not old_directory.exists()


def test_per_session_storage_cap_prevents_upload(app, client):
    app.config["MAX_SESSION_BYTES"] = 7
    assert upload(client, b"Size\n12\n").status_code == 413
    assert client.get("/api/session").get_json()["files"] == []


def test_results_storage_overflow_rolls_back_new_files_and_state(tmp_path):
    class LargeOutputAgent(FakeAgent):
        def chat(self, message, state, files):
            response = super().chat(message, state, files)
            state["downloads"]["result-1"].write_bytes(b"x" * 1000)
            return response

    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path), "MAX_SESSION_BYTES": 100},
                     service=FakeService(), agent=LargeOutputAgent())
    client = app.test_client()
    upload(client, b"Size\n1\n")
    response = client.post("/api/chat", json={"message": "classify"}, headers=csrf(client))
    assert response.status_code == 413
    details = client.get("/api/session").get_json()
    assert len(details["files"]) == 1
    assert details["results"] == []
    assert client.get("/api/download/result-1").status_code == 404
    with client.session_transaction() as cookie:
        directory = app.extensions["session_store"].records[cookie["sid"]].directory
    assert len(list(directory.iterdir())) == 1


@pytest.mark.parametrize("overflow", [False, True])
def test_download_retention_preserves_prior_file_on_quota_rollback(tmp_path, overflow):
    class EvictingAgent(FakeAgent):
        def chat(self, message, state, files):
            response = super().chat(message, state, files)
            state["downloads"].pop("prior")
            if overflow:
                state["downloads"]["result-1"].write_bytes(b"x" * 1000)
            return response

    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path), "MAX_SESSION_BYTES": 100},
                     service=FakeService(), agent=EvictingAgent())
    client = app.test_client()
    upload(client, b"Size\n1\n")
    with client.session_transaction() as cookie:
        record = app.extensions["session_store"].records[cookie["sid"]]
    prior = record.directory / "prior.csv"
    prior.write_bytes(b"row_id,prediction\nold,0\n")
    record.state["downloads"]["prior"] = prior
    response = client.post("/api/chat", json={"message": "classify"}, headers=csrf(client))
    assert response.status_code == (413 if overflow else 200)
    assert len(record.files) == 1
    assert next(iter(record.files.values()))["path"].is_file()
    if overflow:
        assert prior.is_file()
        assert b"old,0" in client.get("/api/download/prior").data
        assert client.get("/api/download/result-1").status_code == 404
    else:
        assert not prior.exists()
        assert client.get("/api/download/prior").status_code == 404
        assert client.get("/api/download/result-1").status_code == 200


def test_http_agent_model_batch_flow_with_mocked_openai(tmp_path):
    """Exercise HTTP -> Responses tool call -> CSV/model -> private download."""
    import json
    from types import SimpleNamespace
    import numpy as np
    from ml_project.agent import Agent
    from ml_project.tools import ToolService

    class TestModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.tile([0.1, 0.9], (len(frame), 1))

    class ScriptedClient:
        def __init__(self):
            self.responses = self
            self.script = []

        def create(self, **kwargs):
            return self.script.pop(0)

    metadata = {**FakeService.metadata, "threshold": 0.5, "class_mapping": {"0": "goodware", "1": "malware"}}
    service = ToolService({"pipeline": TestModel(), "metadata": metadata})
    provider = ScriptedClient()
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path)}, service=service,
                     agent=Agent(service, client=provider))
    client = app.test_client()
    file_id = upload(client, b"Size,Label,SHA1\n10,1,valid-row\nbad,0,invalid-row\n").get_json()["file"]["id"]
    provider.script = [
        SimpleNamespace(output=[{"type": "function_call", "call_id": "call-test", "name": "predict_batch",
                                 "arguments": json.dumps({"file_id": file_id})}], output_text=""),
        SimpleNamespace(output=[], output_text=json.dumps({"kind": "results", "result_ids": [], "focus": "summary"})),
    ]
    response = client.post("/api/chat", json={"message": f"Classify all rows in file {file_id}."}, headers=csrf(client))
    assert response.status_code == 200
    result = response.get_json()["results"][0]
    assert (result["valid_count"], result["invalid_count"], result["malware_count"]) == (1, 1, 1)
    download = client.get(result["download_url"])
    assert download.status_code == 200
    assert b"valid-row,valid,1,0.9" in download.data
    assert b"invalid-row,invalid" in download.data


def test_per_session_chat_lock_serializes_concurrent_turns(app, client):
    class TrackingAgent(FakeAgent):
        active = 0
        maximum = 0
        lock = Lock()

        def chat(self, message, state, files):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return {"reply": "ok", "activity": [], "results": []}

    tracking = TrackingAgent()
    # The factory closes over the injected agent, so create a fresh application.
    concurrent_app = create_app({"TESTING": True, "SECRET_KEY": "test", "SESSION_ROOT": app.config["SESSION_ROOT"] + "-concurrent"},
                                service=FakeService(), agent=tracking)
    owner = concurrent_app.test_client()
    token = csrf(owner)
    cookie = owner.get_cookie("session").value

    def request_chat():
        with concurrent_app.test_client() as worker:
            worker.set_cookie("session", cookie)
            return worker.post("/api/chat", json={"message": "hello"}, headers=token).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: request_chat(), range(2))) == [200, 200]
    assert tracking.maximum == 1

"""HTTP integration checks using an injected model service and fake agent."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import re
from threading import Event, Lock
import time

import numpy as np
import pytest

from ml_project.tools import ToolService
from ml_project.web import create_app


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.tile([0.1, 0.9], (len(frame), 1))


class FakeService(ToolService):
    metadata = {
        "model_version": "test-model", "selected_model": "LightGBM",
        "features": [{"name": "Size", "dtype": "int64", "allow_missing": False}],
    }

    def __init__(self):
        super().__init__({"metadata": {**self.metadata, "threshold": 0.5}, "pipeline": FakeModel()})


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
    assert "openai_model" in response.get_json()
    assert app.extensions["chat_agent"].calls == []
    assert "Content-Security-Policy" in response.headers


def test_every_sample_button_names_a_bundled_sample(client):
    names = re.findall(r'data-sample="([^"]+)"', client.get("/").text)
    samples = Path(__file__).resolve().parents[1] / "samples"
    assert sorted(names) == sorted(path.stem for path in samples.glob("*.csv"))


def test_chat_page_groups_side_panels_beside_one_chat(client):
    # The side column scrolls on its own beside the chat; phones reorder it in CSS.
    page = client.get("/").text
    side = page.index('class="side-column"')
    assert side < page.index('class="panel upload-panel"') < page.index('class="panel conditional-panel"') < page.index('class="conversation panel"')
    assert page.count('class="conversation panel"') == 1
    assert 'id="file-input" class="file-input" accept=".csv,text/csv" multiple' in page
    assert '<label class="upload-zone" for="file-input"' in page
    css = client.get("/static/style.css").text
    assert ".side-column{display:contents}" in css and ".conversation{order:2}" in css


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
    (b"Size,Unexpected\n1,value\n", "features.csv"),
    (b" Size\n1\n", "features.csv"),
    (b"Size\n1\x00\n", "features.csv"),
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


def test_upload_size_and_file_limits(tmp_path):
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path),
                      "MAX_UPLOAD_BYTES": 12, "MAX_FILES_PER_SESSION": 1},
                     service=FakeService(), agent=FakeAgent())
    client = app.test_client()
    assert upload(client, b"Size\n123456789\n").status_code == 413
    assert upload(client, b"Size\n1\n").status_code == 201
    assert upload(client, b"Size\n2\n").status_code == 413


def test_upload_uses_shared_inspector_and_configured_limits(tmp_path):
    class InspectingService(FakeService):
        def __init__(self):
            super().__init__()
            self.inspections = 0

        def inspect_csv(self, path):
            self.inspections += 1
            return super().inspect_csv(path)

    service = InspectingService()
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path / "sessions"),
                      "MAX_UPLOAD_BYTES": 100, "MAX_CSV_ROWS": 1, "MAX_CELL_LENGTH": 3},
                     service=service, agent=FakeAgent())
    client = app.test_client()
    assert (service.max_bytes, service.max_rows, service.max_cell_length) == (100, 1, 3)
    assert upload(client, b"Size\n1\n2\n").status_code == 400
    assert service.inspections == 1
    path = tmp_path / "same-input.csv"
    path.write_bytes(b"Size\n1\n2\n")
    with pytest.raises(ValueError, match="row limit"):
        service.predict_batch(path, tmp_path / "output.csv")
    path.write_bytes(b"Size\n" + b"1" * 100)
    with pytest.raises(ValueError, match="byte"):
        service.inspect_csv(path)


def test_oversized_cell_is_uploaded_then_reported_as_invalid_row(tmp_path):
    service = FakeService()
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path / "sessions"), "MAX_CELL_LENGTH": 3},
                     service=service, agent=FakeAgent())
    client = app.test_client()
    response = upload(client, b"Size,Label\n1234,1\n1,1\n")
    assert response.status_code == 201
    assert response.json["file"]["rows"] == 2
    with client.session_transaction() as cookie:
        record = app.extensions["session_store"].records[cookie["sid"]]
    path = next(iter(record.files.values()))["path"]
    output = service.predict_batch(path, record.directory / "result.csv")
    assert (output["total_count"], output["valid_count"], output["invalid_count"]) == (2, 1, 1)
    assert output["invalid_rows"][0]["row_index"] == 0
    assert service.evaluate(path)["invalid_count"] == 1
    with pytest.raises(ValueError, match="character limit"):
        service.predict_single(path, 0)


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


def test_invalid_requests_and_public_pages_do_not_allocate_sessions(app, client):
    store = app.extensions["session_store"]
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/download/unknown").status_code == 404
    assert client.post("/api/reset").status_code == 403
    assert client.post("/api/upload", data={"file": (BytesIO(b"Size\n1\n"), "file.csv")}).status_code == 403
    assert store.records == {}
    assert not list(store.root.iterdir())

    csrf(client)
    with client.session_transaction() as cookie:
        record = store.records[cookie["sid"]]
    before = record.touched
    assert client.post("/api/reset", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert record.touched == before
    assert record.active_requests == 0
    assert len(store.records) == 1


def test_capacity_reclaims_oldest_empty_session_with_new_private_identifiers(app, client):
    store = app.extensions["session_store"]
    store.maximum = 2
    old_token = csrf(client)
    with client.session_transaction() as cookie:
        first_id = cookie["sid"]
    first = store.records[first_id]
    second_client = app.test_client()
    csrf(second_client)
    with second_client.session_transaction() as cookie:
        second_id = cookie["sid"]
    first.touched = time.monotonic() - 2
    store.records[second_id].touched = time.monotonic() - 1

    newcomer = app.test_client()
    new_token = csrf(newcomer)
    with newcomer.session_transaction() as cookie:
        new_id = cookie["sid"]
    assert first_id not in store.records
    assert not first.directory.exists()
    assert second_id in store.records
    assert new_id not in {first_id, second_id}
    assert new_token != old_token
    assert len(store.records) == 2
    assert client.post("/api/reset", headers=old_token).status_code == 403
    assert len(store.records) == 2


@pytest.mark.parametrize("protected", ["history", "results", "activity", "downloads", "disk_file"])
def test_capacity_keeps_sessions_with_conversation_or_results(app, client, protected):
    csrf(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    store = app.extensions["session_store"]
    store.maximum = 1
    record = store.records[identifier]
    if protected in {"downloads", "disk_file"}:
        path = record.directory / "result.csv"
        path.write_text("row_id,prediction\nold,0\n")
        if protected == "downloads":
            record.state["downloads"]["saved"] = path
    else:
        record.state[protected].append({"saved": True})
    assert app.test_client().get("/api/session").status_code == 503
    assert store.records[identifier] is record
    assert record.directory.exists()
    if protected == "downloads":
        assert b"old,0" in client.get("/api/download/saved").data


def test_capacity_and_expiry_keep_a_leased_session_before_its_endpoint_lock(app, client):
    csrf(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    store = app.extensions["session_store"]
    store.maximum = 1
    _, record = store.get(identifier)
    try:
        # This is the interval between retrieving the record and taking its lock.
        record.touched -= store.ttl + 1
        assert app.test_client().get("/api/session").status_code == 503
        assert store.records[identifier] is record
        assert record.directory.exists()
    finally:
        store.release(record)
    assert record.active_requests == 0
    assert app.test_client().get("/api/session").status_code == 200
    assert identifier not in store.records


def test_capacity_and_expiry_do_not_remove_a_locked_session(app, client):
    csrf(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    store = app.extensions["session_store"]
    store.maximum = 1
    record = store.records[identifier]
    locked, release = Event(), Event()

    def hold_lock():
        with record.lock:
            locked.set()
            assert release.wait(3)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(hold_lock)
        assert locked.wait(3)
        try:
            record.touched -= store.ttl + 1
            assert app.test_client().get("/api/session").status_code == 503
            assert store.records[identifier] is record
            assert record.directory.exists()
        finally:
            release.set()
        worker.result()
    assert app.test_client().get("/api/session").status_code == 200
    assert identifier not in store.records


def test_request_leases_are_released_after_success_and_failure(app, client):
    token = csrf(client)
    with client.session_transaction() as cookie:
        record = app.extensions["session_store"].records[cookie["sid"]]
    assert client.post("/api/chat", json={"message": "hello"}, headers=token).status_code == 200
    assert record.active_requests == 0
    assert client.post("/api/chat", json={"message": "fail"}, headers=token).status_code == 502
    assert record.active_requests == 0
    assert client.post("/api/chat", json={}, headers=token).status_code == 400
    assert record.active_requests == 0


def test_expired_session_cannot_create_replacement_on_an_old_csrf_write(app, client):
    upload(client)
    token = csrf(client)
    with client.session_transaction() as cookie:
        identifier = cookie["sid"]
    store = app.extensions["session_store"]
    old = store.records[identifier]
    old.touched -= store.ttl + 1
    assert client.post("/api/reset", headers=token).status_code == 403
    assert not old.directory.exists()
    assert store.records == {}
    assert csrf(client) != token
    assert len(store.records) == 1


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


def stream_events(response):
    import json
    return [json.loads(line) for line in response.data.splitlines()]


def test_stream_uses_shared_chat_transaction_and_keeps_downloads(app, client):
    headers = csrf(client)
    upload(client, headers=headers)
    response = client.post("/api/chat/stream", json={"message": "Classify"}, headers=headers)
    assert response.mimetype == "application/x-ndjson"
    assert response.headers["X-Accel-Buffering"] == "no"
    events = stream_events(response)
    assert [event["type"] for event in events] == ["progress", "result"]
    assert events[-1]["response"]["results"][0]["download_id"] == "result-1"
    assert client.get("/api/download/result-1").status_code == 200
    assert len(app.extensions["chat_agent"].calls) == 1
    response.close()
    assert next(iter(app.extensions["session_store"].records.values())).active_requests == 0


def test_stream_actual_started_event_arrives_before_tool_executes(app, client):
    import json
    from types import SimpleNamespace
    from ml_project.agent import Agent
    headers = csrf(client)
    file_id = upload(client, headers=headers).get_json()["file"]["id"]
    called = []
    service = app.extensions["tool_service"]
    original = service.predict_single
    def predict(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)
    service.predict_single = predict
    responses = iter([
        SimpleNamespace(output=[{"type": "function_call", "name": "predict_single", "call_id": "one",
                                 "arguments": json.dumps({"file_id": file_id, "row_index": 0})}]),
        SimpleNamespace(output=[], output_text='{"kind":"results","result_ids":[],"focus":"summary"}')])
    provider = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: next(responses)))
    # The factory closes over this injected object; use its real incremental implementation.
    original_agent = app.extensions["chat_agent"]
    real_agent = Agent(service, provider)
    original_agent.chat_events = real_agent.chat_events
    response = client.post("/api/chat/stream", json={"message": "Predict row 0"}, headers=headers, buffered=False)
    iterator = iter(response.response)
    trail = []
    while not trail:
        event = json.loads(next(iterator))
        if event["type"] == "tool":
            trail.append(event)
    assert trail[0]["activity"]["status"] == "started"
    assert not called
    assert json.loads(next(iterator))["activity"]["status"] == "success"
    assert called
    rest = [json.loads(chunk) for chunk in iterator]
    assert rest[-1]["type"] == "result"
    assert client.get("/api/session").get_json()["results"][0]["prediction"] == 1
    response.close()


@pytest.mark.parametrize("cancel_stage", ["accepted", "modified", "complete"])
def test_stream_disconnect_releases_lease_and_rolls_back_unfinished_turn(app, client, cancel_stage):
    import json
    headers = csrf(client)
    upload(client, headers=headers)
    record = next(iter(app.extensions["session_store"].records.values()))
    before = record.state.copy()
    made = record.directory / "unfinished.csv"
    def events(message, state, files):
        state["history"] = [{"role": "user", "content": "unfinished"}]
        made.write_text("unfinished")
        yield {"type": "progress", "stage": "modified"}
        return {"reply": "Done", "activity": [], "results": []}
    app.extensions["chat_agent"].chat_events = events
    response = client.post("/api/chat/stream", json={"message": "Predict"}, headers=headers, buffered=False)
    iterator = iter(response.response)
    assert json.loads(next(iterator))["stage"] == "accepted"
    if cancel_stage != "accepted":
        assert json.loads(next(iterator))["stage"] == "modified"
    if cancel_stage == "complete":
        assert json.loads(next(iterator))["type"] == "result"
    response.close()
    assert record.active_requests == 0
    def can_lock():
        acquired = record.lock.acquire(blocking=False)
        if acquired:
            record.lock.release()
        return acquired
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(can_lock).result(timeout=2)
    if cancel_stage == "complete":
        assert made.exists() and record.state != before
    else:
        assert not made.exists() and record.state == before


def test_stream_close_without_iteration_releases_retained_session(app):
    # Directly close the view's Response without ever starting its generator.
    client = app.test_client()
    headers = csrf(client)
    cookie = client.get_cookie("session").value
    with app.test_request_context("/api/chat/stream", method="POST", json={"message": "Predict"},
                                  headers={**headers, "Cookie": f"session={cookie}"}):
        assert app.preprocess_request() is None
        response = app.view_functions["chat_stream"]()
        record = next(iter(app.extensions["session_store"].records.values()))
        assert record.active_requests == 2
        response.close()
        assert record.active_requests == 1
    assert record.active_requests == 0


def test_stream_quota_failure_withholds_results_and_restores_files(app, client):
    headers = csrf(client)
    upload(client, headers=headers)
    record = next(iter(app.extensions["session_store"].records.values()))
    before_paths = set(record.directory.iterdir())
    before_state = deepcopy(record.state)
    app.config["MAX_SESSION_BYTES"] = sum(path.stat().st_size for path in before_paths) + 1
    response = client.post("/api/chat/stream", json={"message": "Classify"}, headers=headers)
    events = stream_events(response)
    assert events[-1]["type"] == "error" and events[-1]["status"] == 413
    assert not any(event["type"] == "result" for event in events)
    assert set(record.directory.iterdir()) == before_paths
    assert record.state == before_state
    response.close()


def test_stream_failure_after_headers_is_sanitized(app, client):
    headers = csrf(client)
    response = client.post("/api/chat/stream", json={"message": "fail"}, headers=headers)
    events = stream_events(response)
    assert events[-1]["type"] == "error" and events[-1]["status"] == 502
    assert "secret error text" not in response.text
    response.close()


@pytest.mark.parametrize("payload,status", [({}, 400), ({"message": ""}, 400), ({"message": "x" * 4001}, 400)])
def test_stream_rejects_bad_requests_before_streaming(client, payload, status):
    response = client.post("/api/chat/stream", json=payload, headers=csrf(client))
    assert response.status_code == status and response.is_json


def test_stream_csrf_and_chat_limits_are_shared_with_json(app, client):
    headers = csrf(client)
    app.config["CHAT_REQUESTS_PER_MINUTE"] = 1
    assert client.post("/api/chat/stream", json={"message": "help"}).status_code == 403
    first = client.post("/api/chat/stream", json={"message": "help"}, headers=headers)
    assert stream_events(first)[-1]["type"] == "result"
    first.close()
    assert client.post("/api/chat", json={"message": "help"}, headers=headers).status_code == 429


def test_stream_disconnect_rolls_back_and_releases_even_if_agent_cleanup_fails(app, client):
    headers = csrf(client)
    upload(client, headers=headers)
    record = next(iter(app.extensions["session_store"].records.values()))
    before = deepcopy(record.state)
    path = record.directory / "partial.csv"
    def events(message, state, files):
        try:
            path.write_text("partial")
            state["history"] = ["partial"]
            yield {"type": "progress", "stage": "partial"}
        finally:
            raise RuntimeError("private cleanup error")
    app.extensions["chat_agent"].chat_events = events
    response = client.post("/api/chat/stream", json={"message": "Predict"}, headers=headers, buffered=False)
    iterator = iter(response.response)
    next(iterator)  # Accepted.
    next(iterator)  # Partial state created.
    with pytest.raises(RuntimeError, match="private cleanup error"):
        response.close()
    assert record.active_requests == 0 and record.state == before and not path.exists()
    def unlocked():
        if not record.lock.acquire(blocking=False):
            return False
        record.lock.release()
        return True
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(unlocked).result(timeout=2)


def test_eviction_cleanup_failure_does_not_restore_deleted_download_registry(app, client, monkeypatch):
    headers = csrf(client)
    upload(client, headers=headers)
    record = next(iter(app.extensions["session_store"].records.values()))
    old = record.directory / "old.csv"
    old.write_text("old result")
    record.state["downloads"]["old"] = old
    original_chat = app.extensions["chat_agent"].chat
    def chat(message, state, files):
        state["downloads"].pop("old")
        return original_chat(message, state, files)
    app.extensions["chat_agent"].chat = chat
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == old:
            raise PermissionError("cleanup temporarily unavailable")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    response = client.post("/api/chat/stream", json={"message": "Classify"}, headers=headers)
    assert stream_events(response)[-1]["type"] == "result"
    assert "old" not in record.state["downloads"]
    assert client.get("/api/download/result-1").status_code == 200
    response.close()

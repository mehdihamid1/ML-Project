"""Run the scenario harness against real tools and the pinned SDK, without a network.

These are mocked-provider tests. Their temporary reports are never evidence of
a real OpenAI run and are never written into the project's documentation.
"""
import json
from pathlib import Path
import re
import threading

import httpx
import joblib
import pytest
from openai import OpenAI
from werkzeug.serving import make_server

from ml_project.tools import ToolService
from ml_project.web import create_app
from scripts.run_agent_evaluation import run


class ScriptedBackend:
    def __init__(self, mode="success"):
        self.mode = mode
        self.requests = []

    @staticmethod
    def response(output):
        return httpx.Response(200, json={
            "id": "resp_mock", "object": "response", "created_at": 1,
            "status": "completed", "model": "mock-provider",
            "output": output, "parallel_tool_calls": False,
            "error": None, "incomplete_details": None,
        })

    def final(self, kind="results", ids=None, focus="summary"):
        return self.response([{
            "type": "message", "id": "msg_mock", "status": "completed",
            "role": "assistant", "content": [{"type": "output_text", "annotations": [],
                "text": json.dumps({"kind": kind, "result_ids": ids or [], "focus": focus})}],
        }])

    def function(self, name, args):
        return self.response([{
            "type": "function_call", "id": "fc_mock", "status": "completed",
            "call_id": "call_mock", "name": name, "arguments": json.dumps(args),
        }])

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.mode == "provider_failure":
            return httpx.Response(503, json={"error": {
                "message": "PRIVATE_PROVIDER_ERROR_TOKEN", "type": "server_error", "code": "test",
            }})
        inputs = body["input"]
        prompt = [item["content"] for item in inputs if item.get("role") == "user"][-1]
        brief = [item["content"] for item in inputs if item.get("role") == "developer" and "Conditional request" in item["content"]]
        if inputs[-1].get("type") == "function_call_output":
            result = json.loads(inputs[-1]["output"])
            if brief and result["tool"] == "evaluate":
                threshold = float(re.search(r"minimum accuracy is ([0-9.e-]+)", brief[-1]).group(1))
                target = re.search(r"predict row (\d+) of file (\w+)", prompt, re.I)
                complete = (result["evaluated_count"] == result["total_count"] and not any(
                    result[key] for key in ("invalid_count", "missing_label_count", "invalid_label_count")))
                # A correct model reads the returned accuracy before deciding; a misrouted one predicts anyway.
                if self.mode == "misrouted_conditional" or (complete and result["accuracy"] >= threshold):
                    return self.function("predict_single", {"file_id": target.group(2), "row_index": int(target.group(1))})
            return self.final(ids=[result["result_id"]])
        if body.get("tool_choice") != "auto":
            # The server forces evaluate first for a conditional request.
            return self.function("evaluate", {"file_id": re.search(r"(?:evaluate file|accuracy of file) (\w+)", prompt, re.I).group(1)})
        if "false negatives" in prompt:
            context = json.loads(inputs[0]["content"].partition(": ")[2])
            result_id = context["session_results"][-1]["result_id"] if context["session_results"] else "no-result"
            if self.mode == "stale_followup":
                result_id = "different-session-result"
            return self.final(ids=[result_id], focus="false_negatives")
        if "individual features" in prompt:
            return self.final(kind="explanation_unavailable")
        file_id = re.search(r"file (\w+)", prompt).group(1)
        if prompt.startswith("Evaluate"):
            return self.function("evaluate", {"file_id": file_id})
        if "every row" in prompt:
            return self.function("predict_batch", {"file_id": file_id})
        return self.function("predict_single", {"file_id": file_id, "row_index": int(re.search(r"row (\d+)", prompt).group(1))})


@pytest.fixture
def backend_factory(monkeypatch):
    clients = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "mock-provider")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    def install(mode="success"):
        backend = ScriptedBackend(mode)

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(backend.handle)))
            clients.append(client)
            return client

        monkeypatch.setattr("openai.OpenAI", factory)
        return backend

    yield install
    for client in clients:
        client.close()


def read_report(path):
    return json.loads((path / "results.json").read_text())


def record(report, scenario):
    return next(item for item in report["records"] if item["scenario"] == scenario)


def test_runner_exercises_sdk_function_calls_structured_responses_and_real_tools(tmp_path, backend_factory):
    backend = backend_factory()
    output = tmp_path / "report"
    assert run(output) is True
    report = read_report(output)
    assert report["scenario_count"] == len(report["records"]) == 15
    assert report["passed_count"] == 15
    assert all(item["passed"] for item in report["records"])
    assert (output / "report.md").is_file()
    assert "15 of 15 scenarios passed" in (output / "report.md").read_text()
    # Each reply's six-decimal numbers were compared with the tool output.
    assert sum(item["numbers"]["checked"] for item in report["records"]) > 5
    assert record(report, "Conditional: prediction withheld")["observed_tools"] == [["evaluate", "success"], ["predict_single", "skipped"]]
    # These bodies went through OpenAI 2.x serialization and typed responses,
    # rather than a stand-in client exposing a similarly named method.
    for request in backend.requests:
        assert request["store"] is False and request["parallel_tool_calls"] is False
        assert request["text"]["format"]["type"] == "json_schema"
        assert request["text"]["format"]["strict"] is True
        for function in request["tools"]:
            schema = function["parameters"]
            assert function["strict"] is True
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
    assert any(request["input"][-1].get("type") == "function_call_output" for request in backend.requests)
    assert str(Path("samples").resolve()) not in str(backend.requests)


def test_runner_marks_stale_followup_failed_even_when_other_scenarios_pass(tmp_path, backend_factory):
    backend_factory("stale_followup")
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    assert report["passed_count"] == 14
    followup = record(report, "False-negative follow-up")
    assert not followup["passed"]
    assert "unavailable in this session" in followup["observed"]["reply"]


def test_runner_records_a_model_that_predicts_despite_the_condition(tmp_path, backend_factory):
    backend_factory("misrouted_conditional")
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    assert report["scenario_count"] == 15 and report["passed_count"] == 13
    withheld = record(report, "Conditional: prediction withheld")
    assert withheld["observed_tools"] == [["evaluate", "success"], ["predict_single", "blocked"]]
    assert not withheld["passed"] and "differ from the expected tools" in withheld["failure"]
    assert record(report, "Earlier accuracy figure ignored")["observed_tools"][-1] == ["predict_single", "blocked"]
    assert "FAIL" in (tmp_path / "report" / "report.md").read_text()


def test_runner_preserves_failure_report_on_provider_outage(tmp_path, backend_factory):
    backend_factory("provider_failure")
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    # Only the clarification scenario passes: the server answers it before any provider call.
    assert report["scenario_count"] == 15 and report["passed_count"] == 1
    assert record(report, "Ambiguous condition")["passed"]
    assert all(item["observed"]["error"] == "provider" for item in report["records"] if item["scenario"] != "Ambiguous condition")
    assert "PRIVATE_PROVIDER_ERROR_TOKEN" not in json.dumps(report)


def test_runner_rejects_invalid_production_class_order(tmp_path, backend_factory, monkeypatch):
    backend_factory()
    bundle = joblib.load("models/production.joblib")

    class UnknownClasses:
        classes_ = [0, 2]

        def predict_proba(self, frame):
            pytest.fail("Unexpected classes must be rejected before scoring")

    bundle["pipeline"] = UnknownClasses()
    monkeypatch.setattr(joblib, "load", lambda path: bundle)
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    single = report["records"][0]
    assert not single["passed"]
    assert single["observed"]["error"] == "tool"
    assert single["observed"]["results"] == []


def test_live_mode_drives_the_http_app_like_a_browser(tmp_path, backend_factory):
    backend_factory()
    service = ToolService(joblib.load("models/production.joblib"))
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path / "sessions"),
                      "REQUESTS_PER_MINUTE": 1000, "CHAT_REQUESTS_PER_MINUTE": 100}, service=service)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert run(tmp_path / "report", base_url=f"http://127.0.0.1:{server.server_port}/") is True
    finally:
        server.shutdown()
    report = read_report(tmp_path / "report")
    assert report["mode"] == "live" and report["model"] == "mock-provider"
    assert report["passed_count"] == report["scenario_count"] == 15
    # Requests used the server's own upload IDs; the report names the sample files instead.
    identifiers = [value for item in report["records"] for value in item.get("files", {}).values()]
    assert identifiers and all(re.fullmatch(r"[0-9a-f]{32}", value) for value in identifiers)
    text = (tmp_path / "report" / "report.md").read_text()
    assert "live deployment http://127.0.0.1" in text and "file single.csv" in text
    assert not re.search(r"[0-9a-f]{32}", text.split("## Replies")[0])


def test_live_mode_requires_a_configured_server_key(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_agent_evaluation.wake", lambda base: {"openai_configured": False})
    with pytest.raises(ValueError, match="no real-LLM evaluation was run"):
        run(tmp_path / "report", base_url="https://example.invalid")
    assert not (tmp_path / "report").exists()


def test_runner_requires_key_before_any_provider_or_model_work(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no real-LLM evaluation was run"):
        run(tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_runner_preserves_existing_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    output = tmp_path / "report"
    output.mkdir()
    previous = output / "results.json"
    previous.write_text("old report")
    with pytest.raises(ValueError, match="empty evaluation output directory"):
        run(output)
    assert previous.read_text() == "old report"

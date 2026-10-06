"""Run the scenario harness against real tools and the pinned SDK, without a network.

These are mocked-provider tests. Their temporary reports are never evidence of
a real OpenAI run and are never written into the project's documentation.
"""
import json
from pathlib import Path
import re

import httpx
import joblib
import pytest
from openai import OpenAI

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
        if inputs[-1].get("type") == "function_call_output":
            result = json.loads(inputs[-1]["output"])
            return self.final(ids=[result["result_id"]])
        prompt = inputs[-1]["content"]
        if "What was the accuracy" in prompt:
            context = json.loads(inputs[0]["content"].partition(": ")[2])
            result_id = context["session_results"][-1]["result_id"] if context["session_results"] else "no-result"
            if self.mode == "stale_followup":
                result_id = "different-session-result"
            return self.final(ids=[result_id], focus="accuracy")
        if "individual features" in prompt:
            return self.final(kind="explanation_unavailable")
        if "only if" in prompt:
            return self.function("evaluate_then_predict", {
                "evaluation_file_id": "fail" if "file fail," in prompt else "labeled",
                "prediction_file_id": "single", "row_index": 0,
                "min_accuracy": 1 if "at least 1" in prompt else 0,
            })
        if prompt.startswith("Evaluate"):
            if self.mode == "misrouted_followup_initial" and prompt == "Evaluate file labeled.":
                return self.function("predict_single", {"file_id": "single", "row_index": 0})
            file_id = re.search(r"file (\w+)", prompt).group(1)
            return self.function("evaluate", {"file_id": file_id})
        if "every row" in prompt:
            file_id = re.search(r"file (\w+)", prompt).group(1)
            return self.function("predict_batch", {"file_id": file_id})
        return self.function("predict_single", {"file_id": "single", "row_index": 0})


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


def test_runner_exercises_sdk_function_calls_structured_responses_and_real_tools(tmp_path, backend_factory):
    backend = backend_factory()
    output = tmp_path / "report"
    assert run(output) is True
    report = read_report(output)
    assert report["scenario_count"] == len(report["records"]) == 11
    assert report["passed_count"] == 11
    assert all(record["passed"] for record in report["records"])
    assert (output / "report.md").is_file()
    assert "Passed 11 of 11" in (output / "report.md").read_text()
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
    assert report["passed_count"] == 10
    followup = report["records"][-1]
    assert followup["scenario"] == "Session follow-up" and not followup["passed"]
    assert "unavailable in this session" in followup["observed"]["reply"]


def test_runner_reports_misrouted_initial_evaluation_without_crashing(tmp_path, backend_factory):
    backend_factory("misrouted_followup_initial")
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    assert report["scenario_count"] == 11
    assert report["passed_count"] == 10
    followup = report["records"][-1]
    assert followup["initial"]["results"][0]["tool"] == "predict_single"
    assert not followup["passed"]


def test_runner_preserves_failure_report_on_provider_outage(tmp_path, backend_factory):
    backend_factory("provider_failure")
    assert run(tmp_path / "report") is False
    report = read_report(tmp_path / "report")
    assert report["scenario_count"] == 11 and report["passed_count"] == 0
    assert all(record["observed"]["error"] == "provider" for record in report["records"])
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

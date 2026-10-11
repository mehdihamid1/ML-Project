"""The Workflow tab: real limits on the page, and each turn's OpenAI calls recorded without CSV rows."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_project.agent import SYSTEM_INSTRUCTIONS, TOOLS, Agent
from ml_project.web import create_app

ROOT = Path(__file__).resolve().parents[1]


def tool(name, **args):
    return SimpleNamespace(output=[{"type": "function_call", "call_id": "call-test", "name": name,
                                    "arguments": json.dumps(args)}], output_text="")


def final(kind="results", result_ids=None, focus="summary"):
    return SimpleNamespace(output=[], output_text=json.dumps({"kind": kind, "result_ids": result_ids or [], "focus": focus}))


class FakeClient:
    def __init__(self, *responses):
        self.responses = self
        self.script = list(responses)

    def create(self, **kwargs):
        value = self.script.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeService:
    def __init__(self, accuracy=0.9, **coverage):
        self.evaluation = {
            "accuracy": accuracy, "auc": 0.96, "auc_reason": None,
            "confusion_matrix": [[4, 1], [0, 5]], "confusion_matrix_order": [0, 1],
            "class_counts": {"0": 5, "1": 5}, "total_count": 10, "evaluated_count": 10, "valid_count": 10,
            "invalid_count": 0, "missing_label_count": 0, "invalid_label_count": 0,
            "evaluation_coverage": 1, "threshold": 0.5, "model_version": "test", **coverage,
        }

    def predict_single(self, path, row_index=0):
        return {"row_id": "sha-not-a-feature", "row_index": row_index, "prediction": 1, "label": "malware",
                "malware_probability": 0.87, "threshold": 0.5, "model_version": "test",
                "raw_features": "DO_NOT_SEND_ROW_FEATURES"}

    def evaluate(self, path):
        return dict(self.evaluation)


@pytest.fixture
def files(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("CSV_CONTENT_MUST_STAY_LOCAL")
    return {"file-a": {"path": path, "name": "input.csv"}, "file-b": {"path": path, "name": "predict.csv"}}


def test_workflow_page_shows_the_configured_limits_and_real_texts(tmp_path):
    agent = Agent(FakeService(), client=FakeClient(), model="test-model")
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path), "MAX_TOOL_CALLS": 4,
                      "MAX_CSV_ROWS": 1234, "CHAT_REQUESTS_PER_MINUTE": 7}, service=FakeService(), agent=agent)
    page = app.test_client().get("/workflow")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    for expected in ("At most 4 tool calls per request", "1 to 1,234 rows", "at most 7 chat requests",
                     "AI model: test-model", '<a href="/workflow" aria-current="page">Workflow</a>',
                     "/static/workflow.js", "store=false", "30-second timeout"):
        assert expected in text
    assert SYSTEM_INSTRUCTIONS.splitlines()[0] in text
    for function in TOOLS:
        assert f"<code>{function['name']}</code>: {function['description']}" in text


def test_every_page_offers_the_workflow_tab(tmp_path):
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path)}, service=FakeService(),
                     agent=Agent(FakeService(), client=FakeClient()))
    client = app.test_client()
    for path in ("/", "/analytics", "/workflow"):
        assert '<a href="/workflow"' in client.get(path).get_data(as_text=True), path


def test_the_page_script_never_inserts_server_text_as_html():
    script = (ROOT / "static/workflow.js").read_text()
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script


def test_trace_records_each_openai_call_without_csv_rows(files):
    state = {}
    Agent(FakeService(), FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())).chat(
        "Predict row 0 of file-a", state, files)
    trace = state["last_trace"]
    first, second = trace["llm_calls"]
    assert [first["call"], second["call"]] == [1, 2]
    assert first["tools"] == second["tools"] == ["predict_single", "predict_batch", "evaluate"]
    assert first["tool_choice"] == second["tool_choice"] == "auto"
    assert first["returned"]["type"] == "function_call" and first["returned"]["name"] == "predict_single"
    assert second["returned"] == {"type": "answer", "kind": "results", "result_ids": [], "focus": "summary"}
    assert {"part": "message", "text": "Predict row 0 of file-a"} in first["sent"]
    output = next(part["output"] for part in second["sent"] if part["part"] == "tool_output")
    assert output["malware_probability"] == 0.87 and "raw_features" not in output
    assert [(entry["tool"], entry["status"]) for entry in trace["activity"]] == [("predict_single", "success")]
    recorded = json.dumps(trace)
    for private in ("DO_NOT_SEND_ROW_FEATURES", "CSV_CONTENT_MUST_STAY_LOCAL", str(files["file-a"]["path"])):
        assert private not in recorded
    assert "sha-not-a-feature" not in json.dumps(trace["llm_calls"])


def test_trace_shows_the_conditional_steps_and_the_ai_decision(files):
    state = {}
    service = FakeService(accuracy=0, confusion_matrix=[[0, 0], [1, 0]])
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat(
        "Evaluate file-a; predict row 0 of file-b only if accuracy >= 0.9", state, files)
    trace = state["last_trace"]
    first, second = trace["llm_calls"]
    assert (first["tools"], first["tool_choice"]) == (["evaluate"], "evaluate")
    assert (second["tools"], second["tool_choice"]) == (["predict_single"], "auto")
    assert any(part["part"] == "brief" and "minimum accuracy is 0.9" in part["text"] for part in first["sent"])
    assert next(part["output"] for part in second["sent"] if part["part"] == "tool_output")["accuracy"] == 0
    assert second["returned"]["type"] == "answer"
    assert [(entry["tool"], entry["status"]) for entry in trace["activity"]] == [("evaluate", "success"), ("predict_single", "skipped")]


def test_trace_of_a_stored_result_question_offers_no_tools(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate file-a", state, files)
    Agent(service, FakeClient(final(focus="false_negatives"))).chat("How many false negatives were there?", state, files)
    (call,) = state["last_trace"]["llm_calls"]
    assert (call["tools"], call["tool_choice"]) == ([], "none")
    assert next(part for part in call["sent"] if part["part"] == "context")["results"] == ["evaluate"]


def test_trace_of_a_request_stopped_before_the_ai_has_no_calls(files):
    state = {}
    output = Agent(FakeService(), FakeClient()).chat(
        "Predict row 0 of file-b only if the accuracy is good enough.", state, files)
    assert state["last_trace"]["llm_calls"] == []
    assert state["last_trace"]["reply"] == output["reply"] and "accuracy threshold" in output["reply"]


def test_trace_never_keeps_pasted_csv_rows(files):
    state = {}
    pasted = "Classify these rows:\nMachine,SizeOfCode,Label\n332,1024,1\n332,2048,0\n332,4096,1"
    output = Agent(FakeService(), FakeClient()).chat(pasted, state, files)
    assert output["error"] == "input"
    assert state["last_trace"]["message"] == "[CSV paste rejected]" and state["last_trace"]["llm_calls"] == []
    assert "SizeOfCode" not in json.dumps(state["last_trace"])


def test_trace_marks_a_failed_openai_call(files):
    state = {}
    output = Agent(FakeService(), FakeClient(RuntimeError("provider down"))).chat("Predict row 0 of file-a", state, files)
    assert output["error"] == "provider"
    assert state["last_trace"]["llm_calls"][0]["returned"] == {"type": "error"}
    assert state["last_trace"]["error"] == "provider"


def test_session_api_returns_the_last_trace_only_to_its_own_session(tmp_path):
    agent = Agent(FakeService(), client=FakeClient(final(kind="help")), model="test-model")
    app = create_app({"TESTING": True, "SESSION_ROOT": str(tmp_path)}, service=FakeService(), agent=agent)
    client, other = app.test_client(), app.test_client()
    token = client.get("/api/session").get_json()["csrf_token"]
    assert client.get("/api/session").get_json()["last_trace"] is None
    response = client.post("/api/chat", headers={"X-CSRF-Token": token}, json={"message": "What can you do?"})
    assert response.status_code == 200
    trace = client.get("/api/session").get_json()["last_trace"]
    assert trace["message"] == "What can you do?" and trace["model"] == "test-model"
    assert [call["returned"]["kind"] for call in trace["llm_calls"]] == ["help"]
    assert other.get("/api/session").get_json()["last_trace"] is None

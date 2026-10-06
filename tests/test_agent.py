"""Adversarial mocked-provider tests exercise routing and evidence boundaries."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_project.agent import Agent


def tool(name, **args):
    return SimpleNamespace(output=[{
        "type": "function_call", "call_id": "call-test", "name": name,
        "arguments": json.dumps(args),
    }], output_text="")


def final(kind="results", result_ids=None, focus="summary"):
    return SimpleNamespace(output=[], output_text=json.dumps({
        "kind": kind, "result_ids": result_ids or [], "focus": focus,
    }))


class FakeClient:
    def __init__(self, *responses):
        self.responses = self
        self.script = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        value = self.script.pop(0)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(kwargs)
        return value


class FakeService:
    def __init__(self, accuracy=0.9, **coverage):
        self.calls = []
        self.evaluation = {
            "accuracy": accuracy, "auc": 0.96, "auc_reason": None,
            "confusion_matrix": [[4, 1], [0, 5]], "confusion_matrix_order": [0, 1],
            "class_counts": {"0": 5, "1": 5},
            "total_count": 10, "evaluated_count": 10, "valid_count": 10,
            "invalid_count": 0, "missing_label_count": 0, "invalid_label_count": 0,
            "evaluation_coverage": 1, "threshold": 0.5, "model_version": "test",
            **coverage,
        }

    def predict_single(self, path, row_index=0):
        self.calls.append(("predict_single", Path(path), row_index))
        return {"row_id": "sha-not-a-feature", "row_index": row_index, "prediction": 1,
                "label": "malware", "malware_probability": 0.87, "threshold": 0.5,
                "model_version": "test", "raw_features": "DO_NOT_SEND_ROW_FEATURES"}

    def predict_batch(self, path, output_path):
        self.calls.append(("predict_batch", Path(path)))
        Path(output_path).write_text("row_index,prediction\n0,1\n")
        return {"total_count": 2, "valid_count": 1, "invalid_count": 1,
                "malware_count": 1, "goodware_count": 0, "results_path": str(output_path),
                "invalid_rows": [{"row_index": 1, "errors": ["Missing Size"]}]}

    def evaluate(self, path):
        self.calls.append(("evaluate", Path(path)))
        return dict(self.evaluation)


@pytest.fixture
def files(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("CSV_CONTENT_MUST_STAY_LOCAL")
    return {"file-a": {"path": path, "name": "input.csv"},
            "file-b": {"path": path, "name": "predict.csv"}}


def test_single_is_tool_output_not_provider_prose(files):
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0),
                        SimpleNamespace(output=[], output_text="This file is goodware with 100% certainty."))
    service, state = FakeService(), {}
    response = Agent(service, client).chat("Classify row 0", state, files)
    assert "predicts malware" in response["reply"]
    assert "0.870000" in response["reply"]
    assert "100%" not in response["reply"]
    assert len(service.calls) == 1
    assert response["activity"][0]["status"] == "success"
    requests = str(client.requests)
    assert "CSV_CONTENT_MUST_STAY_LOCAL" not in requests
    assert "DO_NOT_SEND_ROW_FEATURES" not in requests
    assert "sha-not-a-feature" not in requests
    assert str(files["file-a"]["path"]) not in requests
    assert all(not r["store"] and not r["parallel_tool_calls"] for r in client.requests)


def test_no_tool_no_classification_can_be_invented(files):
    service = FakeService()
    response = Agent(service, FakeClient(SimpleNamespace(output=[], output_text="Definitely malware; AUC 1.0"))).chat("Please classify", {}, files)
    assert "Definitely" not in response["reply"] and "AUC 1.0" not in response["reply"]
    assert "No model result" in response["reply"]
    assert service.calls == []


def test_batch_registers_download_and_reports_invalid_rows(files):
    service, state = FakeService(), {}
    response = Agent(service, FakeClient(tool("predict_batch", file_id="file-a"), final())).chat("Classify all rows", state, files)
    result = response["results"][0]
    assert result["invalid_count"] == 1
    assert result["invalid_rows"][0]["row_index"] == 1
    assert "1 invalid rows" in response["reply"]
    assert state["downloads"][result["download_id"]].is_file()
    assert result["download_url"].startswith("/api/download/")
    assert "results_path" not in result
    assert "Missing Size" not in str(Agent._summary(result))


@pytest.mark.parametrize("accuracy,predicts", [(0.89, False), (0.9, True), (0.91, True)])
def test_conditional_uses_evaluation_first_and_inclusive_gate(files, accuracy, predicts):
    service = FakeService(accuracy)
    client = FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.9))
    response = Agent(service, client).chat("Evaluate file-a; only if accuracy >= 0.9 predict row 0 of file-b", {}, files)
    assert [c[0] for c in service.calls] == (["evaluate", "predict_single"] if predicts else ["evaluate"])
    assert [a["tool"] for a in response["activity"]] == ["evaluate", "predict_single"]
    assert response["activity"][-1]["status"] == ("success" if predicts else "skipped")
    assert ("Prediction skipped" in response["reply"]) != predicts
    assert client.requests[0]["tool_choice"]["name"] == "evaluate_then_predict"
    assert [t["name"] for t in client.requests[0]["tools"]] == ["evaluate_then_predict"]


@pytest.mark.parametrize("coverage", [
    {"missing_label_count": 1, "evaluated_count": 9, "evaluation_coverage": 0.9},
    {"invalid_count": 1}, {"invalid_label_count": 1},
    {"evaluation_coverage": None}, {"accuracy": None},
])
def test_conditional_fails_closed_for_incomplete_or_undefined_results(files, coverage):
    service = FakeService(**coverage)
    response = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.8))).chat("Evaluate and predict if accuracy >= 0.8", {}, files)
    assert [c[0] for c in service.calls] == ["evaluate"]
    assert "Prediction skipped" in response["reply"]


def test_model_cannot_bypass_conditional_or_lower_user_threshold(files):
    for selected in [
        tool("predict_single", file_id="file-b", row_index=0),
        tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.1),
    ]:
        service = FakeService()
        response = Agent(service, FakeClient(selected)).chat("Only predict if accuracy >= 0.95", {}, files)
        assert response["error"] == "tool"
        assert service.calls == []


def test_conditional_cannot_answer_with_stale_prediction_without_evaluating(files):
    service, state = FakeService(), {}
    Agent(service, FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())).chat("Predict", state, files)
    old_id = state["results"][0]["result_id"]
    output = Agent(service, FakeClient(final(result_ids=[old_id]))).chat("Predict only if accuracy >= 0.95", state, files)
    assert output["error"] == "tool"
    assert "evaluation tool was not called" in output["reply"]
    assert "predicts malware" not in output["reply"]
    assert len(service.calls) == 1


def test_percentage_threshold_and_negative_input(files):
    service = FakeService()
    response = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.9))).chat("Predict only if accuracy is at least 90%", {}, files)
    assert len(service.calls) == 2 and "condition was met" in response["reply"]
    bad = Agent(service, FakeClient()).chat("Predict if accuracy >= -0.1", {}, files)
    assert bad["error"] == "input"


@pytest.mark.parametrize("threshold", ["1e999999999999999999999999", "1e1000000%"])
def test_extreme_user_threshold_is_rejected_before_provider_call(files, threshold):
    client = FakeClient()
    output = Agent(FakeService(), client).chat(f"Predict if accuracy >= {threshold}", {}, files)
    assert output["error"] == "input"
    assert client.requests == []


@pytest.mark.parametrize("message", [
    "Only predict if accuracy for file abc123 is at least 0.95",
    "Only predict if we reach 95% accuracy",
    "Predict if accuracy >= .95",
    "Predict if accuracy is at least 95 percent",
])
def test_numeric_threshold_binding_does_not_consume_upload_id_digits(files, message):
    service = FakeService()
    output = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.1))).chat(message, {}, files)
    assert output["error"] == "tool"
    assert "does not match" in output["reply"]
    assert service.calls == []


def test_classify_word_is_not_a_conditional_trigger(files):
    client = FakeClient(tool("predict_batch", file_id="file-a"), final())
    output = Agent(FakeService(), client).chat("Classify every row and report accuracy", {}, files)
    assert "error" not in output
    assert len(client.requests[0]["tools"]) == 4


@pytest.mark.parametrize("message", [
    "Classify row 0 of file-a and tell me if it is malware",
    "Predict row 0 and report if it is goodware",
    "Classify row 0 using the model's decision threshold",
    "Classify every row and include the minimum malware probability",
    "Classify row 0 and show the threshold and minimum row metadata",
])
def test_descriptive_conditions_and_decision_metadata_allow_classification(files, message):
    service = FakeService()
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())
    output = Agent(service, client).chat(message, {}, files)
    assert "error" not in output
    assert "predicts malware" in output["reply"]
    assert [call[0] for call in service.calls] == ["predict_single"]
    assert client.requests[0]["tool_choice"] == "auto"


def test_independent_evaluation_and_batch_in_one_turn_are_allowed(files):
    service = FakeService()
    client = FakeClient(tool("evaluate", file_id="file-a"),
                        tool("predict_batch", file_id="file-b"), final())
    output = Agent(service, client).chat("Evaluate file-a and classify every row of file-b", {}, files)
    assert "error" not in output
    assert [call[0] for call in service.calls] == ["evaluate", "predict_batch"]
    assert [entry["status"] for entry in output["activity"]] == ["success", "success"]
    assert "Accuracy: 0.900000" in output["reply"] and "Batch:" in output["reply"]


@pytest.mark.parametrize("message", [
    "Only predict if accuracy >= 33.3%",
    "Evaluate first and classify only when accuracy is at least 33.3 percent",
])
def test_decimal_percentage_uses_canonical_user_threshold(files, message):
    service = FakeService(accuracy=0.333)
    output = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a",
                   prediction_file_id="file-b", row_index=0, min_accuracy=0.333))).chat(message, {}, files)
    assert "error" not in output
    assert [call[0] for call in service.calls] == ["evaluate", "predict_single"]


def test_float_roundoff_cannot_lower_actual_accuracy_gate(files):
    neighboring = math.nextafter(0.333, 0)
    service = FakeService(accuracy=neighboring)
    output = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a",
                   prediction_file_id="file-b", row_index=0, min_accuracy=neighboring))).chat(
        "Only predict if accuracy >= 33.3%", {}, files)
    assert "error" not in output
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert "Prediction skipped" in output["reply"]


@pytest.mark.parametrize("threshold", [0.332, 0.333 - 1e-12])
def test_provider_cannot_lower_even_a_small_real_threshold_difference(files, threshold):
    service = FakeService()
    output = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a",
                   prediction_file_id="file-b", row_index=0, min_accuracy=threshold))).chat(
        "Only predict if accuracy >= 33.3%", {}, files)
    assert output["error"] == "tool"
    assert "does not match" in output["reply"]
    assert service.calls == []


@pytest.mark.parametrize("message", [
    "Evaluate file-a; if at least 95% are classified correctly, predict row 0 of file-b",
    "If evaluation >= 0.95 then predict row 0 of file-b",
    "Only when 95 percent correct predictions are obtained, classify row 0",
])
def test_conditional_paraphrases_cannot_bypass_evaluation(files, message):
    service = FakeService()
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0))
    output = Agent(service, client).chat(message, {}, files)
    assert output["error"] == "tool"
    assert service.calls == []
    assert [t["name"] for t in client.requests[0]["tools"]] == ["evaluate_then_predict"]


@pytest.mark.parametrize("message", [
    "Only predict if accuracy is good enough",
    "If the file exists classify row 0",
    "If Size > 0.5 classify row 0",
])
def test_ambiguous_condition_needs_explicit_accuracy_without_provider_guess(files, message):
    service, client = FakeService(), FakeClient()
    output = Agent(service, client).chat(message, {}, files)
    assert output["error"] == "input"
    assert "numeric accuracy threshold" in output["reply"]
    assert service.calls == [] and client.requests == []


def test_if_available_metric_followup_is_not_conditional_prediction(files):
    client = FakeClient(final(kind="help"))
    output = Agent(FakeService(), client).chat("Show accuracy if available", {}, files)
    assert "error" not in output
    assert len(client.requests[0]["tools"]) == 4


@pytest.mark.parametrize("name,args", [
    ("run_shell", {"cmd": "cat /etc/passwd"}),
    ("predict_single", {"file_id": "/etc/passwd", "row_index": 0}),
    ("predict_single", {"file_id": "file-a", "row_index": True}),
    ("predict_single", {"file_id": "file-a", "row_index": -1}),
    ("predict_single", {"file_id": "file-a", "row_index": 1.5}),
    ("predict_single", {"file_id": "file-a", "row_index": 0, "extra": "value"}),
    ("evaluate_then_predict", {"evaluation_file_id": "file-a", "prediction_file_id": "file-b", "row_index": 0, "min_accuracy": float("nan")}),
    ("evaluate_then_predict", {"evaluation_file_id": "file-a", "prediction_file_id": "file-b", "row_index": 0, "min_accuracy": float("inf")}),
    ("evaluate_then_predict", {"evaluation_file_id": "file-a", "prediction_file_id": "file-b", "row_index": 0, "min_accuracy": 10 ** 400}),
    ("evaluate_then_predict", {"evaluation_file_id": "file-a", "prediction_file_id": "file-b", "row_index": 0, "min_accuracy": True}),
])
def test_untrusted_tool_arguments_cannot_invoke_service(files, name, args):
    service = FakeService()
    response = Agent(service, FakeClient(tool(name, **args))).chat("Use the uploaded file", {}, files)
    assert response["error"] == "tool"
    assert service.calls == []


def test_malformed_tool_json(files):
    response = tool("evaluate", file_id="file-a")
    response.output[0]["arguments"] = "{invalid JSON"
    service = FakeService()
    output = Agent(service, FakeClient(response)).chat("Evaluate", {}, files)
    assert output["error"] == "tool" and service.calls == []


def test_failure_does_not_send_exception_secret_or_ask_llm_to_guess(files):
    class BrokenService(FakeService):
        def predict_single(self, *args, **kwargs):
            raise RuntimeError("SECRET_TOKEN_AND_INTERNAL_PATH")
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0),
                        SimpleNamespace(output=[], output_text="Guess malware"))
    output = Agent(BrokenService(), client).chat("Classify", {}, files)
    assert output["error"] == "tool"
    assert "SECRET" not in str(output)
    assert "malware" not in output["reply"]
    assert len(client.requests) == 1


@pytest.mark.parametrize("name,args,message", [
    ("predict_single", {"row_index": 10}, "row_index must identify an existing row (starting at zero)"),
    ("predict_single", {"row_index": 0}, "Invalid row 0: Invalid numeric feature: Size"),
    ("evaluate", {}, "Evaluation requires a Label column containing 0 or 1"),
    ("evaluate", {}, "Evaluation has no valid labeled rows"),
    ("predict_batch", {}, "Missing features: Size"),
    ("predict_batch", {}, "CSV has 1 unexpected column(s)"),
])
def test_known_tool_validation_errors_are_visible_in_reply_and_activity(files, name, args, message):
    class InvalidService(FakeService):
        feature_names = ["Size"]
        def predict_single(self, *args, **kwargs):
            raise ValueError(message)
        def predict_batch(self, *args, **kwargs):
            raise ValueError(message)
        def evaluate(self, *args, **kwargs):
            raise ValueError(message)
    client = FakeClient(tool(name, file_id="file-a", **args))
    output = Agent(InvalidService(), client).chat("Use the uploaded file", {}, files)
    assert output["error"] == "tool"
    assert output["reply"] == message
    assert output["activity"][0]["error"] == message
    assert output["activity"][0]["status"] == "error"
    assert len(client.requests) == 1
    assert not list(files["file-a"]["path"].parent.glob("predictions-*.csv"))


@pytest.mark.parametrize("message", [
    "SECRET_TOKEN_AND_INTERNAL_PATH /private/credentials",
    "Invalid row 0: Invalid numeric feature: RAW_CELL_SECRET",
    "Missing features: /private/credentials",
    "CSV has 1 unexpected column(s): SECRET_UPLOAD_INSTRUCTION",
])
def test_unknown_valueerrors_cannot_leak_cells_paths_or_secrets(files, message):
    class BrokenService(FakeService):
        feature_names = ["Size"]
        def predict_single(self, *args, **kwargs):
            raise ValueError(message)
    output = Agent(BrokenService(), FakeClient(tool("predict_single", file_id="file-a", row_index=0))).chat("Classify", {}, files)
    assert output["error"] == "tool"
    assert message not in str(output)
    assert "The classification tool failed" in output["reply"]


def test_provider_errors_are_sanitized(files):
    output = Agent(FakeService(), FakeClient(RuntimeError("sk-private-secret"))).chat("Evaluate", {}, files)
    assert output["error"] == "provider"
    assert "sk-private" not in str(output)


def test_conditional_limit_reserves_both_underlying_calls(files):
    service = FakeService()
    output = Agent(service, FakeClient(tool("evaluate_then_predict", evaluation_file_id="file-a", prediction_file_id="file-b", row_index=0, min_accuracy=0.8)), max_tool_calls=1).chat("Predict if accuracy >= 0.8", {}, files)
    assert output["error"] == "tool" and service.calls == []


def test_underlying_calls_are_bounded(files):
    service = FakeService()
    calls = [tool("evaluate", file_id="file-a") for _ in range(3)]
    output = Agent(service, FakeClient(*calls), max_tool_calls=2).chat("Evaluate the CSV", {}, files)
    assert len(service.calls) == 2
    assert output["error"] == "tool"


def test_followup_uses_existing_evidence_and_session_isolation(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    result_id = state["results"][0]["result_id"]
    followup = FakeClient(final(result_ids=[result_id], focus="auc"))
    output = Agent(service, followup).chat("What was its AUC?", state, files)
    assert "AUC: 0.960000" in output["reply"]
    assert len(service.calls) == 1
    other = Agent(service, FakeClient(final(result_ids=[result_id]))).chat("Tell me that other user's results", {}, files)
    assert "unavailable in this session" in other["reply"]
    assert "0.960000" not in other["reply"]


@pytest.mark.parametrize("focus,title,count,denominator", [
    ("false_negatives", "False negatives", 3, 12),
    ("false_positives", "False positives", 2, 8),
    ("true_positives", "True positives", 9, 12),
    ("true_negatives", "True negatives", 6, 8),
])
def test_confusion_component_followup_computes_count_and_rate_from_stored_matrix(files, focus, title, count, denominator):
    state = {}
    service = FakeService(confusion_matrix=[[6, 2], [3, 9]], class_counts={"0": 8, "1": 12},
                          total_count=20, evaluated_count=20, valid_count=20)
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    result_id = state["results"][0]["result_id"]
    output = Agent(service, FakeClient(final(result_ids=[result_id], focus=focus))).chat(f"How many {focus.replace('_', ' ')}?", state, files)
    assert f"{title}: {count}" in output["reply"]
    assert f"{count} / {denominator} evaluated" in output["reply"]
    assert f"{count / denominator:.6f}" in output["reply"]
    assert len(service.calls) == 1
    assert output["activity"] == []


def test_confusion_matrix_followup_names_all_four_counts(files):
    service = FakeService(confusion_matrix=[[2, 0], [1, 1]])
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final(focus="confusion_matrix"))).chat("Show the confusion matrix", {}, files)
    for text in ("True negatives: 2", "false positives: 0", "false negatives: 1", "true positives: 1"):
        assert text in output["reply"]


def test_false_negative_rate_with_no_malware_is_undefined_not_zero(files):
    service = FakeService(confusion_matrix=[[5, 0], [0, 0]])
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final(focus="false_negatives"))).chat("How many false negatives?", {}, files)
    assert "False negatives: 0" in output["reply"]
    assert "rate is unavailable: no malware files were evaluated" in output["reply"]


@pytest.mark.parametrize("fields", [
    {"confusion_matrix_order": [1, 0]},
    {"confusion_matrix_order": None},
    {"confusion_matrix_order": [False, True]},
    {"confusion_matrix": [[2, 0], [-1, 1]]},
    {"confusion_matrix": [[2, 0], [1.5, 1]]},
    {"confusion_matrix": [[2], [1, 1]]},
])
def test_confusion_counts_require_known_class_order_and_valid_integer_matrix(files, fields):
    output = Agent(FakeService(**fields), FakeClient(tool("evaluate", file_id="file-a"), final(focus="false_negatives"))).chat("How many false negatives?", {}, files)
    assert "Confusion-matrix counts unavailable" in output["reply"]
    assert "rate is" not in output["reply"]


def test_single_class_auc_rendering_and_missing_labels_count(files):
    service = FakeService(auc=None, missing_label_count=1, evaluated_count=9, evaluation_coverage=0.9)
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", {}, files)
    assert "AUC: unavailable" in output["reply"]
    assert "missing labels 1" in output["reply"]


def test_no_feature_explanations_can_be_invented(files):
    output = Agent(FakeService(), FakeClient(final(kind="explanation_unavailable"))).chat("Which DLL caused the malware classification?", {}, files)
    assert "no explanation tool" in output["reply"]


def test_histories_and_result_context_are_bounded(files):
    state = {"results": [], "history": [], "activity": []}
    service = FakeService()
    for _ in range(23):
        Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    assert len(state["results"]) == 20
    assert len(state["history"]) == 12
    assert len(state["activity"]) <= 100


def test_evicted_batch_download_registry_can_be_rolled_back_by_caller(files):
    service, state = FakeService(), {}
    response = Agent(service, FakeClient(tool("predict_batch", file_id="file-a"), final())).chat("Predict batch", state, files)
    old_id = response["results"][0]["download_id"]
    old_path = state["downloads"][old_id]
    for _ in range(20):
        Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    assert old_path.exists()
    assert old_id not in state["downloads"]
    assert files["file-a"]["path"].is_file()


def test_pasted_csv_and_large_message_never_reach_provider(files):
    client = FakeClient()
    agent = Agent(FakeService(), client)
    for message in ["x" * 4001, "a,b,c,d,e,f\n1,2,3,4,5,6\n1,2,3,4,5,6", "Size,Label\n100,1", "```csv\nSize,Label\n100,1\n```", Path("samples/single.csv").read_text().strip()]:
        assert agent.chat(message, {}, files)["error"] == "input"
    assert client.requests == []


def test_missing_api_key_is_lazy_and_informative(files, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = Agent(FakeService())
    output = agent.chat("Evaluate", {}, files)
    assert output["error"] == "configuration"
    assert "OPENAI_API_KEY is not configured" in output["reply"]


def test_blank_optional_model_uses_documented_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "")
    assert Agent(FakeService()).model == "gpt-4.1-mini"

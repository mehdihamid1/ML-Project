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


def brief(request):
    """The developer message that states a conditional request's threshold."""
    return [item["content"] for item in request["input"] if item.get("role") == "developer"][-1]


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
def test_conditional_evaluates_first_then_model_decides_from_returned_accuracy(files, accuracy, predicts):
    service = FakeService(accuracy)
    # A correct model calls predict_single exactly when the returned accuracy meets 0.9.
    client = FakeClient(tool("evaluate", file_id="file-a"),
                        tool("predict_single", file_id="file-b", row_index=0) if predicts else final())
    response = Agent(service, client).chat("Evaluate file-a; only if accuracy >= 0.9 predict row 0 of file-b", {}, files)
    assert [c[0] for c in service.calls] == (["evaluate", "predict_single"] if predicts else ["evaluate"])
    assert [(a["tool"], a["status"]) for a in response["activity"]] == [
        ("evaluate", "success"), ("predict_single", "success" if predicts else "skipped")]
    assert "error" not in response
    assert response["reply"].startswith("Condition met" if predicts else "Prediction withheld")
    first, second = client.requests
    assert first["tool_choice"] == {"type": "function", "name": "evaluate"}
    assert [t["name"] for t in first["tools"]] == ["evaluate"]
    assert "minimum accuracy is 0.9 " in brief(first)
    # The decision request carries the evaluation the model just received.
    assert second["tool_choice"] == "auto"
    assert [t["name"] for t in second["tools"]] == ["predict_single"]
    returned = second["input"][-1]
    assert returned["type"] == "function_call_output"
    assert json.loads(returned["output"])["accuracy"] == accuracy


@pytest.mark.parametrize("coverage", [
    {"accuracy": 0.89},
    {"missing_label_count": 1, "evaluated_count": 9, "evaluation_coverage": 0.9},
    {"invalid_count": 1}, {"invalid_label_count": 1},
    {"evaluation_coverage": None}, {"accuracy": None},
])
def test_server_blocks_a_prediction_the_condition_does_not_allow(files, coverage):
    service = FakeService(**coverage)
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_single", file_id="file-b", row_index=0))
    response = Agent(service, client).chat("Evaluate file-a; predict row 0 of file-b if accuracy >= 0.9", {}, files)
    assert [c[0] for c in service.calls] == ["evaluate"]
    assert [(a["tool"], a["status"]) for a in response["activity"]] == [("evaluate", "success"), ("predict_single", "blocked")]
    assert response["activity"][-1]["arguments"] == {"file_id": "file-b", "row_index": 0}
    assert "the server blocked it" in response["reply"]
    assert len(response["results"]) == 1


@pytest.mark.parametrize("coverage", [
    {"missing_label_count": 1, "evaluated_count": 9, "evaluation_coverage": 0.9},
    {"invalid_count": 1}, {"invalid_label_count": 1},
    {"evaluation_coverage": None}, {"accuracy": None},
])
def test_withholding_for_incomplete_or_undefined_results_is_accepted(files, coverage):
    service = FakeService(**coverage)
    response = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate file-a; predict row 0 of file-b if accuracy >= 0.8", {}, files)
    assert [c[0] for c in service.calls] == ["evaluate"]
    assert response["activity"][-1]["status"] == "skipped"
    assert "error" not in response
    assert response["reply"].startswith("Prediction withheld")


def test_model_withholding_a_permitted_prediction_is_reported_not_overridden(files):
    service = FakeService(accuracy=0.95)
    response = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat(
        "Evaluate file-a; only predict row 0 of file-b if accuracy >= 0.9", {}, files)
    # The server checks the model's decision; it never predicts on the model's behalf.
    assert [c[0] for c in service.calls] == ["evaluate"]
    assert response["error"] == "tool"
    assert response["activity"][-1]["status"] == "skipped"
    assert "did not call predict_single, although accuracy 0.950000 meets the required 0.900000" in response["reply"]


@pytest.mark.parametrize("selected", [
    tool("predict_single", file_id="file-b", row_index=0),
    tool("predict_batch", file_id="file-b"),
])
def test_conditional_turn_cannot_skip_the_evaluation(files, selected):
    service = FakeService()
    response = Agent(service, FakeClient(selected)).chat("Evaluate file-a; only predict row 0 of file-b if accuracy >= 0.95", {}, files)
    assert response["error"] == "tool"
    assert "must call evaluate first" in response["reply"]
    assert service.calls == []


def test_after_the_evaluation_only_predict_single_can_follow(files):
    service = FakeService()
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_batch", file_id="file-b"))
    response = Agent(service, client).chat("Evaluate file-a; only predict row 0 of file-b if accuracy >= 0.5", {}, files)
    assert response["error"] == "tool"
    assert [c[0] for c in service.calls] == ["evaluate"]
    assert "No prediction was made." in response["reply"]


def test_failed_evaluation_stops_the_conditional_task(files):
    class NoLabels(FakeService):
        def evaluate(self, path):
            self.calls.append(("evaluate", Path(path)))
            raise ValueError("Evaluation requires a Label column containing 0 or 1")
    service, client = NoLabels(), FakeClient(tool("evaluate", file_id="file-a"))
    output = Agent(service, client).chat("Evaluate file-a; only predict row 0 of file-b if accuracy >= 0.9", {}, files)
    assert output["error"] == "tool"
    assert output["reply"] == "Evaluation requires a Label column containing 0 or 1. No prediction was made."
    assert [(a["tool"], a["status"]) for a in output["activity"]] == [("evaluate", "error")]
    assert [c[0] for c in service.calls] == ["evaluate"] and len(client.requests) == 1


def test_conditional_cannot_answer_with_stale_prediction_without_evaluating(files):
    service, state = FakeService(), {}
    Agent(service, FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())).chat("Predict", state, files)
    old_id = state["results"][0]["result_id"]
    output = Agent(service, FakeClient(final(result_ids=[old_id]))).chat("Evaluate file-a; predict row 0 of file-b only if accuracy >= 0.95", state, files)
    assert output["error"] == "tool"
    assert "evaluation tool was not called" in output["reply"]
    assert "predicts malware" not in output["reply"]
    assert len(service.calls) == 1


def test_percentage_threshold_and_negative_input(files):
    service = FakeService()
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_single", file_id="file-b", row_index=0))
    response = Agent(service, client).chat("Evaluate file-a; predict row 0 of file-b only if accuracy is at least 90%", {}, files)
    assert len(service.calls) == 2 and "meets the required 0.900000" in response["reply"]
    assert "minimum accuracy is 0.9 " in brief(client.requests[0])
    bad = Agent(service, FakeClient()).chat("Predict if accuracy >= -0.1", {}, files)
    assert bad["error"] == "input"


@pytest.mark.parametrize("threshold", ["1e999999999999999999999999", "1e1000000%"])
def test_extreme_user_threshold_is_rejected_before_provider_call(files, threshold):
    client = FakeClient()
    output = Agent(FakeService(), client).chat(f"Predict if accuracy >= {threshold}", {}, files)
    assert output["error"] == "input"
    assert client.requests == []


@pytest.mark.parametrize("message", [
    "Predict row 0 of file-b only if accuracy for file abc123 is at least 0.95",
    "Evaluate file-a; only predict row 0 of file-b if we reach 95% accuracy",
    "Evaluate file-a; predict row 0 of file-b if accuracy >= .95",
    "Evaluate file-a; predict row 0 of file-b if accuracy is at least 95 percent",
])
def test_numeric_threshold_binding_does_not_consume_upload_id_digits(files, message):
    files = {**files, "abc123": files["file-a"]}
    evaluation_id = "abc123" if "abc123" in message else "file-a"
    service = FakeService(accuracy=0.94)
    client = FakeClient(tool("evaluate", file_id=evaluation_id), tool("predict_single", file_id="file-b", row_index=0))
    output = Agent(service, client).chat(message, {}, files)
    assert "minimum accuracy is 0.95 " in brief(client.requests[0])
    assert output["activity"][-1]["status"] == "blocked"
    assert "below the required 0.950000" in output["reply"]
    assert [call[0] for call in service.calls] == ["evaluate"]


@pytest.mark.parametrize("message", [
    "Classify every row and report accuracy",
    "Classify every row and report the accuracy over all rows",
])
def test_classify_word_is_not_a_conditional_trigger(files, message):
    client = FakeClient(tool("predict_batch", file_id="file-a"), final())
    output = Agent(FakeService(), client).chat(message, {}, files)
    assert "error" not in output
    assert [t["name"] for t in client.requests[0]["tools"]] == ["predict_single", "predict_batch", "evaluate"]


@pytest.mark.parametrize("message", [
    "Evaluate file-a; predict row 0 of file-b when accuracy is 0.95 or higher",
    "Predict row 0 of file-b when the accuracy of file-a is 0.95 or higher",
    "Evaluate file-a; predict row 0 of file-b once the evaluation accuracy reaches 95%",
    "Evaluate file-a; predict row 0 of file-b as long as accuracy is above 0.95",
    "Evaluate file-a; predict row 0 of file-b whenever accuracy is greater than 0.95",
    "Evaluate file-a; accuracy must reach 0.95 before you predict row 0 of file-b",
])
def test_conditions_worded_without_if_still_evaluate_first(files, message):
    # Regression: these were treated as plain predictions, so a model could predict without evaluating.
    service = FakeService()
    client = FakeClient(tool("predict_single", file_id="file-b", row_index=0))
    output = Agent(service, client).chat(message, {}, files)
    assert output["error"] == "tool" and "must call evaluate first" in output["reply"]
    assert service.calls == []
    assert client.requests[0]["tool_choice"] == {"type": "function", "name": "evaluate"}
    assert "minimum accuracy is 0.95 " in brief(client.requests[0])


@pytest.mark.parametrize("message", [
    "The previous accuracy of 0.80 was too low. Evaluate file-a; only if accuracy is at least 0.95, predict row 0 of file-b.",
    "Previous accuracy is 0.80. Evaluate file-a; predict row 0 of file-b only if accuracy >= 0.95",
    "Tell me if the previous accuracy of 0.80 was good. Evaluate file-a; predict row 0 of file-b only if accuracy is at least 0.95",
])
def test_an_earlier_accuracy_figure_does_not_become_the_threshold(files, message):
    # Regression: the parser bound 0.80, so an evaluation of 0.90 permitted the prediction.
    service = FakeService(accuracy=0.90)
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_single", file_id="file-b", row_index=0))
    output = Agent(service, client).chat(message, {}, files)
    assert "minimum accuracy is 0.95 " in brief(client.requests[0])
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert output["activity"][-1]["status"] == "blocked"
    assert "accuracy 0.900000 is below the required 0.950000" in output["reply"]


def test_two_accuracy_values_in_the_condition_ask_which_one(files):
    service, client = FakeService(), FakeClient()
    output = Agent(service, client).chat(
        "Only predict row 0 if accuracy is at least 0.95 or at least 0.80 accuracy", {}, files)
    assert output["error"] == "input"
    assert "more than one accuracy value" in output["reply"] and "0.95" in output["reply"] and "0.8" in output["reply"]
    assert service.calls == [] and client.requests == []


@pytest.mark.parametrize("message", [
    "Classify row 0 of file-a and tell me if it is malware",
    "Predict row 0 and report if it is goodware",
    "Classify row 0 using the model's decision threshold",
    "Classify every row and include the minimum malware probability",
    "Classify row 0 and show the threshold and minimum row metadata",
    "Classify row 0 of file-a and let me know if it's malware",
    "Predict row 0; I want to know if it is malware",
    "Classify row 0 and explain if the probability is high",
    "Can you classify row 0? If so, show the probability",
    "Classify row 0 if possible",
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
    "If accuracy on file-a is greater than or equal to 0.95, predict row 0 of file-b",
    "Predict row 0 of file-b provided the accuracy on file-a is no less than 0.95",
    "Predict row 0 of file-b if the accuracy of file-a is over 95%",
    "Only predict row 0 of file-b if the accuracy of file-a is 0.95 or higher",
])
def test_comparison_wordings_bind_the_stated_threshold(files, message):
    service = FakeService(accuracy=0.9)
    client = FakeClient(tool("evaluate", file_id="file-a"), final())
    output = Agent(service, client).chat(message, {}, files)
    assert "minimum accuracy is 0.95 " in brief(client.requests[0])
    assert "error" not in output
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert "below the required 0.950000" in output["reply"]


@pytest.mark.parametrize("message", [
    "Evaluate file-a; only predict row 0 of file-b if accuracy >= 33.3%",
    "Evaluate file-a first and classify row 0 of file-b only when accuracy is at least 33.3 percent",
])
def test_decimal_percentage_uses_canonical_user_threshold(files, message):
    service = FakeService(accuracy=0.333)
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"),
                                       tool("predict_single", file_id="file-b", row_index=0))).chat(message, {}, files)
    assert "error" not in output
    assert [call[0] for call in service.calls] == ["evaluate", "predict_single"]


@pytest.mark.parametrize("accuracy", [math.nextafter(0.333, 0), 0.333 - 1e-12, 0.332])
def test_prediction_just_below_the_user_threshold_is_blocked(files, accuracy):
    # Float round-off or a model misreading cannot lower the stated threshold.
    service = FakeService(accuracy=accuracy)
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"),
                                       tool("predict_single", file_id="file-b", row_index=0))).chat(
        "Evaluate file-a; only predict row 0 of file-b if accuracy >= 33.3%", {}, files)
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert output["activity"][-1]["status"] == "blocked"
    assert "Prediction withheld" in output["reply"]


@pytest.mark.parametrize("message", [
    "Evaluate file-a; if at least 95% are classified correctly, predict row 0 of file-b",
    "Evaluate file-a; if evaluation >= 0.95 then predict row 0 of file-b",
    "Evaluate file-a; only when 95 percent correct predictions are obtained, classify row 0 of file-b",
    "Evaluate file-a; if it reaches 95% accuracy, predict row 0 of file-b",
])
def test_conditional_paraphrases_cannot_bypass_evaluation(files, message):
    service = FakeService()
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0))
    output = Agent(service, client).chat(message, {}, files)
    assert output["error"] == "tool"
    assert service.calls == []
    assert [t["name"] for t in client.requests[0]["tools"]] == ["evaluate"]
    assert client.requests[0]["tool_choice"] == {"type": "function", "name": "evaluate"}


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
    assert len(client.requests[0]["tools"]) == 3


@pytest.mark.parametrize("message", [
    "Previous accuracy was 0.8. Evaluate file-a; predict row 0 of file-b only if accuracy is good enough.",
    "Evaluate file-a; predict row 0 of file-b only if accuracy is good enough (previous accuracy of 0.8).",
    "Evaluate file-a; predict row 0 of file-b only if accuracy is good enough (yesterday accuracy of 0.8).",
    "Evaluate file-a; predict row 0 of file-b only if accuracy is good enough, for context accuracy of 0.8.",
    "Evaluate file-a; if Size > 0.5 classify row 0 of file-b and report accuracy.",
])
def test_unstated_accuracy_minimum_never_uses_history_or_feature_conditions(files, message):
    service, client = FakeService(accuracy=0.9), FakeClient()
    response = Agent(service, client).chat(message, {}, files)
    assert response["error"] == "input"
    assert "numeric accuracy threshold" in response["reply"]
    assert not service.calls and not client.requests


def test_when_finished_is_sequencing_not_an_accuracy_condition(files):
    service = FakeService()
    client = FakeClient(tool("evaluate", file_id="file-a"),
                        tool("predict_single", file_id="file-b", row_index=0), final())
    response = Agent(service, client).chat(
        "Evaluate file-a and classify row 0 of file-b; when you are finished, show both results and the accuracy.", {}, files)
    assert "error" not in response
    assert [call[0] for call in service.calls] == ["evaluate", "predict_single"]
    assert client.requests[0]["tool_choice"] == "auto"


@pytest.mark.parametrize("wrong_call", [
    ("evaluate", {"file_id": "file-c"}),
    ("predict_single", {"file_id": "file-a", "row_index": 0}),
    ("predict_single", {"file_id": "file-b", "row_index": 7}),
])
def test_conditional_contract_blocks_substituted_file_or_row(files, wrong_call):
    files = {**files, "file-c": {**files["file-a"], "path": Path("/different-evaluation.csv")}}
    name, args = wrong_call
    responses = ([tool("evaluate", file_id="file-a")] if name != "evaluate" else []) + [tool(name, **args)]
    service, client = FakeService(accuracy=0.99), FakeClient(*responses)
    response = Agent(service, client).chat(
        "Evaluate file file-a; only if accuracy >= 0.95, predict row 0 of file file-b.", {}, files)
    assert response["error"] == "tool"
    assert response["activity"][-1]["status"] == "blocked"
    assert response["activity"][-1]["arguments"] == args
    assert "do not match" in response["reply"] and "No prediction was made" in response["reply"]
    assert [call[0] for call in service.calls] == ([] if name == "evaluate" else ["evaluate"])


@pytest.mark.parametrize("message", [
    "Previous accuracy on file-a was 0.8. Predict row 0 of file-b only if accuracy >= 0.95.",
    "Evaluate file-a; predict row 0 only if accuracy >= 0.95.",
    "Evaluate file-a; predict file-b only if accuracy >= 0.95.",
    "Evaluate file-a and file-b; predict row 0 of file-b only if accuracy >= 0.95.",
    "Evaluate file file-unknown; predict row 0 of file-b only if accuracy >= 0.95.",
    "Evaluate file-a; predict row 0 of file-a2 only if accuracy >= 0.95.",
    "Evaluate file-a; predict row 1.5 of file-b only if accuracy >= 0.95.",
    "Evaluate file-a; predict row -0.5 of file-b only if accuracy >= 0.95.",
    "Evaluate file-a; predict row 0/1 of file-b only if accuracy >= 0.95.",
    "Evaluate file-a; predict row 0 or 1 of file-b only if accuracy >= 0.95.",
])
def test_unclear_conditional_targets_need_clarification_before_provider(files, message):
    service, client = FakeService(), FakeClient()
    response = Agent(service, client).chat(message, {}, files)
    assert response["error"] == "input"
    assert not response["activity"] and not service.calls and not client.requests


@pytest.mark.parametrize("message", [
    "Evaluate file-a; accuracy must be >= 0.95 to predict row 0 of file-b.",
    "Require at least 0.95 accuracy, then predict row 0 of file-b after evaluating file-a.",
    "Use a minimum accuracy of 0.95 to classify row 0 of file-b; evaluate file-a first.",
])
def test_accuracy_requirements_without_condition_words_still_gate_prediction(files, message):
    service, client = FakeService(accuracy=0.9), FakeClient(tool("evaluate", file_id="file-a"), final())
    response = Agent(service, client).chat(message, {}, files)
    assert "error" not in response
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert response["activity"][-1]["status"] == "skipped"


def test_one_upload_infers_only_file_identity_and_exposes_bound_tool_schema(files):
    files = {"file-a": files["file-a"]}
    service = FakeService(accuracy=0.99)
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_single", file_id="file-a", row_index=2))
    response = Agent(service, client).chat("Predict row 2 only if accuracy >= 0.95", {}, files)
    assert "error" not in response
    assert service.calls[-1][2] == 2
    assert client.requests[0]["tools"][0]["parameters"]["properties"]["file_id"]["enum"] == ["file-a"]
    assert client.requests[1]["tools"][0]["parameters"]["properties"]["row_index"]["enum"] == [2]
    assert "row_index 2" in brief(client.requests[0])


@pytest.mark.parametrize("target", ["file-b", "f" * 32, "01234567890123456789012345678901"])
def test_one_upload_never_replaces_an_explicit_unknown_target(files, target):
    files = {"file-a": files["file-a"]}
    client = FakeClient()
    response = Agent(FakeService(), client).chat(
        f"Evaluate file-a; predict row 0 of {target} only if accuracy >= 0.95", {}, files)
    assert response["error"] == "input"
    assert "Unknown upload ID" in response["reply"]
    assert not client.requests


@pytest.mark.parametrize("threshold", ["0.95abc", "0e456abc"])
def test_accuracy_threshold_requires_a_complete_numeric_token(files, threshold):
    client = FakeClient()
    response = Agent(FakeService(), client).chat(
        f"Evaluate file-a; predict row 0 of file-b only if accuracy >= {threshold}", {}, files)
    assert response["error"] == "input" and not client.requests


@pytest.mark.parametrize("history", [
    "Yesterday accuracy on file-a was 0.8. ",
    "For context, accuracy on file-a is 0.8. ",
])
def test_historical_accuracy_files_cannot_supply_missing_evaluation_target(files, history):
    client = FakeClient()
    response = Agent(FakeService(), client).chat(
        history + "Predict row 0 of file-b only if accuracy >= 0.95", {}, files)
    assert response["error"] == "input" and "evaluation file ID" in response["reply"]
    assert not client.requests


def test_explicit_new_evaluation_is_not_hidden_by_a_historical_sentence(files):
    client = FakeClient(tool("evaluate", file_id="file-a"), final())
    response = Agent(FakeService(), client).chat(
        "Previous accuracy was 0.8, but now evaluate file-a; predict row 0 of file-b only if accuracy >= 0.95", {}, files)
    assert "error" not in response
    assert response["activity"][-1]["status"] == "skipped"


@pytest.mark.parametrize("report", [
    "report whether accuracy is at least 0.95",
    "tell me if the accuracy is >= 0.95",
])
def test_reporting_an_accuracy_comparison_does_not_gate_independent_prediction(files, report):
    service = FakeService()
    client = FakeClient(tool("evaluate", file_id="file-a"), tool("predict_single", file_id="file-b", row_index=0), final())
    response = Agent(service, client).chat("Evaluate file-a and classify row 0 of file-b; " + report, {}, files)
    assert "error" not in response
    assert [call[0] for call in service.calls] == ["evaluate", "predict_single"]


@pytest.mark.parametrize("name,args", [
    ("run_shell", {"cmd": "cat /etc/passwd"}),
    ("predict_single", {"file_id": "/etc/passwd", "row_index": 0}),
    ("predict_single", {"file_id": "file-a", "row_index": True}),
    ("predict_single", {"file_id": "file-a", "row_index": -1}),
    ("predict_single", {"file_id": "file-a", "row_index": 1.5}),
    ("predict_single", {"file_id": "file-a", "row_index": 0, "extra": "value"}),
    # The former combined conditional tool no longer exists.
    ("evaluate_then_predict", {"evaluation_file_id": "file-a", "prediction_file_id": "file-b", "row_index": 0, "min_accuracy": 0.5}),
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
    service, client = FakeService(), FakeClient()
    output = Agent(service, client, max_tool_calls=1).chat("Predict if accuracy >= 0.8", {}, files)
    assert output["error"] == "tool" and service.calls == [] and client.requests == []


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
    # A follow-up's focus narrows the answer to the metric asked about.
    assert "Accuracy" not in output["reply"]
    assert followup.requests[0]["tools"] == []
    assert followup.requests[0]["tool_choice"] == "none"
    other = Agent(service, FakeClient(final(result_ids=[result_id]))).chat("Tell me that other user's results", {}, files)
    assert "unavailable in this session" in other["reply"]
    assert "0.960000" not in other["reply"]


@pytest.mark.parametrize("name,args", [
    ("predict_single", {"file_id": "file-b", "row_index": 0}),
    ("predict_batch", {"file_id": "file-b"}),
    ("evaluate", {"file_id": "file-a"}),
])
@pytest.mark.parametrize("message", [
    "How many false negatives were there in that evaluation?",
    "What was its false negative rate?",
    "How many false negatives were files predicted as goodware?",
    "Do not evaluate again. How many false negatives were there in that evaluation?",
    "How many false negatives were there? Don't predict another row.",
    "Don't predict or classify. How many false negatives were there?",
    "Don't predict row 0. How many false negatives were there?",
    "What was the prediction for row 0?",
    "How many malware samples did it miss?",
    "Explain what the evaluation found.",
    "What did the classifier miss?",
    "Did it correctly identify all malware?",
    "What did the model predict in that evaluation?",
    "What does the evaluate function return?",
])
def test_followup_cannot_execute_provider_tools_after_withheld_prediction(files, name, args, message):
    state, service = {}, FakeService(accuracy=0, confusion_matrix=[[0, 0], [1, 0]])
    first = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat(
        "Evaluate file-a; predict row 0 of file-b only if accuracy >= 0.9", state, files)
    assert first["activity"][-1]["status"] == "skipped"
    saved_results, saved_downloads = copy.deepcopy(state["results"]), dict(state["downloads"])
    client = FakeClient(tool(name, **args))  # Ignores the disabled tools deliberately.
    events = Agent(service, client).chat_events(message, state, files)
    observed = []
    while True:
        try:
            observed.append(next(events))
        except StopIteration as finished:
            output = finished.value
            break
    assert [call[0] for call in service.calls] == ["evaluate"]
    assert "false negatives: 1" in output["reply"].lower()
    assert output["activity"] == [] and output["results"] == []
    assert state["results"] == saved_results and state["downloads"] == saved_downloads
    assert all(event["type"] == "progress" for event in observed)
    assert client.requests[0]["tools"] == [] and client.requests[0]["tool_choice"] == "none"


@pytest.mark.parametrize("message,name,args", [
    ("Evaluate file-a again and show false negatives", "evaluate", {"file_id": "file-a"}),
    ("Predict row 0 of file-b", "predict_single", {"file_id": "file-b", "row_index": 0}),
    ("Classify every row of file-b", "predict_batch", {"file_id": "file-b"}),
    ("Run a new evaluation of file-a", "evaluate", {"file_id": "file-a"}),
    ("Could you please evaluate file-a again?", "evaluate", {"file_id": "file-a"}),
    ("I want you to predict row 0 of file-b", "predict_single", {"file_id": "file-b", "row_index": 0}),
])
def test_explicit_fresh_commands_still_execute_after_stored_evaluation(files, message, name, args):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    output = Agent(service, FakeClient(tool(name, **args), final())).chat(message, state, files)
    assert service.calls[-1][0] == name and len(service.calls) == 2
    assert output["activity"][0]["status"] == "success"


def test_followup_scopes_evaluation_evidence_and_explicit_older_result(files):
    state, service = {}, FakeService(accuracy=0.8)
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    older = state["results"][0]["result_id"]
    service.evaluation["accuracy"] = 0.9
    Agent(service, FakeClient(tool("evaluate", file_id="file-b"), final())).chat("Evaluate again", state, files)
    Agent(service, FakeClient(tool("predict_single", file_id="file-b", row_index=0), final())).chat("Predict row 0", state, files)
    client = FakeClient(final(result_ids=[older], focus="accuracy"))
    output = Agent(service, client).chat(f"What was the accuracy for result_id {older}?", state, files)
    assert "Accuracy: 0.800000" in output["reply"]
    context = json.loads(client.requests[0]["input"][0]["content"].partition(": ")[2])
    assert [r["result_id"] for r in context["session_results"]] == [older]
    assert len(service.calls) == 3 and not output["activity"]


@pytest.mark.parametrize("message", [
    "What was the AUC of result_id different-session-result?",
    "What was the accuracy for file-unknown?",
])
def test_followup_missing_reference_never_substitutes_or_executes(files, message):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    client = FakeClient(tool("predict_single", file_id="file-b", row_index=0))
    output = Agent(service, client).chat(message, state, files)
    assert "unavailable in this session" in output["reply"]
    assert len(service.calls) == 1 and client.requests == []
    assert output["activity"] == [] and output["results"] == []


def test_metric_question_can_evaluate_a_different_upload_without_stored_result(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-b"), final())).chat(
        "Show accuracy for file-b", state, files)
    assert len(service.calls) == 2 and output["activity"][0]["tool"] == "evaluate"


def test_followup_requested_metric_and_sole_reference_override_empty_provider_selection(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    output = Agent(service, FakeClient(final(focus="accuracy"))).chat("How many false negatives?", state, files)
    assert "False negatives: 0" in output["reply"] and "Accuracy" not in output["reply"]
    assert len(service.calls) == 1 and not output["results"]


@pytest.mark.parametrize("name,message,expected", [
    ("predict_single", "What is the malware probability?", "malware probability 0.870000"),
    ("predict_single", "What is the malware probability of row 0?", "malware probability 0.870000"),
    ("predict_batch", "How many goodware files were there?", "0 goodware"),
])
def test_prediction_and_batch_followups_also_disable_new_tools(files, name, message, expected):
    state, service = {}, FakeService()
    args = {"file_id": "file-a", **({"row_index": 0} if name == "predict_single" else {})}
    Agent(service, FakeClient(tool(name, **args), final())).chat("Classify", state, files)
    client = FakeClient(tool("evaluate", file_id="file-a"))
    output = Agent(service, client).chat(message, state, files)
    assert expected in output["reply"] and len(service.calls) == 1
    assert client.requests[0]["tool_choice"] == "none"


def test_mixed_stored_prediction_and_batch_metrics_do_not_enable_tools(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())).chat("Classify row 0", state, files)
    Agent(service, FakeClient(tool("predict_batch", file_id="file-b"), final())).chat("Classify all", state, files)
    client = FakeClient(final(result_ids=[r["result_id"] for r in state["results"]]))
    output = Agent(service, client).chat("What is the malware probability and goodware count?", state, files)
    assert "malware probability 0.870000" in output["reply"] and "0 goodware" in output["reply"]
    assert len(service.calls) == 2 and not output["activity"] and not output["results"]
    assert client.requests[0]["tools"] == [] and client.requests[0]["tool_choice"] == "none"


@pytest.mark.parametrize("message", [
    "What is the malware probability of row 0?",
    "Is row 0 malware?",
    "Tell me if row 0 of file-b is malware.",
    "What was the malware probability of row 0 in that batch?",
])
def test_row_question_after_a_batch_classifies_that_row(files, message):
    # Live regression: after a batch, per-record questions got the batch counts
    # or "unavailable" because stored-result mode had disabled every tool.
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("predict_batch", file_id="file-b"), final())).chat(
        "Classify all rows in file-b", state, files)
    client = FakeClient(tool("predict_single", file_id="file-b", row_index=0), final())
    output = Agent(service, client).chat(message, state, files)
    assert [call[0] for call in service.calls] == ["predict_batch", "predict_single"]
    assert [(a["tool"], a["status"]) for a in output["activity"]] == [("predict_single", "success")]
    assert "malware probability 0.870000" in output["reply"]
    assert client.requests[0]["tools"] and client.requests[0]["tool_choice"] == "auto"


def test_row_question_about_an_unpredicted_row_can_classify_it(files):
    state, service = {}, FakeService()
    Agent(service, FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())).chat(
        "Classify row 0 of file-a", state, files)
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=1), final())
    output = Agent(service, client).chat("What is the malware probability of row 1?", state, files)
    assert service.calls[-1] == ("predict_single", files["file-a"]["path"], 1)
    assert output["activity"][0]["status"] == "success"


def test_refused_summary_followup_still_shows_the_stored_result(files):
    # Live finding: "Explain what the evaluation found." got only the refusal.
    state, service = {}, FakeService(accuracy=0, confusion_matrix=[[0, 0], [1, 0]])
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate file-a", state, files)
    client = FakeClient(final(kind="explanation_unavailable"))
    output = Agent(service, client).chat("Explain what the evaluation found.", state, files)
    assert output["reply"].startswith("Feature-level explanations are unavailable")
    assert "Accuracy: 0.000000" in output["reply"] and "false negatives: 1" in output["reply"].lower()
    assert len(service.calls) == 1 and not output["activity"] and not output["results"]
    assert client.requests[0]["tools"] == [] and client.requests[0]["tool_choice"] == "none"


@pytest.mark.parametrize("focus", ["accuracy", "auc", "counts", "false_negatives"])
def test_new_evaluation_reports_every_metric_whatever_the_focus(files, focus):
    # Run 2 regression: the model picked the accuracy focus, so AUC and the matrix went unreported.
    output = Agent(FakeService(), FakeClient(tool("evaluate", file_id="file-a"), final(focus=focus))).chat(
        "Evaluate file-a and report the AUC, accuracy and confusion matrix", {}, files)
    for text in ("Accuracy: 0.900000", "AUC: 0.960000", "Confusion matrix", "Labeled class counts"):
        assert text in output["reply"]


@pytest.mark.parametrize("focus", ["accuracy", "auc"])
def test_unavailable_auc_is_always_explained_with_accuracy(files, focus):
    state, service = {}, FakeService(auc=None)
    Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", state, files)
    result_id = state["results"][0]["result_id"]
    output = Agent(service, FakeClient(final(result_ids=[result_id], focus=focus))).chat("What about that one?", state, files)
    assert "AUC: unavailable (evaluation requires both classes for AUC)." in output["reply"]
    assert "Accuracy: 0.900000." in output["reply"]


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
    # As the tool reports it, the unlabeled row is the one row left out of the metrics.
    service = FakeService(auc=None, missing_label_count=1, invalid_count=1, evaluated_count=9, evaluation_coverage=0.9)
    output = Agent(service, FakeClient(tool("evaluate", file_id="file-a"), final())).chat("Evaluate", {}, files)
    assert "AUC: unavailable" in output["reply"]
    assert "Evaluated 9 of 10 rows; 1 excluded (missing labels 1, invalid labels 0)." in output["reply"]


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


def test_incremental_events_precede_provider_and_actual_tool_execution(files):
    service = FakeService()
    client = FakeClient(tool("predict_single", file_id="file-a", row_index=0), final())
    state = {}
    events = Agent(service, client).chat_events("Predict row 0", state, files)
    assert next(events)["stage"] == "planning"
    assert not client.requests and not service.calls
    assert next(events)["stage"] == "validating"
    started = next(events)
    assert started["activity"]["status"] == "started"
    assert not service.calls
    finished = next(events)
    assert service.calls[0][0] == "predict_single"
    assert finished["index"] == started["index"] == 0
    assert finished["activity"]["status"] == "success"
    assert started["activity"]["status"] == "started"  # Immutable snapshot.
    assert next(events)["stage"] == "responding"
    with pytest.raises(StopIteration) as end:
        next(events)
    assert end.value.value["results"][0]["prediction"] == 1
    assert state["activity"][0]["status"] == "success"


@pytest.mark.parametrize("accuracy,status", [(0.89, "skipped"), (0.9, "success")])
def test_incremental_conditional_events_preserve_order_and_gate(files, accuracy, status):
    service = FakeService(accuracy)
    client = FakeClient(tool("evaluate", file_id="file-a"),
                        tool("predict_single", file_id="file-b", row_index=0) if status == "success" else final())
    events = list(Agent(service, client).chat_events("Evaluate file-a; predict row 0 of file-b only if accuracy >= 0.9", {}, files))
    trail = [event["activity"] for event in events if event["type"] == "tool"]
    assert [(entry["tool"], entry["status"]) for entry in trail] == (
        [("evaluate", "started"), ("evaluate", "success"), ("predict_single", "skipped")]
        if status == "skipped" else
        [("evaluate", "started"), ("evaluate", "success"), ("predict_single", "started"), ("predict_single", "success")])
    assert len(service.calls) == (1 if status == "skipped" else 2)
    # The decision step is visible while the model reads the evaluation.
    stages = [event["stage"] for event in events if event["type"] == "progress"]
    assert stages.index("deciding") > stages.index("planning")


def test_incremental_failed_tool_is_sanitized_and_not_successful(files):
    service = FakeService()
    def broken(*args, **kwargs):
        raise RuntimeError("PRIVATE_ROW_OR_SECRET")
    service.predict_single = broken
    events = list(Agent(service, FakeClient(tool("predict_single", file_id="file-a", row_index=0)))
                  .chat_events("Predict row 0", {}, files))
    trail = [event["activity"] for event in events if event["type"] == "tool"]
    assert [entry["status"] for entry in trail] == ["started", "error"]
    assert "PRIVATE_ROW_OR_SECRET" not in str(events)

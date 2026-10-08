"""The public comparison uses saved results and never calls an LLM or model."""
import json
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_project.web import create_app


ROOT = Path(__file__).resolve().parents[1]


class DashboardElements(HTMLParser):
    """Collect semantic controls and score text using the standard library."""
    def __init__(self, page):
        super().__init__()
        self.controls = {}
        self.folds = []
        self.options = {}
        self.text = {}
        self.stack = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            self.controls[identifier] = attrs
            self.text.setdefault(identifier, "")
        if "data-fold" in attrs:
            self.folds.append(attrs)
        if tag == "option":
            parent = next((name for _, name in reversed(self.stack) if name), None)
            self.options.setdefault(parent, []).append(attrs)
        if tag not in {"input", "link", "meta", "br"}:
            self.stack.append((tag, identifier))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for _, identifier in self.stack:
            if identifier:
                self.text[identifier] += data


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
    for path in ("/static/dashboard.css", "/static/dashboard.js", "/static/walkthrough.css", "/static/favicon.svg"):
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
    gap = production["auc_mean"] - runner_up["auc_mean"]
    assert f"The gap between the top two mean AUCs is {gap:.6f}" in page
    for difference in differences:
        assert f"{difference:+.6f}" in page
    # The production model cannot be compared with itself.
    controls = DashboardElements(page)
    assert {option["value"] for option in controls.options["fold-model"]} == {
        model["model"] for model in recorded["models"] if model["model"] != production["model"]}
    assert next(option["value"] for option in controls.options["fold-model"] if "selected" in option) == runner_up["model"]
    assert "Zoom in on differences" not in page


def test_guided_experiment_uses_saved_split_and_round_one_scores(tmp_path):
    app, recorded = make_app(tmp_path)
    page = app.test_client().get("/analytics").text
    controls = DashboardElements(page)
    winner = next(model for model in recorded["models"] if model["model"] == recorded["selected_model"])
    split = controls.text["data-split"]
    assert f"{recorded['data']['clean_rows']:,}" in split
    assert f"{recorded['training_rows']:,}" in split
    assert f"{recorded['holdout_rows']:,}" in split
    assert "80% training" in split and "20% reserved test" in split
    assert "selected and frozen" in split
    assert len(controls.folds) == recorded["folds"]
    assert [fold["data-fold"] for fold in controls.folds if "is-validation" in fold["class"]] == ["1"]
    assert [fold["data-fold"] for fold in controls.folds if "is-validation" not in fold["class"]] == [str(fold) for fold in range(2, recorded["folds"] + 1)]
    assert controls.text["walkthrough-auc"] == f"{winner['fold_auc'][0]:.6f}"
    assert controls.text["walkthrough-accuracy"] == f"{winner['fold_accuracy'][0] * 100:.3f}%"
    assert {option["value"] for option in controls.options["walkthrough-model"]} == {model["model"] for model in recorded["models"]}
    assert next(option["value"] for option in controls.options["walkthrough-model"] if "selected" in option) == recorded["selected_model"]
    assert "Each round starts with a fresh model and freshly fitted preprocessing" in page
    assert "Nothing learned in a previous round is carried over" in page
    assert "The controls display saved results" in page
    assert controls.controls["walkthrough-round"]["max"] == str(recorded["folds"])
    assert all("disabled" in controls.controls[identifier] for identifier in (
        "walkthrough-model", "walkthrough-round", "walkthrough-previous", "walkthrough-next"))
    assert page.index('id="data-split"') < page.index('id="cv-walkthrough"') < page.index('id="model-comparison"') < page.index('id="final-test"')
    holdout = recorded["holdout_metrics"]
    assert f"{holdout['confusion_matrix'][0][1]:,} legitimate files were wrongly flagged" in controls.text["final-test"]
    assert f"{holdout['confusion_matrix'][1][0]:,} malware files were missed" in controls.text["final-test"]


def test_guided_experiment_shows_cleaning_round_range_and_selection_before_final_test(tmp_path):
    app, recorded = make_app(tmp_path)
    page = app.test_client().get("/analytics").text
    controls = DashboardElements(page)
    data = recorded["data"]
    split = controls.text["data-split"]
    for count in (data["raw_rows"], data["removed_rows"], data["clean_rows"]):
        assert f"{count:,}" in split
    assert f"every row of the {data['conflicting_hashes_removed']} files labelled both malware and goodware" in split
    assert "SHA1 and FirstSeenDate never enter the model" in split
    winner = next(model for model in recorded["models"] if model["model"] == recorded["selected_model"])
    low, high = min(winner["fold_auc"]), max(winner["fold_auc"])
    # Without JavaScript the ten-round strip falls back to its range and the mean that step 3 reports.
    assert controls.text["round-strip"] == ""
    assert controls.text["round-strip-note"] == (
        f"{winner['model']} ranged from {low:.6f} to {high:.6f}. "
        f"Step 3 summarises these scores as a mean of {winner['auc_mean']:.6f} ± {winner['auc_std']:.6f}.")
    # The page follows the experiment: selection is explained before the one final test.
    assert page.index('id="model-comparison"') < page.index('id="model-selection"') < page.index('id="final-test"')
    assert f"Selection rule: {recorded['selection_rule']}. The reserved test files played no part." in controls.text["model-selection"]
    final = controls.text["final-test"]
    holdout = recorded["holdout_metrics"]
    accuracy_low, accuracy_high = min(winner["fold_accuracy"]), max(winner["fold_accuracy"])
    assert (f"ranged from {low:.6f} to {high:.6f} AUC and from "
            f"{accuracy_low * 100:.3f}% to {accuracy_high * 100:.3f}% accuracy") in final
    inside = low <= holdout["auc"] <= high and accuracy_low <= holdout["accuracy"] <= accuracy_high
    assert ("Both hold-out results fall within those ranges." in final) is inside
    matrix = holdout["confusion_matrix"]
    goodware, malware = sum(matrix[0]), sum(matrix[1])
    assert f"({matrix[0][1] / goodware * 100:.2f}% of {goodware:,} goodware files)" in final
    assert f"({matrix[1][0] / malware * 100:.2f}% of {malware:,})" in final
    assert "Wrongly flagged" in final and "false positives" in final
    assert "Missed" in final and "false negatives" in final


@pytest.mark.parametrize("auc, position", [(0.99, "below"), (0.9999, "above")])
def test_final_test_says_where_the_hold_out_auc_falls_against_the_rounds(tmp_path, auc, position):
    recorded = json.loads((ROOT / "models/comparison.json").read_text())
    recorded["holdout_metrics"]["auc"] = auc
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(recorded))
    app, _ = make_app(tmp_path, comparison_path=path)
    final = DashboardElements(app.test_client().get("/analytics").text).text["final-test"]
    assert f"The hold-out accuracy falls within its range; the AUC is {position} its range." in final
    assert "Both hold-out results fall within those ranges." not in final

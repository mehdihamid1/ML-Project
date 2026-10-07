import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ml_project.comparison import MAX_REPORT_BYTES, MODEL_NAMES, fold_comparison, load_comparison
from scripts.export_model_comparison import METADATA_FIELDS, MODEL_FIELDS, export_comparison


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report():
    return json.loads((ROOT / "models/comparison.json").read_text())


def write_report(tmp_path, report):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report))
    return path


def recorded_sources(tmp_path, report):
    """Reconstruct source fixtures entirely from files present in clean checkouts."""
    source = tmp_path / "experiment"
    source.mkdir()
    metadata = json.loads((ROOT / "docs/training-metadata.json").read_text())
    records = copy.deepcopy(list(reversed(report["models"])))
    for record in records:
        record["parameters"] = {"training_only": "must not enter the public export"}
    (source / "cv-results.json").write_text(json.dumps(records))
    (source / "metadata.json").write_text(json.dumps(metadata))
    return source, records, metadata


def test_bundled_report_matches_frozen_metadata_and_source_checksum(report):
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    loaded = load_comparison(ROOT / "models/comparison.json", model_version=manifest["model_version"])
    metadata_bytes = (ROOT / "docs/training-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    assert loaded == report
    for field in METADATA_FIELDS:
        assert loaded[field] == metadata[field]
    assert {row["model"] for row in loaded["models"]} == MODEL_NAMES
    assert loaded["models"][0]["model"] == manifest["selected_model"]
    assert next(source["sha256"] for source in loaded["sources"]
                if source["name"] == "metadata.json") == hashlib.sha256(metadata_bytes).hexdigest()


def test_export_preserves_exact_recorded_metrics_and_is_reproducible(tmp_path, report):
    source, records, metadata = recorded_sources(tmp_path, report)
    output = tmp_path / "public/comparison.json"
    result = export_comparison(source, output)
    first = output.read_bytes()
    export_comparison(source, output)
    assert output.read_bytes() == first
    for row in result["models"]:
        original = next(record for record in records if record["model"] == row["model"])
        assert row == {field: original[field] for field in MODEL_FIELDS}
        assert "parameters" not in row
    for field in METADATA_FIELDS:
        assert result[field] == metadata[field]
    for provenance in result["sources"]:
        assert provenance["sha256"] == hashlib.sha256((source / provenance["name"]).read_bytes()).hexdigest()
    assert load_comparison(output) == result


def test_export_cli_requires_only_standard_library(tmp_path, report):
    source, _, _ = recorded_sources(tmp_path, report)
    output = tmp_path / "comparison.json"
    command = [sys.executable, "-S", str(ROOT / "scripts/export_model_comparison.py"),
               "--artifacts", str(source), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["models"] == len(MODEL_NAMES)
    assert load_comparison(output)["model_version"] == report["model_version"]


@pytest.mark.parametrize("field,value", [
    ("smoke", True), ("selected_model", "CatBoost"), ("folds", 2),
])
def test_export_rejects_incomplete_or_changed_experiment_without_overwriting(
        tmp_path, report, field, value):
    source, _, metadata = recorded_sources(tmp_path, report)
    metadata[field] = value
    (source / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "comparison.json"
    output.write_text("keep the previous report")
    with pytest.raises(ValueError):
        export_comparison(source, output)
    assert output.read_text() == "keep the previous report"


@pytest.mark.parametrize("change,match", [
    (lambda r: r["models"][0].update(auc_mean=0.5), "summary disagrees"),
    (lambda r: r["models"][0].update(accuracy_std=0.5), "summary disagrees"),
    (lambda r: r["models"][0].update(fit_seconds_mean=float("nan")), "Non-finite"),
    (lambda r: r["models"][0].update(fit_seconds_mean=-1), "outside its range"),
    (lambda r: r["models"][0].update(fit_seconds_mean=10 ** 1000), "finite number"),
    (lambda r: r["models"][0]["fold_auc"].pop(), "one auc score per fold"),
    (lambda r: r["models"].pop(), "all seven"),
    (lambda r: r["models"].__setitem__(1, copy.deepcopy(r["models"][0])), "repeated model"),
    (lambda r: r["models"].reverse(), "recorded ranking"),
    (lambda r: r.update(selected_model="XGBoost"), "winner disagrees"),
    (lambda r: r.update(training_rows=r["training_rows"] - 1), "dataset counts"),
    (lambda r: r.update(schema_version=2), "Unsupported"),
    (lambda r: r["holdout_metrics"].update(accuracy=1), "summary disagrees"),
    (lambda r: r["holdout_metrics"]["confusion_matrix"][0].__setitem__(1, 0), "confusion counts"),
    (lambda r: r["holdout_metrics"].update(confusion_matrix_order=[1, 0]), "matrix order"),
    (lambda r: r["sources"][0].update(sha256="bad"), "checksum"),
    (lambda r: r.update(threshold=True), "finite number"),
])
def test_loader_rejects_inconsistent_or_malformed_metrics(tmp_path, report, change, match):
    change(report)
    with pytest.raises(ValueError, match=match):
        load_comparison(write_report(tmp_path, report))


def test_loader_rejects_production_version_mismatch(tmp_path, report):
    with pytest.raises(ValueError, match="production model"):
        load_comparison(write_report(tmp_path, report), model_version="different-model")


@pytest.mark.parametrize("raw", [b"not json", b"\xff", b"[]"])
def test_loader_rejects_non_json_or_non_object_report(tmp_path, raw):
    path = tmp_path / "comparison.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_comparison(path)


def test_loader_rejects_missing_and_oversized_reports(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(OSError):
        load_comparison(path)
    path.write_bytes(b" " * (MAX_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_comparison(path)


def test_fold_comparison_pairs_shared_folds_and_counts_each_direction(report):
    models = {row["model"]: row for row in report["models"]}
    production, other = models[report["selected_model"]], report["models"][1]
    differences = [mine - theirs for mine, theirs in zip(production["fold_auc"], other["fold_auc"])]
    paired = fold_comparison(report, other["model"])
    assert [fold["fold"] for fold in paired["folds"]] == list(range(1, report["folds"] + 1))
    assert [fold["difference"] for fold in paired["folds"]] == differences
    assert paired["production_higher"] == sum(difference > 0 for difference in differences)
    assert paired["production_higher"] + paired["other_higher"] + paired["ties"] == report["folds"]
    # On shared folds the mean paired difference equals the gap between the means.
    assert paired["mean_difference"] == pytest.approx(production["auc_mean"] - other["auc_mean"], abs=1e-12)
    accuracy = fold_comparison(report, other["model"], metric="accuracy")
    assert [fold["production"] for fold in accuracy["folds"]] == production["fold_accuracy"]


@pytest.mark.parametrize("other,metric", [("production", "auc"), ("Unknown model", "auc"), ("XGBoost", "fit_seconds")])
def test_fold_comparison_rejects_self_unknown_models_and_metrics(report, other, metric):
    with pytest.raises(ValueError):
        fold_comparison(report, report["selected_model"] if other == "production" else other, metric=metric)

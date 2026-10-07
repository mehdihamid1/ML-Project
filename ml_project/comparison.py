"""Read the small, public experiment summary without training dependencies."""

import json
import math
from pathlib import Path
import re
import statistics


MODEL_NAMES = frozenset({
    "Logistic Regression", "Decision Tree", "Random Forest", "PyTorch MLP",
    "XGBoost", "LightGBM", "CatBoost",
})
SELECTION_RULE = "CV AUC descending, then accuracy descending, then fit time ascending"
MAX_REPORT_BYTES = 64 * 1024


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Invalid comparison {name}: expected an object")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid comparison {name}: expected an integer >= {minimum}")
    return value


def _number(value, name, minimum=0, maximum=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"Invalid comparison {name}: expected a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid comparison {name}: number outside its range")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Invalid comparison {name}: expected a nonempty bounded string")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid comparison {name}: expected a SHA-256 checksum")


def _same(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Invalid comparison {name}: summary disagrees with recorded values")


def validate_comparison(report, *, model_version=None):
    """Reject malformed or inconsistent metrics; preserve all recorded values."""
    report = _mapping(report, "report")
    if _integer(report.get("schema_version"), "schema_version", 1) != 1:
        raise ValueError("Unsupported comparison schema version")
    version = _string(report.get("model_version"), "model_version")
    if model_version is not None and version != model_version:
        raise ValueError("Comparison model version differs from the production model")
    if _integer(report.get("folds"), "folds", 2) != 10:
        raise ValueError("The comparison must contain the completed 10-fold experiment")
    if _integer(report.get("random_state"), "random_state") != 42:
        raise ValueError("The comparison experiment must use random state 42")
    _number(report.get("threshold"), "threshold", maximum=1)
    training_rows = _integer(report.get("training_rows"), "training_rows", 1)
    holdout_rows = _integer(report.get("holdout_rows"), "holdout_rows", 1)
    data = _mapping(report.get("data"), "data")
    raw_rows = _integer(data.get("raw_rows"), "raw_rows", 1)
    clean_rows = _integer(data.get("clean_rows"), "clean_rows", 1)
    removed_rows = _integer(data.get("removed_rows"), "removed_rows")
    conflicts = _integer(data.get("conflicting_hashes_removed"), "conflicting_hashes_removed")
    if clean_rows != training_rows + holdout_rows or raw_rows != clean_rows + removed_rows:
        raise ValueError("Comparison dataset counts disagree")
    if conflicts > removed_rows:
        raise ValueError("Comparison conflict count exceeds removed rows")
    _sha256(report.get("dataset_sha256"), "dataset_sha256")
    if report.get("selection_rule") != SELECTION_RULE:
        raise ValueError("Unsupported comparison model selection rule")
    _string(report.get("search_scope"), "search_scope")

    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Comparison must identify both recorded source files")
    names = []
    for source in sources:
        source = _mapping(source, "source")
        names.append(_string(source.get("name"), "source name"))
        _sha256(source.get("sha256"), "source checksum")
    if sorted(names) != ["cv-results.json", "metadata.json"]:
        raise ValueError("Comparison source file names disagree")

    rows = report.get("models")
    if not isinstance(rows, list) or len(rows) != len(MODEL_NAMES):
        raise ValueError("Comparison must include all seven models")
    seen = set()
    for row in rows:
        row = _mapping(row, "model record")
        name = _string(row.get("model"), "model name")
        if name not in MODEL_NAMES or name in seen:
            raise ValueError("Comparison has an unknown or repeated model")
        seen.add(name)
        _integer(row.get("configuration"), "configuration")
        _number(row.get("fit_seconds_mean"), "fit_seconds_mean")
        for metric in ("auc", "accuracy"):
            scores = row.get(f"fold_{metric}")
            if not isinstance(scores, list) or len(scores) != report["folds"]:
                raise ValueError(f"Comparison {name} needs one {metric} score per fold")
            for value in scores:
                _number(value, f"fold_{metric}", maximum=1)
            mean = _number(row.get(f"{metric}_mean"), f"{metric}_mean", maximum=1)
            std = _number(row.get(f"{metric}_std"), f"{metric}_std", maximum=1)
            _same(mean, statistics.mean(scores), f"{name} {metric} mean")
            _same(std, statistics.pstdev(scores), f"{name} {metric} standard deviation")
    ranked = sorted(rows, key=lambda row: (
        -row["auc_mean"], -row["accuracy_mean"], row["fit_seconds_mean"],
    ))
    if rows != ranked:
        raise ValueError("Comparison model rows disagree with the recorded ranking")
    if report.get("selected_model") != rows[0]["model"]:
        raise ValueError("Comparison winner disagrees with the recorded CV selection")

    holdout = _mapping(report.get("holdout_metrics"), "holdout_metrics")
    if _integer(holdout.get("sample_count"), "holdout sample_count", 1) != holdout_rows:
        raise ValueError("Comparison hold-out sample count disagrees")
    _number(holdout.get("auc"), "holdout AUC", maximum=1)
    accuracy = _number(holdout.get("accuracy"), "holdout accuracy", maximum=1)
    if holdout.get("class_mapping") != {"0": "goodware", "1": "malware"}:
        raise ValueError("Comparison hold-out class mapping disagrees")
    if holdout.get("confusion_matrix_order") != [0, 1]:
        raise ValueError("Comparison hold-out confusion matrix order disagrees")
    matrix = holdout.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2:
        raise ValueError("Comparison hold-out confusion matrix must be 2 x 2")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Comparison hold-out confusion matrix must be 2 x 2")
        for count in row:
            _integer(count, "holdout confusion count")
    if sum(sum(row) for row in matrix) != holdout_rows:
        raise ValueError("Comparison hold-out confusion counts disagree")
    _same(accuracy, (matrix[0][0] + matrix[1][1]) / holdout_rows, "holdout accuracy")
    return report


def fold_comparison(report, other, metric="auc"):
    """Pair the production model's scores with another model's on the shared CV folds.

    Every model used identical folds, so per-fold differences compare like with
    like. The counts describe those folds; they are not a significance test.
    """
    if metric not in ("auc", "accuracy"):
        raise ValueError("Unsupported fold comparison metric")
    models = {row["model"]: row for row in report["models"]}
    production = report["selected_model"]
    if other not in models or other == production:
        raise ValueError("Choose a different recorded model to compare")
    key = f"fold_{metric}"
    folds = [{"fold": index, "production": mine, "other": theirs, "difference": mine - theirs}
             for index, (mine, theirs) in enumerate(zip(models[production][key], models[other][key]), start=1)]
    return {
        "metric": metric, "production": production, "other": other, "folds": folds,
        "production_higher": sum(fold["difference"] > 0 for fold in folds),
        "other_higher": sum(fold["difference"] < 0 for fold in folds),
        "ties": sum(fold["difference"] == 0 for fold in folds),
        "mean_difference": statistics.mean(fold["difference"] for fold in folds),
    }


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant in comparison report: {value}")


def load_comparison(path, *, model_version=None):
    """Load a bounded runtime summary; OSError and ValueError are intentional."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("Comparison report exceeds the runtime size limit")
    try:
        report = json.loads(raw, parse_constant=_reject_constant)
    except UnicodeDecodeError as error:
        raise ValueError("Comparison report is not valid UTF-8 JSON") from error
    return validate_comparison(report, model_version=model_version)

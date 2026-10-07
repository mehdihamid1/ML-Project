"""Export recorded experiment metrics for the dashboard; never fit any model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_project.comparison import validate_comparison


MODEL_FIELDS = (
    "model", "configuration", "auc_mean", "auc_std", "accuracy_mean",
    "accuracy_std", "fit_seconds_mean", "fold_auc", "fold_accuracy",
)
METADATA_FIELDS = (
    "model_version", "selected_model", "folds", "random_state", "threshold",
    "training_rows", "holdout_rows", "data", "dataset_sha256", "selection_rule",
    "search_scope", "holdout_metrics",
)


def export_comparison(artifacts, output):
    """Project the saved selected CV records, keeping exact numeric precision."""
    artifacts, output = Path(artifacts), Path(output)
    contents = {name: (artifacts / name).read_bytes()
                for name in ("cv-results.json", "metadata.json")}
    metadata = json.loads(contents["metadata.json"])
    records = json.loads(contents["cv-results.json"])
    if not isinstance(metadata, dict) or metadata.get("smoke") is not False:
        raise ValueError("Only a completed full training experiment can be exported")
    if not isinstance(records, list):
        raise ValueError("CV source must contain a list of recorded model results")
    try:
        rows = [{key: record[key] for key in MODEL_FIELDS} for record in records]
        rows.sort(key=lambda row: (
            -row["auc_mean"], -row["accuracy_mean"], row["fit_seconds_mean"],
        ))
        report = {
            "schema_version": 1,
            **{key: metadata[key] for key in METADATA_FIELDS},
            "sources": [{"name": name, "sha256": hashlib.sha256(raw).hexdigest()}
                        for name, raw in contents.items()],
            "models": rows,
        }
    except (KeyError, TypeError) as error:
        raise ValueError("Recorded experiment is missing required dashboard fields") from error
    validate_comparison(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training-final"))
    parser.add_argument("--output", type=Path, default=Path("models/comparison.json"))
    args = parser.parse_args()
    report = export_comparison(args.artifacts, args.output)
    print(json.dumps({
        "output": str(args.output), "models": len(report["models"]),
        "folds": report["folds"], "selected_model": report["selected_model"],
        "model_version": report["model_version"],
        "sources": report["sources"],
    }, indent=2))


if __name__ == "__main__":
    main()

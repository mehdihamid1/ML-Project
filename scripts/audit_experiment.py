"""Audit the frozen experiment without fitting, predicting, or changing its split.

Training membership comes exclusively from training-ids.csv.  Comparisons use
canonical cells, rather than a join that can expand duplicate groups into a
large table.  The report contains aggregate counts and recorded settings only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"SHA1", "FirstSeenDate", "Label", "Magic", "PE_TYPE", "SizeOfOptionalHeader"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_rows(frame, columns):
    # Missing values use None, distinct from a literal text missing-value token.
    # Both partitions have the original CSV's dtypes, so numeric strings cannot
    # accidentally compare differently because a small CSV inferred another type.
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        yield tuple(None if pd.isna(value) else value for value in row)


def exact_overlap(training, holdout, columns):
    """Count exact canonical rows, without materializing matching row pairs."""
    left = Counter(_canonical_rows(training, columns))
    right = Counter(_canonical_rows(holdout, columns))
    common = left.keys() & right.keys()
    return {
        "columns": list(columns),
        "shared_distinct_vectors": len(common),
        "training_rows_with_match": sum(left[key] for key in common),
        "holdout_rows_with_match": sum(right[key] for key in common),
        "matching_row_pairs": sum(left[key] * right[key] for key in common),
    }


def audit_partitions(raw, training_ids, holdout, feature_schema=None):
    """Recover the saved split and check its source data and overlap invariants."""
    if not {"SHA1", "Label"}.issubset(raw.columns):
        raise ValueError("Original CSV must contain SHA1 and Label")
    if raw.SHA1.isna().any() or raw.Label.isna().any():
        raise ValueError("Original SHA1 and Label must not be missing")
    if not set(raw.Label.unique()).issubset({0, 1}):
        raise ValueError("Original labels must be 0 or 1")
    if "SHA1" not in training_ids.columns or "SHA1" not in holdout.columns:
        raise ValueError("Saved training and hold-out partitions must contain SHA1")
    if set(holdout.columns) != set(raw.columns):
        raise ValueError("Saved hold-out columns differ from the original CSV")
    if "partition" in training_ids and not training_ids.partition.eq("training").all():
        raise ValueError("training-ids.csv contains a non-training partition")
    for name, frame in (("training", training_ids), ("hold-out", holdout)):
        if frame.SHA1.isna().any() or not frame.SHA1.is_unique:
            raise ValueError(f"Saved {name} SHA1 IDs must be present and unique")

    label_counts = raw.groupby("SHA1")["Label"].nunique()
    conflicting_ids = set(label_counts[label_counts > 1].index)
    clean = raw.loc[~raw.SHA1.isin(conflicting_ids)].drop_duplicates("SHA1")
    clean_ids = set(clean.SHA1)
    left_ids, right_ids = set(training_ids.SHA1), set(holdout.SHA1)
    unknown = (left_ids | right_ids) - clean_ids
    if unknown:
        raise ValueError(f"Frozen partitions contain {len(unknown)} unknown or conflicting SHA1 IDs")
    missing = clean_ids - (left_ids | right_ids)
    if missing:
        raise ValueError(f"Frozen partitions omit {len(missing)} cleaned dataset SHA1 IDs")

    source = clean.set_index("SHA1", drop=False)
    training = source.loc[training_ids.SHA1].reset_index(drop=True)
    expected_holdout = source.loc[holdout.SHA1].reset_index(drop=True)
    # Apply one schema to both sides. Do not infer types from each partition.
    # Do not let astype(int) silently truncate a changed fractional value.
    for column, dtype in raw.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            values = pd.to_numeric(holdout[column], errors="raise")
            if values.isna().any() or values.mod(1).ne(0).any():
                raise ValueError(f"Saved hold-out integer column has missing or fractional values: {column}")
    holdout = holdout.loc[:, raw.columns].astype(raw.dtypes.to_dict()).reset_index(drop=True)
    equal = holdout.eq(expected_holdout) | (holdout.isna() & expected_holdout.isna())
    feature_columns = [column for column in raw.columns if column not in EXCLUDED]
    if feature_schema is not None:
        saved_columns = [item["name"] for item in feature_schema]
        if saved_columns != feature_columns:
            raise ValueError("Recorded feature schema differs from non-leak, non-constant source inputs")
        for item in feature_schema:
            if str(raw[item["name"]].dtype) != item["dtype"]:
                raise ValueError(f"Recorded dtype differs from original CSV: {item['name']}")

    same_sha1 = len(left_ids & right_ids)
    mismatched_rows = int((~equal.all(axis=1)).sum())
    result = {
        "definitions": {
            "same_file": "An identical SHA1 occurs in the saved training IDs and saved hold-out IDs.",
            "model_input_vector": "All raw model-input cells match after original-CSV dtype normalization; SHA1, FirstSeenDate, Label and the three excluded constants are omitted. This is before fitted encoding.",
            "nonleak_record_with_label": "All original cells except SHA1 and FirstSeenDate match, including Label and the constant columns.",
            "missing_values": "Missing cells match one another; a literal text token remains distinct from a missing cell. Text is compared verbatim and numeric values without rounding or tolerance.",
            "training_recovery": "Only the SHA1 membership and order saved in training-ids.csv; no new split is made.",
        },
        "raw_rows": len(raw),
        "raw_unique_sha1": int(raw.SHA1.nunique()),
        "repeated_sha1_groups": int((raw.groupby("SHA1").size() > 1).sum()),
        "conflicting_sha1_removed": len(conflicting_ids),
        "clean_rows": len(clean),
        "removed_rows": len(raw) - len(clean),
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "training_class_counts": {str(label): int(count) for label, count in training.Label.value_counts().sort_index().items()},
        "holdout_class_counts": {str(label): int(count) for label, count in holdout.Label.value_counts().sort_index().items()},
        "same_sha1_overlap": same_sha1,
        "partition_union_equals_cleaned_data": True,
        "holdout_label_mismatches": int((~equal.Label).sum()),
        "holdout_source_mismatch_rows": mismatched_rows,
        "holdout_source_mismatch_cells": int((~equal).to_numpy().sum()),
        "model_input_overlap": exact_overlap(training, holdout, feature_columns),
        "nonleak_record_with_label_overlap": exact_overlap(training, holdout, [column for column in raw.columns if column not in {"SHA1", "FirstSeenDate"}]),
        "partition_integrity_passed": same_sha1 == 0 and mismatched_rows == 0,
    }
    return result


def _rank(row):
    return -row["auc_mean"], -row["accuracy_mean"], row["fit_seconds_mean"]


def audit_cv(metadata, results, candidates):
    """Check recorded fold summaries and selections; never run CV again."""
    configurations = {}
    summaries = []
    for row in candidates:
        name = row["model"]
        configurations.setdefault(name, []).append(row["configuration"])
        for metric in ("auc", "accuracy"):
            folds = np.asarray(row[f"fold_{metric}"], dtype=float)
            if len(folds) != metadata["folds"] or not np.isfinite(folds).all():
                raise ValueError(f"Invalid saved {metric} folds for {name}")
            for statistic, computed in (("mean", folds.mean()), ("std", folds.std(ddof=0))):
                if not np.isclose(row[f"{metric}_{statistic}"], computed, rtol=0, atol=1e-12):
                    raise ValueError(f"Saved {metric} {statistic} disagrees with folds for {name}")
        params = row["parameters"]
        summaries.append({
            "model": name, "configuration": row["configuration"],
            "auc_mean": row["auc_mean"], "accuracy_mean": row["accuracy_mean"],
            "recorded_parameters": {key: params[key] for key in ("C", "max_depth", "min_samples_leaf", "n_estimators", "hidden", "epochs", "batch_size", "learning_rate", "num_leaves", "iterations", "depth", "random_state", "random_seed", "cat_features") if key in params},
        })
    if len(results) != len(configurations) or len({row["model"] for row in results}) != len(results):
        raise ValueError("CV results must have one selection per recorded model")
    for selected in results:
        best = min((row for row in candidates if row["model"] == selected["model"]), key=_rank)
        if selected["configuration"] != best["configuration"] or _rank(selected) != _rank(best):
            raise ValueError(f"Saved candidate selection disagrees with CV ranking: {selected['model']}")
        for key in ("auc_std", "accuracy_std", "fold_auc", "fold_accuracy"):
            if not np.array_equal(selected[key], best[key]):
                raise ValueError(f"Saved CV summary differs from selected search candidate: {selected['model']}")
    ranked = sorted(results, key=_rank)
    winner, runner_up = ranked[:2]
    if winner["model"] != metadata["selected_model"]:
        raise ValueError("Recorded production model differs from saved CV winner")
    gap = winner["auc_mean"] - runner_up["auc_mean"]
    return {
        "folds": metadata["folds"], "random_state": metadata["random_state"],
        "selection_rule": metadata["selection_rule"], "search_scope": metadata["search_scope"],
        "models_compared": len(results), "candidate_configurations": len(candidates),
        "configurations_by_model": {name: sorted(values) for name, values in configurations.items()},
        "fold_summary_checks_passed": True,
        "configuration_summaries": summaries,
        "selected_model": winner["model"], "selected_configuration": winner["configuration"],
        "selected_recorded_parameters": winner["parameters"],
        "best_auc_mean": winner["auc_mean"], "best_auc_std": winner["auc_std"],
        "runner_up_model": runner_up["model"], "runner_up_auc_mean": runner_up["auc_mean"], "runner_up_auc_std": runner_up["auc_std"],
        "top_two_auc_mean_gap": gap,
        "gap_smaller_than_each_fold_std": gap < min(winner["auc_std"], runner_up["auc_std"]),
        "selected_fit_seconds_mean": winner["fit_seconds_mean"],
        "fastest_selected_candidate_model": min(results, key=lambda row: row["fit_seconds_mean"])["model"],
        "interpretation_limit": "Fold standard deviations describe variability; this comparison does not establish a statistically significant difference or equivalence.",
    }


def audit(data, artifacts, model, manifest):
    artifacts, model = Path(artifacts), Path(model)
    metadata = json.loads((artifacts / "metadata.json").read_text())
    manifest = json.loads(Path(manifest).read_text())
    actual_dataset_sha = sha256(data)
    if actual_dataset_sha != metadata["dataset_sha256"]:
        raise ValueError("Original dataset checksum differs from frozen metadata")
    model_sha = sha256(model)
    frozen_model_sha = sha256(artifacts / "production.joblib")
    if model_sha != manifest["sha256"] or model_sha != frozen_model_sha:
        raise ValueError("Bundled or frozen production artifact checksum mismatch")
    for key in ("selected_model", "model_version", "dataset_sha256", "threshold"):
        if metadata[key] != manifest[key]:
            raise ValueError(f"Frozen metadata differs from model manifest: {key}")

    raw = pd.read_csv(data)
    training_ids = pd.read_csv(artifacts / "training-ids.csv", dtype={"SHA1": raw.SHA1.dtype})
    holdout = pd.read_csv(artifacts / "holdout.csv", dtype=raw.dtypes.to_dict())
    partitions = audit_partitions(raw, training_ids, holdout, metadata["features"])
    expected_counts = {
        "raw_rows": metadata["data"]["raw_rows"],
        "clean_rows": metadata["data"]["clean_rows"],
        "removed_rows": metadata["data"]["removed_rows"],
        "conflicting_sha1_removed": metadata["data"]["conflicting_hashes_removed"],
        "training_rows": metadata["training_rows"], "holdout_rows": metadata["holdout_rows"],
    }
    if any(partitions[key] != count for key, count in expected_counts.items()):
        raise ValueError("Frozen metadata counts disagree with recovered partitions")
    cv = audit_cv(metadata, json.loads((artifacts / "cv-results.json").read_text()), json.loads((artifacts / "search-results.json").read_text()))
    # Loading this trusted, checksum-verified repository artifact is read-only.
    # Do not call fit, predict, predict_proba, or evaluate on it.
    sys.path.insert(0, str(ROOT))
    import joblib
    bundle = joblib.load(model)
    encoder = bundle["pipeline"].named_steps["features"]
    for key in ("selected_model", "model_version", "dataset_sha256", "threshold", "features"):
        if bundle["metadata"][key] != metadata[key]:
            raise ValueError(f"Production bundle metadata disagrees with frozen metadata: {key}")
    metrics = metadata["holdout_metrics"]
    if metrics["confusion_matrix_order"] != [0, 1] or metrics["sample_count"] != partitions["holdout_rows"]:
        raise ValueError("Recorded hold-out metric count or class order differs from saved data")
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    if tn + fp + fn + tp != partitions["holdout_rows"]:
        raise ValueError("Recorded confusion matrix does not total the saved hold-out count")
    if tn + fp != partitions["holdout_class_counts"]["0"] or fn + tp != partitions["holdout_class_counts"]["1"]:
        raise ValueError("Recorded confusion matrix disagrees with saved hold-out labels")
    if not np.isclose((tn + tp) / (tn + fp + fn + tp), metrics["accuracy"], rtol=0, atol=1e-12):
        raise ValueError("Recorded accuracy disagrees with recorded confusion matrix")
    return {
        "audit_version": 1,
        "operations": "Read-only audit of frozen membership, source cells, recorded CV scores/settings, model bytes and recorded hold-out confusion matrix; no training, reselection, tuning, predictions or new hold-out evaluation.",
        "dataset_sha256": actual_dataset_sha,
        "partitions": partitions,
        "cross_validation": cv,
        "production": {
            "model_version": metadata["model_version"], "threshold": metadata["threshold"],
            "model_bytes": model.stat().st_size, "model_sha256": model_sha,
            "matches_manifest_sha256": True, "matches_frozen_training_artifact": True,
            "raw_model_input_columns": len(metadata["features"]),
            "text_vocabulary_limit_per_import_list": encoder.max_features,
            "fitted_import_list_vocabulary_sizes": {name: len(vector.vocabulary_) for name, vector in encoder.vectors_.items()},
            "identify_category_limit_for_non_native_encoding": encoder.categories_.max_categories,
        },
        "recorded_holdout": {
            "auc": metrics["auc"], "accuracy": metrics["accuracy"],
            "threshold": metadata["threshold"], "sample_count": metrics["sample_count"],
            "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
            "goodware_count": tn + fp, "malware_count": fn + tp,
            "false_positive_rate": fp / (tn + fp), "false_negative_rate": fn / (fn + tp),
            "metrics_source": "metadata.json from the original frozen training run; not reevaluated by this audit.",
        },
        "checks_passed": partitions["partition_integrity_passed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/brazilian-malware.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training-final"))
    parser.add_argument("--model", type=Path, default=Path("models/production.joblib"))
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiment-audit.json"))
    args = parser.parse_args()
    report = audit(args.data, args.artifacts, args.model, args.manifest)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

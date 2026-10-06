import numpy as np
import pandas as pd
import pytest

from scripts.audit_experiment import audit_partitions, exact_overlap


def rows():
    return pd.DataFrame({
        "SHA1": ["a", "b", "c", "d"],
        "Label": [0, 1, 0, 1],
        "FirstSeenDate": ["1970", "2019", "1970", "2019"],
        "Magic": [267] * 4,
        "Size": [10, 20, 30, 40],
        "Entropy": [1.25, 2.5, 3.75, 5.0],
        "Identify": [None, "compiler", "packer", "compiler"],
        "ImportedDlls": ["kernel32.dll"] * 4,
    })


def test_different_sha1_with_identical_model_inputs_is_detected():
    raw = rows()
    for column in ["Size", "Entropy", "Identify", "ImportedDlls"]:
        raw.loc[2, column] = raw.loc[0, column]
    # Dates, hashes, and even labels do not distinguish model input vectors.
    raw.loc[2, "Label"] = 1
    result = audit_partitions(raw, raw.iloc[:2][["SHA1"]], raw.iloc[2:])
    assert result["same_sha1_overlap"] == 0
    assert result["model_input_overlap"]["shared_distinct_vectors"] == 1
    assert result["model_input_overlap"]["holdout_rows_with_match"] == 1
    assert result["nonleak_record_with_label_overlap"]["shared_distinct_vectors"] == 0


def test_same_file_overlap_marks_integrity_failure():
    raw = rows()
    result = audit_partitions(raw, raw.iloc[:3][["SHA1"]], raw.iloc[2:])
    assert result["same_sha1_overlap"] == 1
    assert result["partition_integrity_passed"] is False


def test_conflicting_hashes_removed_and_saved_membership_preserved():
    raw = rows()
    raw = pd.concat([raw, raw.iloc[[0]], raw.iloc[[1]].assign(Label=0)], ignore_index=True)
    result = audit_partitions(raw, pd.DataFrame({"SHA1": ["d", "a"]}), raw.iloc[[2]])
    assert result["conflicting_sha1_removed"] == 1
    assert result["clean_rows"] == 3
    assert result["removed_rows"] == 3
    assert result["training_class_counts"] == {"0": 1, "1": 1}
    assert result["partition_integrity_passed"] is True
    with pytest.raises(ValueError, match="unknown or conflicting"):
        audit_partitions(raw, pd.DataFrame({"SHA1": ["a", "b", "d"]}), raw.iloc[[2]])


def test_missing_partition_member_is_rejected():
    raw = rows()
    with pytest.raises(ValueError, match="omit 1 cleaned dataset"):
        audit_partitions(raw, raw.iloc[:2][["SHA1"]], raw.iloc[[2]])


def test_holdout_cell_or_label_changes_are_reported():
    raw = rows()
    holdout = raw.iloc[2:].copy()
    holdout.loc[2, "Label"] = 1
    holdout.loc[3, "Entropy"] = 5.5
    result = audit_partitions(raw, raw.iloc[:2][["SHA1"]], holdout)
    assert result["holdout_label_mismatches"] == 1
    assert result["holdout_source_mismatch_cells"] == 2
    assert result["holdout_source_mismatch_rows"] == 2
    assert result["partition_integrity_passed"] is False


def test_missing_value_matches_but_literal_missing_token_is_distinct():
    training = pd.DataFrame({"Size": [10, 10, 10], "Identify": [None, np.nan, "__missing__"]})
    holdout = pd.DataFrame({"Size": [10.0], "Identify": [None]})
    result = exact_overlap(training, holdout, ["Size", "Identify"])
    assert result["shared_distinct_vectors"] == 1
    assert result["training_rows_with_match"] == 2
    assert result["matching_row_pairs"] == 2


def test_exact_overlap_counts_groups_and_pairs_without_joining():
    training = pd.DataFrame({"Size": [1] * 500})
    holdout = pd.DataFrame({"Size": [1] * 400})
    result = exact_overlap(training, holdout, ["Size"])
    assert result["shared_distinct_vectors"] == 1
    assert result["training_rows_with_match"] == 500
    assert result["holdout_rows_with_match"] == 400
    assert result["matching_row_pairs"] == 200000


def test_duplicate_saved_membership_is_rejected():
    raw = rows()
    with pytest.raises(ValueError, match="present and unique"):
        audit_partitions(raw, pd.DataFrame({"SHA1": ["a", "a", "b"]}), raw.iloc[2:])


def test_fractional_integer_feature_is_not_silently_truncated():
    raw = rows()
    holdout = raw.iloc[2:].astype({"Size": float})
    holdout.loc[2, "Size"] = 30.5
    with pytest.raises(ValueError, match="fractional values: Size"):
        audit_partitions(raw, raw.iloc[:2][["SHA1"]], holdout)

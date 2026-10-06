import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_project.tools import ToolService


class ProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.copy())
        probability = frame["Size"].to_numpy(dtype=float)
        return np.column_stack([1 - probability, probability])

    def predict(self, frame):
        raise AssertionError("All classifications must come from probabilities")


@pytest.fixture
def service():
    return ToolService({"pipeline": ProbabilityModel(), "metadata": {
        "features": [{"name": "Size", "dtype": "float64", "allow_missing": False},
                     {"name": "Identify", "dtype": "object", "allow_missing": True}],
        "threshold": 0.5, "model_version": "test-v1", "class_mapping": {"0": "goodware", "1": "malware"},
    }})


def write_csv(tmp_path, rows, header=("Size", "Identify", "SHA1", "Label"), name="input.csv"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_single_uses_schema_probabilities_and_saved_threshold(service, tmp_path):
    service.threshold = 0.8
    path = write_csv(tmp_path, [[0.7, "", "hash1", "ignored"], [0.9, "compiler", "hash2", 1]])
    result = service.predict_single(path)
    assert result == {"row_id": "hash1", "row_id_truncated": False, "row_index": 0, "source_row": 2,
                      "prediction": 0, "label": "goodware", "malware_probability": 0.7,
                      "threshold": 0.8, "model_version": "test-v1"}
    assert list(service.pipeline.calls[0].columns) == ["Size", "Identify"]
    assert np.isnan(service.pipeline.calls[0]["Identify"].iloc[0])
    assert service.predict_single(path, 1)["prediction"] == 1


@pytest.mark.parametrize("row_index", [-1, 2, True, "0", 0.5])
def test_single_rejects_invalid_index(service, tmp_path, row_index):
    path = write_csv(tmp_path, [[0.1, "compiler", "h", 0]])
    with pytest.raises(ValueError, match="row_index"):
        service.predict_single(path, row_index)


def test_batch_preserves_invalid_width_missing_and_nonnumeric_rows(service, tmp_path):
    path = write_csv(tmp_path, [[0.8, "compiler", "h1", 1], ["do not print me", "compiler", "h2", 0],
                               [0.2, "compiler", "h3", 0], ["", "compiler", "h4", 0], [0.4, "compiler"]])
    output = tmp_path / "nested" / "results.csv"
    result = service.predict_batch(path, output)
    assert (result["total_count"], result["valid_count"], result["invalid_count"]) == (5, 2, 3)
    assert (result["malware_count"], result["goodware_count"]) == (1, 1)
    assert [row["row_index"] for row in result["invalid_rows"]] == [1, 3, 4]
    assert "do not print me" not in json.dumps(result)
    assert result["results_path"] == str(output)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert [row["row_id"] for row in rows] == ["h1", "h2", "h3", "h4", "4"]
    assert [row["status"] for row in rows] == ["valid", "invalid", "valid", "invalid", "invalid"]
    assert rows[1]["prediction"] == rows[1]["malware_probability"] == ""
    assert [row["row_index"] for row in rows] == [str(index) for index in range(5)]


def test_batch_bounded_preview_and_all_invalid_rows(service, tmp_path):
    path = write_csv(tmp_path, [["invalid", "", str(index), 0] for index in range(12)])
    result = service.predict_batch(path, tmp_path / "results.csv")
    assert result["invalid_count"] == 12
    assert len(result["invalid_rows"]) == 10
    assert result["invalid_rows_truncated"] is True
    assert result["valid_count"] == result["malware_count"] == result["goodware_count"] == 0
    assert service.pipeline.calls == []
    with pytest.raises(ValueError, match="no valid labeled rows"):
        service.evaluate(path)


@pytest.mark.parametrize("identifier", ["=SUM(A1:A2)", "+cmd", "-cmd", "@cmd", "\t=cmd", "  =cmd", "\r=cmd"])
def test_batch_neutralizes_spreadsheet_formula_identifiers(service, tmp_path, identifier):
    path = write_csv(tmp_path, [[0.7, "compiler", identifier, 1]])
    output = tmp_path / "result.csv"
    service.predict_batch(path, output)
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["row_id"] == "'" + identifier


def test_evaluation_auc_uses_probability_scores_and_reports_partial_coverage(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "compiler", "h1", 0], [0.4, "compiler", "h2", 1],
                               [0.9, "compiler", "h3", ""], [0.7, "compiler", "h4", 2],
                               ["bad", "compiler", "h5", 1]])
    result = service.evaluate(path)
    assert result["auc"] == 1.0
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]
    assert result["confusion_matrix_order"] == [0, 1]
    assert result["class_counts"] == {"0": 1, "1": 1}
    assert result["total_count"] == 5
    assert result["evaluated_count"] == result["valid_count"] == 2
    assert result["invalid_count"] == 3
    assert result["missing_label_count"] == result["invalid_label_count"] == 1
    assert result["feature_valid_count"] == 4
    assert result["evaluation_coverage"] == 0.4
    assert result["auc_reason"] is None


def test_evaluation_single_class_has_explicit_undefined_auc(service, tmp_path):
    path = write_csv(tmp_path, [[0.6, "compiler", "h1", 1], [0.2, "compiler", "h2", 1]])
    result = service.evaluate(path)
    assert result["auc"] is None
    assert "both classes" in result["auc_reason"]
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == [[0, 0], [1, 1]]
    assert result["evaluation_coverage"] == 1


def test_evaluation_requires_label_column(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "compiler"]], header=("Size", "Identify"))
    with pytest.raises(ValueError, match="Label"):
        service.evaluate(path)
    assert service.predict_single(path)["row_id"] == "0"


def test_labels_are_exact_classes_without_float_rounding(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "compiler", "h0", "0.0"], [0.9, "compiler", "h1", "1.0"],
                               [0.8, "compiler", "h2", "0.9999999999999999999999999999"],
                               [0.1, "compiler", "h3", "1e-999"], [0.2, "compiler", "h4", "NaN"]])
    result = service.evaluate(path)
    assert result["evaluated_count"] == 2
    assert result["invalid_label_count"] == result["invalid_count"] == 3
    assert result["evaluation_coverage"] == 0.4
    assert result["accuracy"] == 1


def test_integer_schema_checks_exact_integrality_and_declared_range(service, tmp_path):
    service.features[0]["dtype"] = "int64"
    service = ToolService(service.bundle)
    path = write_csv(tmp_path, [[0, "compiler", "h0", 0], [1, "compiler", "h1", 1],
                               ["1.00000000000000000001", "compiler", "h2", 0],
                               ["1e-999", "compiler", "h3", 0],
                               ["9223372036854775808", "compiler", "h4", 0]])
    result = service.predict_batch(path, tmp_path / "result.csv")
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 3
    assert result["malware_count"] == result["goodware_count"] == 1


@pytest.mark.parametrize('dtype,valid,invalid', [
    ('int8', ['-128', '127', '1e1'], ['-129', '128', '0.00000000000000000001']),
    ('uint8', ['0', '255', '1.0'], ['-1', '256', '1.00000000000000000001']),
    ('Int64', ['-9223372036854775808', '9223372036854775807', '0'],
     ['-9223372036854775809', '9223372036854775808', '1e-999']),
    ('UInt64', ['0', '18446744073709551615', '1.0'],
     ['-1', '18446744073709551616', '1.00000000000000000001']),
    ('bool', ['0', '1', '1.0'], ['2', '-1', '0.99999999999999999999']),
    ('boolean', ['0', '1', '0.0'], ['NaN', 'inf', '1e-999']),
])
def test_compiled_schema_preserves_exact_numeric_boundaries(tmp_path, dtype, valid, invalid):
    service = ToolService({'pipeline': ProbabilityModel(), 'metadata': {
        'features': [{'name': 'Size', 'dtype': dtype, 'allow_missing': False}],
        'threshold': 0.5, 'model_version': 'numeric-boundaries',
    }})
    path = write_csv(tmp_path, [[value] for value in valid + invalid], header=('Size',))
    data = service._read_csv(path)
    for row, value in zip(data.rows[:len(valid)], valid):
        features, errors = service._validate_features(row, data.header)
        assert errors == []
        assert 'Size' in features
        if dtype not in {'bool', 'boolean'}:
            assert isinstance(features['Size'], int)
    for row in data.rows[len(valid):]:
        features, errors = service._validate_features(row, data.header)
        assert features == {}
        assert errors == ['Invalid numeric feature: Size']


def test_compiled_schema_matches_frozen_model_probabilities(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(root / 'models/production.joblib')
    pipeline = bundle['pipeline']
    before = pipeline.get_params(deep=True)
    service = ToolService(bundle)
    source = root / 'samples/labeled.csv'
    expected = pipeline.predict_proba(pd.read_csv(source)[service.feature_names])[:, 1]

    results = service.predict_batch(source, tmp_path / 'results.csv')
    with (tmp_path / 'results.csv').open(newline='') as stream:
        actual = np.array([float(row['malware_probability']) for row in csv.DictReader(stream)])

    assert results['invalid_count'] == 0
    np.testing.assert_array_equal(actual, expected)
    assert pipeline.get_params(deep=True) == before


@pytest.mark.parametrize("contents,message", [
    ("Size,Identify,Size\n0.1,x,0.1\n", "duplicate"),
    ("Size,Identify,unknown\n0.1,x,value\n", "unexpected"),
    ("Identify\nx\n", "Missing features"),
    ("Size,\n0.1,x\n", "header"),
    ("Size,Identify \n0.1,x\n", "exact column names"),
    ("Size,Identify\n", "no data"),
    ('Size,Identify\n0.1,"unterminated\n', "malformed"),
    ("", "header"),
])
def test_csv_file_and_schema_errors(service, tmp_path, contents, message):
    path = tmp_path / "input.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        service.inspect_csv(path)
    assert service.pipeline.calls == []


def test_utf8_type_byte_and_row_limits(service, tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"Size,Identify\n0.1,\xff\n")
    with pytest.raises(ValueError, match="UTF-8"):
        service.predict_single(path)
    path = write_csv(tmp_path, [[0.1, "compiler"], [0.2, "compiler"]], header=("Size", "Identify"))
    service.max_rows = 1
    with pytest.raises(ValueError, match="row limit"):
        service.inspect_csv(path)
    service.max_rows = 10
    service.max_bytes = 8
    with pytest.raises(ValueError, match="byte"):
        service.inspect_csv(path)
    with pytest.raises(ValueError, match="CSV file"):
        service.inspect_csv(tmp_path / "input.exe")


@pytest.mark.parametrize("size", ["NaN", "inf", "-inf", "", "not-a-number"])
def test_invalid_numeric_features_are_reported_without_values(service, tmp_path, size):
    path = write_csv(tmp_path, [[size, "compiler", "h", 0]])
    result = service.predict_batch(path, tmp_path / "results.csv")
    assert result["invalid_count"] == 1
    assert service.pipeline.calls == []
    with pytest.raises(ValueError, match="Invalid row"):
        service.predict_single(path)


def test_cell_limit_and_blank_line_are_invalid_rows(service, tmp_path):
    service.max_cell_length = 8
    path = write_csv(tmp_path, [[0.1, "long-text-value", "h1", 0], []])
    assert service.inspect_csv(path)["total_count"] == 2
    result = service.predict_batch(path, tmp_path / "results.csv")
    assert result["invalid_count"] == result["total_count"] == 2
    text = json.dumps(result)
    assert "long-text-value" not in text


def test_nul_is_rejected_as_a_malformed_file(service, tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"Size,Identify\n0.1,nul\x00\n")
    with pytest.raises(ValueError, match="NUL"):
        service.inspect_csv(path)


def test_multiline_source_rows_and_bom(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "line1\nline2", "h1", 0], [0.9, "compiler", "h2", 1]])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert service.predict_single(path, 0)["source_row"] == 2
    assert service.predict_single(path, 1)["source_row"] == 4


def test_predict_single_row_id_is_bounded(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "compiler", "h" * 500, 0]])
    result = service.predict_single(path)
    assert len(result["row_id"]) == 128
    assert result["row_id_truncated"] is True


def test_model_failure_does_not_leave_partial_output(service, tmp_path):
    class BrokenModel:
        def predict_proba(self, frame):
            raise ValueError("untrusted model error detail")

    service.pipeline = BrokenModel()
    path = write_csv(tmp_path, [[0.1, "compiler", "h", 0]])
    output = tmp_path / "result.csv"
    with pytest.raises(RuntimeError, match="production model"):
        service.predict_batch(path, output)
    assert not output.exists()


@pytest.mark.parametrize("probabilities", [[[0.2, 0.9]], [[float("nan"), 0.5]], [[1.2, -0.2]], [[0.5]]])
def test_invalid_model_probability_output_is_a_tool_failure(service, tmp_path, probabilities):
    class InvalidModel:
        def predict_proba(self, frame):
            return probabilities

    service.pipeline = InvalidModel()
    path = write_csv(tmp_path, [[0.1, "compiler", "h", 0]])
    with pytest.raises(RuntimeError, match="production model"):
        service.predict_single(path)


def test_batch_cannot_overwrite_input(service, tmp_path):
    path = write_csv(tmp_path, [[0.1, "compiler", "h", 0]])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different path"):
        service.predict_batch(path, path)
    assert path.read_bytes() == before

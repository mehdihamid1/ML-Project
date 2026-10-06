"""Bounded CSV tools backed exclusively by the saved production model."""

from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


_METADATA_COLUMNS = {"Label", "SHA1", "FirstSeenDate", "Magic", "PE_TYPE", "SizeOfOptionalHeader"}
_PREVIEW_LIMIT = 10
_ROW_ID_LIMIT = 128


@dataclass
class _CsvRow:
    index: int
    source_row: int
    cells: list[str]


@dataclass
class _CsvData:
    header: list[str]
    rows: list[_CsvRow]


@dataclass(frozen=True)
class _FeatureRule:
    name: str
    allow_missing: bool
    numeric: bool
    integer: bool
    boolean: bool
    minimum: int | None
    maximum: int | None

    @classmethod
    def from_schema(cls, feature: dict) -> _FeatureRule:
        """Resolve the trusted, fixed schema once rather than for every row."""
        dtype = feature.get("dtype", "object")
        integer = is_integer_dtype(dtype)
        minimum = maximum = None
        if integer:
            declared = pd.api.types.pandas_dtype(dtype)
            bounds = np.iinfo(getattr(declared, "numpy_dtype", declared))
            minimum, maximum = int(bounds.min), int(bounds.max)
        boolean = is_bool_dtype(dtype)
        return cls(feature["name"], feature.get("allow_missing", False),
                   is_numeric_dtype(dtype) or boolean, integer, boolean, minimum, maximum)


def _positive_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _spreadsheet_safe(value: str) -> str:
    """Keep untrusted identifiers inert when a results CSV opens in a spreadsheet."""
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


class ToolService:
    def __init__(self, bundle: dict, *, max_bytes: int = 5 * 1024 * 1024,
                 max_rows: int = 10_000, max_cell_length: int = 4096):
        self.bundle = bundle
        self.metadata = bundle["metadata"]
        self.pipeline = bundle["pipeline"]
        self.configure_limits(max_bytes=max_bytes, max_rows=max_rows, max_cell_length=max_cell_length)
        self.features = self.metadata["features"]
        self.feature_names = [feature["name"] for feature in self.features]
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("The production feature schema must contain unique feature names")
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("The production feature schema has an invalid name")
        self._feature_rules = tuple(_FeatureRule.from_schema(feature) for feature in self.features)
        self.threshold = float(self.metadata["threshold"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("The production decision threshold must be between zero and one")
        self.model_version = self.metadata["model_version"]
        self.class_mapping = self.metadata.get("class_mapping", {"0": "goodware", "1": "malware"})

    def configure_limits(self, *, max_bytes: int, max_rows: int, max_cell_length: int):
        """Keep upload inspection and every model tool on the same configured limits."""
        limits = (_positive_limit(max_bytes, "max_bytes"), _positive_limit(max_rows, "max_rows"),
                  _positive_limit(max_cell_length, "max_cell_length"))
        self.max_bytes, self.max_rows, self.max_cell_length = limits

    def _read_csv(self, csv_path: Path) -> _CsvData:
        path = Path(csv_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("Input must be a CSV file")
        try:
            with path.open("rb") as stream:
                raw = stream.read(self.max_bytes + 1)
        except OSError as exc:
            raise ValueError("The CSV file could not be read") from exc
        if len(raw) > self.max_bytes:
            raise ValueError(f"CSV exceeds the {self.max_bytes}-byte upload limit")
        try:
            contents = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must use UTF-8 encoding") from exc
        if "\x00" in contents:
            raise ValueError("CSV contains a NUL character")
        reader = csv.reader(io.StringIO(contents, newline=""), strict=True)
        try:
            header = next(reader, None)
            if not header or any(not name or name != name.strip() for name in header):
                raise ValueError("CSV requires a nonempty header with exact column names")
            if len(set(header)) != len(header):
                raise ValueError("CSV has duplicate column names")
            missing = set(self.feature_names) - set(header)
            if missing:
                raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
            unexpected = set(header) - set(self.feature_names) - _METADATA_COLUMNS
            if unexpected:
                # Headers are untrusted too: do not echo an uploaded instruction to the LLM.
                raise ValueError(f"CSV has {len(unexpected)} unexpected column(s)")
            rows = []
            previous_line = reader.line_num
            for cells in reader:
                rows.append(_CsvRow(len(rows), previous_line + 1, cells))
                previous_line = reader.line_num
                if len(rows) > self.max_rows:
                    raise ValueError(f"CSV exceeds the {self.max_rows}-row limit")
        except csv.Error as exc:
            raise ValueError("CSV syntax is malformed") from exc
        if not rows:
            raise ValueError("CSV has no data rows")
        return _CsvData(header, rows)

    def inspect_csv(self, csv_path: Path) -> dict:
        """Check file-level limits and schema without dropping or rejecting individual rows."""
        data = self._read_csv(csv_path)
        return {"total_count": len(data.rows), "columns": data.header, "max_rows": self.max_rows}

    def _row_id(self, row: _CsvRow, header: list[str]) -> str:
        if "SHA1" in header:
            position = header.index("SHA1")
            if position < len(row.cells) and row.cells[position].strip():
                return row.cells[position]
        return str(row.index)

    def _validate_features(self, row: _CsvRow, header: list[str]) -> tuple[dict, list[str]]:
        errors = []
        if len(row.cells) != len(header):
            errors.append("Row has a different number of fields from the header")
        long_positions = {index for index, cell in enumerate(row.cells) if len(cell) > self.max_cell_length}
        if long_positions:
            errors.append(f"A field exceeds the {self.max_cell_length}-character limit")
        if errors:
            return {}, errors
        values = dict(zip(header, row.cells))
        features = {}
        for rule in self._feature_rules:
            name = rule.name
            value = values[name]
            if not value.strip():
                if rule.allow_missing:
                    features[name] = np.nan
                else:
                    errors.append(f"Missing required feature: {name}")
                continue
            if rule.numeric:
                try:
                    exact = Decimal(value)
                    if not exact.is_finite():
                        raise ValueError
                    if rule.integer:
                        if exact != exact.to_integral_value() or not rule.minimum <= exact <= rule.maximum:
                            raise ValueError
                    if rule.boolean and exact not in (0, 1):
                        raise ValueError
                    number = float(exact)
                    if not math.isfinite(number):
                        raise ValueError
                    features[name] = int(exact) if rule.integer else number
                except (InvalidOperation, ValueError, OverflowError):
                    errors.append(f"Invalid numeric feature: {name}")
            else:
                features[name] = value
        return features, errors

    @staticmethod
    def _label(row: _CsvRow, header: list[str]) -> tuple[int | None, str | None]:
        position = header.index("Label")
        value = row.cells[position].strip() if position < len(row.cells) else ""
        if not value:
            return None, "Missing Label"
        try:
            label = Decimal(value)
        except (InvalidOperation, ValueError):
            return None, "Label must be 0 or 1"
        if not label.is_finite() or label not in (0, 1):
            return None, "Label must be 0 or 1"
        return int(label), None

    def _predict(self, features: list[dict]) -> np.ndarray:
        if not features:
            return np.empty(0, dtype=float)
        frame = pd.DataFrame(features, columns=self.feature_names)
        try:
            classes = getattr(self.pipeline, "classes_", None)
            if classes is not None and list(classes) != [0, 1]:
                raise ValueError("Unexpected production class order")
            probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
            if probabilities.shape != (len(features), 2):
                raise ValueError("Unexpected production probability shape")
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Invalid production probabilities")
            if not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError("Production probabilities do not sum to one")
        except Exception as exc:
            raise RuntimeError("The production model could not score the validated rows") from exc
        return probabilities[:, 1]

    def _common(self) -> dict:
        return {"model_version": self.model_version, "threshold": self.threshold}

    @staticmethod
    def _invalid_preview(invalid: list[dict]) -> dict:
        return {"invalid_rows": invalid[:_PREVIEW_LIMIT],
                "invalid_rows_truncated": len(invalid) > _PREVIEW_LIMIT}

    def predict_single(self, csv_path: Path, row_index: int = 0) -> dict:
        data = self._read_csv(csv_path)
        if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < len(data.rows):
            raise ValueError("row_index must identify an existing row (starting at zero)")
        row = data.rows[row_index]
        features, errors = self._validate_features(row, data.header)
        if errors:
            raise ValueError(f"Invalid row {row.index}: {'; '.join(errors)}")
        probability = float(self._predict([features])[0])
        prediction = int(probability >= self.threshold)
        row_id = self._row_id(row, data.header)
        return {"row_id": row_id[:_ROW_ID_LIMIT], "row_id_truncated": len(row_id) > _ROW_ID_LIMIT,
                "row_index": row.index, "source_row": row.source_row,
                "prediction": prediction, "label": self.class_mapping[str(prediction)],
                "malware_probability": probability, **self._common()}

    def predict_batch(self, csv_path: Path, output_path: Path) -> dict:
        data = self._read_csv(csv_path)
        destination = Path(output_path)
        if destination.resolve() == Path(csv_path).resolve():
            raise ValueError("Results CSV must have a different path from the input")
        valid_features = []
        valid_positions = []
        output_rows = []
        invalid = []
        for row in data.rows:
            features, errors = self._validate_features(row, data.header)
            output_rows.append({"row_index": row.index, "source_row": row.source_row,
                                "row_id": _spreadsheet_safe(self._row_id(row, data.header)),
                                "status": "invalid" if errors else "valid", "prediction": "",
                                "malware_probability": "", "errors": _spreadsheet_safe("; ".join(errors))})
            if errors:
                invalid.append({"row_index": row.index, "source_row": row.source_row, "errors": errors})
            else:
                valid_features.append(features)
                valid_positions.append(row.index)
        probabilities = self._predict(valid_features)
        predictions = (probabilities >= self.threshold).astype(int)
        for position, probability, prediction in zip(valid_positions, probabilities, predictions):
            output_rows[position].update(prediction=int(prediction), malware_probability=float(probability))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=destination.parent,
                                             prefix=".results-", suffix=".csv", delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=["row_index", "source_row", "row_id", "status",
                                                           "prediction", "malware_probability", "errors"])
                writer.writeheader()
                writer.writerows(output_rows)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"total_count": len(data.rows), "valid_count": len(valid_positions), "invalid_count": len(invalid),
                "malware_count": int(predictions.sum()), "goodware_count": int(len(predictions) - predictions.sum()),
                "results_path": str(destination), **self._invalid_preview(invalid), **self._common()}

    def evaluate(self, csv_path: Path) -> dict:
        data = self._read_csv(csv_path)
        if "Label" not in data.header:
            raise ValueError("Evaluation requires a Label column containing 0 or 1")
        features = []
        labels = []
        invalid = []
        missing_labels = 0
        invalid_labels = 0
        feature_valid_count = 0
        for row in data.rows:
            values, errors = self._validate_features(row, data.header)
            feature_valid_count += not errors
            label, label_error = self._label(row, data.header)
            if label_error:
                errors.append(label_error)
                missing_labels += label_error == "Missing Label"
                invalid_labels += label_error != "Missing Label"
            if errors:
                invalid.append({"row_index": row.index, "source_row": row.source_row, "errors": errors})
            else:
                features.append(values)
                labels.append(label)
        if not labels:
            raise ValueError("Evaluation has no valid labeled rows")
        probabilities = self._predict(features)
        predicted = (probabilities >= self.threshold).astype(int)
        both_classes = len(set(labels)) == 2
        return {"auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
                "auc_reason": None if both_classes else "AUC requires both classes among the evaluated labels",
                "accuracy": float(accuracy_score(labels, predicted)),
                "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
                "confusion_matrix_order": [0, 1], "class_mapping": self.class_mapping,
                "class_counts": {"0": labels.count(0), "1": labels.count(1)},
                "total_count": len(data.rows), "evaluated_count": len(labels), "valid_count": len(labels),
                "invalid_count": len(invalid), "feature_valid_count": feature_valid_count,
                "missing_label_count": missing_labels, "invalid_label_count": invalid_labels,
                "evaluation_coverage": len(labels) / len(data.rows),
                **self._invalid_preview(invalid), **self._common()}

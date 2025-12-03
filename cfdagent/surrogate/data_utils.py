"""Utility helpers for loading CFD dataset CSV files."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable


@dataclass
class DatasetSchema:
    """Schema describing the input and output columns for a CFD dataset."""

    input_columns: list[str]
    output_columns: list[str]


def infer_schema(csv_path: Path) -> DatasetSchema:
    """
    Inspect the header of a dataset CSV from run_param_sweep and return
    the inferred DatasetSchema.

    Rules:
    - Input columns are all columns that do NOT start with "out_".
    - Output columns are all columns that DO start with "out_".
    - Preserve the order from the CSV header.
    """

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    input_columns: list[str] = []
    output_columns: list[str] = []

    for name in fieldnames:
        if name.startswith("out_"):
            output_columns.append(name)
        else:
            input_columns.append(name)

    return DatasetSchema(input_columns=input_columns, output_columns=output_columns)


def parse_value(text: str | None) -> float | int | None:
    """
    Convert a CSV cell to a numeric value where possible.

    Rules:
    - If text is empty or whitespace, return None.
    - Try float(text); if fails, return None.
    - If the float is an integer (num.is_integer()), return int(num).
    - Otherwise, return the float.
    """

    if text is None:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        num = float(stripped)
    except ValueError:
        return None

    if num.is_integer():
        return int(num)
    return num


def _row_empty(values: Iterable[str | None]) -> bool:
    for value in values:
        if value and value.strip():
            return False
    return True


def load_dataset(
    csv_path: Path,
    schema: DatasetSchema | None = None,
) -> tuple[list[list[float | int | None]], list[list[float | int | None]], DatasetSchema]:
    """
    Load a CFD dataset CSV into memory.

    Returns:
        (X, Y, schema)

        X: list of rows, each row is a list of parsed values for input_columns.
        Y: list of rows, each row is a list of parsed values for output_columns.
        schema: the DatasetSchema used (inferred if not provided).

    Rules:
    - If schema is None, call infer_schema(csv_path).
    - For each row in the CSV:
        * Use parse_value on each cell in input_columns and output_columns.
    - Rows are kept in the original CSV order.
    """

    resolved_schema = schema or infer_schema(csv_path)
    X: list[list[float | int | None]] = []
    Y: list[list[float | int | None]] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            if _row_empty(row.values()):
                continue

            x_row = [parse_value(row.get(col, "")) for col in resolved_schema.input_columns]
            y_row = [parse_value(row.get(col, "")) for col in resolved_schema.output_columns]

            X.append(x_row)
            Y.append(y_row)

    return X, Y, resolved_schema

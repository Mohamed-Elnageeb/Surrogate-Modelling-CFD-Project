"""Simple CLI to inspect CFD dataset CSV files produced by cfdagent.cfd.param_sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import statistics

from .data_utils import infer_schema, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a CFD dataset CSV produced by cfdagent.cfd.param_sweep"
    )
    parser.add_argument("csv_path", help="Path to the dataset CSV file")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    schema = infer_schema(csv_path)
    X, Y, _ = load_dataset(csv_path, schema)

    samples = len(X)
    print(f"Dataset path: {csv_path}")
    print(f"Samples: {samples}")
    print(f"Inputs (d = {len(schema.input_columns)}): {', '.join(schema.input_columns)}")
    print(f"Outputs (k = {len(schema.output_columns)}): {', '.join(schema.output_columns)}")

    for idx, col in enumerate(schema.output_columns):
        column_values = [row[idx] for row in Y if row[idx] is not None]
        numeric_values = [value for value in column_values if isinstance(value, (int, float))]

        if not numeric_values:
            print(f"{col}: no numeric data")
            continue

        col_min = min(numeric_values)
        col_max = max(numeric_values)
        col_mean = statistics.fmean(numeric_values)
        print(
            f"{col}: min={col_min}, max={col_max}, mean={col_mean}, count={len(numeric_values)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

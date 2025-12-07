from pathlib import Path
from typing import Any, Dict
import csv
import numpy as np


def _clean_field_name(name: str) -> str:
    """Return a normalized SU2 history column name.

    SU2 history exports sometimes include extra whitespace and quotes around
    column names (e.g., ``"CL"``). Normalizing the keys lets downstream code use
    consistent lookups regardless of formatting quirks.
    """

    return name.strip().strip('"')

def extract_metrics(history_file: Path) -> Dict[str, Any]:
    """
    Extract the final Cl, Cd, and residual values from an SU2 history CSV.
    Returns None for any missing metric.
    """
    if not history_file.exists():
        return {"Cl": None, "Cd": None, "residual": None}

    with history_file.open("r", newline="") as f:
        raw_rows = list(csv.reader(f))

    if not raw_rows:
        return {"Cl": None, "Cd": None, "residual": None}

    header = [_clean_field_name(h) for h in raw_rows[0] if h is not None]
    last_row = None
    for raw_row in raw_rows[1:]:
        if not any(raw_row):
            continue
        cleaned_row = {h: (raw_row[idx] if idx < len(raw_row) else "") for idx, h in enumerate(header)}
        last_row = cleaned_row

    if last_row is None:
        return {"Cl": None, "Cd": None, "residual": None}

    def _as_float(value: str | None) -> float | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    cl = None
    for key in ("CL", "CLtot", "cl", "Cl"):
        cl = _as_float(last_row.get(key))
        if cl is not None:
            break

    cd = None
    for key in ("CD", "CDtot", "cd", "Cd"):
        cd = _as_float(last_row.get(key))
        if cd is not None:
            break

    residual = None
    for key in ("RMS_RES", "RMS_DENSITY", "residual"):
        residual = _as_float(last_row.get(key))
        if residual is not None:
            break

    def _sanitize_positive(value: float | None) -> float | None:
        return value if value is None or value >= 0 else None

    return {"Cl": _sanitize_positive(cl), "Cd": _sanitize_positive(cd), "residual": residual}


def save_flowfield(solution_dir: Path, out_path: Path) -> None:
    """
    Aggregate CSV-based solution exports into a compressed ``.npz`` archive.

    The helper looks for CSV files inside ``solution_dir`` (e.g. SU2 PARAVIEW
    exports) and stores every numeric column as an array in ``out_path``. File
    stems are prefixed to the column names to avoid collisions.
    """
    solution_dir = Path(solution_dir)
    if not solution_dir.exists():
        raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

    arrays: Dict[str, Any] = {}
    csv_files = sorted(solution_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV solution files found in {solution_dir}")

    for csv_file in csv_files:
        with csv_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            columns: Dict[str, list[float]] = {}
            for row in reader:
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    try:
                        numeric_value = float(value)
                    except ValueError:
                        # Skip non-numeric columns silently.
                        continue
                    columns.setdefault(key, []).append(numeric_value)

        for col_name, values in columns.items():
            arrays[f"{csv_file.stem}_{col_name}"] = np.asarray(values, dtype=np.float32)

    if not arrays:
        raise ValueError(f"No numeric columns found while parsing {solution_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

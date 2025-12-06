from pathlib import Path
from typing import Any, Dict
import csv
import numpy as np

def extract_metrics(history_file: Path) -> Dict[str, Any]:
    """
    Extract the final Cl, Cd, and residual values from an SU2 history CSV.
    Returns None for any missing metric.
    """
    if not history_file.exists():
        return {"Cl": None, "Cd": None, "residual": None}

    last_row = None
    with history_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                last_row = row

    if last_row is None:
        return {"Cl": None, "Cd": None, "residual": None}

    cl = float(last_row.get("CL", last_row.get("CLtot", last_row.get("cl", None))) or 0) if last_row.get("CL") or last_row.get("CLtot") or last_row.get("cl") else None
    cd = float(last_row.get("CD", last_row.get("CDtot", last_row.get("cd", None))) or 0) if last_row.get("CD") or last_row.get("CDtot") or last_row.get("cd") else None
    residual = None
    for key in ("RMS_RES", "RMS_DENSITY", "residual"):
        if key in last_row:
            residual_value = last_row[key]
            residual = float(residual_value) if residual_value else None
            break

    return {"Cl": cl, "Cd": cd, "residual": residual}


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

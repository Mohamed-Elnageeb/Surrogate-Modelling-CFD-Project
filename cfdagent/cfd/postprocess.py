from pathlib import Path
from typing import Any, Dict
import csv

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
    Placeholder to parse SU2 solution files, sample onto a fixed grid,
    and save arrays to out_path as .npz. Not implemented in Phase 0.
    """
    raise NotImplementedError("flowfield extraction not implemented yet")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import csv
import shutil
import re
from typing import Iterable

from .postprocess import extract_metrics


@dataclass
class Su2RunConfig:
    """Configuration for running a SU2 CFD case from Python."""

    workdir: Path
    base_config_name: str = "config.cfg"
    su2_executable: str = "SU2_CFD"
    history_file_name: str = "history.csv"


def create_modified_config(base_cfg: Path, out_cfg: Path, param_overrides: dict[str, float | int | str]) -> None:
    """
    Read a SU2 config file, apply overrides to specific SU2 keywords,
    and write the modified config to out_cfg.

    Rules:
    - For every key in param_overrides, the key must match a line in the config that
      begins with "<KEY>=" exactly.
    - Replace the entire RHS with the new value.
    - Preserve all other lines unchanged.
    - If a keyword is missing in the file, raise a KeyError.
    """

    lines = base_cfg.read_text().splitlines(keepends=True)
    overrides_applied: set[str] = set()

    for idx, line in enumerate(lines):
        for key, value in param_overrides.items():
            if line.startswith(f"{key}="):
                lines[idx] = f"{key}= {value}\n"
                overrides_applied.add(key)

    missing_keys = set(param_overrides) - overrides_applied
    if missing_keys:
        raise KeyError(f"Missing keys in config: {sorted(missing_keys)}")

    out_cfg.write_text("".join(lines))


def parse_history_file(path: Path) -> dict:
    """
    Parse a SU2 history file (.csv or .dat).
    Return the last meaningful data row as a dict.
    If the file does not exist or is empty, return {}.
    """

    if not path or not path.exists():
        return {}

    with path.open("r", newline="") as csvfile:
        data_lines = [line for line in csvfile if line.strip() and not line.lstrip().startswith(("%", "#"))]

    if not data_lines:
        return {}

    reader = csv.reader(data_lines)
    header = next(reader, None)
    if not header:
        return {}

    last_row: list[str] | None = None
    for row in reader:
        if row:
            last_row = row

    if not last_row:
        return {}

    def convert(value: str) -> float | int | str:
        try:
            num = float(value)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            return value

    return {key: convert(val) for key, val in zip(header, last_row)}


def run_su2_case(cfg: Su2RunConfig, param_overrides: dict[str, float | int | str] | None = None, timeout: int = 3600) -> dict:
    """
    Run a SU2 CFD simulation using the config file in cfg.workdir.

    Steps:
    - Determine the working directory: cfg.workdir.
    - If param_overrides is provided:
        * Create a new config file in workdir named "config_override.cfg".
        * Call create_modified_config(...) to apply changes.
        * Run SU2_CFD on that overridden file.
      Else:
        * Run SU2_CFD on cfg.base_config_name.
    - Use subprocess.run([...], cwd=cfg.workdir, capture_output=True, text=True).
    - If returncode != 0, raise a RuntimeError with useful stderr information.
    - After the run, detect the history file:
        * If cfg.history_file_name exists, use it.
        * Otherwise, find the newest file starting with "history".
    - Parse the history file:
        * Skip comment lines (% or #).
        * Assume first non-comment line is header.
        * Read last row.
        * Convert numeric fields where possible.
    - Return a dict containing:
        {
          "config_path": Path to the config used,
          "stdout": command stdout,
          "stderr": command stderr,
          "history_path": path to history file or None,
          "history_data": last-row dict or {},
        }
    """

    workdir = cfg.workdir
    base_cfg = workdir / cfg.base_config_name

    if param_overrides:
        override_cfg = workdir / "config_override.cfg"
        create_modified_config(base_cfg, override_cfg, param_overrides)
        config_to_run = override_cfg
    else:
        config_to_run = base_cfg

    cmd = [cfg.su2_executable, str(config_to_run)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(f"SU2_CFD failed with code {proc.returncode}: {proc.stderr}")

    history_path: Path | None = workdir / cfg.history_file_name
    if not history_path.exists():
        history_candidates = sorted(workdir.glob("history*"), key=lambda p: p.stat().st_mtime, reverse=True)
        history_path = history_candidates[0] if history_candidates else None

    history_data: dict = {}
    if history_path:
        history_data = parse_history_file(history_path)

    return {
        "config_path": config_to_run,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "history_path": history_path,
        "history_data": history_data,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "base_case"
    cfg = Su2RunConfig(workdir=base_dir)
    result = run_su2_case(
        cfg,
        param_overrides={
            "MACH_NUMBER": 0.15,
            "AOA": 10.0,
        },
    )
    print("History file:", result["history_path"])
    print("Last row:", result["history_data"])


def _extract_first(history: dict, *keys: str):
    for key in keys:
        if key in history:
            value = history[key]
            if value is None or value == "":
                continue
            return value
    return None


def run_cfd(
    design_id: str,
    design_vec: Iterable[float],
    workdir: Path | None = None,
    su2_executable: str | None = None,
) -> dict:
    """
    Lightweight convenience wrapper for running a single SU2 case.

    The helper mirrors the legacy ``run_cfd`` interface that higher-level
    scripts import. It delegates to :func:`run_su2_case` using the default
    ``cfdagent/cfd/base_case`` directory (unless ``workdir`` is provided),
    and returns a dictionary containing lift/drag metrics plus a success flag.

    Args:
        design_id: Identifier for the design being evaluated (used for logging).
        design_vec: Ten-parameter design vector. It is accepted for API
            compatibility but not manipulated inside this helper; callers can
            persist it alongside the returned metrics if needed.
        workdir: Optional path to a prepared SU2 case directory.
        su2_executable: Optional override for the SU2 binary name/path.
    """

    case_dir = workdir or (Path(__file__).resolve().parent / "base_case")
    cfg = Su2RunConfig(workdir=case_dir)
    if su2_executable:
        cfg.su2_executable = su2_executable

    base_cfg = case_dir / cfg.base_config_name
    mesh_file = case_dir / "mesh.su2"

    missing_reqs: list[str] = []
    if not base_cfg.exists():
        missing_reqs.append(f"Missing SU2 base config: {base_cfg}")
    if not mesh_file.exists():
        missing_reqs.append(f"Missing SU2 mesh file: {mesh_file}")

    su2_path = shutil.which(cfg.su2_executable)
    if not su2_path:
        missing_reqs.append(
            f"SU2 executable '{cfg.su2_executable}' not found in PATH."
        )

    if missing_reqs:
        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": None,
            "Cd": None,
            "residual": None,
            "success": False,
            "error": "; ".join(missing_reqs),
        }

    try:
        result = run_su2_case(cfg)
        history = result.get("history_data") or {}
        metrics = extract_metrics(result.get("history_path")) if result.get("history_path") else {}

        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": metrics.get("Cl") if metrics else _extract_first(history, "CL", "CLtot", "cl", "Cl"),
            "Cd": metrics.get("Cd") if metrics else _extract_first(history, "CD", "CDtot", "cd", "Cd"),
            "residual": metrics.get("residual") if metrics else _extract_first(history, "RMS_RES", "RMS_DENSITY", "residual"),
            "success": True,
            "history_data": history,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "config_path": result.get("config_path"),
            "history_path": result.get("history_path"),
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": None,
            "Cd": None,
            "residual": None,
            "success": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    main()

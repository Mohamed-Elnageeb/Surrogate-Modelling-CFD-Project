from __future__ import annotations

import subprocess
import csv
import shutil
import re
from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

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
    Parse a SU2 history file (.csv or .dat) and return the last data row.

    The parser tolerates trailing delimiters and both comma- and whitespace-
    separated tables, which occur across SU2 versions. Comment/blank lines are
    skipped. If the file does not exist or no data rows are present, an empty
    dict is returned.
    """

    if not path or not path.exists():
        return {}

    with path.open("r", newline="") as csvfile:
        data_lines = [line for line in csvfile if line.strip() and not line.lstrip().startswith(("%", "#"))]

    if not data_lines:
        return {}

    def split_line(line: str) -> list[str]:
        stripped = line.strip()
        if "," in stripped:
            return [tok for tok in stripped.split(",") if tok != ""]
        return [tok for tok in re.split(r"\s+", stripped) if tok != ""]

    header = split_line(data_lines[0])
    if not header:
        return {}

    last_row: list[str] | None = None
    for line in data_lines[1:]:
        row = split_line(line)
        if not row:
            continue
        # Ignore extra trailing tokens and pad missing ones with empty strings
        row = (row + [""] * len(header))[: len(header)]
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
    base_dir = _default_case_dir()
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


def _config_value(cfg_path: Path, key: str, default: str | None = None) -> str | None:
    """Return the value assigned to ``key`` in a SU2 config, if present."""

    if not cfg_path.exists():
        return default

    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*(.+)$")
    for line in cfg_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()
    return default


def _latest_output(workdir: Path, stem: str) -> Path | None:
    """Find the newest output file matching a given stem in ``workdir``."""

    extensions = (".csv", ".dat", ".vtu", ".vtk", ".su2", ".txt")
    candidates = list(workdir.glob(f"{stem}*"))
    candidates = [c for c in candidates if c.suffix in extensions]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _default_case_dir() -> Path:
    """Return the canonical SU2 case directory bundled with the repository."""

    return Path(__file__).resolve().parents[2] / "TestCases" / "airfoil_naca0012_opt"


def run_cfd(
    design_id: str,
    design_vec: Iterable[float],
    workdir: Path | None = None,
    su2_executable: str | None = None,
) -> dict:
    """
    Lightweight convenience wrapper for running a single SU2 case.

    The helper mirrors the legacy ``run_cfd`` interface that higher-level
    scripts import. It delegates to :func:`run_su2_case` using the canonical
    ``TestCases/airfoil_naca0012_opt`` directory (unless ``workdir`` is
    provided), and returns a dictionary containing lift/drag metrics plus a
    success flag.

    Args:
        design_id: Identifier for the design being evaluated (used for logging).
        design_vec: Ten-parameter design vector. It is accepted for API
            compatibility but not manipulated inside this helper; callers can
            persist it alongside the returned metrics if needed.
        workdir: Optional path to a prepared SU2 case directory.
        su2_executable: Optional override for the SU2 binary name/path.
    """

    case_dir = workdir or _default_case_dir()
    cfg = Su2RunConfig(workdir=case_dir)
    if su2_executable:
        cfg.su2_executable = su2_executable

    base_cfg = case_dir / cfg.base_config_name
    volume_stem = _config_value(base_cfg, "VOLUME_FILENAME", "flow_fields")
    surface_stem = _config_value(base_cfg, "SURFACE_FILENAME", "surface_airfoil")

    missing_reqs = []
    if not base_cfg.exists():
        missing_reqs.append(f"Missing SU2 base config: {base_cfg}")

    mesh_file = case_dir / "mesh.su2"
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

        cl = metrics.get("Cl") if metrics else None
        cd = metrics.get("Cd") if metrics else None
        residual = metrics.get("residual") if metrics else None

        if cl is None:
            cl = _extract_first(history, "CL", "CLtot", "cl", "Cl")
        if cd is None:
            cd = _extract_first(history, "CD", "CDtot", "cd", "Cd")
        if residual is None:
            residual = _extract_first(history, "RMS_RES", "RMS_DENSITY", "residual")

        volume_output = _latest_output(case_dir, volume_stem or "flow_fields")
        surface_output = _latest_output(case_dir, surface_stem or "surface_airfoil")

        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": cl,
            "Cd": cd,
            "residual": residual,
            "success": True,
            "history_data": history,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "config_path": result.get("config_path"),
            "history_path": result.get("history_path"),
            "volume_output": volume_output,
            "surface_output": surface_output,
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

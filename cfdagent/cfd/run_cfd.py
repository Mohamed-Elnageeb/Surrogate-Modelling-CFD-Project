from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import csv
import shutil
import re


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


if __name__ == "__main__":
    main()

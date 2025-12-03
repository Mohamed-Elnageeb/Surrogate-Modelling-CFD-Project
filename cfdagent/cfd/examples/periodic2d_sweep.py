"""Example script to run a small parameter sweep on SU2's periodic2D test case.

The SU2 TestCases directory is not bundled with this package. Set the
``CFDAGENT_SU2_BASEDIR`` environment variable to point to your SU2 TestCases
root before running this script. If the variable is not provided, the script
falls back to a relative path that may need adjustment depending on your
installation layout.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

from ..param_sweep import generate_param_grid, run_param_sweep
from ..run_cfd import Su2RunConfig


DEFAULT_TESTCASES_ROOT = Path(__file__).resolve().parents[3] / "SU2" / "TestCases"


def resolve_base_dir() -> Path:
    """Resolve the base directory for the periodic2D SU2 case."""

    env_base = os.getenv("CFDAGENT_SU2_BASEDIR")
    if env_base:
        return Path(env_base) / "navierstokes" / "periodic2D"
    return DEFAULT_TESTCASES_ROOT / "navierstokes" / "periodic2D"


def main() -> None:
    base_dir = resolve_base_dir()
    cfg = Su2RunConfig(workdir=base_dir)

    machs = [0.1, 0.15, 0.2]
    aoas = [0.0, 2.5, 5.0]
    samples = generate_param_grid(machs, aoas)

    out_csv = base_dir / "dataset_periodic2d.csv"

    try:
        run_param_sweep(cfg, samples, out_csv, shuffle=False, max_cases=None)
    except RuntimeError as exc:  # Raised when SU2_CFD fails
        print(f"Error running SU2 case: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Ran {len(samples)} samples.")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

from .param_sweep import CaseSample, generate_param_grid, run_param_sweep
from .run_cfd import Su2RunConfig


def _parse_float_list(value: str) -> List[float]:
    try:
        return [float(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {value}") from exc


def _build_samples(
    mach_values: Iterable[float],
    aoa_values: Iterable[float],
    reynolds_values: Iterable[float] | None,
) -> list[CaseSample]:
    if reynolds_values is not None:
        return generate_param_grid(mach_values, aoa_values, reynolds_values)
    return generate_param_grid(mach_values, aoa_values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CFD parameter sweep and generate a dataset")
    parser.add_argument("workdir", help="Path to SU2 case directory")
    parser.add_argument("--out-csv", help="Output CSV path", default=None)
    parser.add_argument("--mach-values", required=True, type=_parse_float_list, help="Comma-separated Mach numbers")
    parser.add_argument("--aoa-values", required=True, type=_parse_float_list, help="Comma-separated angles of attack")
    parser.add_argument(
        "--reynolds-values", type=_parse_float_list, help="Comma-separated Reynolds numbers", default=None
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases to run")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle case order")
    parser.add_argument("--su2-executable", type=str, default=None, help="Path to SU2 executable")

    args = parser.parse_args(argv)

    try:
        workdir = Path(args.workdir).expanduser().resolve()
        if not workdir.exists():
            raise FileNotFoundError(f"Workdir not found: {workdir}")

        out_csv = Path(args.out_csv) if args.out_csv else workdir / "dataset.csv"

        cfg = Su2RunConfig(workdir=workdir)
        if args.su2_executable:
            cfg.su2_executable = args.su2_executable

        samples = _build_samples(args.mach_values, args.aoa_values, args.reynolds_values)

        print(f"Running param sweep in workdir: {workdir}")
        print(f"Mach values: {args.mach_values}")
        print(f"AoA values: {args.aoa_values}")
        print(f"Reynolds values: {args.reynolds_values if args.reynolds_values is not None else 'none'}")
        print(f"Total samples: {len(samples)}")
        print(f"Writing dataset CSV to: {out_csv}")

        run_param_sweep(cfg, samples, out_csv, shuffle=args.shuffle, max_cases=args.max_cases)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Param sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

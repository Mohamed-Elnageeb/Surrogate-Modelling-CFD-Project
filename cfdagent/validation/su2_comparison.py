"""Compare SU2 RANS results against the transition-matched experiment.

SU2 with Spalart-Allmaras and no transition model is turbulent from the leading
edge, so it represents the *tripped* condition. It should therefore agree with
Ladson and disagree with the untripped Abbott data -- the mirror image of
NeuralFoil, which models natural transition.

Together the two halves form a 2x2 whose diagonal should be accurate and whose
off-diagonal should not. Populate the SU2 half with
``scripts/run_su2_validation_sweep.sh``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .experimental import load_abbott, load_ladson


def read_history(path: Path) -> dict[str, float] | None:
    """Return the final row of an SU2 history file as floats."""

    if not path.exists():
        return None
    rows = list(csv.reader(path.open()))
    if len(rows) < 2:
        return None
    header = [h.strip().strip('"') for h in rows[0]]
    out: dict[str, float] = {}
    for key, value in zip(header, rows[-1]):
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out or None


def collect_runs(runs_dir: Path, rms_target: float = -6.0) -> list[dict]:
    """Gather converged SU2 runs from a sweep directory."""

    results = []
    for case in sorted(runs_dir.glob("alpha_*")):
        hist = read_history(case / "history.csv")
        if hist is None or "CL" not in hist or "CD" not in hist:
            continue
        try:
            alpha = float(case.name.split("_", 1)[1])
        except ValueError:
            continue
        rms = hist.get("rms[Rho]", 0.0)
        results.append({
            "alpha": alpha, "cl": hist["CL"], "cd": hist["CD"],
            "rms": rms, "converged": rms <= rms_target,
        })
    return results


def compare(runs: list[dict]) -> dict:
    """Drag error at matched lift against both experimental conditions."""

    if len(runs) < 2:
        raise ValueError("Need at least two SU2 runs to interpolate a polar")

    cl = np.array([r["cl"] for r in runs])
    cd = np.array([r["cd"] for r in runs])
    order = np.argsort(cl)
    cl, cd = cl[order], cd[order]

    out = {}
    for key, dataset in (("tripped", load_ladson()), ("free", load_abbott())):
        inside = (dataset.cl >= cl.min()) & (dataset.cl <= cl.max())
        if inside.sum() == 0:
            out[key] = None
            continue
        exp_cl, exp_cd = dataset.cl[inside], dataset.cd[inside]
        pred = np.interp(exp_cl, cl, cd)
        pct = 100.0 * (pred - exp_cd) / exp_cd
        out[key] = {
            "dataset": dataset.name, "n": int(inside.sum()),
            "mean_abs_pct": float(np.mean(np.abs(pct))),
            "bias_pct": float(np.mean(pct)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs/su2_validation"))
    a = p.parse_args()

    if not a.runs.exists():
        raise SystemExit(
            f"No sweep at {a.runs}. Run scripts/run_su2_validation_sweep.sh first."
        )

    runs = collect_runs(a.runs)
    print(f"SU2 RANS-SA (fully turbulent), NACA0012, Re=6e6 -- {len(runs)} runs\n")
    print(f"  {'alpha':>6}{'Cl':>10}{'Cd':>11}{'rms':>9}  converged")
    for r in sorted(runs, key=lambda x: x["alpha"]):
        print(f"  {r['alpha']:>6.1f}{r['cl']:>10.4f}{r['cd']:>11.5f}"
              f"{r['rms']:>9.2f}  {'yes' if r['converged'] else 'NO'}")

    unconverged = [r for r in runs if not r["converged"]]
    if unconverged:
        print(f"\n  WARNING: {len(unconverged)} run(s) below the convergence target;"
              "\n  their coefficients should not be treated as final.")

    if len(runs) >= 2:
        print("\nDrag error at matched lift:")
        for key, res in compare(runs).items():
            if res is None:
                print(f"  vs {key:<8}: no overlap in Cl")
            else:
                print(f"  vs {res['dataset']:<26} [{key:>7}]  n={res['n']:>3}  "
                      f"mean|err|={res['mean_abs_pct']:>5.1f}%  bias={res['bias_pct']:>+6.1f}%")
        print(
            "\n  Expected: SU2 (fully turbulent) should track the TRIPPED data and"
            "\n  overpredict drag against the untripped data -- the mirror of"
            "\n  NeuralFoil in cfdagent.validation.transition_study."
        )


if __name__ == "__main__":
    main()

"""Quantify the aerodynamic error caused by airfoil coordinate-ordering convention.

`design_to_airfoil_coords` emits a closed loop anchored at the leading edge
(LE -> TE upper, TE -> LE lower). External aerodynamic tools expect Selig order,
anchored at the trailing edge. Passing the LE-anchored array to such a tool
raises no error; it simply describes a different, blunt-nosed body.

This script measures the resulting error against a reference NACA0012, and
confirms that the underlying geometry is itself correct -- the defect is purely
in the ordering, which is what makes it invisible to validity checks.
"""

from __future__ import annotations

import numpy as np

from ..geometry.airfoil_param import design_to_airfoil_coords, to_selig_order

ALPHA = 4.0
RE = 1.0e6


def _scalar(v) -> float:
    return float(np.atleast_1d(np.asarray(v)).ravel()[0])


def _polar(coords, alpha: float = ALPHA, re: float = RE):
    import neuralfoil as nf

    aero = nf.get_aero_from_coordinates(
        coordinates=np.asarray(coords, dtype=float), alpha=alpha, Re=re,
        model_size="medium",
    )
    return _scalar(aero["CL"]), _scalar(aero["CD"])


def run() -> dict:
    import aerosandbox as asb

    baseline = design_to_airfoil_coords(np.zeros(10))
    selig = to_selig_order(baseline)
    reference = asb.Airfoil("naca0012").coordinates

    cl_bad, cd_bad = _polar(baseline)
    cl_ok, cd_ok = _polar(selig)
    cl_ref, cd_ref = _polar(reference)

    # Geometry check: compare upper surfaces on a common chordwise grid.
    n = (len(baseline) + 1) // 2
    half = len(reference) // 2
    xs = np.linspace(0.01, 0.99, 50)
    y_ours = np.interp(xs, baseline[:n, 0], baseline[:n, 1])
    y_ref = np.interp(xs, reference[:half, 0][::-1], reference[:half, 1][::-1])
    geom_err = float(np.max(np.abs(y_ours - y_ref)))

    return {
        "le_anchored": {"Cl": cl_bad, "Cd": cd_bad, "LD": cl_bad / cd_bad},
        "selig": {"Cl": cl_ok, "Cd": cd_ok, "LD": cl_ok / cd_ok},
        "reference": {"Cl": cl_ref, "Cd": cd_ref, "LD": cl_ref / cd_ref},
        "drag_error_factor": cd_bad / cd_ok,
        "max_geometry_deviation": geom_err,
    }


def main() -> None:
    r = run()
    print(f"NACA0012 baseline, alpha={ALPHA} deg, Re={RE:.0e}\n")
    print(f"  {'ordering':<24}{'Cl':>9}{'Cd':>11}{'L/D':>9}")
    print("  " + "-" * 53)
    for key, label in [
        ("le_anchored", "LE-anchored (as emitted)"),
        ("selig", "Selig (corrected)"),
        ("reference", "reference NACA0012"),
    ]:
        d = r[key]
        print(f"  {label:<24}{d['Cl']:>9.4f}{d['Cd']:>11.5f}{d['LD']:>9.1f}")
    print()
    print(f"  Drag overprediction from ordering alone : {r['drag_error_factor']:.1f}x")
    print(f"  Max geometry deviation vs NACA0012      : {r['max_geometry_deviation']:.2e}")
    print("\n  The geometry is exact; only the point ordering differs. No validity")
    print("  check in the pipeline rejects the LE-anchored form.")


if __name__ == "__main__":
    main()

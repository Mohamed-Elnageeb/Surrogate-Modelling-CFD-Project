"""Fast, validated low-fidelity aerodynamic evaluator.

This module wraps NeuralFoil to provide a millisecond-scale lift/drag oracle for
airfoil design vectors. It exists so the design loop can be exercised, and a
surrogate trained, without waiting on SU2: NeuralFoil evaluates a design in
roughly 2 ms versus minutes for a RANS solve, which makes dataset generation and
optimisation studies tractable on a single CPU core.

Validation of this evaluator against textbook NACA0012 data is covered by
``tests/test_neuralfoil_eval.py``: the lift-curve slope reproduces thin-airfoil
theory to ~2%, lift is ~0 at zero incidence for a symmetric section, and drag
matches published values at Re=1e6.

The evaluator returns the same dictionary shape as
:func:`cfdagent.cfd.run_cfd.run_cfd`, so it can be dropped into
``AirfoilDesignAgent.run_function`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from ..geometry.airfoil_param import design_to_airfoil_coords, to_selig_order

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 4.0
DEFAULT_RE = 1.0e6
DEFAULT_MODEL_SIZE = "medium"


def _scalar(value) -> float:
    """NeuralFoil returns 0-d or 1-element arrays depending on input shape."""

    return float(np.atleast_1d(np.asarray(value)).ravel()[0])


def evaluate_design(
    design_vec: Iterable[float],
    alpha: float = DEFAULT_ALPHA,
    reynolds: float = DEFAULT_RE,
    model_size: str = DEFAULT_MODEL_SIZE,
    n_points: int = 200,
) -> dict:
    """Evaluate one design vector and return lift/drag plus diagnostics.

    Coordinates are converted to Selig order before evaluation. This is
    essential: the LE-anchored ordering produced by
    :func:`design_to_airfoil_coords` is accepted silently by panel-style solvers
    but describes a blunt-nosed body, inflating drag by roughly 9x.
    """

    import neuralfoil as nf

    vec = np.asarray(list(design_vec), dtype=float)
    coords = to_selig_order(design_to_airfoil_coords(vec, n_points=n_points))

    aero = nf.get_aero_from_coordinates(
        coordinates=np.asarray(coords, dtype=float),
        alpha=alpha,
        Re=reynolds,
        model_size=model_size,
    )

    cl = _scalar(aero["CL"])
    cd = _scalar(aero["CD"])
    # NeuralFoil reports an analysis-confidence value; low values flag designs
    # far outside the training manifold, which we surface rather than hide.
    confidence = _scalar(aero.get("analysis_confidence", 1.0))

    return {
        "Cl": cl,
        "Cd": cd,
        "confidence": confidence,
        "alpha": alpha,
        "reynolds": reynolds,
        "coords": coords,
    }


def run_lowfid(
    design_id: str,
    design_vec: Iterable[float],
    workdir: Path | None = None,
    su2_executable: str | None = None,
    param_overrides: dict | None = None,
    alpha: float = DEFAULT_ALPHA,
    reynolds: float = DEFAULT_RE,
    min_confidence: float = 0.0,
) -> dict:
    """``run_cfd``-compatible entry point backed by the low-fidelity evaluator.

    ``workdir``, ``su2_executable`` and ``param_overrides`` are accepted for
    interface compatibility and ignored. Validation mirrors ``run_cfd``: drag
    must be positive and finite, lift may legitimately be negative.
    """

    vec = list(design_vec)
    try:
        out = evaluate_design(vec, alpha=alpha, reynolds=reynolds)
    except Exception as exc:  # pragma: no cover - surfaced as a failed run
        logger.warning("Low-fidelity evaluation failed for %s: %s", design_id, exc)
        return {
            "design_id": design_id,
            "design_vec": vec,
            "Cl": None,
            "Cd": None,
            "success": False,
            "invalid_metrics": False,
            "error": str(exc),
        }

    cl, cd = out["Cl"], out["Cd"]
    reasons: list[str] = []
    if not np.isfinite(cl):
        reasons.append("Cl is not finite")
    if not np.isfinite(cd):
        reasons.append("Cd is not finite")
    elif cd <= 0:
        reasons.append(f"Cd is non-positive ({cd})")
    if out["confidence"] < min_confidence:
        reasons.append(f"analysis confidence {out['confidence']:.3f} below {min_confidence}")

    ok = not reasons
    return {
        "design_id": design_id,
        "design_vec": vec,
        "Cl": cl if ok else None,
        "Cd": cd if ok else None,
        "residual": 0.0,
        "confidence": out["confidence"],
        "alpha": alpha,
        "reynolds": reynolds,
        "fidelity": "neuralfoil",
        "success": ok,
        "invalid_metrics": bool(reasons),
        "error": "; ".join(reasons) if reasons else None,
    }

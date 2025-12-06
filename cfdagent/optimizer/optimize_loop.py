from __future__ import annotations

from typing import Callable, Dict, List, Any
import numpy as np

from cfdagent.geometry.airfoil_param import sample_random_design


def optimize(
    objective: Callable[[np.ndarray], float | Dict[str, Any]],
    n_samples: int = 20,
    bounds: tuple[float, float] = (-0.02, 0.02),
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Lightweight random-search optimiser used for Phase 0.

    Args:
        objective: Callable returning a scalar score (lower is better) or a
            dictionary containing ``loss``/``score``. It receives a design
            vector with shape ``(10,)``.
        n_samples: Number of random designs to evaluate.
        bounds: Tuple of ``(low, high)`` for the uniform design sampler.
        seed: Optional seed for reproducibility.

    Returns:
        Dictionary with ``best_design``, ``best_score`` and a ``history`` list
        capturing every evaluation.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    history: List[Dict[str, Any]] = []
    best_design: np.ndarray | None = None
    best_score: float | None = None

    for idx in range(n_samples):
        design = sample_random_design(bounds[0], bounds[1], rng)
        result = objective(design)

        if isinstance(result, dict):
            score = result.get("loss") or result.get("score") or result.get("objective")
            if score is None:
                raise ValueError("Objective dictionary must include 'loss', 'score', or 'objective'.")
            score_val = float(score)
        else:
            score_val = float(result)

        history.append({"iter": idx, "design": design, "score": score_val})

        if best_score is None or score_val < best_score:
            best_score = score_val
            best_design = design.copy()

    return {"best_design": best_design, "best_score": best_score, "history": history}

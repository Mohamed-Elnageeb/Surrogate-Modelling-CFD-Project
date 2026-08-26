"""Multi-fidelity optimisation study: does surrogate screening save solver calls?

Compares two budget-matched strategies for maximising L/D over the 10-D design
space, both allowed the same number of *expensive* evaluations:

  baseline  : draw N designs at random, evaluate all N with the solver.
  surrogate : draw N * pool_factor designs, rank them with the CNN surrogate
              (a forward pass, essentially free), and spend the N solver calls
              only on the top-ranked candidates.

Reporting the best L/D found per solver-call budget, over several seeds, is the
comparison that shows whether the surrogate is doing useful work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..geometry.airfoil_param import design_to_airfoil_coords, to_selig_order
from ..lowfid.neuralfoil_eval import evaluate_design
from ..scripts.generate_lowfid_dataset import rasterize
from ..surrogate.train_cnn_surrogate import GeometryToCoefficients


def load_surrogate(path: Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GeometryToCoefficients().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["norm_mean"], ckpt["norm_std"]


def _images_for(vectors: np.ndarray, res: int = 64) -> np.ndarray:
    out = np.empty((len(vectors), 1, res, res), dtype=np.float32)
    for i, v in enumerate(vectors):
        out[i, 0] = rasterize(to_selig_order(design_to_airfoil_coords(v)), res=res)
    return out


def surrogate_ld(model, mean, std, vectors, device="cpu") -> np.ndarray:
    imgs = torch.from_numpy(_images_for(vectors)).to(device)
    with torch.no_grad():
        pred = model(imgs).cpu().numpy() * std + mean
    cl, cd = pred[:, 0], pred[:, 1]
    return np.where(cd > 1e-6, cl / np.maximum(cd, 1e-6), -np.inf)


def true_ld(vec) -> float:
    out = evaluate_design(vec)
    if not np.isfinite(out["Cl"]) or not np.isfinite(out["Cd"]) or out["Cd"] <= 0:
        return -np.inf
    return out["Cl"] / out["Cd"]


def run_study(model_path: Path, budgets, pool_factor=20, seeds=10, low=-0.02, high=0.02):
    model, mean, std = load_surrogate(model_path)
    results = {"baseline": {b: [] for b in budgets}, "surrogate": {b: [] for b in budgets}}

    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        for b in budgets:
            # --- baseline: spend the whole budget on random designs
            cand = rng.uniform(low, high, size=(b, 10))
            results["baseline"][b].append(max(true_ld(v) for v in cand))

            # --- surrogate: screen a large pool, solve only the top b
            pool = rng.uniform(low, high, size=(b * pool_factor, 10))
            ranked = np.argsort(surrogate_ld(model, mean, std, pool))[::-1][:b]
            results["surrogate"][b].append(max(true_ld(pool[i]) for i in ranked))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/cnn_surrogate.pt"))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--pool-factor", type=int, default=20)
    a = p.parse_args()

    budgets = [5, 10, 20, 40]
    res = run_study(a.model, budgets, pool_factor=a.pool_factor, seeds=a.seeds)

    print(f"\nBest L/D found vs solver-call budget "
          f"(mean +/- std over {a.seeds} seeds, pool factor {a.pool_factor}x)")
    print(f"  NACA0012 baseline L/D = 58.0\n")
    print(f"  {'budget':>7} | {'random search':>18} | {'surrogate-screened':>19} | {'gain':>7}")
    print("  " + "-" * 62)
    for b in budgets:
        rb = np.array(res["baseline"][b]); rs = np.array(res["surrogate"][b])
        print(f"  {b:>7} | {rb.mean():>8.2f} +/- {rb.std():>5.2f} | "
              f"{rs.mean():>9.2f} +/- {rs.std():>5.2f} | {rs.mean()-rb.mean():>+7.2f}")

    # Equivalent-budget question: how many random draws match the surrogate at b=10?
    s10 = np.array(res["surrogate"][10]).mean()
    for b in budgets:
        if np.array(res["baseline"][b]).mean() >= s10:
            print(f"\n  Random search needs >= {b} calls to match "
                  f"surrogate-screened at 10 calls ({s10:.2f}).")
            break
    else:
        print(f"\n  Random search did not match surrogate-at-10 ({s10:.2f}) "
              f"within {max(budgets)} calls.")


if __name__ == "__main__":
    main()

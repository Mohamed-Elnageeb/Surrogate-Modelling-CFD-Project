"""Generate a labelled airfoil design dataset using the low-fidelity evaluator.

Produces a CSV of design vectors and their lift/drag coefficients, plus an
optional .npz of rasterised geometry images suitable for training a CNN/U-Net
surrogate. Runs in seconds rather than the hours a comparable SU2 campaign
would need, which makes it usable as the exploration tier of a multi-fidelity
workflow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry.airfoil_param import design_to_airfoil_coords, to_selig_order
from ..lowfid.neuralfoil_eval import evaluate_design

N_PARAMS = 10


def rasterize(coords: np.ndarray, res: int = 64) -> np.ndarray:
    """Rasterise an airfoil outline into a signed occupancy image.

    The section is sampled on a fixed [-0.1, 1.1] x [-0.3, 0.3] window so that
    every design shares a common frame of reference, which is what a
    convolutional surrogate needs.
    """

    xs = np.linspace(-0.1, 1.1, res)
    ys = np.linspace(-0.3, 0.3, res)
    gx, gy = np.meshgrid(xs, ys)

    n = len(coords) // 2
    upper = coords[: n + 1][::-1]  # LE -> TE
    lower = coords[n:]  # LE -> TE
    yu = np.interp(gx, upper[:, 0], upper[:, 1], left=np.nan, right=np.nan)
    yl = np.interp(gx, lower[:, 0], lower[:, 1], left=np.nan, right=np.nan)
    inside = (gy <= yu) & (gy >= yl)
    return inside.astype(np.float32)


def generate(
    n_samples: int,
    out_csv: Path,
    out_npz: Path | None = None,
    alpha: float = 4.0,
    reynolds: float = 1e6,
    low: float = -0.02,
    high: float = 0.02,
    seed: int = 0,
    res: int = 64,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    images: list[np.ndarray] = []
    t0 = time.time()
    failures = 0

    for i in range(n_samples):
        vec = rng.uniform(low, high, size=N_PARAMS)
        try:
            out = evaluate_design(vec, alpha=alpha, reynolds=reynolds)
        except Exception:
            failures += 1
            continue

        cl, cd = out["Cl"], out["Cd"]
        if not (np.isfinite(cl) and np.isfinite(cd) and cd > 0):
            failures += 1
            continue

        row = {"design_id": f"d{i:06d}"}
        row.update({f"dc{j+1}": float(vec[j]) for j in range(5)})
        row.update({f"dt{j+1}": float(vec[5 + j]) for j in range(5)})
        row.update(
            {
                "Cl": cl,
                "Cd": cd,
                "LD": cl / cd,
                "confidence": out["confidence"],
                "alpha": alpha,
                "reynolds": reynolds,
                "success": True,
            }
        )
        rows.append(row)
        if out_npz is not None:
            images.append(rasterize(out["coords"], res=res))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if out_npz is not None and images:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            images=np.stack(images).astype(np.float32),
            params=df[[f"dc{i+1}" for i in range(5)] + [f"dt{i+1}" for i in range(5)]]
            .to_numpy(dtype=np.float32),
            cl=df["Cl"].to_numpy(dtype=np.float32),
            cd=df["Cd"].to_numpy(dtype=np.float32),
        )

    dt = time.time() - t0
    print(f"Generated {len(df)} samples ({failures} rejected) in {dt:.1f}s "
          f"({dt / max(n_samples, 1) * 1000:.1f} ms/design)")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--out-csv", type=Path, default=Path("data/lowfid/designs.csv"))
    p.add_argument("--out-npz", type=Path, default=Path("data/lowfid/snapshots.npz"))
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--reynolds", type=float, default=1e6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--res", type=int, default=64)
    args = p.parse_args()
    generate(
        args.n_samples, args.out_csv, args.out_npz,
        alpha=args.alpha, reynolds=args.reynolds, seed=args.seed, res=args.res,
    )


if __name__ == "__main__":
    main()

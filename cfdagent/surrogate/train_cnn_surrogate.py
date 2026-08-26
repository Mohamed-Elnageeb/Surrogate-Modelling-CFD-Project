"""Train a CNN surrogate mapping rasterised airfoil geometry to (Cl, Cd).

This is the screening model used by the multi-fidelity design loop: it replaces
an aerodynamic solve with a forward pass, so candidate designs can be ranked in
bulk and only the most promising ones sent to a real solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class GeometryToCoefficients(nn.Module):
    """Small conv encoder -> (Cl, Cd) regression head."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(base * 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class Normalizer:
    """Standardises targets so Cl and Cd contribute comparably to the loss."""

    mean: np.ndarray
    std: np.ndarray

    def encode(self, y):
        return (y - self.mean) / self.std

    def decode(self, y):
        return y * self.std + self.mean


def train(
    npz_path: Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str | None = None,
):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(npz_path)
    images = z["images"][:, None, :, :].astype(np.float32)
    targets = np.stack([z["cl"], z["cd"]], axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_val = int(len(idx) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    norm = Normalizer(targets[train_idx].mean(0), targets[train_idx].std(0))
    y_all = norm.encode(targets)

    xt = torch.from_numpy(images[train_idx]).to(device)
    yt = torch.from_numpy(y_all[train_idx]).to(device)
    xv = torch.from_numpy(images[val_idx]).to(device)
    yv = torch.from_numpy(y_all[val_idx]).to(device)

    model = GeometryToCoefficients().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(xv), yv).item()
            print(f"  epoch {ep+1:3d}  val_loss(norm)={vl:.5f}")

    model.eval()
    with torch.no_grad():
        pred = norm.decode(model(xv).cpu().numpy())
    true = targets[val_idx]

    def r2(a, b):
        return 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()

    metrics = {
        "cl_r2": float(r2(pred[:, 0], true[:, 0])),
        "cd_r2": float(r2(pred[:, 1], true[:, 1])),
        "cl_mae": float(np.abs(pred[:, 0] - true[:, 0]).mean()),
        "cd_mae": float(np.abs(pred[:, 1] - true[:, 1]).mean()),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    return model, norm, metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, default=Path("data/lowfid/snapshots.npz"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("models/cnn_surrogate.pt"))
    a = p.parse_args()

    model, norm, metrics = train(a.npz, epochs=a.epochs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "norm_mean": norm.mean, "norm_std": norm.std}, a.out)
    print("\n=== held-out surrogate accuracy ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")

"""Loaders for the NASA Turbulence Modeling Resource NACA0012 datasets.

The datasets differ in one respect that dominates drag: whether the wind-tunnel
model was *tripped*. A tripped model is turbulent from near the leading edge; an
untripped model runs laminar over the forward chord and has substantially lower
drag. Any comparison between a prediction and an experiment is only meaningful
when the transition states match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "experimental"


@dataclass(frozen=True)
class ExperimentalDataset:
    """A digitised experimental polar."""

    name: str
    reynolds: float
    tripped: bool
    cl: np.ndarray
    cd: np.ndarray | None = None
    alpha: np.ndarray | None = None
    note: str = ""

    @property
    def transition(self) -> str:
        return "tripped" if self.tripped else "free"

    def __len__(self) -> int:
        return len(self.cl)


def _read_columns(path: Path, ncols: int) -> np.ndarray:
    rows: list[tuple[float, ...]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "variables", "zone")):
            continue
        parts = line.split()
        if len(parts) != ncols:
            continue
        try:
            rows.append(tuple(float(p) for p in parts))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")
    return np.array(rows)


def load_ladson(data_dir: Path | None = None) -> ExperimentalDataset:
    """Ladson, NASA TM 4074 (1988). Re=6e6, M=0.15, transition tripped."""

    d = _read_columns((data_dir or DATA_DIR) / "CLCD_Ladson_expdata.dat", 3)
    return ExperimentalDataset(
        name="Ladson 1988", reynolds=6.0e6, tripped=True,
        alpha=d[:, 0], cl=d[:, 1], cd=d[:, 2],
        note="80-grit trip, M=0.15",
    )


def load_abbott(data_dir: Path | None = None) -> ExperimentalDataset:
    """Abbott & von Doenhoff drag polar. Re=6e6, untripped (free transition).

    Digitised from a printed figure; the NASA source file notes the digitisation
    is only approximate, so this dataset is noisier than Ladson.
    """

    d = _read_columns((data_dir or DATA_DIR) / "0012.abbottdata.cd.dat", 2)
    order = np.argsort(d[:, 0])
    return ExperimentalDataset(
        name="Abbott & von Doenhoff", reynolds=6.0e6, tripped=False,
        cl=d[order, 0], cd=d[order, 1],
        note="digitised from printed plot; approximate",
    )


def load_gregory(data_dir: Path | None = None) -> ExperimentalDataset:
    """Gregory & O'Reilly (1970). Re=3e6, transition tripped. Lift only here."""

    d = _read_columns((data_dir or DATA_DIR) / "CL_Gregory_expdata.dat", 2)
    return ExperimentalDataset(
        name="Gregory & O'Reilly 1970", reynolds=3.0e6, tripped=True,
        alpha=d[:, 0], cl=d[:, 1],
    )


def load_all(data_dir: Path | None = None) -> dict[str, ExperimentalDataset]:
    return {
        "ladson": load_ladson(data_dir),
        "abbott": load_abbott(data_dir),
        "gregory": load_gregory(data_dir),
    }

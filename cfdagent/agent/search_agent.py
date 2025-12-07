"""Agent-driven loop for exploring airfoil designs with CFD feedback.

The :class:`AirfoilDesignAgent` learns from historical SU2 runs stored in
``data/designs.csv`` and proposes new candidates by training a lightweight
regressor on the observed lift/drag performance. It perturbs the best designs
found so far, runs CFD evaluations, and appends the outcomes back to the dataset
for continual improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor

from ..cfd.run_cfd import run_cfd
from ..geometry.airfoil_param import sample_random_design
from ..utils.io_utils import DESIGN_LOG

DesignVector = Sequence[float]
RunFunction = Callable[[str, Iterable[float], Path | None, str | None], dict]


@dataclass
class AirfoilDesignAgent:
    """Agent that proposes and evaluates promising airfoil designs.

    The agent maintains a growing dataset of design parameters and resulting
    aerodynamic metrics. On each iteration it:

    1. Loads historical designs from ``designs.csv``.
    2. Fits a regressor to predict ``score = Cl - 0.1 * Cd`` from the 10 design
       parameters.
    3. Identifies high-performing designs and samples new candidates near them
       via Gaussian perturbations.
    4. Runs CFD (or a user-provided surrogate) on each candidate and logs the
       results back to ``designs.csv`` for future iterations.
    """

    design_log: Path = Path(DESIGN_LOG)
    noise_scale: float = 0.003
    top_k: int = 5
    design_bounds: tuple[float, float] = (-0.05, 0.05)
    random_state: int | None = None
    run_function: RunFunction = run_cfd

    def __post_init__(self) -> None:
        self.design_log = Path(self.design_log)
        self.rng = np.random.default_rng(self.random_state)

    def load_history(self) -> pd.DataFrame:
        """Return the historical design log (may be empty)."""

        if not self.design_log.exists():
            return pd.DataFrame()
        return pd.read_csv(self.design_log)

    def _design_columns(self, df: pd.DataFrame) -> list[str]:
        candidates = [
            col
            for col in df.columns
            if col.lower().startswith(("dc", "dt", "design_", "param_"))
        ]
        if candidates:
            return sorted(candidates, key=self._column_sort_key)

        return [f"dc{i+1}" for i in range(5)] + [f"dt{i+1}" for i in range(5)]

    @staticmethod
    def _column_sort_key(name: str) -> tuple[str, int]:
        match = re.match(r"([a-zA-Z_]+)(\d+)$", name)
        if match:
            return match.group(1), int(match.group(2))
        return name, 0

    @staticmethod
    def _score_series(df: pd.DataFrame) -> pd.Series:
        cl_col = next((c for c in ["Cl", "CL", "cl"] if c in df.columns), None)
        cd_col = next((c for c in ["Cd", "CD", "cd"] if c in df.columns), None)
        if cl_col is None or cd_col is None:
            return pd.Series(dtype=float)
        return df[cl_col].astype(float) - 0.1 * df[cd_col].astype(float)

    def _fit_regressor(self, df: pd.DataFrame, columns: list[str]):
        if df.empty:
            return None

        if not set(columns).issubset(df.columns):
            return None

        scores = self._score_series(df)
        valid = df[columns].copy()
        valid["score"] = scores
        valid = valid.dropna(subset=columns + ["score"])
        if len(valid) < 2:
            return None

        X = valid[columns].to_numpy(dtype=float)
        y = valid["score"].to_numpy(dtype=float)

        if len(np.unique(y)) <= 1:
            model = DummyRegressor(strategy="mean")
        else:
            model = GradientBoostingRegressor(random_state=self.random_state)
        model.fit(X, y)
        return model

    def _select_seed_designs(
        self, df: pd.DataFrame, columns: list[str], model
    ) -> list[np.ndarray]:
        if df.empty or not set(columns).issubset(df.columns):
            return []

        seeds = df[columns].dropna()
        if seeds.empty:
            return []

        scores = self._score_series(df)
        seeds = seeds.copy()
        if not scores.empty:
            seeds["score"] = scores

        if model is not None:
            preds = model.predict(seeds[columns].to_numpy(dtype=float))
            seeds["score"] = seeds.get("score", pd.Series(dtype=float)).fillna(pd.Series(preds, index=seeds.index))
            seeds["score"] = seeds["score"].fillna(pd.Series(preds, index=seeds.index))

        if "score" not in seeds or seeds["score"].isna().all():
            return []

        seeds = seeds.dropna(subset=["score"])
        seeds = seeds.sort_values("score", ascending=False)
        seeds = seeds.head(self.top_k) if self.top_k else seeds
        return [row.to_numpy(dtype=float) for _, row in seeds[columns].iterrows()]

    def _sample_candidates(self, seeds: list[np.ndarray], num_candidates: int) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []
        lower, upper = self.design_bounds

        while len(candidates) < num_candidates:
            if seeds:
                seed = seeds[len(candidates) % len(seeds)]
                perturb = self.rng.normal(scale=self.noise_scale, size=len(seed))
                candidate = np.clip(seed + perturb, lower, upper)
            else:
                candidate = sample_random_design(rng=self.rng)
            candidates.append(candidate.astype(float))

        return candidates

    def _build_row(self, design_id: str, vec: np.ndarray, columns: list[str], result: dict) -> dict:
        row = {"design_id": design_id}
        for name, value in zip(columns, vec):
            row[name] = float(value)

        row.update(
            {
                "Cl": result.get("Cl"),
                "Cd": result.get("Cd"),
                "residual": result.get("residual"),
                "success": result.get("success", False),
            }
        )
        if result.get("error"):
            row["error"] = result.get("error")
        return row

    def _append_row(self, row: dict) -> None:
        self.design_log.parent.mkdir(parents=True, exist_ok=True)
        if self.design_log.exists():
            df = pd.read_csv(self.design_log)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(self.design_log, index=False)

    def run_iteration(
        self,
        num_candidates: int = 3,
        workdir: Path | None = None,
        su2_executable: str | None = None,
    ) -> list[dict]:
        """Run an agent iteration: propose candidates, evaluate them, log results."""

        history = self.load_history()
        design_columns = self._design_columns(history)
        model = self._fit_regressor(history, design_columns)
        seeds = self._select_seed_designs(history, design_columns, model)
        candidates = self._sample_candidates(seeds, num_candidates)

        results: list[dict] = []
        for vec in candidates:
            design_id = str(uuid.uuid4())[:8]
            sim_result = self.run_function(design_id, vec, workdir=workdir, su2_executable=su2_executable)
            row = self._build_row(design_id, np.asarray(vec), design_columns, sim_result)
            self._append_row(row)
            results.append(row)

        return results

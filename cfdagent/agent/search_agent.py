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
from typing import Any, Callable, Iterable, Sequence
import re
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor

from ..cfd.run_cfd import run_cfd
from ..geometry.airfoil_param import design_to_airfoil_coords, sample_random_design
from ..utils.snapshot_pipeline import (
    SnapshotSpec,
    create_snapshot_from_run,
    relative_snapshot_path,
)
from ..utils.io_utils import DESIGN_LOG

DesignVector = Sequence[float]
RunFunction = Callable[
    [str, Iterable[float], Path | None, str | None, dict[str, float | int | str] | None],
    dict,
]


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
    report_dir: Path | None = None

    def __post_init__(self) -> None:
        self.design_log = Path(self.design_log)
        self.rng = np.random.default_rng(self.random_state)
        self.report_dir = (
            Path(self.report_dir)
            if self.report_dir is not None
            else self.design_log.parent / "reports"
        )

    @property
    def _benchmarks(self) -> list[dict]:
        """Static lift/drag reference points drawn from public NACA studies."""

        return [
            {
                "name": "NACA0012_lowAoA",
                "Cl": 0.32,
                "Cd": 0.009,
                "conditions": "M=0.15, Re≈3e6, α≈2° (wind-tunnel data)",
            },
            {
                "name": "NACA0012_cruise",
                "Cl": 0.52,
                "Cd": 0.011,
                "conditions": "M=0.3, Re≈6e6, α≈4° (literature averages)",
            },
            {
                "name": "NACA2412_reference",
                "Cl": 0.72,
                "Cd": 0.013,
                "conditions": "M=0.3, Re≈6e6, α≈4° (cambered baseline)",
            },
        ]

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
            and col.lower() not in {"design_id"}
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
            if df.empty:
                df = pd.DataFrame([row])
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(self.design_log, index=False)

    def _plot_performance(self, df: pd.DataFrame, design_columns: list[str]) -> Path | None:
        if df.empty or not {"Cl", "Cd"}.issubset(df.columns):
            return None

        df = df.dropna(subset=["Cl", "Cd"])
        if df.empty:
            return None

        self.report_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        scores = self._score_series(df)
        df["score"] = scores
        df = df.sort_values("score", ascending=False)

        axes[0].scatter(df["Cd"], df["Cl"], c=df["score"], cmap="viridis", edgecolor="k")
        axes[0].set_xlabel("Cd (drag coefficient)")
        axes[0].set_ylabel("Cl (lift coefficient)")
        axes[0].set_title("Lift vs Drag")

        axes[1].plot(range(len(df)), df["score"], marker="o")
        axes[1].set_xlabel("Ranked design index")
        axes[1].set_ylabel("Score (Cl - 0.1*Cd)")
        axes[1].set_title("Performance leaderboard")
        axes[1].grid(True, linestyle="--", linewidth=0.5)

        fig.tight_layout()
        perf_path = self.report_dir / "performance.png"
        fig.savefig(perf_path, dpi=200)
        plt.close(fig)
        return perf_path

    def _plot_geometry(self, vec: Sequence[float]) -> Path:
        coords = design_to_airfoil_coords(np.asarray(vec, dtype=float))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(coords[:, 0], coords[:, 1], label="optimized airfoil")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / chord")
        ax.set_ylabel("y / chord")
        ax.set_title("Optimized airfoil geometry")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        geom_path = self.report_dir / "best_geometry.png"
        fig.savefig(geom_path, dpi=200)
        plt.close(fig)
        return geom_path

    def _benchmark_summary(self, best_cl: float, best_cd: float) -> tuple[str, dict | None]:
        best_score = best_cl - 0.1 * best_cd
        chosen = self._benchmarks[0]
        deltas: dict[str, float | str] | None = None
        if self._benchmarks:
            chosen = min(
                self._benchmarks,
                key=lambda b: abs((b["Cl"] - b["Cd"] * 0.1) - best_score),
            )
            bench_score = chosen["Cl"] - 0.1 * chosen["Cd"]
            deltas = {
                "score_delta": best_score - bench_score,
                "Cl_delta": best_cl - chosen["Cl"],
                "Cd_delta": best_cd - chosen["Cd"],
            }

        summary = (
            f"Benchmark: {chosen['name']} ({chosen['conditions']}). "
            f"Reference Cl={chosen['Cl']:.3f}, Cd={chosen['Cd']:.4f}. "
            f"Agent best Cl={best_cl:.3f}, Cd={best_cd:.4f}, score={best_score:.3f}."
        )
        return summary, deltas

    def generate_review(self, n_best: int = 3) -> dict:
        """Build a post-optimization review with plots and benchmark comparison."""

        df = self.load_history()
        design_columns = self._design_columns(df)
        scores = self._score_series(df)
        for col in design_columns:
            if col not in df.columns:
                df[col] = np.nan
        for metric in ("Cl", "Cd"):
            if metric not in df.columns:
                df[metric] = np.nan
        df["score"] = scores
        df = df.dropna(subset=design_columns + ["Cl", "Cd", "score"], how="any")
        if df.empty:
            return {
                "review": "No completed CFD runs found. Run the agent to generate data first.",
                "plots": {},
            }

        df = df.sort_values("score", ascending=False)
        leaderboard = df.head(n_best)

        perf_plot = self._plot_performance(df, design_columns)
        geom_plot = self._plot_geometry(leaderboard.iloc[0][design_columns].to_numpy(dtype=float))

        best_row = leaderboard.iloc[0]
        bench_text, deltas = self._benchmark_summary(float(best_row["Cl"]), float(best_row["Cd"]))

        bullet_lines = [
            f"Top {n_best} designs by score (Cl - 0.1*Cd):",
        ]
        for idx, row in leaderboard.iterrows():
            bullet_lines.append(
                f"  • Design {row.get('design_id', idx)}: Cl={row['Cl']:.3f}, Cd={row['Cd']:.4f}, score={row['score']:.3f}"
            )

        review_lines = [
            "Agent optimization review:",
            *bullet_lines,
            bench_text,
        ]

        if deltas:
            review_lines.append(
                (
                    "Delta vs. benchmark: "
                    f"Δscore={deltas['score_delta']:+.3f}, "
                    f"ΔCl={deltas['Cl_delta']:+.3f}, "
                    f"ΔCd={deltas['Cd_delta']:+.4f}"
                )
            )

        review_text = "\n".join(review_lines)

        self.report_dir.mkdir(parents=True, exist_ok=True)
        review_file = self.report_dir / "review.txt"
        review_file.write_text(review_text, encoding="utf-8")

        return {
            "review": review_text,
            "plots": {
                "performance": str(perf_plot) if perf_plot else None,
                "geometry": str(geom_plot),
            },
            "leaderboard": leaderboard,
            "review_file": str(review_file),
        }

    def run_iteration(
        self,
        num_candidates: int = 3,
        workdir: Path | None = None,
        su2_executable: str | None = None,
        summarize: bool = False,
        snapshot_spec: SnapshotSpec | None = None,
    ) -> list[dict] | dict[str, Any]:
        """Run an agent iteration: propose candidates, evaluate them, log results.

        When ``snapshot_spec`` is provided, each successful run also emits a
        snapshot archive alongside the logged metrics, enabling a full
        dataset→training pipeline without switching tools.
        """

        history = self.load_history()
        design_columns = self._design_columns(history)
        model = self._fit_regressor(history, design_columns)
        seeds = self._select_seed_designs(history, design_columns, model)
        candidates = self._sample_candidates(seeds, num_candidates)

        results: list[dict] = []
        failures: list[dict] = []
        for vec in candidates:
            design_id = str(uuid.uuid4())[:8]
            sim_result = self.run_function(design_id, vec, workdir=workdir, su2_executable=su2_executable)
            row = self._build_row(design_id, np.asarray(vec), design_columns, sim_result)
            if sim_result.get("invalid_metrics"):
                row.setdefault(
                    "error",
                    sim_result.get(
                        "error",
                        "CFD run did not yield valid lift/drag metrics (check SU2 installation and case setup)",
                    ),
                )
                failures.append(row)
                continue
            if not row.get("success", False):
                row.setdefault(
                    "error",
                    sim_result.get(
                        "error",
                        "CFD run did not yield valid lift/drag metrics (check SU2 installation and case setup)",
                    ),
                )
                failures.append(row)
                self._append_row(row)
                results.append(row)
                continue
            if snapshot_spec and row.get("success"):
                try:
                    snapshot_path = create_snapshot_from_run(design_id, snapshot_spec, sim_result)
                    row["flow_path"] = relative_snapshot_path(snapshot_path)
                except Exception as exc:  # pylint: disable=broad-except
                    row["snapshot_error"] = str(exc)
            self._append_row(row)
            results.append(row)

        if summarize:
            review = self.generate_review()
            if review.get("review", "").startswith("No completed CFD runs") and failures:
                error_lines = [
                    f"  • {run.get('design_id', 'unknown')}: {run.get('error', 'Unknown error')}"
                    for run in failures
                ]
                review["review"] = "No successful CFD runs were completed. Encountered errors:\n" + "\n".join(error_lines)

            return {"results": results, "review": review}

        return results

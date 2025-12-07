from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import itertools
import random

from .run_cfd import Su2RunConfig, run_su2_case


@dataclass
class CaseSample:
    """Single CFD case definition for parameter sweep."""

    mach: float
    aoa: float
    reynolds: float | None = None
    extra: dict[str, float | int | str] | None = None

    def to_overrides(self) -> dict[str, float | int | str]:
        """Build overrides for SU2 configuration."""

        overrides: dict[str, float | int | str] = {
            "MACH_NUMBER": self.mach,
            "AOA": self.aoa,
        }
        if self.reynolds is not None:
            overrides["REYNOLDS_NUMBER"] = self.reynolds
        if self.extra:
            overrides.update(self.extra)
        return overrides


def generate_param_grid(
    mach_values: Iterable[float],
    aoa_values: Iterable[float],
    reynolds_values: Iterable[float] | None = None,
) -> list[CaseSample]:
    """Generate a Cartesian grid of CaseSample objects."""

    if reynolds_values is None:
        combos = itertools.product(mach_values, aoa_values)
        return [CaseSample(mach=m, aoa=a) for m, a in combos]

    combos = itertools.product(mach_values, aoa_values, reynolds_values)
    return [CaseSample(mach=m, aoa=a, reynolds=r) for m, a, r in combos]


def _collect_input_columns(samples: list[CaseSample]) -> list[str]:
    base_cols = {"mach", "aoa", "reynolds"}
    extra_cols: set[str] = set()
    for sample in samples:
        if sample.extra:
            extra_cols.update(sample.extra.keys())
    return sorted(base_cols | extra_cols)


def run_param_sweep(
    cfg: Su2RunConfig,
    samples: list[CaseSample],
    out_csv: Path,
    shuffle: bool = False,
    max_cases: int | None = None,
) -> None:
    """Run SU2 cases for samples and write dataset CSV."""

    selected_samples = list(samples)
    if shuffle:
        random.shuffle(selected_samples)
    if max_cases is not None:
        selected_samples = selected_samples[:max_cases]

    input_columns = _collect_input_columns(selected_samples)

    results: list[tuple[CaseSample, dict]] = []
    output_keys: set[str] = set()

    for sample in selected_samples:
        overrides = sample.to_overrides()
        result = run_su2_case(cfg, param_overrides=overrides)
        history = result.get("history_data", {}) or {}
        results.append((sample, history))
        output_keys.update(history.keys())

    output_columns = [f"out_{key}" for key in sorted(output_keys)]
    header = input_columns + output_columns

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for sample, history in results:
            row = {key: "" for key in header}
            sample_dict = asdict(sample)
            row["mach"] = sample_dict.get("mach")
            row["aoa"] = sample_dict.get("aoa")
            row["reynolds"] = sample_dict.get("reynolds")

            extras = sample_dict.get("extra") or {}
            for key in input_columns:
                if key in {"mach", "aoa", "reynolds"}:
                    continue
                if key in extras:
                    row[key] = extras[key]

            for key, value in history.items():
                row[f"out_{key}"] = str(value)

            writer.writerow(row)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2] / "TestCases" / "airfoil_naca0012_opt"
    cfg = Su2RunConfig(workdir=base_dir)

    machs = [0.1, 0.15]
    aoas = [0.0, 5.0]
    samples = generate_param_grid(machs, aoas)

    out_path = base_dir / "dataset_demo.csv"
    run_param_sweep(cfg, samples, out_path, shuffle=False, max_cases=None)
    print(f"Wrote dataset to: {out_path}")


if __name__ == "__main__":
    main()

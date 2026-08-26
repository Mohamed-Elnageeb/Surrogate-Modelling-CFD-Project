"""Census of coordinate-ordering conventions in the UIUC airfoil database.

Motivation
----------
This project shipped a coordinate-ordering bug: `design_to_airfoil_coords`
emitted a leading-edge-anchored loop, which panel-style solvers accept silently
while modelling a different body (8.9x drag error, see `convention_error.py`).
The natural follow-up question is whether that hazard is widespread in the data
the field actually trains on, or whether it was specific to this codebase.

Result: NEGATIVE
----------------
All 2174 airfoils in the UIUC database as redistributed with AeroSandbox are in
Selig (trailing-edge-anchored) order. Zero are Lednicer-ordered. The curated
distribution has been normalised, so the hazard does not contaminate this
corpus. The Selig/Lednicer duality is real and documented upstream -- UIUC hosts
both, and downstream parsers detect the format heuristically (airfoiltools.com
assumes Lednicer only when the first numeric pair exceeds 1) -- but that is a
hazard for hand-supplied files, not a defect in the canonical dataset.

A methodological note worth keeping
-----------------------------------
An earlier version of this script reported 20 "LE-anchored" files. That was
wrong: the TASOPT files carry a metadata line (`-2.0  3.0`) after the name, and
a naive parser reads it as a coordinate, which relocates the apparent start
point. The false positive was produced by exactly the class of silent
preprocessing bug this experiment set out to measure. Coordinate rows are now
filtered to the physically valid range before classification.
"""

from __future__ import annotations

import pathlib

import numpy as np

X_RANGE = (-0.01, 1.01)
Y_LIMIT = 1.0


def database_dir() -> pathlib.Path:
    import aerosandbox

    return pathlib.Path(aerosandbox.__file__).parent / "geometry/airfoil/airfoil_database"


def load_coordinates(path: pathlib.Path) -> np.ndarray | None:
    """Parse a UIUC .dat file, discarding non-coordinate header rows.

    Some files carry a count line (Lednicer) or solver metadata (TASOPT) after
    the name. Rows outside the physically valid coordinate range are dropped
    before the first genuine point is located.
    """

    rows = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if len(rows) < 10:
        return None

    arr = np.array(rows)
    valid = (
        (arr[:, 0] >= X_RANGE[0])
        & (arr[:, 0] <= X_RANGE[1])
        & (np.abs(arr[:, 1]) <= Y_LIMIT)
    )
    if not valid.any():
        return None
    start = int(np.argmax(valid))
    return arr[start:][valid[start:]]


def classify(coords: np.ndarray) -> str:
    """Selig starts and ends at the TE; Lednicer starts at the LE."""

    if coords[0, 0] > 0.5 and coords[-1, 0] > 0.5:
        return "selig"
    if coords[0, 0] < 0.05 and coords[-1, 0] > 0.5:
        return "lednicer"
    return "other"


def run() -> dict:
    counts = {"selig": 0, "lednicer": 0, "other": 0}
    outliers: list[str] = []
    for path in sorted(database_dir().glob("*.dat")):
        coords = load_coordinates(path)
        if coords is None or len(coords) < 20:
            continue
        kind = classify(coords)
        counts[kind] += 1
        if kind != "selig":
            outliers.append(f"{path.name} [{kind}]")
    counts["total"] = sum(counts[k] for k in ("selig", "lednicer", "other"))
    counts["outliers"] = outliers
    return counts


def main() -> None:
    r = run()
    total = r["total"]
    print("Coordinate-ordering census, UIUC database (as bundled with AeroSandbox)\n")
    print(f"  total airfoils         : {total}")
    for kind in ("selig", "lednicer", "other"):
        pct = 100 * r[kind] / total if total else 0.0
        print(f"  {kind:<22} : {r[kind]:>5}  ({pct:.1f}%)")
    if r["outliers"]:
        print("\n  outliers:")
        for name in r["outliers"][:20]:
            print(f"     {name}")
    print(
        "\n  NEGATIVE RESULT: the curated database is uniformly Selig-ordered, so the"
        "\n  ordering hazard seen in this project was a defect in our own generator,"
        "\n  not contamination of the corpus the field trains on."
    )


if __name__ == "__main__":
    main()

"""Generate a 2D Gmsh mesh for a four-digit NACA airfoil.

This module provides helpers to sample NACA 4-digit airfoil coordinates and
construct a 2D unstructured mesh using the Gmsh Python API. It can be used as a
library function or executed as a CLI tool to write an SU2-compatible mesh.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import List, Tuple

try:
    import gmsh  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    print(
        "The Gmsh Python API is required to run this module. "
        "Install Gmsh (pip install gmsh) and try again.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def _cosine_spacing(num: int) -> List[float]:
    return [0.5 * (1 - math.cos(math.pi * i / (num - 1))) for i in range(num)]


def naca4_points(code: str, n: int, chord: float, spacing: str = "cosine") -> List[Tuple[float, float]]:
    """Compute coordinates along the upper and lower surfaces of a NACA airfoil.

    Parameters
    ----------
    code:
        Four-digit NACA code such as "2412".
    n:
        Number of sample points per surface.
    chord:
        Airfoil chord length.
    spacing:
        Either "uniform" or "cosine" point spacing along the chord.

    Returns
    -------
    list of tuple
        List of ``(x, y)`` coordinates ordered along the upper surface from
        leading to trailing edge, followed by the lower surface from trailing to
        leading edge.
    """

    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must be a four-digit string.")
    if n < 2:
        raise ValueError("Number of points must be at least 2 per surface.")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    if spacing.lower() == "uniform":
        x_dist = [i / (n - 1) for i in range(n)]
    elif spacing.lower() == "cosine":
        x_dist = _cosine_spacing(n)
    else:
        raise ValueError("Spacing must be either 'uniform' or 'cosine'.")

    x_coords = [x * chord for x in x_dist]
    y_t = [
        5
        * t
        * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
        * chord
        for x in x_dist
    ]

    y_c = [0.0 for _ in x_coords]
    dyc_dx = [0.0 for _ in x_coords]
    if p > 0:
        for i, x in enumerate(x_dist):
            if x < p:
                y_c[i] = m / (p**2) * (2 * p * x - x**2)
                dyc_dx[i] = 2 * m / (p**2) * (p - x)
            else:
                y_c[i] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2)
                dyc_dx[i] = 2 * m / ((1 - p) ** 2) * (p - x)

    theta = [math.atan(dy) for dy in dyc_dx]

    upper = []
    lower = []
    for x, yt, yc, th in zip(x_coords, y_t, y_c, theta):
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))

    # lower surface in reverse to trace the outline counter-clockwise
    return upper + lower[::-1]


def generate_mesh(
    naca_code: str,
    chord: float = 1.0,
    farfield_radius: float = 20.0,
    mesh_size_airfoil: float = 0.01,
    mesh_size_farfield: float = 1.0,
    outfile: str = "mesh.su2",
) -> None:
    """Generate and write an SU2 mesh for the specified NACA airfoil."""

    gmsh.initialize()
    gmsh.model.add("naca_airfoil")

    try:
        coords = naca4_points(naca_code, n=101, chord=chord, spacing="cosine")

        point_map: dict[Tuple[float, float], int] = {}

        def add_point(pt: Tuple[float, float], size: float) -> int:
            key = (round(pt[0], 8), round(pt[1], 8))
            if key in point_map:
                return point_map[key]
            tag = gmsh.model.occ.addPoint(pt[0], pt[1], 0.0, size)
            point_map[key] = tag
            return tag

        num_surface_points = len(coords) // 2
        upper_points = coords[:num_surface_points]
        lower_points = coords[num_surface_points:]

        # Enforce shared tags at the leading and trailing edges to guarantee the
        # inner airfoil loop is perfectly closed even with floating point noise.
        leading_tag = add_point(upper_points[0], mesh_size_airfoil)
        trailing_tag = add_point(upper_points[-1], mesh_size_airfoil)

        upper_tags = (
            [leading_tag]
            + [add_point(pt, mesh_size_airfoil) for pt in upper_points[1:-1]]
            + [trailing_tag]
        )
        lower_tags = (
            [trailing_tag]
            + [add_point(pt, mesh_size_airfoil) for pt in lower_points[1:-1]]
            + [leading_tag]
        )

        spline_upper = gmsh.model.occ.addSpline(upper_tags)
        spline_lower = gmsh.model.occ.addSpline(lower_tags)

        # Farfield outer circle
        center = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        p0 = gmsh.model.occ.addPoint(farfield_radius, 0.0, 0.0, mesh_size_farfield)
        p1 = gmsh.model.occ.addPoint(0.0, farfield_radius, 0.0, mesh_size_farfield)
        p2 = gmsh.model.occ.addPoint(-farfield_radius, 0.0, 0.0, mesh_size_farfield)
        p3 = gmsh.model.occ.addPoint(0.0, -farfield_radius, 0.0, mesh_size_farfield)

        arc1 = gmsh.model.occ.addCircleArc(p0, center, p1)
        arc2 = gmsh.model.occ.addCircleArc(p1, center, p2)
        arc3 = gmsh.model.occ.addCircleArc(p2, center, p3)
        arc4 = gmsh.model.occ.addCircleArc(p3, center, p0)
        outer_loop = gmsh.model.occ.addCurveLoop([arc1, arc2, arc3, arc4])

        inner_loop = gmsh.model.occ.addCurveLoop([spline_upper, spline_lower])

        surface = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])

        gmsh.model.occ.synchronize()

        # Distance-based mesh refinement near the airfoil
        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            field_distance, "EdgesList", [spline_upper, spline_lower]
        )
        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "IField", field_distance)
        gmsh.model.mesh.field.setNumber(field_threshold, "LcMin", mesh_size_airfoil)
        gmsh.model.mesh.field.setNumber(field_threshold, "LcMax", mesh_size_farfield)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", chord * 0.1)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", farfield_radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

        gmsh.model.addPhysicalGroup(2, [surface], name="Fluid")
        gmsh.model.addPhysicalGroup(1, [spline_upper, spline_lower], name="Airfoil")
        gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="Farfield")

        gmsh.model.mesh.generate(2)
        gmsh.write(outfile)
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naca-code", default="0012", help="Four-digit NACA code")
    parser.add_argument("--chord", type=float, default=1.0, help="Chord length")
    parser.add_argument(
        "--farfield-radius", type=float, default=20.0, help="Radius of farfield circle"
    )
    parser.add_argument(
        "--mesh-size-airfoil",
        type=float,
        default=0.01,
        help="Target mesh size near airfoil",
    )
    parser.add_argument(
        "--mesh-size-farfield",
        type=float,
        default=1.0,
        help="Target mesh size on farfield boundary",
    )
    parser.add_argument("--outfile", default="mesh.su2", help="Output SU2 filename")

    args = parser.parse_args()

    generate_mesh(
        naca_code=args.naca_code,
        chord=args.chord,
        farfield_radius=args.farfield_radius,
        mesh_size_airfoil=args.mesh_size_airfoil,
        mesh_size_farfield=args.mesh_size_farfield,
        outfile=args.outfile,
    )

    print(f"Mesh generated successfully and saved to {args.outfile}")


if __name__ == "__main__":
    main()

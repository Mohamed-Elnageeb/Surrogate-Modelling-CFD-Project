"""Boundary-layer-resolved 2-D airfoil mesh generation for viscous SU2 runs.

The mesh produced by :func:`cfdagent.cfd.run_cfd._regenerate_mesh_for_design`
is a uniform unstructured triangulation with ~1e-2 chord spacing at the wall and
a 20-chord farfield. That is adequate for inviscid lift but cannot resolve a
boundary layer: viscous drag at Re=1e6 needs a first-cell height around
2.4e-5 chords (y+ ~ 1), roughly 400x finer, and RANS practice puts the farfield
at 50-100 chords to avoid contaminating circulation.

This module builds a mesh with an explicit quad boundary-layer extrusion around
the section, geometric growth into an unstructured far field, and the marker
names SU2 expects (``Airfoil`` / ``Farfield``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..geometry.airfoil_param import design_to_airfoil_coords, to_selig_order

logger = logging.getLogger(__name__)


def first_cell_height(reynolds: float, y_plus: float = 1.0, chord: float = 1.0) -> float:
    """Estimate the wall-normal size of the first cell for a target y+.

    Uses the flat-plate turbulent skin-friction correlation
    ``Cf = 0.026 / Re^(1/7)``, which is the standard rule of thumb for sizing an
    airfoil boundary-layer mesh. At Re=1e6 and y+=1 this gives ~2.4e-5 chords.
    """

    cf = 0.026 / reynolds ** (1.0 / 7.0)
    u_tau_over_u = math.sqrt(0.5 * cf)
    return chord * y_plus / (reynolds * u_tau_over_u)


@dataclass
class BLMeshConfig:
    """Parameters controlling boundary-layer mesh generation."""

    reynolds: float = 1.0e6
    y_plus: float = 1.0
    n_layers: int = 35
    growth_ratio: float = 1.18
    farfield_radius: float = 60.0
    n_surface_points: int = 220
    te_size: float = 4.0e-4
    le_size: float = 4.0e-4
    farfield_size: float = 6.0
    wake_refine: bool = True

    def first_layer(self) -> float:
        return first_cell_height(self.reynolds, self.y_plus)

    def bl_thickness(self) -> float:
        """Total extruded thickness implied by the layer count and growth."""

        h = self.first_layer()
        if abs(self.growth_ratio - 1.0) < 1e-9:
            return h * self.n_layers
        return h * (self.growth_ratio ** self.n_layers - 1.0) / (self.growth_ratio - 1.0)


def build_bl_mesh(
    design_vec: Iterable[float],
    out_path: Path,
    cfg: BLMeshConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Generate a boundary-layer-resolved SU2 mesh for one design vector.

    Returns a dict with the mesh path and the realised mesh metrics, so callers
    can assert the resolution actually achieved rather than assuming it.
    """

    import gmsh

    cfg = cfg or BLMeshConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Selig order (TE -> upper -> LE -> lower -> TE) is what meshing tools expect.
    coords = to_selig_order(
        design_to_airfoil_coords(
            np.asarray(list(design_vec), dtype=float), n_points=cfg.n_surface_points
        )
    )
    coords = np.asarray(coords, dtype=float)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 0)
        gmsh.model.add("airfoil_bl")

        # --- airfoil outline -------------------------------------------------
        # Drop the duplicated closing point so the loop is not degenerate.
        pts = coords[:-1] if np.allclose(coords[0], coords[-1], atol=1e-9) else coords

        # Cluster points toward LE and TE via the existing cosine distribution;
        # a single spline through all of them keeps the surface smooth.
        tags = [gmsh.model.geo.addPoint(x, y, 0.0, cfg.le_size) for x, y in pts]
        curve = gmsh.model.geo.addSpline(tags + [tags[0]])
        airfoil_loop = gmsh.model.geo.addCurveLoop([curve])

        # --- far field -------------------------------------------------------
        r = cfg.farfield_radius
        cx = 0.5
        c = gmsh.model.geo.addPoint(cx, 0.0, 0.0, cfg.farfield_size)
        f0 = gmsh.model.geo.addPoint(cx + r, 0.0, 0.0, cfg.farfield_size)
        f1 = gmsh.model.geo.addPoint(cx, r, 0.0, cfg.farfield_size)
        f2 = gmsh.model.geo.addPoint(cx - r, 0.0, 0.0, cfg.farfield_size)
        f3 = gmsh.model.geo.addPoint(cx, -r, 0.0, cfg.farfield_size)
        arcs = [
            gmsh.model.geo.addCircleArc(f0, c, f1),
            gmsh.model.geo.addCircleArc(f1, c, f2),
            gmsh.model.geo.addCircleArc(f2, c, f3),
            gmsh.model.geo.addCircleArc(f3, c, f0),
        ]
        outer_loop = gmsh.model.geo.addCurveLoop(arcs)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil_loop])
        gmsh.model.geo.synchronize()

        # --- boundary layer extrusion ---------------------------------------
        bl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(bl, "CurvesList", [curve])
        gmsh.model.mesh.field.setNumber(bl, "Size", cfg.first_layer())
        gmsh.model.mesh.field.setNumber(bl, "Ratio", cfg.growth_ratio)
        gmsh.model.mesh.field.setNumber(bl, "Thickness", cfg.bl_thickness())
        gmsh.model.mesh.field.setNumber(bl, "Quads", 1)
        # Fan the layers around the sharp trailing edge so the extrusion closes.
        gmsh.model.mesh.field.setNumbers(bl, "FanPointsList", [tags[0]])
        gmsh.model.mesh.field.setAsBoundaryLayer(bl)

        # --- wake / near-field sizing ---------------------------------------
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", [curve])
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 400)

        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", cfg.te_size * 6)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", cfg.farfield_size)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.02)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", cfg.farfield_radius * 0.5)

        fields = [thr]
        if cfg.wake_refine:
            wake = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(wake, "VIn", cfg.te_size * 20)
            gmsh.model.mesh.field.setNumber(wake, "VOut", cfg.farfield_size)
            gmsh.model.mesh.field.setNumber(wake, "XMin", 0.9)
            gmsh.model.mesh.field.setNumber(wake, "XMax", 6.0)
            gmsh.model.mesh.field.setNumber(wake, "YMin", -0.6)
            gmsh.model.mesh.field.setNumber(wake, "YMax", 0.6)
            gmsh.model.mesh.field.setNumber(wake, "Thickness", 1.0)
            fields.append(wake)

        bg = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(bg, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

        # --- SU2 markers ------------------------------------------------------
        gmsh.model.addPhysicalGroup(1, [curve], name="Airfoil")
        gmsh.model.addPhysicalGroup(1, arcs, name="Farfield")
        gmsh.model.addPhysicalGroup(2, [surface], name="Fluid")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_path))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(node_coords).reshape(-1, 3)[:, :2]
        etypes, etags, _ = gmsh.model.mesh.getElements(dim=2)
        n_elem = int(sum(len(t) for t in etags))

        metrics = {
            "mesh_path": out_path,
            "n_nodes": int(len(node_tags)),
            "n_elements": n_elem,
            "first_layer": cfg.first_layer(),
            "bl_thickness": cfg.bl_thickness(),
            "farfield_radius": cfg.farfield_radius,
            "wall_normal_min": _min_wall_normal_spacing(xy, coords),
        }
        return metrics
    finally:
        gmsh.finalize()


def _min_wall_normal_spacing(nodes: np.ndarray, surface: np.ndarray) -> float:
    """Smallest non-zero distance from a mesh node to the airfoil surface."""

    from scipy.spatial import cKDTree

    tree = cKDTree(surface)
    d, _ = tree.query(nodes, k=1)
    nz = d[d > 1e-12]
    return float(nz.min()) if len(nz) else float("nan")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("meshes/naca0012_bl.su2"))
    p.add_argument("--reynolds", type=float, default=1e6)
    p.add_argument("--farfield", type=float, default=60.0)
    p.add_argument("--layers", type=int, default=35)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    cfg = BLMeshConfig(reynolds=a.reynolds, farfield_radius=a.farfield, n_layers=a.layers)
    m = build_bl_mesh(np.zeros(10), a.out, cfg, verbose=a.verbose)
    print("=== BL mesh generated ===")
    for k, v in m.items():
        print(f"  {k}: {v:.3e}" if isinstance(v, float) else f"  {k}: {v}")

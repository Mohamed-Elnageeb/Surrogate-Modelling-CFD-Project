from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


from .postprocess import extract_metrics
from ..geometry.airfoil_param import design_to_airfoil_coords


logger = logging.getLogger(__name__)


@dataclass
class Su2RunConfig:
    """Configuration for running a SU2 CFD case from Python."""

    workdir: Path
    base_config_name: str = "config.cfg"
    su2_executable: str = "SU2_CFD"
    history_file_name: str = "history.csv"


def create_modified_config(base_cfg: Path, out_cfg: Path, param_overrides: dict[str, float | int | str]) -> None:
    """
    Read a SU2 config file, apply overrides to specific SU2 keywords,
    and write the modified config to out_cfg.

    Rules:
    - For every key in param_overrides, the key must match a line in the config that
      begins with "<KEY>=" exactly.
    - Replace the entire RHS with the new value.
    - Preserve all other lines unchanged.
    - If a keyword is missing in the file, raise a KeyError.
    """

    lines = base_cfg.read_text().splitlines(keepends=True)
    overrides_applied: set[str] = set()

    for idx, line in enumerate(lines):
        for key, value in param_overrides.items():
            if line.startswith(f"{key}="):
                lines[idx] = f"{key}= {value}\n"
                overrides_applied.add(key)

    missing_keys = set(param_overrides) - overrides_applied
    if missing_keys:
        raise KeyError(f"Missing keys in config: {sorted(missing_keys)}")

    out_cfg.write_text("".join(lines))


def parse_history_file(path: Path) -> dict:
    """
    Parse a SU2 history file (.csv or .dat) and return the last data row.

    The parser tolerates trailing delimiters and both comma- and whitespace-
    separated tables, which occur across SU2 versions. Comment/blank lines are
    skipped. If the file does not exist or no data rows are present, an empty
    dict is returned.
    """

    if not path or not path.exists():
        return {}

    with path.open("r", newline="") as csvfile:
        data_lines = [line for line in csvfile if line.strip() and not line.lstrip().startswith(("%", "#"))]

    if not data_lines:
        return {}

    def split_line(line: str) -> list[str]:
        stripped = line.strip()
        if "," in stripped:
            return [tok for tok in stripped.split(",") if tok != ""]
        return [tok for tok in re.split(r"\s+", stripped) if tok != ""]

    header = split_line(data_lines[0])
    if not header:
        return {}

    last_row: list[str] | None = None
    for line in data_lines[1:]:
        row = split_line(line)
        if not row:
            continue
        # Ignore extra trailing tokens and pad missing ones with empty strings
        row = (row + [""] * len(header))[: len(header)]
        last_row = row

    if not last_row:
        return {}

    def convert(value: str) -> float | int | str:
        try:
            num = float(value)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            return value

    return {key: convert(val) for key, val in zip(header, last_row)}


def run_su2_case(cfg: Su2RunConfig, param_overrides: dict[str, float | int | str] | None = None, timeout: int = 3600) -> dict:
    """
    Run a SU2 CFD simulation using the config file in cfg.workdir.

    Steps:
    - Determine the working directory: cfg.workdir.
    - If param_overrides is provided:
        * Create a new config file in workdir named "config_override.cfg".
        * Call create_modified_config(...) to apply changes.
        * Run SU2_CFD on that overridden file.
      Else:
        * Run SU2_CFD on cfg.base_config_name.
    - Use subprocess.run([...], cwd=cfg.workdir, capture_output=True, text=True).
    - If returncode != 0, raise a RuntimeError with useful stderr information.
    - After the run, detect the history file:
        * If cfg.history_file_name exists, use it.
        * Otherwise, find the newest file starting with "history".
    - Parse the history file:
        * Skip comment lines (% or #).
        * Assume first non-comment line is header.
        * Read last row.
        * Convert numeric fields where possible.
    - Return a dict containing:
        {
          "config_path": Path to the config used,
          "stdout": command stdout,
          "stderr": command stderr,
          "history_path": path to history file or None,
          "history_data": last-row dict or {},
        }
    """

    workdir = cfg.workdir
    base_cfg = workdir / cfg.base_config_name

    if param_overrides:
        override_cfg = workdir / "config_override.cfg"
        create_modified_config(base_cfg, override_cfg, param_overrides)
        config_to_run = override_cfg
    else:
        config_to_run = base_cfg

    cmd = [cfg.su2_executable, str(config_to_run)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(f"SU2_CFD failed with code {proc.returncode}: {proc.stderr}")

    history_path: Path | None = workdir / cfg.history_file_name
    if not history_path.exists():
        history_candidates = sorted(workdir.glob("history*"), key=lambda p: p.stat().st_mtime, reverse=True)
        history_path = history_candidates[0] if history_candidates else None

    history_data: dict = {}
    if history_path:
        history_data = parse_history_file(history_path)

    return {
        "config_path": config_to_run,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "history_path": history_path,
        "history_data": history_data,
    }


def main() -> None:
    base_dir = _default_case_dir()
    cfg = Su2RunConfig(workdir=base_dir)
    result = run_su2_case(
        cfg,
        param_overrides={
            "MACH_NUMBER": 0.15,
            "AOA": 10.0,
        },
    )
    print("History file:", result["history_path"])
    print("Last row:", result["history_data"])


def _extract_first(history: dict, *keys: str):
    for key in keys:
        if key in history:
            value = history[key]
            if value is None or value == "":
                continue
            return value
    return None


def _sanitize_positive(value: float | int | None) -> float | None:
    """Return a non-negative aerodynamic metric when the sign is unreliable."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return abs(numeric)


def _apply_su2_sign_convention(
    cl: float | int | None, cd: float | int | None, cfg_path: Path | None
) -> tuple[float | int | None, float | int | None]:
    """Align lift/drag signs with SU2's aerodynamic convention.

    Per the SU2 user guide, a positive ``AOA`` rotates the freestream vector
    clockwise around the z-axis (for 2D airfoils in the *xy*-plane). When users
    supply ``AOA`` using the opposite "nose-up" convention, the reported
    coefficients can appear flipped (negative lift and drag). To present
    physically meaningful magnitudes, we detect this mismatch and flip the
    signs accordingly.
    """

    if cl is None and cd is None:
        return cl, cd

    try:
        aoa = float(_config_value(cfg_path, "AOA", "nan")) if cfg_path else float("nan")
    except (TypeError, ValueError):
        aoa = float("nan")

    def _abs_or_none(val: float | int | None) -> float | int | None:
        return None if val is None else abs(val)

    if np.isnan(aoa):
        return cl, _abs_or_none(cd)

    # If the angle-of-attack and lift signs disagree, flip both coefficients to
    # match the SU2 documentation (positive lift for positive AOA).
    if cl is not None and float(cl) * aoa < 0:
        cl = -float(cl)
        cd = _abs_or_none(cd)
    else:
        cd = _abs_or_none(cd)

    return cl, cd


def _config_value(cfg_path: Path, key: str, default: str | None = None) -> str | None:
    """Return the value assigned to ``key`` in a SU2 config, if present."""

    if not cfg_path.exists():
        return default

    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*(.+)$")
    for line in cfg_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()
    return default


def _latest_output(
    workdir: Path, stem: str, preferred_exts: tuple[str, ...] | None = None
) -> Path | None:
    """Find the newest output file matching a given stem in ``workdir``.

    When multiple file formats are present (e.g., CSV and VTU exports), the search
    prefers text-friendly formats first so that downstream readers avoid brittle
    binary parsers.
    """

    preferred_exts = preferred_exts or (".csv", ".dat", ".su2", ".txt", ".vtu", ".vtk")
    candidates = list(workdir.glob(f"{stem}*"))
    candidates = [c for c in candidates if c.suffix in preferred_exts]
    if not candidates:
        return None

    def _priority(path: Path) -> tuple[int, float]:
        try:
            rank = preferred_exts.index(path.suffix)
        except ValueError:
            rank = len(preferred_exts)
        return (rank, -path.stat().st_mtime)

    return sorted(candidates, key=_priority)[0]


def _default_case_dir() -> Path:
    """Return the canonical SU2 case directory bundled with the repository."""

    return Path(__file__).resolve().parents[2] / "TestCases" / "airfoil_naca0012_opt"


def _prepare_isolated_case(base_case_dir: Path, design_id: str) -> Path:
    """Return a fresh, design-specific working directory for a CFD run.

    SU2 writes outputs such as ``history.csv`` directly inside the working
    directory. When multiple designs reuse the same directory (especially in
    parallel), the history file accumulates rows from all runs and downstream
    readers end up extracting identical metrics regardless of the sampled
    design. To avoid the cross-run contamination, we copy the canonical case
    into a dedicated subdirectory and clear any pre-existing history files
    before launching SU2.
    """

    runs_root = Path(__file__).resolve().parents[1] / "data" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Place the copied case in a predictable location so callers can inspect
    # the raw SU2 outputs if needed. Recreate the folder if it already exists
    # to guarantee a clean slate for every design.
    isolated_dir = runs_root / design_id
    if isolated_dir.exists():
        shutil.rmtree(isolated_dir)

    shutil.copytree(base_case_dir, isolated_dir)

    # Ensure no stale history files leak into the next run's metrics.
    for history_file in isolated_dir.glob("history*"):
        try:
            history_file.unlink()
        except OSError:
            pass

    return isolated_dir


def run_cfd(
    design_id: str,
    design_vec: Iterable[float],
    workdir: Path | None = None,
    su2_executable: str | None = None,
    param_overrides: dict[str, float | int | str] | None = None,
    regenerate_mesh: bool = True,
) -> dict:
    """
    Lightweight convenience wrapper for running a single SU2 case.

    The helper mirrors the legacy ``run_cfd`` interface that higher-level
    scripts import. It delegates to :func:`run_su2_case` using the canonical
    ``TestCases/airfoil_naca0012_opt`` directory (unless ``workdir`` is
    provided), and returns a dictionary containing lift/drag metrics plus a
    success flag.

    To support varying flow conditions across runs (e.g. Mach/AoA sweeps),
    callers can provide ``param_overrides`` which are forwarded directly to
    :func:`create_modified_config`. This ensures each run actually reflects the
    requested setup instead of silently reusing the baseline configuration.

    Args:
        design_id: Identifier for the design being evaluated (used for logging).
        design_vec: Ten-parameter design vector. It is accepted for API
            compatibility but not manipulated inside this helper; callers can
            persist it alongside the returned metrics if needed.
        workdir: Optional path to a prepared SU2 case directory.
        su2_executable: Optional override for the SU2 binary name/path.
        param_overrides: Optional mapping of SU2 config keys to override for
            this run (e.g., {"MACH_NUMBER": 0.2, "AOA": 5}).
        regenerate_mesh: When True (default), rebuild the SU2 mesh from the
            supplied design vector so each run uses its own geometry instead of
            sharing the baseline case mesh. Mesh and geometry preview PNGs are
            stored alongside the run directory.
    """

    base_case_dir = workdir or _default_case_dir()
    case_dir = _prepare_isolated_case(base_case_dir, design_id) if workdir is None else base_case_dir

    airfoil_plot: Path | None = None
    mesh_plot: Path | None = None
    mesh_path: Path | None = case_dir / "mesh.su2"

    if regenerate_mesh:
        regen = _regenerate_mesh_for_design(design_id, design_vec, case_dir, mesh_basename="mesh.su2")
        airfoil_plot = regen.get("airfoil_plot")
        mesh_plot = regen.get("mesh_plot")
        mesh_path = regen.get("mesh_path", mesh_path)
        if not regen.get("success", False):
            return {
                "design_id": design_id,
                "design_vec": list(design_vec),
                "Cl": None,
                "Cd": None,
                "residual": None,
                "success": False,
                "error": regen.get("error", "Failed to regenerate mesh"),
                "airfoil_plot": airfoil_plot,
                "mesh_plot": mesh_plot,
                "mesh_path": mesh_path,
            }
    cfg = Su2RunConfig(workdir=case_dir)
    if su2_executable:
        cfg.su2_executable = su2_executable

    base_cfg = case_dir / cfg.base_config_name
    volume_stem = _config_value(base_cfg, "VOLUME_FILENAME", "flow_fields")
    surface_stem = _config_value(base_cfg, "SURFACE_FILENAME", "surface_airfoil")

    missing_reqs = []
    if not base_cfg.exists():
        missing_reqs.append(f"Missing SU2 base config: {base_cfg}")

    mesh_file = mesh_path or (case_dir / "mesh.su2")
    if not mesh_file.exists():
        missing_reqs.append(f"Missing SU2 mesh file: {mesh_file}")

    su2_path = shutil.which(cfg.su2_executable)
    if not su2_path:
        missing_reqs.append(
            f"SU2 executable '{cfg.su2_executable}' not found in PATH."
        )

    if missing_reqs:
        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": None,
            "Cd": None,
            "residual": None,
            "success": False,
            "error": "; ".join(missing_reqs),
        }

    try:
        result = run_su2_case(cfg, param_overrides=param_overrides)
        history = result.get("history_data") or {}
        metrics = extract_metrics(result.get("history_path")) if result.get("history_path") else {}

        cl = metrics.get("Cl") if metrics else None
        cd = metrics.get("Cd") if metrics else None
        residual = metrics.get("residual") if metrics else None

        if cl is None:
            cl = _extract_first(history, "CL", "CLtot", "cl", "Cl")
        if cd is None:
            cd = _extract_first(history, "CD", "CDtot", "cd", "Cd")
        if residual is None:
            residual = _extract_first(history, "RMS_RES", "RMS_DENSITY", "residual")

        cl, cd = _apply_su2_sign_convention(cl, cd, result.get("config_path"))
        cl = _sanitize_positive(cl)
        cd = _sanitize_positive(cd)

        volume_output = _latest_output(case_dir, volume_stem or "flow_fields")
        surface_output = _latest_output(case_dir, surface_stem or "surface_airfoil")

        validation_reasons: list[str] = []

        for name, value in {"Cl": cl, "Cd": cd}.items():
            if value is None:
                validation_reasons.append(f"Missing {name} metric")
            else:
                try:
                    numeric = float(value)
                    if np.isnan(numeric):
                        validation_reasons.append(f"{name} is NaN")
                    elif numeric < 0:
                        validation_reasons.append(f"{name} is negative ({numeric})")
                    elif numeric > 10:
                        validation_reasons.append(f"{name} exceeds limit ({numeric})")
                except (TypeError, ValueError):
                    validation_reasons.append(f"{name} is not numeric")

        try:
            residual_value = float(residual) if residual is not None else None
        except (TypeError, ValueError):
            residual_value = None
        if residual_value is None:
            validation_reasons.append("Residual is missing or non-numeric")
        elif np.isnan(residual_value):
            validation_reasons.append("Residual is NaN")
        elif residual_value > 1e-2:
            validation_reasons.append(f"Residual above threshold ({residual_value})")

        invalid_metrics = bool(validation_reasons)
        success = not invalid_metrics
        error_msg = "; ".join(validation_reasons) if validation_reasons else None

        if invalid_metrics:
            logger.warning("Design %s failed validation: %s", design_id, error_msg)

        def _persist_debug_artifacts():
            debug_dir = case_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for artifact in (airfoil_plot, mesh_plot, mesh_path):
                if artifact and Path(artifact).exists():
                    try:
                        shutil.copy(Path(artifact), debug_dir / Path(artifact).name)
                    except OSError:
                        pass

        if not success:
            _persist_debug_artifacts()

        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": cl,
            "Cd": cd,
            "residual": residual,
            "success": success,
            "invalid_metrics": invalid_metrics,
            "error": error_msg,
            "history_data": history,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "config_path": result.get("config_path"),
            "history_path": result.get("history_path"),
            "volume_output": volume_output,
            "surface_output": surface_output,
            "airfoil_plot": airfoil_plot,
            "mesh_plot": mesh_plot,
            "mesh_path": mesh_file,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "design_id": design_id,
            "design_vec": list(design_vec),
            "Cl": None,
            "Cd": None,
            "residual": None,
            "success": False,
            "invalid_metrics": False,
            "error": str(exc),
            "airfoil_plot": airfoil_plot,
            "mesh_plot": mesh_plot,
            "mesh_path": mesh_file,
        }


def _plot_airfoil_geometry(coords: np.ndarray, outfile: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(coords[:, 0], coords[:, 1], color="navy", linewidth=1.2, label="airfoil")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def _plot_mesh_outline(gmsh_module, outfile: Path, airfoil_coords: np.ndarray | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_tags, node_coords, _ = gmsh_module.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise RuntimeError("No mesh nodes generated; cannot plot mesh")

    xy = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
    tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}

    elem_types, _, elem_nodes = gmsh_module.model.mesh.getElements(dim=2)
    triangles: list[np.ndarray] = []
    for etype, nodes in zip(elem_types, elem_nodes):
        if len(nodes) == 0:
            continue
        properties = gmsh_module.model.mesh.getElementProperties(etype)
        dim = properties[1]
        num_nodes = properties[3]
        if dim != 2 or num_nodes < 3:
            continue
        conn = np.asarray(nodes, dtype=int).reshape(-1, num_nodes)
        tri_conn = conn[:, :3]
        tri_indices = np.vectorize(tag_to_idx.get)(tri_conn)
        triangles.append(tri_indices)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if triangles:
        tri = np.vstack(triangles)
        ax.triplot(xy[:, 0], xy[:, 1], tri, linewidth=0.3, color="0.25")
    ax.scatter(xy[:, 0], xy[:, 1], s=1, color="0.55", alpha=0.6)
    if airfoil_coords is not None:
        ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], color="crimson", linewidth=1.0, label="Airfoil")
        ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Generated mesh (triangulation)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def _regenerate_mesh_for_design(
    design_id: str, design_vec: Iterable[float], case_dir: Path, mesh_basename: str = "mesh.su2"
) -> dict:
    """
    Build a fresh mesh and quick-look plots for a given design vector.

    The helper uses the parametric airfoil description to rebuild the mesh in
    the isolated SU2 working directory so that each design evaluation is tied
    to its own geometry rather than a shared baseline mesh. A pair of PNG
    previews are emitted alongside the mesh for easy inspection.
    """

    coords = design_to_airfoil_coords(np.asarray(list(design_vec), dtype=float))
    airfoil_plot = _plot_airfoil_geometry(coords, case_dir / f"{design_id}_airfoil.png")

    try:
        import gmsh  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "success": False,
            "error": "The Gmsh Python API is required to regenerate meshes. Install gmsh to continue.",
            "airfoil_plot": airfoil_plot,
        }

    chord = max(float(coords[:, 0].max() - coords[:, 0].min()), 1e-3)
    farfield_radius = max(20.0 * chord, 5.0)
    mesh_size_airfoil = max(chord * 0.01, 1e-4)
    mesh_size_farfield = max(farfield_radius * 0.05, mesh_size_airfoil * 5)
    mesh_path = case_dir / mesh_basename

    gmsh.initialize()
    # Silence the verbose meshing progress messages so downstream callers do
    # not get flooded with terminal updates during batch runs. The calls must
    # occur *after* initialization to avoid "Gmsh has not been initialized"
    # stderr noise that confused users and cluttered logs.
    try:  # pragma: no cover - optional convenience
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
    except Exception:
        pass
    gmsh.model.add(f"airfoil_{design_id}")

    try:
        point_map: dict[tuple[float, float], int] = {}

        def add_point(pt: tuple[float, float], size: float) -> int:
            key = (round(pt[0], 8), round(pt[1], 8))
            if key in point_map:
                return point_map[key]
            tag = gmsh.model.occ.addPoint(pt[0], pt[1], 0.0, size)
            point_map[key] = tag
            return tag

        num_surface_points = (len(coords) + 1) // 2
        upper_points = [tuple(pt) for pt in coords[:num_surface_points]]
        lower_points = [tuple(pt) for pt in coords[num_surface_points:]]

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

        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_distance, "EdgesList", [spline_upper, spline_lower])
        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "IField", field_distance)
        gmsh.model.mesh.field.setNumber(field_threshold, "LcMin", mesh_size_airfoil)
        gmsh.model.mesh.field.setNumber(field_threshold, "LcMax", mesh_size_farfield)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", farfield_radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

        gmsh.model.addPhysicalGroup(2, [surface], name="Fluid")
        gmsh.model.addPhysicalGroup(1, [spline_upper, spline_lower], name="Airfoil")
        gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="Farfield")
        gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="Symmetry")

        gmsh.model.mesh.generate(2)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(mesh_path))

        mesh_plot = _plot_mesh_outline(gmsh, case_dir / f"{design_id}_mesh.png", coords)

    finally:
        gmsh.finalize()

    return {
        "success": True,
        "mesh_path": mesh_path,
        "airfoil_plot": airfoil_plot,
        "mesh_plot": mesh_plot,
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class SnapshotConfig:
    """
    Configuration describing how to build a UNet snapshot from SU2 solution tables.

    - input_fields: names of scalar fields to be stacked into the "input" tensor.
    - target_fields: names of scalar fields to be stacked into the "target_fields" tensor.
    - grid_shape: (ny, nx) grid dimensions. The total number of points must be ny * nx.
    - cl_name / cd_name: column names in the surface-force file that contain CL and CD.
    """

    input_fields: Sequence[str]
    target_fields: Sequence[str]
    grid_shape: Tuple[int, int]
    cl_name: str = "CL"
    cd_name: str = "CD"


def _is_comment_or_empty(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith(("#", "%"))


def _looks_binary(raw: bytes) -> bool:
    sample = raw[:4096]
    if not sample:
        return False
    # Heuristic: if we encounter any NUL bytes or more than 30% non-printable
    # characters, treat the content as binary. This prevents us from spamming
    # NaN conversion logs on files such as VTU with appended binary payloads.
    if b"\x00" in sample:
        return True

    printable = bytes(c for c in sample if 32 <= c <= 126 or c in (9, 10, 13))
    return len(printable) / len(sample) < 0.7


def _strip_inline_comment(line: str) -> str:
    """Remove trailing inline comments introduced with '#' or '%'"""

    for marker in ("#", "%"):
        comment_idx = line.find(marker)
        if comment_idx != -1:
            return line[:comment_idx]
    return line


def _tokenize_line(line: str, delimiter: str) -> list[str]:
    """Split a line using a known delimiter while preserving empty tokens."""

    line = _strip_inline_comment(line)
    if delimiter == ",":
        # Preserve positional empties so we can accurately align with the header.
        return [token.strip() for token in line.split(",")]
    return line.split()


def read_su2_table(path: Path) -> Dict[str, np.ndarray]:
    """
    Read a simple SU2-style ASCII table into a dict of column name -> 1D numpy array.

    Assumptions (kept deliberately simple and generic):

    * The file is text, encoded as UTF-8.
    * Lines starting with '#' or '%' or empty/whitespace-only lines are ignored
      until we find the header.
    * The first non-comment, non-empty line is the header, with column names
      separated by commas or whitespace (e.g. 'x, y, p, u, v' OR 'x y p u v').
    * All following non-empty, non-comment lines are numeric rows with the same
      delimiter pattern (commas or whitespace).
    * Each data row must have the same number of columns as the header.

    Returns:
        dict mapping column name (str) to a 1D numpy.ndarray of shape (N,).

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError if the file has no header, inconsistent columns, or no data.
    """

    raw_bytes = path.read_bytes()
    if _looks_binary(raw_bytes):
        raise ValueError(
            f"File appears to be binary or not a plain-text SU2 table: {path}"
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    lines = text.splitlines()

    header_fields: list[str] | None = None
    delimiter: str = ","
    columns: dict[str, list[float]] | None = None

    for idx, line in enumerate(lines, start=1):
        if _is_comment_or_empty(line):
            continue

        if header_fields is None:
            stripped = line.lstrip()
            if stripped.startswith("<"):
                raise ValueError(
                    "File appears to be XML/VTK rather than an SU2 table: "
                    f"{path}"
                )
            header_fields = _tokenize_line(line, ",")
            # Use the header to decide whether the file is comma- or whitespace-separated.
            delimiter = "," if "," in line else ""
            if delimiter == "":
                delimiter = " "
                header_fields = line.split()
            if not header_fields:
                raise ValueError(f"No header found in table: {path}")
            columns = {name: [] for name in header_fields}
            continue

        tokens = _tokenize_line(line, delimiter)

        # Align row tokens with the header length. Extra tokens are ignored, and missing
        # tokens are filled with NaN so downstream consumers can decide how to handle
        # incomplete rows without the parser failing outright.
        if len(tokens) < len(header_fields):
            logger.debug(
                "Row %d in %s has %d columns but %d expected; padding with empty values",
                idx, path, len(tokens), len(header_fields),
            )
            tokens = tokens + [""] * (len(header_fields) - len(tokens))
        elif len(tokens) > len(header_fields):
            logger.debug(
                "Row %d in %s has %d columns but %d expected; truncating extra tokens",
                idx, path, len(tokens), len(header_fields),
            )
            tokens = tokens[: len(header_fields)]

        for name, tok in zip(header_fields, tokens):
            tok = tok.strip()
            try:
                value = float(tok)
            except ValueError:
                if tok != "":
                    logger.debug(
                        "Row %d column '%s' in %s is non-numeric token '%s'; converting to NaN",
                        idx, name, path, tok,
                    )
                value = np.nan if tok == "" else np.nan
            columns[name].append(value)

    if header_fields is None:
        raise ValueError(f"No header found in table: {path}")
    if columns is None or not any(columns.values()):
        raise ValueError(f"No data found in table: {path}")

    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _load_vtu_table(path: Path) -> Dict[str, np.ndarray]:
    """Parse a VTU file into a flat dict of point-data arrays.

    MeshIO is used at runtime (imported lazily) so that environments which do not
    need VTU support do not pay the import cost.
    """

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - exercised in integration
        raise ImportError("meshio is required to read VTU files") from exc

    table: Dict[str, np.ndarray] = {}

    def _convert_mesh(mesh_obj) -> Dict[str, np.ndarray]:
        converted: Dict[str, np.ndarray] = {}

        # Expose nodal coordinates so downstream code can interpolate the
        # unstructured solution onto a regular grid. SU2 volume meshes are not
        # Cartesian, so the raw node ordering cannot simply be reshaped.
        points = np.asarray(getattr(mesh_obj, "points", []), dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2:
            converted["x"] = points[:, 0]
            converted["y"] = points[:, 1]
            if points.shape[1] >= 3:
                converted["z"] = points[:, 2]

        for name, array in mesh_obj.point_data.items():
            data = np.asarray(array)
            if data.ndim == 1:
                converted[name] = data.astype(float)
                continue

            if data.ndim == 2:
                for idx in range(data.shape[1]):
                    converted[f"{name}_{idx}"] = data[:, idx].astype(float)

                lower = name.lower()
                if lower in {"velocity", "momentum"}:
                    component_names = ("u", "v", "w")
                    for idx, comp in enumerate(component_names):
                        if idx < data.shape[1]:
                            converted[comp] = data[:, idx].astype(float)
                continue

            # Skip higher dimensional arrays; they are unlikely to be scalar fields.
        return converted

    try:
        mesh = meshio.read(path)
        table = _convert_mesh(mesh)
    except Exception as exc:  # pragma: no cover - defensive fallback for truncated VTU
        msg = str(exc).lower()
        if "buffer size" not in msg and "not enough data" not in msg:
            raise

        logger.warning(
            "meshio could not read VTU %s (%s); attempting pyvista fallback", path, exc
        )
        try:
            import pyvista as pv

            dataset = pv.read(path)
            table = _convert_mesh(dataset)
        except Exception as pv_exc:  # pragma: no cover - optional dependency
            raise ValueError(
                f"Failed to read VTU {path} with meshio and pyvista: {exc}; {pv_exc}"
            ) from exc

    return table


def _load_volume_table(volume_solution: Path) -> Dict[str, np.ndarray]:
    if volume_solution.suffix.lower() == ".vtu":
        return _load_vtu_table(volume_solution)
    return read_su2_table(volume_solution)


def _load_history_forces(history_path: Path, cl_name: str, cd_name: str) -> Tuple[float, float]:
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError(f"History file is empty: {history_path}")
    for col in (cl_name, cd_name):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in history file {history_path}")

    last_row = df.iloc[-1]
    return float(last_row[cl_name]), float(last_row[cd_name])


def _load_force_values(
    surface_forces: Path | None,
    history_path: Path,
    cl_name: str,
    cd_name: str,
) -> Tuple[float, float]:
    if surface_forces is not None:
        surf_table = read_su2_table(surface_forces)
        return extract_forces(surf_table, cl_name, cd_name)

    return _load_history_forces(history_path, cl_name, cd_name)


_COORD_KEY_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("x", "y"),
    ("X", "Y"),
    ("Points_0", "Points_1"),
    ("x_coord", "y_coord"),
    ("Points:0", "Points:1"),
)


def _find_coordinates(table: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray] | None:
    """Return nodal (x, y) coordinate arrays from a table, if present."""

    for x_key, y_key in _COORD_KEY_CANDIDATES:
        if x_key in table and y_key in table:
            x = np.asarray(table[x_key], dtype=float)
            y = np.asarray(table[y_key], dtype=float)
            if x.shape == y.shape and x.ndim == 1:
                return x, y
    return None


def _interpolate_to_grid(
    columns: Sequence[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Interpolate scattered nodal fields onto a regular (H, W) grid.

    SU2 volume solutions live on an unstructured mesh, so the flat node ordering
    is not a Cartesian grid and must not be reshaped directly. We sample each
    field onto a uniform grid spanning the mesh bounding box using linear
    interpolation, then backfill any points outside the convex hull with a
    nearest-neighbour value so the resulting image has no holes.
    """

    from scipy.interpolate import griddata

    h, w = grid_shape
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        raise ValueError("Need at least 3 valid nodes to interpolate onto a grid")

    x = x[finite]
    y = y[finite]
    pts = np.column_stack([x, y])

    x_lin = np.linspace(float(x.min()), float(x.max()), w)
    y_lin = np.linspace(float(y.min()), float(y.max()), h)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)  # both (H, W), row index = y

    channels = []
    for col in columns:
        values = col[finite]
        gridded = griddata(pts, values, (grid_x, grid_y), method="linear")
        holes = ~np.isfinite(gridded)
        if holes.any():
            filler = griddata(pts, values, (grid_x, grid_y), method="nearest")
            gridded[holes] = filler[holes]
        channels.append(gridded)
    return np.stack(channels, axis=0)


def build_field_tensor(
    table: Dict[str, np.ndarray],
    field_names: Sequence[str],
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """
    From a table of column arrays and a list of field names, build a 3D tensor.

    When the table carries nodal coordinates (``x``/``y`` or an equivalent pair),
    the scattered fields are interpolated onto a uniform ``grid_shape`` grid.
    This is required for SU2 volume solutions, whose nodes are unstructured and
    therefore cannot be reshaped into an image directly. If no coordinates are
    available, the function falls back to a plain row-major reshape (used for
    already-structured inputs), which requires ``len(column) == H * W``.

    Args:
        table: dict from column name to 1D array.
        field_names: ordered list of columns to stack.
        grid_shape: (H, W) of the output grid.

    Returns:
        A numpy array of shape (len(field_names), H, W).

    Raises:
        KeyError if a requested field is missing.
        ValueError if column lengths mismatch, or (reshape fallback) the length
            does not match H * W.
    """

    if not field_names:
        return np.empty((0,) + tuple(grid_shape), dtype=float)

    columns = []
    for name in field_names:
        if name not in table:
            raise KeyError(f"Missing field '{name}' in table")
        columns.append(np.asarray(table[name], dtype=float).copy())

    lengths = {col.shape[0] for col in columns}
    if len(lengths) != 1:
        raise ValueError("Selected columns have mismatched lengths")

    coords = _find_coordinates(table)
    if coords is not None:
        x, y = coords
        (length,) = lengths
        if x.shape[0] != length:
            raise ValueError(
                f"Coordinate arrays have {x.shape[0]} points but fields provide {length}"
            )
        return _interpolate_to_grid(columns, x, y, grid_shape)

    # No coordinates: fall back to a direct reshape for structured inputs.
    h, w = grid_shape
    expected_size = h * w
    (length,) = lengths
    if length != expected_size:
        raise ValueError(
            f"Grid shape {grid_shape} expects {expected_size} points but table provides "
            f"{length}, and no nodal coordinates were found to interpolate from"
        )

    try:
        reshaped = [col.reshape(h, w) for col in columns]
    except ValueError as exc:
        raise ValueError(
            f"Failed to reshape fields into grid {grid_shape}; verify the SU2 export resolution"
        ) from exc
    return np.stack(reshaped, axis=0)


def extract_forces(
    table: Dict[str, np.ndarray],
    cl_name: str,
    cd_name: str,
) -> Tuple[float, float]:
    """
    Extract CL and CD scalars from a surface-force table.

    Rules:
    * Look up cl_name and cd_name in the table; raise KeyError if missing.
    * Each column is a 1D array; we take the LAST entry in each column as the
      representative value (common in SU2 histories).
    * Return them as Python floats.
    """

    if cl_name not in table:
        raise KeyError(f"Missing column '{cl_name}' in surface table")
    if cd_name not in table:
        raise KeyError(f"Missing column '{cd_name}' in surface table")

    cl = float(table[cl_name][-1])
    cd = float(table[cd_name][-1])
    return cl, cd


def su2_to_unet_snapshot(
    volume_solution: Path,
    surface_forces: Path | None,
    out_path: Path,
    cfg: SnapshotConfig,
    history_path: Path | None = None,
) -> None:
    """
    Convert SU2-style volume and surface solution files into a UNet snapshot .npz file.

    This reads:
    * 'volume_solution' as a SU2-style table with the flowfield variables or a VTU
      file containing point data.
    * 'surface_forces' as a SU2-style table with CL / CD columns when provided, or
      falls back to reading the last row of a 'history.csv' file.

    It then:

    * Builds an 'input' tensor by stacking cfg.input_fields from the volume solution,
      reshaped to cfg.grid_shape, giving shape (C_in, H, W).
    * Builds a 'target_fields' tensor by stacking cfg.target_fields from the same
      volume solution, reshaped to cfg.grid_shape, giving shape (C_out, H, W).
    * Extracts scalar CL and CD from the surface_forces table using cfg.cl_name / cfg.cd_name.
    * Saves everything into 'out_path' with keys:
        - 'input'
        - 'target_fields'
        - 'cl'
        - 'cd'
    """

    vol_path = Path(volume_solution)
    surf_path = Path(surface_forces) if surface_forces is not None else None

    if not vol_path.exists():
        raise FileNotFoundError(
            "Volume solution not found: "
            f"{vol_path}. Ensure you replaced 'path/to/…' with your SU2 output file."
        )
    if surf_path is not None and not surf_path.exists():
        raise FileNotFoundError(
            "Surface force file not found: "
            f"{surf_path}. Ensure you replaced 'path/to/…' with your SU2 surface file."
        )

    try:
        vol_table = _load_volume_table(volume_solution)
    except Exception as exc:  # pragma: no cover - defensive logging for runtime exports
        raise ValueError(f"Failed to read volume solution {volume_solution}: {exc}") from exc

    history_csv = history_path if history_path is not None else volume_solution.parent / "history.csv"
    cl, cd = _load_force_values(surface_forces, history_csv, cfg.cl_name, cfg.cd_name)

    inputs = build_field_tensor(vol_table, cfg.input_fields, cfg.grid_shape)
    target_fields = build_field_tensor(vol_table, cfg.target_fields, cfg.grid_shape)

    np.savez(
        out_path,
        input=inputs,
        target_fields=target_fields,
        cl=np.array(cl, dtype=np.float32),
        cd=np.array(cd, dtype=np.float32),
    )


def save_snapshot_from_arrays(
    out_path: Path,
    input_array: np.ndarray,
    target_fields_array: np.ndarray,
    cl: float,
    cd: float,
) -> None:
    """
    Convenience wrapper to save a snapshot directly from precomputed arrays.

    This just enforces a consistent .npz structure used by CFDSnapshotDataset.
    """

    if input_array.ndim != 3:
        raise ValueError("input_array must be 3D (C, H, W)")
    if target_fields_array.ndim != 3:
        raise ValueError("target_fields_array must be 3D (C, H, W)")

    np.savez(
        out_path,
        input=input_array,
        target_fields=target_fields_array,
        cl=np.array(cl, dtype=np.float32),
        cd=np.array(cd, dtype=np.float32),
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


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
            print(
                f"Row {idx} in {path} has {len(tokens)} columns but {len(header_fields)} expected; padding with empty values"
            )
            tokens = tokens + [""] * (len(header_fields) - len(tokens))
        elif len(tokens) > len(header_fields):
            print(
                f"Row {idx} in {path} has {len(tokens)} columns but {len(header_fields)} expected; truncating extra tokens"
            )
            tokens = tokens[: len(header_fields)]

        for name, tok in zip(header_fields, tokens):
            tok = tok.strip()
            try:
                value = float(tok)
            except ValueError:
                if tok != "":
                    print(
                        f"Row {idx} column '{name}' in {path} is non-numeric token '{tok}'; converting to NaN"
                    )
                value = np.nan if tok == "" else np.nan
            columns[name].append(value)

    if header_fields is None:
        print(f"No header found in table: {path}")
        raise ValueError(f"No header found in table: {path}")
    if columns is None or not any(columns.values()):
        print(f"No data found in table: {path}")
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

    mesh = meshio.read(path)
    table: Dict[str, np.ndarray] = {}

    for name, array in mesh.point_data.items():
        data = np.asarray(array)
        if data.ndim == 1:
            table[name] = data.astype(float)
            continue

        if data.ndim == 2:
            # Expose each component separately, both generically (name_0, name_1)
            # and with common aliases for velocity-like vectors.
            for idx in range(data.shape[1]):
                table[f"{name}_{idx}"] = data[:, idx].astype(float)

            lower = name.lower()
            if lower in {"velocity", "momentum"}:
                component_names = ("u", "v", "w")
                for idx, comp in enumerate(component_names):
                    if idx < data.shape[1]:
                        table[comp] = data[:, idx].astype(float)
            continue

        # Skip higher dimensional arrays; they are unlikely to be scalar fields.

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


def build_field_tensor(
    table: Dict[str, np.ndarray],
    field_names: Sequence[str],
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """
    From a table of column arrays and a list of field names, build a 3D tensor.

    Args:
        table: dict from column name to 1D array of length N = H * W.
        field_names: ordered list of columns to stack.
        grid_shape: (H, W). The product must match the length of each selected column.

    Returns:
        A numpy array of shape (len(field_names), H, W), where each channel corresponds
        to one column in 'field_names', reshaped in row-major order (C, H, W).

    Raises:
        KeyError if a requested field is missing.
        ValueError if column lengths mismatch or do not match H * W.
    """

    if not field_names:
        return np.empty((0,) + tuple(grid_shape), dtype=float)

    columns = []
    for name in field_names:
        if name not in table:
            raise KeyError(f"Missing field '{name}' in table")
        columns.append(table[name])

    lengths = {col.shape[0] for col in columns}
    if len(lengths) != 1:
        raise ValueError("Selected columns have mismatched lengths")

    h, w = grid_shape
    expected_size = h * w
    (length,) = lengths
    if length != expected_size:
        raise ValueError("Grid shape does not match column length")

    reshaped = [col.reshape(h, w) for col in columns]
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

    vol_table = _load_volume_table(volume_solution)

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

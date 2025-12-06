from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


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


def _strip_inline_comment(line: str) -> str:
    """Remove trailing inline comments introduced with '#' or '%'."""

    for marker in ("#", "%"):
        comment_idx = line.find(marker)
        if comment_idx != -1:
            return line[:comment_idx]
    return line


def _split_line(line: str) -> list[str]:
    line = _strip_inline_comment(line)
    tokens = [token.strip() for token in line.split(",")] if "," in line else line.split()
    # Some SU2 tables include trailing delimiters that yield empty tokens; drop them so
    # row-length validation does not incorrectly fail.
    return [tok for tok in tokens if tok]


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

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    lines = text.splitlines()

    header_fields: list[str] | None = None
    data_rows: list[list[float]] = []

    for line in lines:
        if _is_comment_or_empty(line):
            continue

        if header_fields is None:
            header_fields = _split_line(line)
            if not header_fields:
                raise ValueError(f"No header found in table: {path}")
            continue

        tokens = _split_line(line)

        numeric_tokens: list[str] = []
        first_non_numeric_found = False
        for tok in tokens:
            try:
                float(tok)
            except ValueError:
                first_non_numeric_found = True
                break
            numeric_tokens.append(tok)

        if len(numeric_tokens) < len(header_fields):
            raise ValueError("Row column count does not match header")
        if len(numeric_tokens) > len(header_fields) and not first_non_numeric_found:
            raise ValueError("Row column count does not match header")

        row_tokens = numeric_tokens[: len(header_fields)]
        try:
            row = [float(tok) for tok in row_tokens]
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("Non-numeric value encountered") from exc
        data_rows.append(row)

    if header_fields is None:
        raise ValueError(f"No header found in table: {path}")
    if not data_rows:
        raise ValueError(f"No data found in table: {path}")

    data = np.asarray(data_rows, dtype=float)
    table: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(header_fields):
        table[name] = data[:, idx]
    return table


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
    surface_forces: Path,
    out_path: Path,
    cfg: SnapshotConfig,
) -> None:
    """
    Convert SU2-style volume and surface solution files into a UNet snapshot .npz file.

    This reads:
    * 'volume_solution' as a SU2-style table with the flowfield variables.
    * 'surface_forces' as a SU2-style table with CL / CD columns.

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

    vol_table = read_su2_table(volume_solution)
    surf_table = read_su2_table(surface_forces)

    inputs = build_field_tensor(vol_table, cfg.input_fields, cfg.grid_shape)
    target_fields = build_field_tensor(vol_table, cfg.target_fields, cfg.grid_shape)
    cl, cd = extract_forces(surf_table, cfg.cl_name, cfg.cd_name)

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

"""Shared helpers for building snapshot artifacts from CFD runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..surrogate.snapshot_builder import SnapshotConfig, su2_to_unet_snapshot
from ..utils.io_utils import DESIGN_LOG


@dataclass
class SnapshotSpec:
    """Description of how to persist a CFD snapshot."""

    output_dir: Path
    grid_shape: tuple[int, int]
    input_fields: list[str]
    target_fields: list[str]
    cl_name: str
    cd_name: str


def relative_snapshot_path(path: Path, *, data_root: Path | None = None) -> str:
    """Return ``path`` relative to ``data_root`` (defaults to ``DESIGN_LOG`` parent)."""

    data_root = data_root or DESIGN_LOG.parent
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def create_snapshot_from_run(design_id: str, spec: SnapshotSpec, run_result: dict) -> Path:
    """Convert solver outputs from ``run_result`` into a saved snapshot archive."""

    volume_path = run_result.get("volume_output")
    surface_path = run_result.get("surface_output")

    if not volume_path or not Path(volume_path).exists():
        raise FileNotFoundError("Volume solution file not found; cannot build snapshot.")

    volume_path = Path(volume_path)
    output_path = spec.output_dir / f"{design_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = SnapshotConfig(
        input_fields=spec.input_fields,
        target_fields=spec.target_fields,
        grid_shape=spec.grid_shape,
        cl_name=spec.cl_name,
        cd_name=spec.cd_name,
    )

    history_path = run_result.get("history_path")
    su2_to_unet_snapshot(
        volume_path,
        Path(surface_path) if surface_path and Path(surface_path).exists() else None,
        output_path,
        cfg,
        history_path=Path(history_path) if history_path else None,
    )

    return output_path

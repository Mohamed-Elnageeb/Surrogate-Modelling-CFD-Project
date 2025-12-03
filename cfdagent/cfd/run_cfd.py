import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict

from .postprocess import extract_metrics
from ..geometry.airfoil_param import design_to_airfoil_coords
from ..utils.io_utils import append_design_result

BASE_CASE = Path(__file__).resolve().parents[0] / "base_case"
RUN_ROOT = Path(__file__).resolve().parents[0] / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

def update_geometry_and_mesh(run_dir: Path, design_vec):
    """
    Update geometry and mesh files inside run_dir based on design_vec.
    Placeholder for Phase 0.
    """
    raise NotImplementedError("geometry/mesh update not implemented yet")


def run_cfd(design_id: str, design_vec) -> Dict[str, Any]:
    run_dir = RUN_ROOT / f"run_{design_id}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(BASE_CASE, run_dir)

    try:
        design_to_airfoil_coords(design_vec)
        update_geometry_and_mesh(run_dir, design_vec)
    except NotImplementedError:
        pass

    config_path = run_dir / "config.cfg"

    result = {
        "design_id": design_id,
        "success": False,
        "Cl": None,
        "Cd": None,
        "residual": None,
        "run_dir": str(run_dir),
    }

    proc = subprocess.run(
        ["SU2_CFD", str(config_path)],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )

    if proc.returncode == 0:
        history_file = run_dir / "history.csv"
        metrics = extract_metrics(history_file)
        result.update({"success": True, **metrics})
    else:
        result["success"] = False

    meta_path = run_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(result, f, indent=2)

    return result

from pathlib import Path
from typing import Iterable

from pathlib import Path
from typing import Iterable

import pandas as pd

from cfdagent.agent.search_agent import AirfoilDesignAgent
from cfdagent.utils.snapshot_pipeline import SnapshotSpec


def _fake_run(design_id: str, vec: Iterable[float], workdir=None, su2_executable=None):
    return {"Cl": 1.0, "Cd": 0.1, "residual": 1e-3, "success": True}


def test_agent_trains_and_appends(tmp_path: Path) -> None:
    design_log = tmp_path / "designs.csv"
    history = pd.DataFrame(
        {
            "dc1": [0.0, 0.001, -0.001],
            "dc2": [0.0, 0.001, -0.001],
            "dc3": [0.0, 0.001, -0.001],
            "dc4": [0.0, 0.001, -0.001],
            "dc5": [0.0, 0.001, -0.001],
            "dt1": [0.0, 0.001, -0.001],
            "dt2": [0.0, 0.001, -0.001],
            "dt3": [0.0, 0.001, -0.001],
            "dt4": [0.0, 0.001, -0.001],
            "dt5": [0.0, 0.001, -0.001],
            "Cl": [0.8, 0.9, 0.7],
            "Cd": [0.05, 0.04, 0.06],
        }
    )
    history.to_csv(design_log, index=False)

    agent = AirfoilDesignAgent(
        design_log=design_log, run_function=_fake_run, noise_scale=0.0, random_state=42
    )

    results = agent.run_iteration(num_candidates=2)

    df = pd.read_csv(design_log)
    assert len(results) == 2
    assert len(df) == len(history) + 2
    assert {"Cl", "Cd", "residual", "success"}.issubset(df.columns)


def test_agent_handles_empty_history(tmp_path: Path) -> None:
    design_log = tmp_path / "designs.csv"
    recorded: list[list[float]] = []

    def run_and_record(design_id: str, vec: Iterable[float], workdir=None, su2_executable=None):
        recorded.append(list(vec))
        return {"Cl": 0.9, "Cd": 0.05, "residual": 5e-4, "success": True}

    agent = AirfoilDesignAgent(design_log=design_log, run_function=run_and_record, random_state=0)
    results = agent.run_iteration(num_candidates=1)

    df = pd.read_csv(design_log)
    assert len(results) == 1
    assert len(df) == 1
    assert len(recorded[0]) == 10
    assert set([f"dc{i+1}" for i in range(5)] + [f"dt{i+1}" for i in range(5)]).issubset(df.columns)
    assert {"Cl", "Cd"}.issubset(df.columns)


def test_agent_generates_review(tmp_path: Path) -> None:
    design_log = tmp_path / "designs.csv"
    history = pd.DataFrame(
        {
            "design_id": ["a", "b", "c"],
            "dc1": [0.0, 0.002, -0.001],
            "dc2": [0.0, 0.001, -0.001],
            "dc3": [0.0, 0.0015, -0.0005],
            "dc4": [0.0, 0.0005, -0.001],
            "dc5": [0.0, 0.0003, -0.0008],
            "dt1": [0.0, 0.001, -0.001],
            "dt2": [0.0, 0.001, -0.001],
            "dt3": [0.0, 0.001, -0.001],
            "dt4": [0.0, 0.001, -0.001],
            "dt5": [0.0, 0.001, -0.001],
            "Cl": [0.9, 1.0, 0.7],
            "Cd": [0.04, 0.05, 0.06],
        }
    )
    history.to_csv(design_log, index=False)

    agent = AirfoilDesignAgent(design_log=design_log, random_state=0)
    review = agent.generate_review(n_best=2)

    assert "Agent optimization review" in review["review"]
    assert Path(review["review_file"]).exists()
    assert Path(review["plots"]["geometry"]).exists()
    # Performance plot can be None when metrics are missing; ensure it exists here.
    assert Path(review["plots"]["performance"]).exists()


def test_agent_optionally_builds_snapshots(tmp_path: Path) -> None:
    design_log = tmp_path / "designs.csv"

    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
# Comment line
x,y,p,u,v
0.0,0.0,1.0,0.1,0.2
1.0,0.0,2.0,0.3,0.4
0.0,1.0,3.0,0.5,0.6
1.0,1.0,4.0,0.7,0.8
""".strip()
    )
    surface_path = tmp_path / "surface.dat"
    surface_path.write_text("CL,CD\n0.5,0.01\n0.6,0.02\n")

    def run_with_outputs(design_id: str, vec: Iterable[float], workdir=None, su2_executable=None):
        return {
            "design_id": design_id,
            "design_vec": list(vec),
            "Cl": 0.6,
            "Cd": 0.02,
            "residual": 1e-3,
            "success": True,
            "volume_output": volume_path,
            "surface_output": surface_path,
        }

    snapshot_spec = SnapshotSpec(
        output_dir=tmp_path / "snapshots",
        grid_shape=(2, 2),
        input_fields=["p", "u"],
        target_fields=["v"],
        cl_name="CL",
        cd_name="CD",
    )

    agent = AirfoilDesignAgent(design_log=design_log, run_function=run_with_outputs, random_state=1)
    results = agent.run_iteration(num_candidates=1, snapshot_spec=snapshot_spec)

    assert results[0]["success"] is True
    assert "snapshot_error" not in results[0]
    assert Path(results[0]["flow_path"]).exists()

    df = pd.read_csv(design_log)
    assert "flow_path" in df.columns

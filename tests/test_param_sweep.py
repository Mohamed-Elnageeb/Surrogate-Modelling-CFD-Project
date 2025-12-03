from __future__ import annotations

from pathlib import Path
import csv

import pytest

from cfdagent.cfd.param_sweep import CaseSample, _collect_input_columns, generate_param_grid, run_param_sweep
from cfdagent.cfd.run_cfd import Su2RunConfig


def test_generate_param_grid_without_reynolds() -> None:
    machs = [0.1, 0.2]
    aoas = [0.0, 5.0]
    samples = generate_param_grid(machs, aoas)

    assert len(samples) == len(machs) * len(aoas)
    for sample in samples:
        assert isinstance(sample, CaseSample)
        assert sample.reynolds is None
        assert sample.mach in machs
        assert sample.aoa in aoas


def test_generate_param_grid_with_reynolds() -> None:
    machs = [0.1]
    aoas = [0.0, 5.0]
    reynolds = [1e6, 2e6]
    samples = generate_param_grid(machs, aoas, reynolds)

    assert len(samples) == len(machs) * len(aoas) * len(reynolds)
    for sample in samples:
        assert sample.reynolds in reynolds
        assert sample.mach in machs
        assert sample.aoa in aoas


def test_collect_input_columns_includes_extras() -> None:
    samples = [
        CaseSample(mach=0.1, aoa=0.0, extra={"BETA": 1.0, "AOA": 2.0}),
        CaseSample(mach=0.2, aoa=5.0, reynolds=1e6, extra={"BETA": 2.0, "GAMMA": 0.5}),
    ]

    columns = _collect_input_columns(samples)
    assert columns == sorted({"mach", "aoa", "reynolds", "BETA", "AOA", "GAMMA"})


def test_run_param_sweep_writes_expected_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [
        CaseSample(mach=0.1, aoa=0.0, extra={"BETA": 1.0}),
        CaseSample(mach=0.2, aoa=5.0, reynolds=1e6),
    ]

    def fake_run_su2_case(cfg: Su2RunConfig, param_overrides: dict[str, float | int | str], timeout: int = 3600):
        lift = param_overrides.get("MACH_NUMBER", 0) * 10
        return {"history_data": {"LIFT": lift, "DRAG": 0.5}, "config_path": cfg.workdir}

    monkeypatch.setattr("cfdagent.cfd.param_sweep.run_su2_case", fake_run_su2_case)

    out_csv = tmp_path / "out.csv"
    cfg = Su2RunConfig(workdir=tmp_path)
    run_param_sweep(cfg, samples, out_csv, shuffle=False, max_cases=None)

    assert out_csv.exists()
    with out_csv.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        assert header is not None
        assert "mach" in header and "aoa" in header and "reynolds" in header
        assert "BETA" in header
        assert "out_LIFT" in header and "out_DRAG" in header
        rows = list(reader)
        assert len(rows) == len(samples)
        assert rows[0]["mach"] == "0.1"
        assert rows[0]["BETA"] == "1.0"
        assert rows[0]["out_LIFT"] == "1.0"
        assert rows[1]["reynolds"] == "1000000.0"


def test_run_param_sweep_respects_max_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [CaseSample(mach=m, aoa=0.0) for m in [0.1, 0.2, 0.3]]

    call_count = 0

    def fake_run_su2_case(cfg: Su2RunConfig, param_overrides: dict[str, float | int | str], timeout: int = 3600):
        nonlocal call_count
        call_count += 1
        return {"history_data": {"LIFT": 1.0}}

    monkeypatch.setattr("cfdagent.cfd.param_sweep.run_su2_case", fake_run_su2_case)

    out_csv = tmp_path / "out.csv"
    cfg = Su2RunConfig(workdir=tmp_path)
    run_param_sweep(cfg, samples, out_csv, shuffle=False, max_cases=2)

    assert call_count == 2
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2


def test_run_param_sweep_with_shuffle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [CaseSample(mach=m, aoa=0.0) for m in [0.1, 0.2, 0.3]]

    def fake_run_su2_case(cfg: Su2RunConfig, param_overrides: dict[str, float | int | str], timeout: int = 3600):
        return {"history_data": {"LIFT": param_overrides["MACH_NUMBER"]}}

    monkeypatch.setattr("cfdagent.cfd.param_sweep.run_su2_case", fake_run_su2_case)

    out_csv = tmp_path / "out.csv"
    cfg = Su2RunConfig(workdir=tmp_path)
    run_param_sweep(cfg, samples, out_csv, shuffle=True, max_cases=None)

    assert out_csv.exists()
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == len(samples)

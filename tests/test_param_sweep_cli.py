from __future__ import annotations

from pathlib import Path
import csv

import pytest

from cfdagent.cfd.param_sweep_cli import main as param_sweep_main


def test_param_sweep_cli_invokes_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "case"
    workdir.mkdir()
    (workdir / "config.cfg").write_text("test")

    calls: list[tuple] = []

    def fake_run_param_sweep(cfg, samples, out_csv, shuffle=False, max_cases=None):
        calls.append((cfg, samples, out_csv, shuffle, max_cases))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["mach", "aoa", "reynolds"])
            writer.writerow(["0.1", "0.0", "1000000.0"])

    monkeypatch.setattr("cfdagent.cfd.param_sweep_cli.run_param_sweep", fake_run_param_sweep)

    out_csv = tmp_path / "data.csv"
    exit_code = param_sweep_main(
        [
            str(workdir),
            "--out-csv",
            str(out_csv),
            "--mach-values",
            "0.1,0.2",
            "--aoa-values",
            "0.0,5.0",
            "--reynolds-values",
            "1e6,2e6",
            "--max-cases",
            "2",
            "--shuffle",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    cfg, samples, passed_out_csv, shuffle_flag, max_cases = calls[0]
    assert cfg.workdir == workdir.resolve()
    assert passed_out_csv == out_csv
    assert shuffle_flag is True
    assert max_cases == 2
    assert out_csv.exists()


def test_param_sweep_cli_without_reynolds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "case"
    workdir.mkdir()
    (workdir / "config.cfg").write_text("test")

    called = False

    def fake_run_param_sweep(cfg, samples, out_csv, shuffle=False, max_cases=None):
        nonlocal called
        called = True
        out_csv.write_text("mach,aoa\n0.1,0.0\n")

    monkeypatch.setattr("cfdagent.cfd.param_sweep_cli.run_param_sweep", fake_run_param_sweep)

    exit_code = param_sweep_main(
        [
            str(workdir),
            "--mach-values",
            "0.1",
            "--aoa-values",
            "0.0",
        ]
    )

    assert exit_code == 0
    assert called

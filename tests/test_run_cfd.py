from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib
import subprocess

import pytest

run_cfd_module = importlib.import_module("cfdagent.cfd.run_cfd")

from cfdagent.cfd.run_cfd import (
    Su2RunConfig,
    create_modified_config,
    parse_history_file,
    run_cfd,
    run_su2_case,
)


def test_create_modified_config_overrides_and_preserves_lines(tmp_path: Path) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text(
        """MACH_NUMBER= 0.1
AOA= 0.0
REYNOLDS_NUMBER= 1e6
SOLVER= RANS
% Comment line
"""
    )
    out_cfg = tmp_path / "config_override.cfg"

    create_modified_config(
        base_cfg, out_cfg, {"MACH_NUMBER": 0.2, "AOA": 5.0}
    )

    assert out_cfg.exists()
    content = out_cfg.read_text().splitlines()
    assert "MACH_NUMBER= 0.2" in content
    assert "AOA= 5.0" in content
    assert "REYNOLDS_NUMBER= 1e6" in content
    assert "% Comment line" in content


def test_create_modified_config_missing_key_raises(tmp_path: Path) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text("MACH_NUMBER= 0.1\n")
    out_cfg = tmp_path / "config_override.cfg"

    with pytest.raises(KeyError):
        create_modified_config(base_cfg, out_cfg, {"AOA": 5.0})


def test_parse_history_file_returns_last_row_and_converts(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text(
        """# Comment\n%Another comment\nIter,Value,Note\n1,1.0,first\n2,2.5,second\n3,3.0,third\n"""
    )

    parsed = parse_history_file(history)
    assert parsed == {"Iter": 3, "Value": 3.0, "Note": "third"}


def test_parse_history_file_empty_or_missing(tmp_path: Path) -> None:
    history = tmp_path / "missing.csv"
    assert parse_history_file(history) == {}

    history.write_text("")
    assert parse_history_file(history) == {}


def test_run_su2_case_uses_override_and_parses_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text("MACH_NUMBER= 0.1\nAOA= 0.0\n")

    history = tmp_path / "history.csv"
    history.write_text("Iter,Value\n1,1.0\n")

    calls: list[dict] = []

    def fake_run(cmd: list[str], cwd: Path, capture_output: bool, text: bool, timeout: int):
        calls.append({"cmd": cmd, "cwd": cwd, "capture_output": capture_output, "text": text, "timeout": timeout})
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = Su2RunConfig(workdir=tmp_path)
    result = run_su2_case(cfg, param_overrides={"MACH_NUMBER": 0.2})

    assert calls
    first_call = calls[0]
    assert cfg.su2_executable in first_call["cmd"][0]
    assert str(tmp_path / "config_override.cfg") == first_call["cmd"][1]
    assert result["history_data"] == {"Iter": 1, "Value": 1.0}
    assert (tmp_path / "config_override.cfg").exists()


def test_run_su2_case_without_overrides_uses_base_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text("MACH_NUMBER= 0.1\n")
    history = tmp_path / "history.csv"
    history.write_text("Iter,Value\n1,1.0\n")

    called_cmd: list[str] = []

    def fake_run(cmd: list[str], cwd: Path, capture_output: bool, text: bool, timeout: int):
        called_cmd[:] = cmd
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = Su2RunConfig(workdir=tmp_path)
    result = run_su2_case(cfg)

    assert called_cmd == [cfg.su2_executable, str(base_cfg)]
    assert result["config_path"] == base_cfg
    assert result["history_data"] == {"Iter": 1, "Value": 1.0}


def test_run_su2_case_raises_on_nonzero_return(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text("MACH_NUMBER= 0.1\n")

    def fake_run(cmd: list[str], cwd: Path, capture_output: bool, text: bool, timeout: int):
        return SimpleNamespace(stdout="", stderr="bad", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = Su2RunConfig(workdir=tmp_path)
    with pytest.raises(RuntimeError):
        run_su2_case(cfg)


def test_run_cfd_flags_missing_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_cfg = tmp_path / "config.cfg"
    base_cfg.write_text("VOLUME_FILENAME=flow\nSURFACE_FILENAME=surface\n")
    (tmp_path / "mesh.su2").write_text("")

    monkeypatch.setattr(run_cfd_module.shutil, "which", lambda name: "/usr/bin/SU2_CFD")

    def fake_run(cfg):  # pragma: no cover - simple stub
        return {"history_data": {}, "history_path": None, "stdout": "", "stderr": "", "config_path": base_cfg}

    monkeypatch.setattr(run_cfd_module, "run_su2_case", fake_run)
    monkeypatch.setattr(run_cfd_module, "extract_metrics", lambda _: {})

    result = run_cfd(design_id="abc", design_vec=[0.0] * 10, workdir=tmp_path, su2_executable="SU2_CFD")

    assert result["Cl"] is None and result["Cd"] is None
    assert result["error"]
    assert result["success"] is False

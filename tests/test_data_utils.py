from __future__ import annotations

from pathlib import Path

import pytest

from cfdagent.surrogate import inspect_dataset
from cfdagent.surrogate.data_utils import infer_schema, load_dataset, parse_value


def test_infer_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        """mach,aoa,reynolds,BETA,out_LIFT,out_DRAG
0.1,2.0,1000000,0.0,1.0,0.1
0.2,3.0,2000000,0.1,1.1,0.2
"""
    )

    schema = infer_schema(csv_path)
    assert schema.input_columns == ["mach", "aoa", "reynolds", "BETA"]
    assert schema.output_columns == ["out_LIFT", "out_DRAG"]


def test_parse_value() -> None:
    assert parse_value("") is None
    assert parse_value("   ") is None
    assert parse_value("1.0") == 1
    assert parse_value("2.5") == 2.5
    assert parse_value("not_a_number") is None


def test_load_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        """mach,aoa,reynolds,out_CL,out_CD
0.1,2.0,,0.5,0.01
0.2,3.0,1000000,0.6,0.02
,,,
"""
    )

    X, Y, schema = load_dataset(csv_path)

    assert schema.input_columns == ["mach", "aoa", "reynolds"]
    assert schema.output_columns == ["out_CL", "out_CD"]
    assert len(X) == 2
    assert len(Y) == 2
    assert X[0] == [0.1, 2.0, None]
    assert Y[0] == [0.5, 0.01]
    assert X[1] == [0.2, 3.0, 1000000]
    assert Y[1] == [0.6, 0.02]


def test_inspect_dataset_main_integration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        """mach,aoa,out_CL
0.1,2.0,0.5
0.2,3.0,0.6
"""
    )

    exit_code = inspect_dataset.main([str(csv_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Samples: 2" in captured
    assert "Inputs (d = 2): mach, aoa" in captured
    assert "Outputs (k = 1): out_CL" in captured
    assert "out_CL:" in captured

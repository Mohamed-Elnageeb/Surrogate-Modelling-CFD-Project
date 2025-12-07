from pathlib import Path

from cfdagent.cfd.postprocess import extract_metrics


def test_extract_metrics_handles_quoted_headers(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text(
        '\"rms[Rho]\",\"rms[RhoE]\",\"CD\",\"CL\"\n'
        ' -2.48, 2.99, 0.094, 0.097\n'
        ' -2.88, 2.58, 0.177, 0.242\n'
    )

    metrics = extract_metrics(history)

    assert metrics == {"Cl": 0.242, "Cd": 0.177, "residual": None}


def test_extract_metrics_discards_negative_coefficients(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text(
        """CL,CD
-0.01,-0.02
"""
    )

    metrics = extract_metrics(history)

    assert metrics == {"Cl": None, "Cd": None, "residual": None}

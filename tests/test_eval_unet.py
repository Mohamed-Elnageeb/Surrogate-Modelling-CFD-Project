from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from cfdagent.surrogate.eval_unet import evaluate_unet_dir
from cfdagent.surrogate.eval_unet_cli import main as eval_unet_main
from cfdagent.surrogate.model_unet import CFDSurrogateUNet


def _make_synthetic_unet_data(path: Path, num_samples: int, in_channels: int, out_channels: int):
    h, w = 8, 8
    rng = np.random.default_rng(0)
    for i in range(num_samples):
        inputs = rng.random((in_channels, h, w), dtype=np.float32)
        targets = rng.random((out_channels, h, w), dtype=np.float32)
        cl = rng.random((), dtype=np.float32)
        cd = rng.random((), dtype=np.float32)
        np.savez(path / f"sample_{i}.npz", input=inputs, target_fields=targets, cl=cl, cd=cd)


def test_evaluate_unet_dir_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_unet_data(data_dir, num_samples=4, in_channels=2, out_channels=3)

    model = CFDSurrogateUNet(in_channels=2, base_channels=4, out_channels=3)

    metrics = evaluate_unet_dir(model, data_dir, device="cpu", batch_size=2)

    assert set(metrics.keys()) == {"field_mse", "cl_mse", "cd_mse"}
    assert all(value >= 0 for value in metrics.values())


def test_eval_unet_cli_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_unet_data(data_dir, num_samples=3, in_channels=2, out_channels=3)

    model = CFDSurrogateUNet(in_channels=2, base_channels=4, out_channels=3)
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    exit_code = eval_unet_main(
        [
            str(data_dir),
            "--model-path",
            str(model_path),
            "--device",
            "cpu",
            "--in-channels",
            "2",
            "--out-channels",
            "3",
            "--base-channels",
            "4",
            "--batch-size",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "field_mse" in captured.out

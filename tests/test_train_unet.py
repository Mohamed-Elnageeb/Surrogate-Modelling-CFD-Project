from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from cfdagent.surrogate.train_unet import CFDSnapshotDataset, UNetTrainConfig, train_unet
from cfdagent.surrogate.train_unet_cli import main as train_unet_main


@pytest.fixture
def synthetic_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_unet_data(data_dir, num_samples=4, in_channels=2, out_channels=3)
    return data_dir


def _make_synthetic_unet_data(path: Path, num_samples: int, in_channels: int, out_channels: int):
    h, w = 8, 8
    rng = np.random.default_rng(0)
    for i in range(num_samples):
        inputs = rng.random((in_channels, h, w), dtype=np.float32)
        targets = rng.random((out_channels, h, w), dtype=np.float32)
        cl = rng.random((), dtype=np.float32)
        cd = rng.random((), dtype=np.float32)
        np.savez(path / f"sample_{i}.npz", input=inputs, target_fields=targets, cl=cl, cd=cd)


def test_dataset_shapes(synthetic_data_dir):
    dataset = CFDSnapshotDataset(str(synthetic_data_dir))
    sample = dataset[0]
    assert len(dataset) == 4
    assert sample[0].shape == (2, 8, 8)
    assert sample[1].shape == (3, 8, 8)
    assert sample[2].shape == ()
    assert sample[3].shape == ()


def test_train_unet_runs(synthetic_data_dir):
    cfg = UNetTrainConfig(
        data_dir=str(synthetic_data_dir),
        epochs=2,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        base_channels=4,
        in_channels=2,
        out_channels=3,
        val_fraction=0.25,
    )
    result = train_unet(cfg)
    assert result.train_losses
    assert all(loss >= 0 for loss in result.train_losses)
    if result.val_losses:
        assert all(loss >= 0 for loss in result.val_losses)


def test_train_unet_cli(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_unet_data(data_dir, num_samples=3, in_channels=2, out_channels=3)
    model_out = tmp_path / "model.pt"
    exit_code = train_unet_main(
        [
            str(data_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "1e-3",
            "--device",
            "cpu",
            "--base-channels",
            "4",
            "--in-channels",
            "2",
            "--out-channels",
            "3",
            "--val-fraction",
            "0.0",
            "--model-out",
            str(model_out),
        ]
    )
    assert exit_code == 0
    assert model_out.exists()

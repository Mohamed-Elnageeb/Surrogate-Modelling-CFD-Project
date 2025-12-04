from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model_unet import CFDSurrogateUNet
from .train_unet import CFDSnapshotDataset


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    return device if torch.cuda.is_available() else "cpu"


def evaluate_unet_dir(
    model: CFDSurrogateUNet,
    data_dir: str | Path,
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict[str, float]:
    """Evaluate a ``CFDSurrogateUNet`` over a directory of snapshot ``.npz`` files.

    If the dataset is empty, returns an empty dictionary.
    """

    dataset = CFDSnapshotDataset(str(data_dir))
    if len(dataset) == 0:
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = _select_device(device)
    model = model.to(device)
    model.eval()

    loss_fn = nn.MSELoss()

    field_total = 0.0
    cl_total = 0.0
    cd_total = 0.0
    count = 0

    with torch.no_grad():
        for inputs, target_fields, cl_true, cd_true in loader:
            inputs = inputs.to(device)
            target_fields = target_fields.to(device)
            cl_true = cl_true.to(device)
            cd_true = cd_true.to(device)

            fields_pred, cl_pred, cd_pred = model(inputs)

            batch_size_actual = inputs.shape[0]
            field_total += loss_fn(fields_pred, target_fields).item() * batch_size_actual
            cl_total += loss_fn(cl_pred.squeeze(-1), cl_true).item() * batch_size_actual
            cd_total += loss_fn(cd_pred.squeeze(-1), cd_true).item() * batch_size_actual
            count += batch_size_actual

    if count == 0:
        return {}

    return {
        "field_mse": field_total / count,
        "cl_mse": cl_total / count,
        "cd_mse": cd_total / count,
    }

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model_unet import CFDSurrogateUNet


class CFDSnapshotDataset(Dataset):
    """Dataset for loading CFD snapshots stored as .npz files."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(
            f
            for f in os.listdir(data_dir)
            if f.endswith(".npz") and os.path.isfile(os.path.join(data_dir, f))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = os.path.join(self.data_dir, self.files[idx])
        with np.load(file_path) as data:
            inputs = torch.from_numpy(data["input"]).float()
            target_fields = torch.from_numpy(data["target_fields"]).float()
            cl = torch.as_tensor(data["cl"], dtype=torch.float32)
            cd = torch.as_tensor(data["cd"], dtype=torch.float32)
        return inputs, target_fields, cl, cd


def split_indices(n: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    """Split indices into train and validation sets."""

    val_size = int(n * val_fraction)
    if val_size >= n:
        val_size = max(n - 1, 0)
    indices = list(range(n))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


@dataclass
class UNetTrainConfig:
    data_dir: str
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    base_channels: int = 32
    in_channels: int = 3
    out_channels: int = 3
    val_fraction: float = 0.1
    field_loss_weight: float = 1.0
    cl_loss_weight: float = 1.0
    cd_loss_weight: float = 1.0


@dataclass
class UNetTrainResult:
    model: CFDSurrogateUNet
    train_losses: List[float]
    val_losses: List[float]


def _compute_loss(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: Tuple[float, float, float],
    loss_fn: nn.Module,
) -> torch.Tensor:
    fields_pred, cl_pred, cd_pred = outputs
    target_fields, cl_true, cd_true = targets
    field_loss = loss_fn(fields_pred, target_fields)
    cl_loss = loss_fn(cl_pred.squeeze(-1), cl_true)
    cd_loss = loss_fn(cd_pred.squeeze(-1), cd_true)
    w_field, w_cl, w_cd = weights
    total = w_field * field_loss + w_cl * cl_loss + w_cd * cd_loss
    return total


def evaluate_unet(model: CFDSurrogateUNet, loader: DataLoader, device: str = "cuda") -> float:
    """Evaluate the model over a dataloader and return mean loss."""

    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, target_fields, cl, cd in loader:
            inputs = inputs.to(device)
            target_fields = target_fields.to(device)
            cl = cl.to(device)
            cd = cd.to(device)

            outputs = model(inputs)
            loss = _compute_loss(
                outputs,
                (target_fields, cl, cd),
                (1.0, 1.0, 1.0),
                loss_fn,
            )
            total_loss += loss.item()
            num_batches += 1
    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


def train_unet(cfg: UNetTrainConfig) -> UNetTrainResult:
    dataset = CFDSnapshotDataset(cfg.data_dir)
    train_indices, val_indices = split_indices(len(dataset), cfg.val_fraction)

    if not train_indices:
        raise ValueError("No training samples available. Adjust val_fraction or dataset size.")

    train_subset: Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = Subset(
        dataset, train_indices
    )
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)

    val_loader = None
    if val_indices:
        val_subset: Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = Subset(
            dataset, val_indices
        )
        val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model = CFDSurrogateUNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        out_channels=cfg.out_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    for _ in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for inputs, target_fields, cl, cd in train_loader:
            inputs = inputs.to(device)
            target_fields = target_fields.to(device)
            cl = cl.to(device)
            cd = cd.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = _compute_loss(
                outputs,
                (target_fields, cl, cd),
                (cfg.field_loss_weight, cfg.cl_loss_weight, cfg.cd_loss_weight),
                loss_fn,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        train_losses.append(epoch_loss / max(num_batches, 1))

        if val_loader is not None:
            val_loss = evaluate_unet(model, val_loader, device=device)
            val_losses.append(val_loss)

    return UNetTrainResult(model=model, train_losses=train_losses, val_losses=val_losses)

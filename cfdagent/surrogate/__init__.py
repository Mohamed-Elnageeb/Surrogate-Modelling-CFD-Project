"""Surrogate modeling utilities for CFD datasets."""

from .data_utils import DatasetSchema, infer_schema, load_dataset, parse_value
from .model_unet import CFDSurrogateUNet
from .train_unet import (
    CFDSnapshotDataset,
    UNetTrainConfig,
    UNetTrainResult,
    evaluate_unet,
    split_indices,
    train_unet,
)

__all__ = [
    "DatasetSchema",
    "infer_schema",
    "load_dataset",
    "parse_value",
    "CFDSurrogateUNet",
    "CFDSnapshotDataset",
    "UNetTrainConfig",
    "UNetTrainResult",
    "train_unet",
    "evaluate_unet",
    "split_indices",
]

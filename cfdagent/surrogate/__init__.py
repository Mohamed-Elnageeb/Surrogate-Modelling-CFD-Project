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
]

try:  # pragma: no cover - optional torch dependency
    from .model_unet import CFDSurrogateUNet

    __all__.append("CFDSurrogateUNet")
except ImportError:  # pragma: no cover
    CFDSurrogateUNet = None  # type: ignore[misc]

try:  # pragma: no cover - optional torch dependency
    from .train_unet import (
        CFDSnapshotDataset,
        UNetTrainConfig,
        UNetTrainResult,
        evaluate_unet,
        split_indices,
        train_unet,
    )
    from .eval_unet import evaluate_unet_dir

    __all__.extend(
        [
            "CFDSnapshotDataset",
            "UNetTrainConfig",
            "UNetTrainResult",
            "train_unet",
            "evaluate_unet",
            "evaluate_unet_dir",
            "split_indices",
        ]
    )
except ImportError:  # pragma: no cover
    CFDSnapshotDataset = None  # type: ignore[misc]
    UNetTrainConfig = None  # type: ignore[misc]
    UNetTrainResult = None  # type: ignore[misc]
    evaluate_unet = None  # type: ignore[misc]
    evaluate_unet_dir = None  # type: ignore[misc]
    split_indices = None  # type: ignore[misc]
    train_unet = None  # type: ignore[misc]

"""Surrogate modeling utilities for CFD datasets."""

from .data_utils import DatasetSchema, infer_schema, load_dataset, parse_value
from .snapshot_builder import (
    SnapshotConfig,
    build_field_tensor,
    extract_forces,
    read_su2_table,
    save_snapshot_from_arrays,
    su2_to_unet_snapshot,
)

__all__ = [
    "DatasetSchema",
    "infer_schema",
    "load_dataset",
    "parse_value",
    "SnapshotConfig",
    "read_su2_table",
    "build_field_tensor",
    "extract_forces",
    "su2_to_unet_snapshot",
    "save_snapshot_from_arrays",
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

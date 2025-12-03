from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .eval_unet import evaluate_unet_dir
from .model_unet import CFDSurrogateUNet
from .train_unet import CFDSnapshotDataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CFDSurrogateUNet")
    parser.add_argument("data_dir", help="Directory containing .npz snapshot files")
    parser.add_argument("--model-path", required=True, help="Path to model state_dict (.pt)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out-channels", type=int, default=3, help="Number of output channels")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for UNet")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        dataset = CFDSnapshotDataset(str(args.data_dir))
        model = CFDSurrogateUNet(
            in_channels=args.in_channels,
            base_channels=args.base_channels,
            out_channels=args.out_channels,
        )
        state_dict = torch.load(Path(args.model_path), map_location="cpu")
        model.load_state_dict(state_dict)

        metrics = evaluate_unet_dir(
            model,
            data_dir=args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

        print(f"Evaluated model on {len(dataset)} snapshots from: {args.data_dir}")
        for key, value in metrics.items():
            print(f"{key}={value}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

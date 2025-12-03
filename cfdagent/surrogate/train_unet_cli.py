import argparse
import sys
import torch

from .train_unet import UNetTrainConfig, train_unet


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CFDSurrogateUNet on CFD snapshots")
    parser.add_argument("data_dir", help="Directory containing .npz snapshot files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for UNet")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out-channels", type=int, default=3, help="Number of output channels")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--field-loss-weight", type=float, default=1.0, help="Weight for field loss")
    parser.add_argument("--cl-loss-weight", type=float, default=1.0, help="Weight for cl loss")
    parser.add_argument("--cd-loss-weight", type=float, default=1.0, help="Weight for cd loss")
    parser.add_argument("--model-out", type=str, default="cfd_unet.pt", help="Path to save model state")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cfg = UNetTrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        base_channels=args.base_channels,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        val_fraction=args.val_fraction,
        field_loss_weight=args.field_loss_weight,
        cl_loss_weight=args.cl_loss_weight,
        cd_loss_weight=args.cd_loss_weight,
    )

    try:
        result = train_unet(cfg)
        torch.save(result.model.state_dict(), args.model_out)
        print(
            f"Training completed. epochs={len(result.train_losses)}, "
            f"final_train_loss={result.train_losses[-1]:.4f}" +
            (f", final_val_loss={result.val_losses[-1]:.4f}" if result.val_losses else "")
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

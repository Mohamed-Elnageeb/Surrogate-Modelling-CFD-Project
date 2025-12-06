import argparse
import concurrent.futures
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ..geometry.airfoil_param import sample_random_design
from ..cfd.run_cfd import run_cfd
from ..surrogate.snapshot_builder import (
    SnapshotConfig,
    build_field_tensor,
    read_su2_table,
    save_snapshot_from_arrays,
    su2_to_unet_snapshot,
)
from ..utils.io_utils import DESIGN_LOG, append_design_result


@dataclass
class SnapshotSpec:
    output_dir: Path
    grid_shape: tuple[int, int]
    input_fields: list[str]
    target_fields: list[str]
    cl_name: str
    cd_name: str


def _relative_snapshot_path(path: Path) -> str:
    data_root = DESIGN_LOG.parent
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def _create_snapshot(design_id: str, spec: SnapshotSpec, run_result: dict) -> Path:
    volume_path = run_result.get("volume_output")
    surface_path = run_result.get("surface_output")

    if not volume_path or not Path(volume_path).exists():
        raise FileNotFoundError("Volume solution file not found; cannot build snapshot.")

    volume_path = Path(volume_path)
    output_path = spec.output_dir / f"{design_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if surface_path and Path(surface_path).exists():
        cfg = SnapshotConfig(
            input_fields=spec.input_fields,
            target_fields=spec.target_fields,
            grid_shape=spec.grid_shape,
            cl_name=spec.cl_name,
            cd_name=spec.cd_name,
        )
        su2_to_unet_snapshot(volume_path, Path(surface_path), output_path, cfg)
        return output_path

    table = read_su2_table(volume_path)
    inputs = build_field_tensor(table, spec.input_fields, spec.grid_shape)
    targets = build_field_tensor(table, spec.target_fields, spec.grid_shape)

    cl = run_result.get("Cl")
    cd = run_result.get("Cd")
    if cl is None or cd is None:
        raise FileNotFoundError("Surface forces missing and Cl/Cd unavailable for snapshot.")

    save_snapshot_from_arrays(output_path, inputs, targets, cl, cd)
    return output_path


def _run_single_simulation(_: int, snapshot_spec: SnapshotSpec | None = None) -> Tuple[dict, float]:
    """Run one CFD simulation and return its row dict and wall time."""

    start = time.time()
    design_vec = sample_random_design()
    design_id = str(uuid.uuid4())[:8]
    result = run_cfd(design_id, design_vec)

    row = {"design_id": design_id}
    for idx in range(5):
        row[f"dc{idx+1}"] = float(design_vec[idx])
    for idx in range(5):
        row[f"dt{idx+1}"] = float(design_vec[5 + idx])
    row.update(
        {
            "Cl": result.get("Cl"),
            "Cd": result.get("Cd"),
            "residual": result.get("residual"),
            "success": result.get("success", False),
        }
    )
    if snapshot_spec and row["success"]:
        try:
            snapshot_path = _create_snapshot(design_id, snapshot_spec, result)
            row["flow_path"] = _relative_snapshot_path(snapshot_path)
        except Exception as exc:  # pylint: disable=broad-except
            row["snapshot_error"] = str(exc)
    if not row["success"] and result.get("error"):
        row["error"] = result.get("error")

    duration = time.time() - start
    return row, duration


def main():
    parser = argparse.ArgumentParser(description="Generate CFD dataset samples")
    parser.add_argument(
        "--n_samples", type=int, default=20, help="Number of random designs to simulate"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel simulations to run (uses process pool)",
    )
    parser.add_argument(
        "--target_successes",
        type=int,
        default=None,
        help=(
            "Stop after this many successful simulations, even if fewer than n_samples"
        ),
    )
    parser.add_argument(
        "--time_limit_minutes",
        type=float,
        default=None,
        help="Stop once this many minutes have elapsed",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional directory to store per-design snapshot .npz files",
    )
    parser.add_argument(
        "--snapshot-grid-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Grid shape (ny, nx) expected by snapshot tensors",
    )
    parser.add_argument(
        "--snapshot-input-fields",
        nargs="+",
        default=["MACH", "PRESSURE", "TEMPERATURE"],
        help="Volume solution columns to stack as model inputs",
    )
    parser.add_argument(
        "--snapshot-target-fields",
        nargs="+",
        default=["MACH", "PRESSURE", "TEMPERATURE"],
        help="Volume solution columns to stack as targets",
    )
    parser.add_argument(
        "--snapshot-cl-column",
        default="CL",
        help="Column name carrying CL in the surface-force table",
    )
    parser.add_argument(
        "--snapshot-cd-column",
        default="CD",
        help="Column name carrying CD in the surface-force table",
    )
    args = parser.parse_args()

    start_time = time.time()
    successes = 0

    snapshot_spec: SnapshotSpec | None = None
    if args.snapshot_dir:
        if not args.snapshot_grid_shape:
            raise SystemExit("--snapshot-grid-shape is required when using --snapshot-dir")

        snapshot_spec = SnapshotSpec(
            output_dir=args.snapshot_dir,
            grid_shape=(args.snapshot_grid_shape[0], args.snapshot_grid_shape[1]),
            input_fields=list(args.snapshot_input_fields),
            target_fields=list(args.snapshot_target_fields),
            cl_name=args.snapshot_cl_column,
            cd_name=args.snapshot_cd_column,
        )

    if args.max_workers <= 1:
        for i in range(args.n_samples):
            row, duration = _run_single_simulation(i, snapshot_spec)
            append_design_result(row)
            if row["success"]:
                successes += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = max(args.n_samples - (i + 1), 0)
            eta = remaining * avg_time

            status = f"[{i+1}/{args.n_samples}] design_id={row['design_id']} success={row['success']}"
            if row.get("flow_path"):
                status += f" snapshot={row['flow_path']}"
            if row.get("snapshot_error"):
                status += f" snapshot_error={row['snapshot_error']}"
            if not row["success"] and row.get("error"):
                status += f" error={row['error']}"
            status += f" elapsed={elapsed:.1f}s runtime={duration:.1f}s"
            if remaining:
                status += f" eta={eta:.1f}s"
            print(status)

            if args.target_successes is not None and successes >= args.target_successes:
                print(
                    f"Reached target_successes={args.target_successes} after {i+1} runs; stopping early."
                )
                break
            if args.time_limit_minutes is not None and elapsed >= args.time_limit_minutes * 60:
                print(
                    f"Time limit of {args.time_limit_minutes} minutes reached after {i+1} runs; stopping early."
                )
                break
    else:
        total_completed = 0
        total_duration = 0.0
        stop_early = False

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = set()

            def submit_one(job_id: int) -> None:
                futures.add(executor.submit(_run_single_simulation, job_id, snapshot_spec))

            for job_id in range(min(args.max_workers, args.n_samples)):
                submit_one(job_id)
            submitted = len(futures)

            while futures:
                completed_future = next(concurrent.futures.as_completed(futures))
                futures.remove(completed_future)
                row, duration = completed_future.result()

                total_completed += 1
                total_duration += duration
                append_design_result(row)
                if row["success"]:
                    successes += 1

                elapsed = time.time() - start_time
                avg_time = total_duration / total_completed
                remaining = max(args.n_samples - total_completed, 0)
                eta = remaining * avg_time

                status = (
                    f"[{total_completed}/{args.n_samples}] design_id={row['design_id']} "
                    f"success={row['success']} elapsed={elapsed:.1f}s runtime={duration:.1f}s"
                )
                if row.get("flow_path"):
                    status += f" snapshot={row['flow_path']}"
                if row.get("snapshot_error"):
                    status += f" snapshot_error={row['snapshot_error']}"
                if not row["success"] and row.get("error"):
                    status += f" error={row['error']}"
                if remaining:
                    status += f" eta={eta:.1f}s"
                status += f" workers={args.max_workers}"
                print(status)

                if args.target_successes is not None and successes >= args.target_successes:
                    print(
                        f"Reached target_successes={args.target_successes} after {total_completed} runs; stopping early."
                    )
                    stop_early = True
                if (
                    args.time_limit_minutes is not None
                    and elapsed >= args.time_limit_minutes * 60
                ):
                    print(
                        f"Time limit of {args.time_limit_minutes} minutes reached after {total_completed} runs; stopping early."
                    )
                    stop_early = True

                if stop_early:
                    for future in futures:
                        future.cancel()
                    break

                if submitted < args.n_samples:
                    submit_one(submitted)
                    submitted += 1


if __name__ == "__main__":
    main()

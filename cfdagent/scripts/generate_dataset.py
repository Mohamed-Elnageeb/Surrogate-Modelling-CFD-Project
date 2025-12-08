import argparse
import concurrent.futures
import logging
import time
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np

from ..geometry.airfoil_param import sample_random_design
from ..cfd.run_cfd import run_cfd
from ..utils.snapshot_pipeline import (
    SnapshotSpec,
    create_snapshot_from_run,
    relative_snapshot_path,
)
from ..utils.io_utils import append_design_result


logger = logging.getLogger(__name__)


def _create_snapshot(design_id: str, spec: SnapshotSpec, run_result: dict) -> Path:
    # Always delegate snapshot construction to the unified helper so that VTU volume
    # files and history.csv fallbacks are both supported. When surface forces are
    # missing, we explicitly pass the discovered history path (if any) so CL/CD can
    # still be recovered from the run output instead of failing with binary file
    # errors on VTU data.
    return create_snapshot_from_run(design_id, spec, run_result)


def _run_single_simulation(
    _: int, snapshot_spec: SnapshotSpec | None = None
) -> Tuple[dict | None, float, str | None]:
    """Run one CFD simulation and return its row dict and wall time.

    When lift/drag metrics are missing or negative, the run is discarded
    entirely to keep the dataset free from invalid samples.
    """

    start = time.time()
    design_vec = np.clip(sample_random_design(), -0.02, 0.02)
    design_id = str(uuid.uuid4())[:8]
    result = run_cfd(design_id, design_vec)

    if result.get("invalid_metrics") or not result.get("success"):
        duration = time.time() - start
        skip_reason = result.get("error") or "invalid_metrics"
        logger.warning("Skipping design %s: %s", design_id, skip_reason)
        return None, duration, skip_reason

    row = {"design_id": design_id}
    for idx in range(5):
        row[f"dc{idx+1}"] = float(design_vec[idx])
    for idx in range(5):
        row[f"dt{idx+1}"] = float(design_vec[5 + idx])
    row.update(
        {
            "Cl": result.get("Cl"),
            "Cd": result.get("Cd"),
            "success": result.get("success", False),
        }
    )
    if snapshot_spec and row["success"]:
        try:
            snapshot_path = _create_snapshot(design_id, snapshot_spec, result)
            row["flow_path"] = relative_snapshot_path(snapshot_path)
        except Exception as exc:  # pylint: disable=broad-except
            duration = time.time() - start
            skip_reason = f"snapshot failed: {exc}"
            logger.warning("Skipping design %s due to snapshot failure: %s", design_id, exc)
            return None, duration, skip_reason
    if not row["success"] and result.get("error"):
        row["error"] = result.get("error")

    duration = time.time() - start
    return row, duration, None


def main():
    parser = argparse.ArgumentParser(description="Generate CFD dataset samples")
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of random designs to simulate"
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
            row, duration, skip_reason = _run_single_simulation(i, snapshot_spec)
            if skip_reason:
                print(
                    f"[{i+1}/{args.n_samples}] discarded run (reason={skip_reason}) elapsed={time.time() - start_time:.1f}s"
                )
                continue

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
                row, duration, skip_reason = completed_future.result()

                if skip_reason:
                    total_completed += 1
                    total_duration += duration
                    elapsed = time.time() - start_time
                    avg_time = total_duration / total_completed
                    remaining = max(args.n_samples - total_completed, 0)
                    eta = remaining * avg_time
                    print(
                        f"[{total_completed}/{args.n_samples}] discarded run (reason={skip_reason}) elapsed={elapsed:.1f}s runtime={duration:.1f}s eta={eta:.1f}s workers={args.max_workers}"
                    )
                else:
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

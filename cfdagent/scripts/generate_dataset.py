import argparse
import concurrent.futures
import time
import uuid
from typing import Tuple

from ..geometry.airfoil_param import sample_random_design
from ..cfd.run_cfd import run_cfd
from ..utils.io_utils import append_design_result


def _run_single_simulation(_: int) -> Tuple[dict, float]:
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
    args = parser.parse_args()

    start_time = time.time()
    successes = 0

    if args.max_workers <= 1:
        for i in range(args.n_samples):
            row, duration = _run_single_simulation(i)
            append_design_result(row)
            if row["success"]:
                successes += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = max(args.n_samples - (i + 1), 0)
            eta = remaining * avg_time

            status = f"[{i+1}/{args.n_samples}] design_id={row['design_id']} success={row['success']}"
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
                futures.add(executor.submit(_run_single_simulation, job_id))

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

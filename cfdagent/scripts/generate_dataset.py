import argparse
import uuid

from ..geometry.airfoil_param import sample_random_design
from ..cfd.run_cfd import run_cfd
from ..utils.io_utils import append_design_result


def main():
    parser = argparse.ArgumentParser(description="Generate CFD dataset samples")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of random designs to simulate")
    args = parser.parse_args()

    for i in range(args.n_samples):
        design_vec = sample_random_design()
        design_id = str(uuid.uuid4())[:8]
        result = run_cfd(design_id, design_vec)

        row = {"design_id": design_id}
        for idx in range(5):
            row[f"dc{idx+1}"] = float(design_vec[idx])
        for idx in range(5):
            row[f"dt{idx+1}"] = float(design_vec[5 + idx])
        row.update({
            "Cl": result.get("Cl"),
            "Cd": result.get("Cd"),
            "residual": result.get("residual"),
            "success": result.get("success", False),
        })
        if not row["success"] and result.get("error"):
            row["error"] = result.get("error")

        append_design_result(row)
        status = f"[{i+1}/{args.n_samples}] design_id={design_id} success={row['success']}"
        if not row["success"] and result.get("error"):
            status += f" error={result['error']}"
        print(status)


if __name__ == "__main__":
    main()

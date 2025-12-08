"""Generate a structured O-grid mesh for a NACA0012 airfoil using SU2_GEO."""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    config_path = Path(__file__).resolve().parent / "mesh_config.cfg"
    try:
        subprocess.run(["SU2_GEO", str(config_path)], check=True)
    except FileNotFoundError:
        print("SU2_GEO not found. Ensure SU2 is installed and in your PATH.")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"SU2_GEO failed with return code {exc.returncode}.")
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())

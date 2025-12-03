from pathlib import Path
import pandas as pd

DESIGN_LOG = Path(__file__).resolve().parents[1] / "data" / "designs.csv"

def append_design_result(row_dict):
    """
    Append a single design result to the central CSV log.
    Ensures the directory exists and preserves headers when creating the file.
    """
    DESIGN_LOG.parent.mkdir(parents=True, exist_ok=True)

    if DESIGN_LOG.exists():
        df = pd.read_csv(DESIGN_LOG)
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([row_dict])

    df.to_csv(DESIGN_LOG, index=False)

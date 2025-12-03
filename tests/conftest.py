from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

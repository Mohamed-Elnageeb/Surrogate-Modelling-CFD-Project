from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CFDDataset(Dataset):
    """
    Dataset for CFD surrogate training.

    The loader is intentionally forgiving: it can operate on a structured
    ``designs.csv`` metadata file or, if that file is missing, it will scan a
    ``flowfields`` directory for ``.npz`` files. Optional geometry images are
    paired automatically when they share the same stem as the flowfield file.
    """

    def __init__(self, data_root: Path | str, split: str = "train", split_ratio: float = 0.8):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must be between 0 and 1")

        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")

        entries = self._load_entries()
        if not entries:
            raise ValueError(f"No CFD samples found under {self.data_root}")

        cutoff = max(1, int(len(entries) * split_ratio))
        if cutoff >= len(entries):
            cutoff = len(entries) - 1

        if split == "train":
            self.samples = entries[:cutoff]
        else:
            self.samples = entries[cutoff:]

        # If the split produced an empty list (e.g., only one sample), reuse the full set
        # to avoid surprising IndexErrors during experimentation.
        if not self.samples:
            self.samples = entries

    def _load_entries(self) -> List[Dict[str, Any]]:
        metadata = self.data_root / "designs.csv"
        entries: List[Dict[str, Any]] = []

        if metadata.exists():
            with metadata.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = (row.get("status") or "success").lower()
                    if status not in {"success", "ok", "completed", ""}:
                        continue

                    # Skip rows that explicitly reported a snapshot creation failure.
                    if row.get("snapshot_error"):
                        continue

                    flow_path = row.get("flow_path") or row.get("flowfile") or row.get("flow")
                    if flow_path is None:
                        continue
                    flow_path = self.data_root / flow_path
                    geometry_path = row.get("geometry_path") or row.get("image_path")
                    geometry_full = self.data_root / geometry_path if geometry_path else None

                    if flow_path.exists():
                        entries.append({"flow": flow_path, "geometry": geometry_full, "meta": row})

        if entries:
            return sorted(entries, key=lambda e: e["flow"].name)

        # Fallback: scan the flowfields directory or top-level for .npz files.
        flow_dir = self.data_root / "flowfields"
        candidate_dir = flow_dir if flow_dir.exists() else self.data_root
        for flow_file in sorted(candidate_dir.glob("*.npz")):
            entries.append({"flow": flow_file, "geometry": self._matching_image(flow_file), "meta": {}})

        return entries

    def _matching_image(self, flow_file: Path) -> Optional[Path]:
        stem = flow_file.stem
        for folder in (self.data_root / "geometry", self.data_root / "images", self.data_root):
            if folder.exists():
                for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                    candidate = folder / f"{stem}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        flow_tensor = self._load_flow(record["flow"])
        geom_tensor = self._load_geometry(record.get("geometry"))
        return {"flow": flow_tensor, "geometry": geom_tensor, "meta": record.get("meta", {})}

    def _load_flow(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Flowfield file not found: {path}")

        try:
            data = np.load(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load flowfield file {path}: {exc}") from exc
        if isinstance(data, np.lib.npyio.NpzFile):
            if "flow" in data:
                flow_array = data["flow"]
            else:
                first_key = data.files[0]
                flow_array = data[first_key]
        else:
            flow_array = data

        return torch.as_tensor(flow_array, dtype=torch.float32)

    def _load_geometry(self, path: Optional[Path]) -> torch.Tensor:
        if path is None or not path.exists():
            return torch.zeros(1, 1, dtype=torch.float32)

        try:
            from PIL import Image
        except ImportError:
            # Pillow is optional; if unavailable, return a placeholder tensor.
            return torch.zeros(1, 1, dtype=torch.float32)

        with Image.open(path) as img:
            img_arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(img_arr).unsqueeze(0)

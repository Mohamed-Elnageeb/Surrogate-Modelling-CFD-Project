from pathlib import Path
import torch
from torch.utils.data import Dataset


class CFDDataset(Dataset):
    """
    Dataset for CFD surrogate training.
    In future phases, this will load design metadata and flowfield arrays.
    """

    def __init__(self, data_root: Path, split: str = "train", split_ratio: float = 0.8):
        """
        Initialize dataset.
        Expected to:
          - load data/designs.csv
          - filter successful simulations
          - load corresponding .npz flowfield files and geometry images
          - split into train/validation using split_ratio
        """
        raise NotImplementedError("dataset loading not implemented yet")

    def __len__(self):
        raise NotImplementedError("dataset length not implemented yet")

    def __getitem__(self, idx):
        raise NotImplementedError("dataset item retrieval not implemented yet")

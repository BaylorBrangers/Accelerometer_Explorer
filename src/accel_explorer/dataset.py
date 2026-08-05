"""PyTorch Dataset helpers for windowed accelerometer tensors."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class AccelerometerWindowDataset(Dataset):
    """Windows shaped (channels, length) with optional integer labels."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        label_to_idx: dict[str, int] | None = None,
    ) -> None:
        if windows.ndim != 3:
            raise ValueError(f"Expected windows (N, C, L), got {windows.shape}")
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels: torch.Tensor | None = None
        self.label_to_idx = label_to_idx
        self.idx_to_label: dict[int, str] | None = None

        if labels is not None:
            if label_to_idx is None:
                unique = sorted({str(x) for x in labels})
                label_to_idx = {lab: i for i, lab in enumerate(unique)}
            self.label_to_idx = label_to_idx
            self.idx_to_label = {i: lab for lab, i in label_to_idx.items()}
            encoded = [label_to_idx[str(x)] for x in labels]
            self.labels = torch.as_tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        if self.labels is None:
            return x
        return x, self.labels[idx]

    @property
    def num_classes(self) -> int | None:
        if self.label_to_idx is None:
            return None
        return len(self.label_to_idx)

    @property
    def input_length(self) -> int:
        return int(self.windows.shape[-1])

    @property
    def num_channels(self) -> int:
        return int(self.windows.shape[1])


def train_val_split(
    dataset: Dataset,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    n_train = n - n_val
    if n_train < 1:
        n_train, n_val = n, 0
    generator = torch.Generator().manual_seed(seed)
    if n_val == 0:
        return Subset(dataset, list(range(n))), Subset(dataset, [])
    return random_split(dataset, [n_train, n_val], generator=generator)


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset | None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return train_loader, val_loader


def build_label_mapping(labels: Sequence[str]) -> dict[str, int]:
    unique = sorted({str(x) for x in labels})
    return {lab: i for i, lab in enumerate(unique)}

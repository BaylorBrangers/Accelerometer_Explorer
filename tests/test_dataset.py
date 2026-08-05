"""Tests for window datasets."""

from pathlib import Path

import numpy as np

from accel_explorer.dataset import AccelerometerWindowDataset, make_loaders, train_val_split
from accel_explorer.io import load_csv
from accel_explorer.preprocess import create_windows

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_accel.csv"


def test_create_windows_and_dataset():
    df = load_csv(SAMPLE)
    X, y, meta = create_windows(df, window_size=64, stride=32)
    assert X.ndim == 3
    assert X.shape[1] == 3
    assert X.shape[2] == 64
    assert y is not None
    assert len(meta) == len(X) == len(y)

    ds = AccelerometerWindowDataset(X, y)
    assert ds.num_classes >= 2
    assert ds.num_channels == 3
    x0, y0 = ds[0]
    assert tuple(x0.shape) == (3, 64)

    train_ds, val_ds = train_val_split(ds, val_fraction=0.2)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=8)
    batch = next(iter(train_loader))
    assert batch[0].shape[0] <= 8
    assert val_loader is not None

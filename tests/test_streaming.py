"""Tests for disk-backed streaming windows (no full DataFrame load)."""

from pathlib import Path

import torch

from accel_explorer.streaming import (
    StreamSource,
    discover_labels,
    iter_windows_from_file,
    make_streaming_loader,
    preview_rows,
)

ANOMARK = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_anomark_weardata.csv"
GENERIC = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_accel.csv"


def test_preview_and_windows_anomark():
    rows = preview_rows(ANOMARK, n=50)
    assert len(rows) == 50
    assert {"x", "y", "z"} <= set(rows[0])

    windows = list(iter_windows_from_file(ANOMARK, window_size=64, stride=32, kind="anomark"))
    assert len(windows) > 10
    w, lab = windows[0]
    assert w.shape == (3, 64)


def test_streaming_loader_generic_labels():
    labels = discover_labels([GENERIC], kind="generic")
    assert len(labels) >= 2
    loader = make_streaming_loader(
        [StreamSource(path=str(GENERIC), kind="generic")],
        window_size=64,
        stride=64,
        batch_size=8,
        label_to_idx=labels,
        max_windows=40,
        require_label=True,
    )
    batch = next(iter(loader))
    x, y = batch
    assert x.shape[0] <= 8
    assert x.shape[1:] == (3, 64)
    assert y.dtype == torch.long


def test_storage_format_size():
    from accel_explorer.storage.huggingface import format_size

    assert format_size(None) == "?"
    assert "KB" in format_size(2048) or "B" in format_size(100)

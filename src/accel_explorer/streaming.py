"""
Streaming readers for large accelerometer files.

Never loads an entire multi‑GB CSV into RAM. Reads line-by-line (or chunked),
keeps a small ring buffer, and emits fixed-length windows for training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from accel_explorer.formats.anomark import (
    is_anomark_weardata_header,
    parse_anomark_timestamp,
    _split_csv_line,
)


@dataclass
class StreamSource:
    """Pointer to on-disk data (local path or HF cache path)."""

    path: str
    kind: str = "auto"  # auto | anomark | generic
    label: str | None = None  # constant label if file-level


def detect_kind(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if is_anomark_weardata_header(header):
        return "anomark"
    return "generic"


def _parse_generic_header(header: str) -> dict[str, int]:
    cols = [c.strip().lower().strip('"') for c in header.strip().split(",")]
    idx: dict[str, int] = {}
    aliases = {
        "x": ("x", "acc_x", "accel_x", "ax"),
        "y": ("y", "acc_y", "accel_y", "ay"),
        "z": ("z", "acc_z", "accel_z", "az"),
        "label": ("label", "activity", "class", "behavior", "behaviour"),
        "timestamp": ("timestamp", "time", "t"),
    }
    for canon, names in aliases.items():
        for i, c in enumerate(cols):
            if c in names:
                idx[canon] = i
                break
    if not {"x", "y", "z"} <= set(idx):
        raise ValueError(f"Generic CSV missing x/y/z columns in header: {header.strip()!r}")
    return idx


def iter_samples(
    path: Path | str,
    *,
    kind: str = "auto",
    constant_label: str | None = None,
) -> Iterator[tuple[np.ndarray, str | None]]:
    """
    Yield (xyz float32[3], label|None) one sample at a time from disk.
    """
    path = Path(path)
    if kind == "auto":
        kind = detect_kind(path)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if kind == "anomark":
            yield from _iter_anomark(f, constant_label=constant_label)
        else:
            col_idx = _parse_generic_header(header)
            yield from _iter_generic(f, col_idx, constant_label=constant_label)


def _iter_anomark(f: TextIO, *, constant_label: str | None) -> Iterator[tuple[np.ndarray, str | None]]:
    for line in f:
        if not line.strip():
            continue
        parts = _split_csv_line(line)
        if len(parts) < 5:
            continue
        try:
            xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
        except ValueError:
            continue
        yield xyz, constant_label


def _iter_generic(
    f: TextIO,
    col_idx: dict[str, int],
    *,
    constant_label: str | None,
) -> Iterator[tuple[np.ndarray, str | None]]:
    xi, yi, zi = col_idx["x"], col_idx["y"], col_idx["z"]
    li = col_idx.get("label")
    for line in f:
        if not line.strip():
            continue
        parts = _split_csv_line(line)
        try:
            xyz = np.array([float(parts[xi]), float(parts[yi]), float(parts[zi])], dtype=np.float32)
        except (ValueError, IndexError):
            continue
        label = constant_label
        if li is not None and li < len(parts) and parts[li]:
            label = parts[li]
        yield xyz, label


def iter_windows_from_file(
    path: Path | str,
    *,
    window_size: int,
    stride: int,
    kind: str = "auto",
    constant_label: str | None = None,
    label_to_idx: dict[str, int] | None = None,
) -> Iterator[tuple[np.ndarray, int | None]]:
    """
    Sliding windows over a file stream.

    Yields (window[3, L], label_idx|None). Majority vote label within the window
    when per-sample labels exist.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    buf: deque[np.ndarray] = deque(maxlen=window_size)
    lab_buf: deque[str | None] = deque(maxlen=window_size)
    seen = 0

    for xyz, lab in iter_samples(path, kind=kind, constant_label=constant_label):
        buf.append(xyz)
        lab_buf.append(lab)
        seen += 1
        if len(buf) < window_size:
            continue
        # emit when the newest sample completes a stride-aligned window
        if (seen - window_size) % stride != 0:
            continue
        window = np.stack(buf, axis=1).astype(np.float32)  # (3, L)
        label_idx = None
        labels = [x for x in lab_buf if x is not None]
        if labels and label_to_idx is not None:
            # majority vote
            values, counts = np.unique(np.array(labels, dtype=object), return_counts=True)
            maj = str(values[int(np.argmax(counts))])
            if maj in label_to_idx:
                label_idx = label_to_idx[maj]
        elif labels and label_to_idx is None:
            # caller will remap later; encode as None and keep string via side channel unused
            label_idx = None
        yield window, label_idx


def preview_rows(path: Path | str, *, n: int = 200, kind: str = "auto") -> list[dict]:
    """Peek at the first n samples without loading the whole file."""
    rows: list[dict] = []
    for i, (xyz, lab) in enumerate(iter_samples(path, kind=kind)):
        rows.append({"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]), "label": lab})
        if i + 1 >= n:
            break
    return rows


def discover_labels(
    paths: list[Path | str],
    *,
    kind: str = "auto",
    max_rows_per_file: int = 200_000,
) -> dict[str, int]:
    """Scan (bounded) stream to build a label vocabulary without full load."""
    found: set[str] = set()
    for path in paths:
        count = 0
        for _, lab in iter_samples(path, kind=kind):
            if lab is not None:
                found.add(str(lab))
            count += 1
            if count >= max_rows_per_file:
                break
    return {lab: i for i, lab in enumerate(sorted(found))}


class StreamingWindowDataset(IterableDataset):
    """
    PyTorch IterableDataset over one or more on-disk accelerometer files.

    Designed for Hub-cached ~60GB corpora: each worker streams from disk.
    """

    def __init__(
        self,
        sources: list[StreamSource],
        *,
        window_size: int = 64,
        stride: int = 32,
        label_to_idx: dict[str, int] | None = None,
        max_windows: int | None = None,
        require_label: bool = False,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.window_size = window_size
        self.stride = stride
        self.label_to_idx = label_to_idx or {}
        self.max_windows = max_windows
        self.require_label = require_label

    def __iter__(self):
        emitted = 0
        for src in self.sources:
            kind = src.kind
            for window, label_idx in iter_windows_from_file(
                src.path,
                window_size=self.window_size,
                stride=self.stride,
                kind=kind,
                constant_label=src.label,
                label_to_idx=self.label_to_idx or None,
            ):
                x = torch.from_numpy(window)
                if self.require_label:
                    if label_idx is None and src.label and src.label in self.label_to_idx:
                        label_idx = self.label_to_idx[src.label]
                    if label_idx is None:
                        continue
                    yield x, int(label_idx)
                else:
                    if label_idx is None:
                        yield x
                    else:
                        yield x, int(label_idx)
                emitted += 1
                if self.max_windows is not None and emitted >= self.max_windows:
                    return


def make_streaming_loader(
    sources: list[StreamSource],
    *,
    window_size: int,
    stride: int,
    batch_size: int = 32,
    label_to_idx: dict[str, int] | None = None,
    max_windows: int | None = None,
    require_label: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = StreamingWindowDataset(
        sources,
        window_size=window_size,
        stride=stride,
        label_to_idx=label_to_idx,
        max_windows=max_windows,
        require_label=require_label,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

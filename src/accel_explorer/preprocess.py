"""Cleaning, optional resampling, and sliding windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from accel_explorer.config import CHANNELS, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STRIDE


def clean_dataframe(df: pd.DataFrame, *, interpolate: bool = True) -> pd.DataFrame:
    """Drop empty rows and fill NaNs on accel channels."""
    out = df.copy()
    cols = [c for c in CHANNELS if c in out.columns]
    if interpolate:
        out[cols] = out[cols].interpolate(limit_direction="both")
    out = out.dropna(subset=cols)
    return out.reset_index(drop=True)


def resample_dataframe(
    df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample irregular timestamps to a fixed rate (linear interp on channels)."""
    if timestamp_col not in df.columns:
        raise ValueError("timestamp column required for resampling")
    work = df.copy()
    ts = pd.to_numeric(work[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError("timestamp column could not be parsed as numeric")
    work[timestamp_col] = ts
    work = work.sort_values(timestamp_col).drop_duplicates(timestamp_col)
    t0, t1 = float(work[timestamp_col].iloc[0]), float(work[timestamp_col].iloc[-1])
    new_t = np.arange(t0, t1, 1.0 / sample_rate_hz)
    if len(new_t) < 2:
        return work.reset_index(drop=True)

    out: dict[str, Any] = {timestamp_col: new_t}
    for col in CHANNELS:
        out[col] = np.interp(new_t, work[timestamp_col].to_numpy(), work[col].to_numpy())
    if "label" in work.columns:
        # nearest-neighbor label assignment
        idx = np.searchsorted(work[timestamp_col].to_numpy(), new_t, side="right") - 1
        idx = np.clip(idx, 0, len(work) - 1)
        out["label"] = work["label"].to_numpy()[idx]
    if "source_file" in work.columns:
        out["source_file"] = work["source_file"].iloc[0]
    return pd.DataFrame(out)


def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["magnitude"] = np.sqrt(out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2)
    return out


def create_windows(
    df: pd.DataFrame,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:
    """
    Slice the signal into overlapping windows.

    Returns
    -------
    X : ndarray, shape (n_windows, 3, window_size)
    y : ndarray of labels (majority vote) or None
    meta : per-window metadata (start/end index, source)
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    work = clean_dataframe(df)
    signal = work[list(CHANNELS)].to_numpy(dtype=np.float32)
    n = len(signal)
    if n < window_size:
        raise ValueError(f"Need at least {window_size} samples, got {n}")

    labels = work["label"].astype(str).to_numpy() if "label" in work.columns else None
    sources = (
        work["source_file"].astype(str).to_numpy()
        if "source_file" in work.columns
        else np.array(["unknown"] * n)
    )

    windows: list[np.ndarray] = []
    y_list: list[str] = []
    meta: list[dict] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        chunk = signal[start:end].T  # (3, window_size)
        windows.append(chunk)
        info = {
            "start_idx": start,
            "end_idx": end,
            "source_file": str(sources[start]),
        }
        if labels is not None:
            vals, counts = np.unique(labels[start:end], return_counts=True)
            lab = str(vals[int(np.argmax(counts))])
            y_list.append(lab)
            info["label"] = lab
        meta.append(info)

    X = np.stack(windows, axis=0)
    y = np.array(y_list) if labels is not None else None
    return X, y, meta

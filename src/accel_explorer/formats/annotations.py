"""Behavior annotation loaders and timestamp alignment."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO, Union

import numpy as np
import pandas as pd

from accel_explorer.config import COLUMN_ALIASES

PathLike = Union[str, Path]
FileLike = Union[BinaryIO, TextIO, bytes]

START_ALIASES = (
    "start",
    "start_time",
    "start_timestamp",
    "t_start",
    "begin",
    "onset",
    "from",
    "start_s",
    "start_sec",
    "start_us",
    "start_microseconds",
)
END_ALIASES = (
    "end",
    "end_time",
    "end_timestamp",
    "t_end",
    "finish",
    "offset",
    "to",
    "end_s",
    "end_sec",
    "end_us",
    "end_microseconds",
)
LABEL_ALIASES = COLUMN_ALIASES["label"] + (
    "behavior",
    "behaviour",
    "event",
    "annotation",
    "ethogram",
)


def _lower_map(columns) -> dict[str, str]:
    return {str(c).lower().strip(): c for c in columns}


def _pick(columns_lower: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns_lower:
            return columns_lower[alias]
    return None


def _to_seconds(series: pd.Series, *, unit_hint: str | None = None) -> pd.Series:
    """Convert numeric or datetime-like times to seconds."""
    if np.issubdtype(series.dtype, np.datetime64):
        return (series.view("int64") - series.view("int64").iloc[0]) / 1e9

    numeric = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    if numeric.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            base = parsed.dropna().iloc[0]
            return (parsed - base).dt.total_seconds()
        raise ValueError(f"Could not parse annotation times from column values like {series.iloc[0]!r}")

    med = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
    if unit_hint == "us" or med > 1e10:
        return numeric / 1_000_000.0
    if unit_hint == "ms" or med > 1e8:
        return numeric / 1_000.0
    return numeric


def load_behavior_annotations(source: PathLike | FileLike) -> pd.DataFrame:
    """
    Load an interval annotation CSV with start/end/label columns.

    Accepts common aliases (start_time/end_time/behavior, etc.). Times are
    normalized to seconds. Raise a clear error if the schema is unrecognized
    so callers can request the real Anomark annotation layout.
    """
    df = pd.read_csv(source)
    lower = _lower_map(df.columns)
    start_col = _pick(lower, START_ALIASES)
    end_col = _pick(lower, END_ALIASES)
    label_col = _pick(lower, LABEL_ALIASES)

    if not start_col or not label_col:
        raise ValueError(
            "Unrecognized behavior annotation schema. Expected columns like "
            "start/end/label (or start_time, end_time, behavior). "
            f"Found: {list(df.columns)}"
        )

    unit_hint = None
    if "us" in start_col.lower() or "micro" in start_col.lower():
        unit_hint = "us"
    elif start_col.lower().endswith("_ms") or "millis" in start_col.lower():
        unit_hint = "ms"

    out = pd.DataFrame(
        {
            "start": _to_seconds(df[start_col], unit_hint=unit_hint),
            "label": df[label_col].astype(str),
        }
    )
    if end_col:
        out["end"] = _to_seconds(df[end_col], unit_hint=unit_hint)
    else:
        # Point events: treat as instantaneous labels spanning to next start.
        starts = out["start"].to_numpy()
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = starts[-1] + 1.0
        out["end"] = ends

    out = out.dropna(subset=["start", "end", "label"]).sort_values("start").reset_index(drop=True)
    if (out["end"] < out["start"]).any():
        raise ValueError("Annotation intervals contain end < start")
    return out


def apply_annotations(signal: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Label each signal row whose timestamp falls inside an annotation interval."""
    if "timestamp" not in signal.columns:
        raise ValueError("Signal dataframe needs a timestamp column")
    out = signal.copy()
    labels = np.array([None] * len(out), dtype=object)
    ts = out["timestamp"].to_numpy(dtype=float)
    for _, ann in annotations.iterrows():
        mask = (ts >= float(ann["start"])) & (ts < float(ann["end"]))
        labels[mask] = ann["label"]
    out["label"] = labels
    return out

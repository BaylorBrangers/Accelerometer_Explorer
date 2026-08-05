"""Load accelerometer CSV from uploads, paths, or directories."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd

from accel_explorer.config import COLUMN_ALIASES
from accel_explorer.formats.anomark import load_anomark_weardata, looks_like_anomark_weardata
from accel_explorer.formats.annotations import apply_annotations, load_behavior_annotations

PathLike = Union[str, Path]
FileLike = Union[BinaryIO, TextIO, BytesIO, StringIO, bytes]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                rename[lower_map[alias]] = canonical
                break
    out = df.rename(columns=rename)
    required = {"x", "y", "z"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Missing required accelerometer columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    if "timestamp" not in out.columns:
        out = out.copy()
        out.insert(0, "timestamp", range(len(out)))
    return out


def load_csv(source: PathLike | FileLike, *, source_name: str | None = None) -> pd.DataFrame:
    """Load a single CSV into a normalized DataFrame (generic or Anomark weardata)."""
    name: str
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = source_name or path.name
        if looks_like_anomark_weardata(path):
            return load_anomark_weardata(path, source_name=name)
        df = pd.read_csv(path)
    elif isinstance(source, bytes):
        name = source_name or "upload.csv"
        if looks_like_anomark_weardata(source):
            return load_anomark_weardata(source, source_name=name)
        df = pd.read_csv(BytesIO(source))
    else:
        name = source_name or getattr(source, "name", "upload.csv")
        # Peek without consuming if possible
        raw = source.read() if hasattr(source, "read") else None
        if raw is None:
            raise TypeError(f"Unsupported file-like source: {type(source)}")
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = raw
        if looks_like_anomark_weardata(raw_bytes):
            return load_anomark_weardata(raw_bytes, source_name=Path(name).name)
        df = pd.read_csv(BytesIO(raw_bytes))

    df = _normalize_columns(df)
    df = df.copy()
    df["source_file"] = Path(name).name
    return df


def load_path(path: PathLike) -> pd.DataFrame:
    """Load a CSV file or all CSVs under a directory (concatenated)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_file():
        return load_csv(p)
    csv_files = sorted(p.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {p}")
    frames = [load_csv(f) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def load_labeled_recording(
    signal_source: PathLike | FileLike,
    annotation_source: PathLike | FileLike,
    *,
    signal_name: str | None = None,
) -> pd.DataFrame:
    """Load accelerometer data and merge interval behavior annotations by timestamp."""
    signal = load_csv(signal_source, source_name=signal_name)
    annotations = load_behavior_annotations(annotation_source)
    return apply_annotations(signal, annotations)


def dataframe_summary(df: pd.DataFrame) -> dict:
    """Compact summary for UI preview."""
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sources": sorted(df["source_file"].unique().tolist()) if "source_file" in df else [],
        "has_label": bool("label" in df.columns and df["label"].notna().any()),
    }
    if summary["has_label"]:
        summary["labels"] = sorted(df["label"].dropna().astype(str).unique().tolist())
    return summary

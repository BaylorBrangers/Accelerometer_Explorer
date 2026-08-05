"""Load accelerometer CSV from uploads, paths, or directories."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd

from accel_explorer.config import COLUMN_ALIASES

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
    """Load a single CSV into a normalized DataFrame."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        df = pd.read_csv(path)
        name = source_name or path.name
    elif isinstance(source, bytes):
        df = pd.read_csv(BytesIO(source))
        name = source_name or "upload.csv"
    else:
        df = pd.read_csv(source)
        name = source_name or getattr(source, "name", "upload.csv")

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


def dataframe_summary(df: pd.DataFrame) -> dict:
    """Compact summary for UI preview."""
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sources": sorted(df["source_file"].unique().tolist()) if "source_file" in df else [],
        "has_label": "label" in df.columns,
    }
    if "label" in df.columns:
        summary["labels"] = sorted(df["label"].astype(str).unique().tolist())
    return summary

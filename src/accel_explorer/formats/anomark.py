"""Parsers for Anomark wearable accelerometer exports."""

from __future__ import annotations

from io import StringIO, TextIOBase
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
FileLike = Union[BinaryIO, TextIO, bytes]

# Header seen in 170531Anomark_acc_weardata.csv (short header; rows have more fields).
ANOMARK_WEARDATA_MARKERS = ("registeraddress", "timestampmicroseconds", "dataelement0")

# After timestamp: accel XYZ, then typically gyro XYZ, then a third triad (mag/aux).
CHANNEL_NAMES = (
    "x",
    "y",
    "z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "aux_0",
    "aux_1",
    "aux_2",
)


def is_anomark_weardata_header(header_line: str) -> bool:
    cols = [c.strip().lower().strip('"') for c in header_line.strip().split(",")]
    return all(m in cols for m in ANOMARK_WEARDATA_MARKERS)


def _split_csv_line(line: str) -> list[str]:
    """Split a CSV line, stripping quotes and preserving commas inside quotes."""
    parts: list[str] = []
    cur = ""
    in_quotes = False
    for ch in line.rstrip("\n\r"):
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if ch == "," and not in_quotes:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def parse_anomark_timestamp(raw: str) -> float:
    """
    Parse Anomark TimestampMicroseconds values.

    Exports often use thousand separators inside quotes, e.g. ``\"304,607500\"``.
    Returns seconds (float) from the raw microsecond counter.
    """
    cleaned = raw.strip().strip('"').replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError(f"Empty Anomark timestamp: {raw!r}")
    return float(cleaned) / 1_000_000.0


def _read_text_preview(source: PathLike | FileLike, *, n_chars: int = 512) -> tuple[str, str | None]:
    """Return (preview_text, resolved_name)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            preview = f.read(n_chars)
        return preview, path.name
    if isinstance(source, bytes):
        return source[:n_chars].decode("utf-8", errors="replace"), None
    # file-like
    name = getattr(source, "name", None)
    if hasattr(source, "seek") and hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        data = source.read(n_chars)
        if isinstance(data, bytes):
            preview = data.decode("utf-8", errors="replace")
            if pos is not None:
                source.seek(pos)
            return preview, Path(name).name if name else None
        preview = data
        if pos is not None:
            source.seek(pos)
        return preview, Path(name).name if name else None
    raise TypeError(f"Unsupported source type: {type(source)}")


def looks_like_anomark_weardata(source: PathLike | FileLike) -> bool:
    preview, _ = _read_text_preview(source)
    first_line = preview.splitlines()[0] if preview else ""
    return is_anomark_weardata_header(first_line)


def _iter_lines(source: PathLike | FileLike):
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8", errors="replace") as f:
            yield from f
        return
    if isinstance(source, bytes):
        yield from StringIO(source.decode("utf-8", errors="replace"))
        return
    if isinstance(source, TextIOBase) or hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if isinstance(data, bytes):
            yield from StringIO(data.decode("utf-8", errors="replace"))
        else:
            yield from StringIO(data)
        return
    raise TypeError(f"Unsupported source type: {type(source)}")


def load_anomark_weardata(
    source: PathLike | FileLike,
    *,
    source_name: str | None = None,
    register_filter: int | None = 34,
) -> pd.DataFrame:
    """
    Load an Anomark ``*_acc_weardata.csv`` export into a normalized frame.

    Output columns: ``timestamp`` (seconds), ``x``, ``y``, ``z``, optional
    ``gyro_*`` / ``aux_*``, ``register_address``, ``source_file``.
    """
    if source_name is None:
        if isinstance(source, (str, Path)):
            source_name = Path(source).name
        else:
            source_name = Path(getattr(source, "name", "anomark_weardata.csv")).name

    lines = _iter_lines(source)
    try:
        header = next(lines)
    except StopIteration as exc:
        raise ValueError("Empty Anomark weardata file") from exc
    if not is_anomark_weardata_header(header):
        raise ValueError(
            "Not an Anomark weardata CSV "
            f"(expected RegisterAddress,TimestampMicroseconds,DataElement0). Got: {header.strip()!r}"
        )

    rows: list[dict] = []
    for line_no, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        parts = _split_csv_line(line)
        if len(parts) < 5:
            continue
        try:
            register = int(float(parts[0]))
            if register_filter is not None and register != register_filter:
                continue
            ts = parse_anomark_timestamp(parts[1])
            values = [float(v) for v in parts[2:]]
        except ValueError:
            continue

        row: dict = {
            "timestamp": ts,
            "register_address": register,
        }
        for i, name in enumerate(CHANNEL_NAMES):
            if i < len(values):
                row[name] = values[i]
        # Ensure accel channels exist even if truncated
        for axis in ("x", "y", "z"):
            row.setdefault(axis, np.nan)
        rows.append(row)

    if not rows:
        raise ValueError("No usable Anomark weardata rows parsed")

    df = pd.DataFrame(rows)
    df["source_file"] = source_name
    # Prefer canonical column order
    ordered = [
        "timestamp",
        "x",
        "y",
        "z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "aux_0",
        "aux_1",
        "aux_2",
        "register_address",
        "source_file",
    ]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[cols]

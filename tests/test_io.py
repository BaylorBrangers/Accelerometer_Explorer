"""Tests for CSV IO."""

from pathlib import Path

import pandas as pd
import pytest

from accel_explorer.io import dataframe_summary, load_csv, load_path

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_accel.csv"


def test_load_sample_csv():
    df = load_csv(SAMPLE)
    assert {"timestamp", "x", "y", "z", "label", "source_file"} <= set(df.columns)
    assert len(df) > 100


def test_load_bytes_and_aliases(tmp_path: Path):
    raw = "time,acc_x,acc_y,acc_z,activity\n0,1,0,0,sit\n1,0,1,0,walk\n"
    df = load_csv(raw.encode("utf-8"), source_name="alias.csv")
    assert list(df[["x", "y", "z", "label"]].columns) == ["x", "y", "z", "label"]
    assert df["source_file"].iloc[0] == "alias.csv"


def test_load_directory(tmp_path: Path):
    df = load_path(SAMPLE.parent)
    summary = dataframe_summary(df)
    assert summary["rows"] >= len(pd.read_csv(SAMPLE))
    assert summary["has_label"] is True


def test_missing_columns_raises():
    with pytest.raises(ValueError, match="Missing required"):
        load_csv(b"a,b\n1,2\n")

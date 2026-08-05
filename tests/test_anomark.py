"""Tests for Anomark weardata and annotation merging."""

from pathlib import Path

import pandas as pd

from accel_explorer.formats.annotations import apply_annotations, load_behavior_annotations
from accel_explorer.formats.anomark import load_anomark_weardata, looks_like_anomark_weardata
from accel_explorer.io import load_csv, load_labeled_recording

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_anomark_weardata.csv"


def test_detect_and_load_anomark_sample():
    assert looks_like_anomark_weardata(SAMPLE)
    df = load_csv(SAMPLE)
    assert {"timestamp", "x", "y", "z", "source_file"} <= set(df.columns)
    assert len(df) > 100
    assert df["timestamp"].is_monotonic_increasing or df["timestamp"].iloc[-1] > df["timestamp"].iloc[0]
    # Accel channels populated
    assert df[["x", "y", "z"]].notna().all().all()


def test_load_anomark_preserves_extra_channels():
    df = load_anomark_weardata(SAMPLE)
    assert "gyro_x" in df.columns
    assert "aux_0" in df.columns


def test_annotation_merge(tmp_path: Path):
    signal = load_csv(SAMPLE)
    t0 = float(signal["timestamp"].iloc[0])
    t1 = float(signal["timestamp"].iloc[len(signal) // 3])
    t2 = float(signal["timestamp"].iloc[2 * len(signal) // 3])
    t3 = float(signal["timestamp"].iloc[-1]) + 0.1
    ann = pd.DataFrame(
        {
            "start_time": [t0, t1, t2],
            "end_time": [t1, t2, t3],
            "behavior": ["rest", "walk", "graze"],
        }
    )
    ann_path = tmp_path / "ann.csv"
    ann.to_csv(ann_path, index=False)

    loaded = load_behavior_annotations(ann_path)
    assert list(loaded.columns) == ["start", "label", "end"] or set(loaded.columns) >= {
        "start",
        "end",
        "label",
    }

    labeled = load_labeled_recording(SAMPLE, ann_path)
    assert "label" in labeled.columns
    assert set(labeled["label"].dropna().unique()) <= {"rest", "walk", "graze"}
    assert labeled["label"].notna().mean() > 0.9

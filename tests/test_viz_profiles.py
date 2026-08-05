"""Tests for behavior profile visualization helpers."""

from pathlib import Path

from accel_explorer.io import load_csv
from accel_explorer.preprocess import clean_dataframe
from accel_explorer.viz.profiles import (
    behavior_channel_boxplot_figure,
    behavior_counts,
    behavior_segment_overlay_figure,
    behavior_stats_table,
    behavior_timeline_figure,
    mean_behavior_profile_figure,
    multi_axis_behavior_figure,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_accel.csv"


def test_behavior_profile_figures():
    df = clean_dataframe(load_csv(SAMPLE))
    counts = behavior_counts(df)
    assert len(counts) >= 2
    assert counts["samples"].sum() == len(df)

    assert behavior_timeline_figure(df).data
    assert behavior_segment_overlay_figure(df, channel="magnitude").data
    assert mean_behavior_profile_figure(df, window_size=64, channel="magnitude").data
    assert behavior_channel_boxplot_figure(df).data
    assert multi_axis_behavior_figure(df, behaviors=counts["behavior"].tolist()[:2]).data

    stats = behavior_stats_table(df)
    assert "behavior" in stats.columns
    assert "magnitude_rms" in stats.columns

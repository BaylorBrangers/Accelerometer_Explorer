"""
CLI for behavior / accelerometer profile plots.

Examples
--------
python -m accel_explorer.viz.cli data/samples/sample_accel.csv --out artifacts/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accel_explorer.io import load_csv, load_labeled_recording
from accel_explorer.preprocess import clean_dataframe
from accel_explorer.viz.profiles import (
    behavior_channel_boxplot_figure,
    behavior_segment_overlay_figure,
    behavior_stats_table,
    behavior_timeline_figure,
    mean_behavior_profile_figure,
    multi_axis_behavior_figure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot behavior accelerometer profiles")
    parser.add_argument("signal", type=Path, help="Accelerometer CSV (generic or Anomark)")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional interval annotation CSV (start/end/label)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/plots"), help="Output directory")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--channel", default="magnitude", choices=["x", "y", "z", "magnitude"])
    args = parser.parse_args(argv)

    if args.annotations:
        df = load_labeled_recording(args.signal, args.annotations)
    else:
        df = load_csv(args.signal)
    df = clean_dataframe(df)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    figures = {
        "timeline": behavior_timeline_figure(df),
        "segments": behavior_segment_overlay_figure(df, channel=args.channel),
        "mean_profile": mean_behavior_profile_figure(
            df, window_size=args.window_size, channel=args.channel
        ),
        "boxplots": behavior_channel_boxplot_figure(df),
        "multi_axis": multi_axis_behavior_figure(df),
    }
    for name, fig in figures.items():
        target = out / f"{name}.html"
        fig.write_html(str(target), include_plotlyjs="cdn")
        print(f"Wrote {target}")

    stats = behavior_stats_table(df)
    stats_path = out / "behavior_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

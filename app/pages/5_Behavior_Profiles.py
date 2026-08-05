"""Visualize accelerometer profiles grouped by annotated behaviors."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.preprocess import clean_dataframe  # noqa: E402
from accel_explorer.viz.profiles import (  # noqa: E402
    behavior_channel_boxplot_figure,
    behavior_counts,
    behavior_segment_overlay_figure,
    behavior_stats_table,
    behavior_timeline_figure,
    mean_behavior_profile_figure,
    multi_axis_behavior_figure,
)
from session_state import SESSION_WINDOW_KEY, init_defaults, require_dataframe  # noqa: E402

st.set_page_config(page_title="Behavior Profiles", layout="wide")
init_defaults()
df = require_dataframe()

st.title("Behavior Profiles")
st.write(
    "Explore how accelerometer signals differ across annotated behaviors — "
    "timelines, segment overlays, mean window profiles, and channel distributions."
)

hf_src = st.session_state.get("data_source")
if hf_src and hf_src.get("type") == "hf_hub":
    st.info(
        "Using the Hub **preview** frame in session. For full-corpus plots, load a labeled "
        "subset via Upload Data or run the CLI on cached files."
    )

clean = clean_dataframe(df)

try:
    counts = behavior_counts(clean)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader("Behavior coverage")
c1, c2 = st.columns([1, 2])
c1.dataframe(counts, use_container_width=True)
c2.bar_chart(counts.set_index("behavior"))

behaviors = counts["behavior"].tolist()
selected = st.multiselect("Behaviors to highlight", behaviors, default=behaviors[: min(4, len(behaviors))])
channel = st.selectbox("Profile channel", ["magnitude", "x", "y", "z"])
window_size = st.number_input(
    "Window size for mean profiles",
    min_value=8,
    max_value=2048,
    value=int(st.session_state.get(SESSION_WINDOW_KEY, 64)),
    step=8,
)

st.subheader("Timeline")
st.plotly_chart(behavior_timeline_figure(clean), use_container_width=True)

st.subheader("Per-behavior axis profiles")
if selected:
    st.plotly_chart(multi_axis_behavior_figure(clean, behaviors=selected), use_container_width=True)

st.subheader("Segment shape overlays")
st.plotly_chart(
    behavior_segment_overlay_figure(clean, channel=channel),
    use_container_width=True,
)

st.subheader("Mean window profile ± std")
try:
    st.plotly_chart(
        mean_behavior_profile_figure(clean, window_size=int(window_size), channel=channel),
        use_container_width=True,
    )
except ValueError as exc:
    st.warning(str(exc))

st.subheader("Channel distributions")
st.plotly_chart(behavior_channel_boxplot_figure(clean), use_container_width=True)

st.subheader("Summary table")
stats = behavior_stats_table(clean)
st.dataframe(stats, use_container_width=True)
st.download_button(
    "Download behavior stats CSV",
    stats.to_csv(index=False).encode("utf-8"),
    file_name="behavior_stats.csv",
    mime="text/csv",
)

st.caption(
    "CLI alternative: `python -m accel_explorer.viz.cli data/samples/sample_accel.csv --out artifacts/plots`"
)

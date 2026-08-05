"""Explore accelerometer signals and choose windowing."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.analyze import signal_figure, summary_stats  # noqa: E402
from accel_explorer.preprocess import clean_dataframe, create_windows  # noqa: E402
from session_state import (  # noqa: E402
    SESSION_STRIDE_KEY,
    SESSION_WINDOW_KEY,
    init_defaults,
    require_dataframe,
)

st.set_page_config(page_title="Explore", layout="wide")
init_defaults()
df = require_dataframe()

st.title("Explore")
st.write("Inspect the loaded signal and configure sliding windows for training.")

hf_src = st.session_state.get("data_source")
if hf_src and hf_src.get("type") == "hf_hub":
    st.info(
        f"Showing a **preview** of Hub dataset `{hf_src['repo_id']}`. "
        "Full files remain on disk in the HF cache; Train streams from those paths."
    )

c1, c2 = st.columns(2)
window_size = c1.number_input(
    "Window size (samples)",
    min_value=8,
    max_value=2048,
    value=int(st.session_state[SESSION_WINDOW_KEY]),
    step=8,
)
stride = c2.number_input(
    "Stride (samples)",
    min_value=1,
    max_value=2048,
    value=int(st.session_state[SESSION_STRIDE_KEY]),
    step=1,
)
st.session_state[SESSION_WINDOW_KEY] = int(window_size)
st.session_state[SESSION_STRIDE_KEY] = int(stride)

clean = clean_dataframe(df)
st.plotly_chart(signal_figure(clean), use_container_width=True)
st.subheader("Channel stats")
st.dataframe(summary_stats(clean), use_container_width=True)

try:
    X, y, meta = create_windows(clean, window_size=int(window_size), stride=int(stride))
    st.success(f"Windows: {X.shape[0]} × shape {tuple(X.shape[1:])}")
    if y is not None:
        import pandas as pd

        st.write(pd.Series(y).value_counts().rename("count"))
except ValueError as exc:
    st.warning(str(exc))

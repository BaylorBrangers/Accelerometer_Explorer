"""Shared Streamlit session helpers."""

from __future__ import annotations

import streamlit as st

SESSION_DF_KEY = "accel_df"
SESSION_WINDOW_KEY = "window_size"
SESSION_STRIDE_KEY = "window_stride"
SESSION_TRAIN_RESULT_KEY = "train_result"
SESSION_CHECKPOINT_KEY = "checkpoint_path"


def require_dataframe():
    df = st.session_state.get(SESSION_DF_KEY)
    if df is None or len(df) == 0:
        st.warning("Load accelerometer data on **Upload Data** first.")
        st.stop()
    return df


def init_defaults() -> None:
    st.session_state.setdefault(SESSION_WINDOW_KEY, 64)
    st.session_state.setdefault(SESSION_STRIDE_KEY, 32)

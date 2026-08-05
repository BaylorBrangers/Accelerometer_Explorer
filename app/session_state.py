"""Shared Streamlit session helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

SESSION_DF_KEY = "accel_df"
SESSION_DATA_SOURCE_KEY = "data_source"  # local_df | hf_hub stream config
SESSION_WINDOW_KEY = "window_size"
SESSION_STRIDE_KEY = "window_stride"
SESSION_TRAIN_RESULT_KEY = "train_result"
SESSION_CHECKPOINT_KEY = "checkpoint_path"


def init_defaults() -> None:
    st.session_state.setdefault(SESSION_WINDOW_KEY, 64)
    st.session_state.setdefault(SESSION_STRIDE_KEY, 32)


def require_dataframe():
    df = st.session_state.get(SESSION_DF_KEY)
    if df is None or len(df) == 0:
        st.warning("Load accelerometer data on **Upload Data** or connect a Hub dataset first.")
        st.stop()
    return df


def get_data_source() -> dict[str, Any] | None:
    return st.session_state.get(SESSION_DATA_SOURCE_KEY)


def set_hf_data_source(
    *,
    repo_id: str,
    files: list[str],
    local_paths: list[str],
    repo_type: str = "dataset",
    revision: str | None = None,
    kind: str = "auto",
) -> None:
    st.session_state[SESSION_DATA_SOURCE_KEY] = {
        "type": "hf_hub",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "files": files,
        "local_paths": local_paths,
        "kind": kind,
    }


def require_trainable_source() -> dict[str, Any]:
    """Return either an hf_hub source or a local_df wrapper."""
    src = get_data_source()
    if src and src.get("type") == "hf_hub" and src.get("local_paths"):
        return src
    df = st.session_state.get(SESSION_DF_KEY)
    if df is not None and len(df) > 0:
        return {"type": "local_df", "dataframe": df}
    st.warning(
        "Connect a Hugging Face dataset on **Hugging Face Data** or load a CSV on **Upload Data**."
    )
    st.stop()

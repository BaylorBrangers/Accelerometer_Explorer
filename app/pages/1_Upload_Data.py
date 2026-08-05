"""Load accelerometer CSV via upload or filesystem path."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.config import DATA_DIR  # noqa: E402
from accel_explorer.io import dataframe_summary, load_csv, load_path  # noqa: E402
from session_state import SESSION_DF_KEY, init_defaults  # noqa: E402

st.set_page_config(page_title="Upload Data", layout="wide")
init_defaults()

st.title("Upload Data")
st.write("Provide accelerometer CSV via browser upload or a local / mounted path.")

mode = st.radio("Data source", ["Upload file(s)", "Local path / directory"], horizontal=True)

df = None
error = None

if mode == "Upload file(s)":
    uploads = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)
    if uploads:
        frames = []
        for up in uploads:
            frames.append(load_csv(up.getvalue(), source_name=up.name))
        if frames:
            import pandas as pd

            df = pd.concat(frames, ignore_index=True)
else:
    default = str(DATA_DIR / "samples" / "sample_accel.csv")
    if not Path(default).exists():
        default = str(DATA_DIR)
    path = st.text_input("Path to CSV file or directory", value=default)
    if st.button("Load path", type="primary") and path:
        try:
            df = load_path(path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

if error:
    st.error(error)

if df is not None:
    st.session_state[SESSION_DF_KEY] = df
    summary = dataframe_summary(df)
    st.success(f"Loaded {summary['rows']} rows from {len(summary['sources'])} source(s).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", summary["rows"])
    c2.metric("Has labels", "yes" if summary["has_label"] else "no")
    c3.metric("Sources", len(summary["sources"]))
    if summary.get("labels"):
        st.write("Labels:", ", ".join(summary["labels"]))
    st.dataframe(df.head(200), use_container_width=True)
elif SESSION_DF_KEY in st.session_state:
    st.info("Using previously loaded data from this session.")
    st.dataframe(st.session_state[SESSION_DF_KEY].head(200), use_container_width=True)

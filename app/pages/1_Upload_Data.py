"""Load accelerometer CSV via upload or filesystem path."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.config import DATA_DIR  # noqa: E402
from accel_explorer.io import (  # noqa: E402
    dataframe_summary,
    load_csv,
    load_labeled_recording,
    load_path,
)
from session_state import SESSION_DF_KEY, init_defaults  # noqa: E402

st.set_page_config(page_title="Upload Data", layout="wide")
init_defaults()

st.title("Upload Data")
st.write(
    "Provide accelerometer CSV via browser upload or a local / mounted path. "
    "Supports generic `timestamp,x,y,z` CSVs and **Anomark** `*_acc_weardata.csv` exports."
)

mode = st.radio("Data source", ["Upload file(s)", "Local path / directory"], horizontal=True)

df = None
error = None

if mode == "Upload file(s)":
    uploads = st.file_uploader(
        "Accelerometer CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="accel_uploads",
    )
    ann_upload = st.file_uploader(
        "Optional behavior annotation CSV (start/end/label intervals)",
        type=["csv"],
        accept_multiple_files=False,
        key="ann_upload",
    )
    if uploads:
        try:
            frames = []
            for up in uploads:
                if ann_upload is not None and len(uploads) == 1:
                    frames.append(
                        load_labeled_recording(
                            up.getvalue(),
                            ann_upload.getvalue(),
                            signal_name=up.name,
                        )
                    )
                else:
                    frames.append(load_csv(up.getvalue(), source_name=up.name))
            if frames:
                df = pd.concat(frames, ignore_index=True)
                if ann_upload is not None and len(uploads) > 1:
                    st.warning(
                        "Annotation merge is applied only when a single accelerometer file is uploaded."
                    )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
else:
    default = str(DATA_DIR / "samples" / "sample_anomark_weardata.csv")
    if not Path(default).exists():
        default = str(DATA_DIR / "samples" / "sample_accel.csv")
    path = st.text_input("Path to CSV file or directory", value=default)
    ann_path = st.text_input(
        "Optional annotation CSV path",
        value="",
        placeholder="/data/annotations.csv",
    )
    if st.button("Load path", type="primary") and path:
        try:
            if ann_path.strip():
                df = load_labeled_recording(path, ann_path.strip())
            else:
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

"""Accelerometer Explorer — home."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from accel_explorer import __version__  # noqa: E402
from accel_explorer.config import ARTIFACTS_DIR, DATA_DIR  # noqa: E402

st.set_page_config(
    page_title="Accelerometer Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Accelerometer Explorer")
st.write(
    "Connect a **Hugging Face dataset** (recommended for large open corpora), "
    "or upload / path-load CSV data. Explore signals, then train **custom PyTorch** "
    "models or fine-tune **Hugging Face** encoders — streaming from disk cache so "
    "~60GB files never need to fit in RAM."
)

col1, col2, col3 = st.columns(3)
col1.metric("Package", f"v{__version__}")
col2.metric("Data dir", str(DATA_DIR))
col3.metric("Artifacts", str(ARTIFACTS_DIR))

st.markdown(
    """
### Workflow
1. **Hugging Face Data** — connect a Hub dataset repo; cache files to disk
2. **Upload Data** — optional local/Anomark CSV upload or path (`/data`)
3. **Explore** — plot axes (preview for Hub sources), set window size
4. **Train** — stream windows from Hub cache or in-memory sample; custom or HF models
5. **Analyze** — run the checkpoint and export predictions

### Expected CSV columns
`timestamp`, `x`, `y`, `z`, and optionally `label` for supervised training.
"""
)

sample = ROOT / "data" / "samples" / "sample_accel.csv"
if sample.exists():
    st.info(f"Sample data available at `{sample}`")

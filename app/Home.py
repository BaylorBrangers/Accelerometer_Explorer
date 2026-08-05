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
    "Upload or point to accelerometer CSV data, explore signals, then train "
    "**custom PyTorch** models or fine-tune **Hugging Face** encoders for "
    "activity recognition and anomaly scoring."
)

col1, col2, col3 = st.columns(3)
col1.metric("Package", f"v{__version__}")
col2.metric("Data dir", str(DATA_DIR))
col3.metric("Artifacts", str(ARTIFACTS_DIR))

st.markdown(
    """
### Workflow
1. **Upload Data** — file upload or local/Docker path (`/data`)
2. **Explore** — plot axes, set window size
3. **Train** — Hugging Face or custom Conv1D / LSTM / autoencoder
4. **Analyze** — run the checkpoint and export predictions

### Expected CSV columns
`timestamp`, `x`, `y`, `z`, and optionally `label` for supervised training.
"""
)

sample = ROOT / "data" / "samples" / "sample_accel.csv"
if sample.exists():
    st.info(f"Sample data available at `{sample}`")

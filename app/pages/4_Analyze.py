"""Run a trained checkpoint and export predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.analyze import (  # noqa: E402
    classification_metrics,
    prediction_timeline_figure,
    results_dataframe,
)
from accel_explorer.config import ARTIFACTS_DIR  # noqa: E402
from accel_explorer.models.registry import build_model  # noqa: E402
from accel_explorer.models.train import (  # noqa: E402
    load_checkpoint,
    predict_activity,
    score_anomaly,
)
from accel_explorer.preprocess import create_windows  # noqa: E402
from session_state import (  # noqa: E402
    SESSION_CHECKPOINT_KEY,
    SESSION_STRIDE_KEY,
    SESSION_WINDOW_KEY,
    init_defaults,
    require_dataframe,
)

st.set_page_config(page_title="Analyze", layout="wide")
init_defaults()
df = require_dataframe()

st.title("Analyze")
st.write("Load a checkpoint, run inference on the current data, and download results.")

ckpts = sorted(Path(ARTIFACTS_DIR).glob("checkpoint_*.pt"), reverse=True)
default_ckpt = st.session_state.get(SESSION_CHECKPOINT_KEY)
options = [str(p) for p in ckpts]
if default_ckpt and default_ckpt not in options:
    options = [default_ckpt] + options

if not options:
    st.warning("No checkpoints found. Train a model first.")
    st.stop()

ckpt_path = st.selectbox("Checkpoint", options, index=0)
window_size = int(st.session_state[SESSION_WINDOW_KEY])
stride = int(st.session_state[SESSION_STRIDE_KEY])

if st.button("Run inference", type="primary"):
    payload = load_checkpoint(ckpt_path)
    cfg = payload.get("config", {})
    task = cfg.get("task", "activity")
    source = cfg.get("source", "custom")
    architecture = cfg.get("architecture")
    hf_model_id = cfg.get("hf_model_id")
    label_to_idx = payload.get("label_to_idx") or cfg.get("label_to_idx") or {}
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    X, y, meta = create_windows(df, window_size=window_size, stride=stride)
    num_classes = len(label_to_idx) if label_to_idx else 2

    model = build_model(
        source=source if task == "activity" else "custom",
        task=task,
        num_classes=num_classes if task == "activity" else None,
        architecture=architecture or "Conv1D",
        hf_model_id=hf_model_id,
        in_channels=3,
        window_length=window_size,
    )
    model.load_state_dict(payload["model_state"])

    scores = None
    if task == "anomaly":
        scores = score_anomaly(model, torch.as_tensor(X, dtype=torch.float32))
        predictions = ["anomaly" if s > float(st.session_state.get("anom_thr", 0.1)) else "normal" for s in scores]
    else:
        _, predictions = predict_activity(
            model,
            torch.as_tensor(X, dtype=torch.float32),
            idx_to_label=idx_to_label or None,
        )

    st.plotly_chart(
        prediction_timeline_figure(meta, predictions, scores=scores),
        use_container_width=True,
    )

    if task == "activity" and y is not None:
        metrics = classification_metrics(y, predictions)
        st.subheader("Classification report")
        st.json(metrics["report"])
        st.write("Confusion matrix (rows=true, cols=pred)", metrics["labels"])
        st.dataframe(metrics["confusion_matrix"])

    if task == "anomaly":
        thr = st.slider("Anomaly threshold", 0.0, float(max(scores) if scores else 1.0), float(sorted(scores)[len(scores) // 2]) if scores else 0.1)
        st.session_state["anom_thr"] = thr
        predictions = ["anomaly" if s > thr else "normal" for s in scores]
        st.plotly_chart(
            prediction_timeline_figure(meta, predictions, scores=scores),
            use_container_width=True,
        )

    results = results_dataframe(meta, predictions, scores=scores)
    st.dataframe(results, use_container_width=True)
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

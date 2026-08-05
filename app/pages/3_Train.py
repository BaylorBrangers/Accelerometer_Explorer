"""Train custom PyTorch or Hugging Face models on windowed signals."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.analyze import loss_curve_figure  # noqa: E402
from accel_explorer.dataset import (  # noqa: E402
    AccelerometerWindowDataset,
    make_loaders,
    train_val_split,
)
from accel_explorer.models.registry import (  # noqa: E402
    build_model,
    list_custom_architectures,
    list_hf_presets,
)
from accel_explorer.models.train import TrainConfig, train_model  # noqa: E402
from accel_explorer.preprocess import create_windows  # noqa: E402
from session_state import (  # noqa: E402
    SESSION_CHECKPOINT_KEY,
    SESSION_STRIDE_KEY,
    SESSION_TRAIN_RESULT_KEY,
    SESSION_WINDOW_KEY,
    init_defaults,
    require_dataframe,
)

st.set_page_config(page_title="Train", layout="wide")
init_defaults()
df = require_dataframe()

st.title("Train")
st.write("Fine-tune a Hugging Face encoder or train a custom PyTorch architecture.")

task = st.selectbox("Task", ["activity", "anomaly"])
source = st.selectbox("Model source", ["custom", "huggingface"])

architecture = None
hf_model_id = None

if source == "custom" and task == "activity":
    architecture = st.selectbox("Architecture", list_custom_architectures())
elif source == "huggingface" and task == "activity":
    presets = list_hf_presets()
    preset_label = st.selectbox("HF preset", list(presets.keys()) + ["Custom Hub ID…"])
    if preset_label == "Custom Hub ID…":
        hf_model_id = st.text_input("Hugging Face model ID", value="prajjwal1/bert-tiny")
    else:
        hf_model_id = presets[preset_label]
        st.caption(f"Resolved ID: `{hf_model_id}`")
elif task == "anomaly":
    st.info("Anomaly training uses the built-in PyTorch autoencoder (source ignored).")
    source = "custom"

c1, c2, c3, c4 = st.columns(4)
epochs = c1.number_input("Epochs", min_value=1, max_value=100, value=5)
batch_size = c2.number_input("Batch size", min_value=1, max_value=256, value=32)
lr = c3.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
val_fraction = c4.slider("Val fraction", 0.0, 0.5, 0.2, 0.05)

window_size = int(st.session_state[SESSION_WINDOW_KEY])
stride = int(st.session_state[SESSION_STRIDE_KEY])
st.caption(f"Using window_size={window_size}, stride={stride} from Explore.")

if st.button("Start training", type="primary"):
    try:
        X, y, meta = create_windows(df, window_size=window_size, stride=stride)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if task == "activity":
        if y is None:
            st.error("Activity training requires a `label` column in the data.")
            st.stop()
        dataset = AccelerometerWindowDataset(X, y)
        num_classes = dataset.num_classes
        label_to_idx = dataset.label_to_idx or {}
    else:
        dataset = AccelerometerWindowDataset(X, None)
        num_classes = None
        label_to_idx = {}

    train_ds, val_ds = train_val_split(dataset, val_fraction=float(val_fraction))
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=int(batch_size))

    progress = st.progress(0.0, text="Training…")
    status = st.empty()

    def on_progress(epoch: int, row: dict) -> None:
        progress.progress(epoch / int(epochs), text=f"Epoch {epoch}/{int(epochs)}")
        status.write(row)

    with st.spinner("Building model…"):
        model = build_model(
            source=source,
            task=task,
            num_classes=num_classes,
            architecture=architecture,
            hf_model_id=hf_model_id,
            in_channels=3,
            window_length=window_size,
        )

    cfg = TrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
        task=task,
        source=source,
        architecture=architecture,
        hf_model_id=hf_model_id,
        label_to_idx=label_to_idx,
    )

    try:
        result = train_model(
            model,
            train_loader,
            val_loader,
            config=cfg,
            progress_callback=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")
        st.stop()

    st.session_state[SESSION_TRAIN_RESULT_KEY] = result
    st.session_state[SESSION_CHECKPOINT_KEY] = result.checkpoint_path
    progress.progress(1.0, text="Done")
    st.success(f"Saved checkpoint: `{result.checkpoint_path}`")
    st.plotly_chart(loss_curve_figure(result.history), use_container_width=True)
    st.json(
        {
            "best_val_loss": result.best_val_loss,
            "task": result.task,
            "source": result.source,
            "architecture": result.architecture,
            "hf_model_id": result.hf_model_id,
        }
    )

if SESSION_TRAIN_RESULT_KEY in st.session_state:
    prev = st.session_state[SESSION_TRAIN_RESULT_KEY]
    st.subheader("Last run in this session")
    st.write(f"Checkpoint: `{prev.checkpoint_path}`")

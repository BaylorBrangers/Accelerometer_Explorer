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
from accel_explorer.streaming import (  # noqa: E402
    StreamSource,
    discover_labels,
    make_streaming_loader,
)
from session_state import (  # noqa: E402
    SESSION_CHECKPOINT_KEY,
    SESSION_STRIDE_KEY,
    SESSION_TRAIN_RESULT_KEY,
    SESSION_WINDOW_KEY,
    init_defaults,
    require_trainable_source,
)

st.set_page_config(page_title="Train", layout="wide")
init_defaults()
data_source = require_trainable_source()

st.title("Train")
st.write("Fine-tune a Hugging Face encoder or train a custom PyTorch architecture.")

if data_source["type"] == "hf_hub":
    st.success(
        f"Streaming from Hub cache: `{data_source['repo_id']}` "
        f"({len(data_source['local_paths'])} file(s)) — not loaded into RAM."
    )
else:
    st.caption("Using in-memory dataframe from Upload Data (fine for small samples).")

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

max_steps = None
max_windows = None
if data_source["type"] == "hf_hub":
    st.subheader("Streaming controls")
    max_windows = st.number_input(
        "Max windows per epoch (caps a pass over huge files)",
        min_value=100,
        max_value=10_000_000,
        value=20_000,
        step=1000,
        help="For ~60GB corpora, start with a window budget instead of a full epoch scan.",
    )
    max_steps = st.number_input(
        "Max optimizer steps per epoch (optional hard cap)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        help="0 = no step cap beyond max windows.",
    )
    max_steps = int(max_steps) or None

if st.button("Start training", type="primary"):
    label_to_idx: dict[str, int] = {}
    num_classes = None
    train_loader = None
    val_loader = None

    if data_source["type"] == "hf_hub":
        paths = data_source["local_paths"]
        kind = data_source.get("kind", "auto")
        sources = [StreamSource(path=p, kind=kind) for p in paths]
        if task == "activity":
            with st.spinner("Scanning labels from stream (bounded)…"):
                label_to_idx = discover_labels(paths, kind=kind, max_rows_per_file=100_000)
            if len(label_to_idx) < 2:
                st.error(
                    "Need at least 2 labels for activity training. "
                    "Add a `label`/`behavior` column or provide annotations."
                )
                st.stop()
            num_classes = len(label_to_idx)
            st.write("Labels:", label_to_idx)
            n_val = max(1, int(int(max_windows) * float(val_fraction))) if max_windows else None
            n_train = int(max_windows) - n_val if max_windows and n_val else max_windows
            train_loader = make_streaming_loader(
                sources,
                window_size=window_size,
                stride=stride,
                batch_size=int(batch_size),
                label_to_idx=label_to_idx,
                max_windows=n_train,
                require_label=True,
            )
            val_loader = make_streaming_loader(
                sources,
                window_size=window_size,
                stride=stride,
                batch_size=int(batch_size),
                label_to_idx=label_to_idx,
                max_windows=n_val,
                require_label=True,
            )
        else:
            train_loader = make_streaming_loader(
                sources,
                window_size=window_size,
                stride=stride,
                batch_size=int(batch_size),
                max_windows=int(max_windows) if max_windows else None,
                require_label=False,
            )
    else:
        df = data_source["dataframe"]
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
        max_steps=max_steps,
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
            "data_source": data_source.get("type"),
        }
    )

if SESSION_TRAIN_RESULT_KEY in st.session_state:
    prev = st.session_state[SESSION_TRAIN_RESULT_KEY]
    st.subheader("Last run in this session")
    st.write(f"Checkpoint: `{prev.checkpoint_path}`")

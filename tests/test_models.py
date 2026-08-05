"""Smoke tests for custom models and training loop."""

from pathlib import Path

import torch

from accel_explorer.dataset import AccelerometerWindowDataset, make_loaders, train_val_split
from accel_explorer.io import load_csv
from accel_explorer.models.custom import build_custom_model
from accel_explorer.models.registry import build_model
from accel_explorer.models.train import TrainConfig, predict_activity, train_model
from accel_explorer.preprocess import create_windows

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_accel.csv"


def test_custom_forward_shapes():
    model = build_custom_model("Conv1D", num_classes=3)
    x = torch.randn(4, 3, 64)
    logits = model(x)
    assert logits.shape == (4, 3)

    lstm = build_custom_model("LSTM", num_classes=3)
    assert lstm(x).shape == (4, 3)


def test_train_custom_one_epoch(tmp_path: Path):
    df = load_csv(SAMPLE)
    X, y, _ = create_windows(df, window_size=64, stride=64)
    ds = AccelerometerWindowDataset(X, y)
    train_ds, val_ds = train_val_split(ds, val_fraction=0.25)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=16)

    model = build_model(
        source="custom",
        task="activity",
        num_classes=ds.num_classes,
        architecture="Conv1D",
    )
    result = train_model(
        model,
        train_loader,
        val_loader,
        config=TrainConfig(
            epochs=1,
            lr=1e-3,
            batch_size=16,
            task="activity",
            source="custom",
            architecture="Conv1D",
            label_to_idx=ds.label_to_idx or {},
        ),
        artifacts_dir=tmp_path,
    )
    assert Path(result.checkpoint_path).exists()
    assert len(result.history) == 1

    preds, labels = predict_activity(
        model,
        torch.as_tensor(X[:8], dtype=torch.float32),
        idx_to_label=ds.idx_to_label,
    )
    assert len(preds) == 8
    assert len(labels) == 8


def test_anomaly_autoencoder_smoke(tmp_path: Path):
    df = load_csv(SAMPLE)
    X, _, _ = create_windows(df, window_size=64, stride=64)
    ds = AccelerometerWindowDataset(X, None)
    train_ds, val_ds = train_val_split(ds, val_fraction=0.2)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=16)
    model = build_model(source="custom", task="anomaly", window_length=64)
    result = train_model(
        model,
        train_loader,
        val_loader,
        config=TrainConfig(epochs=1, task="anomaly", source="custom"),
        artifacts_dir=tmp_path,
    )
    assert Path(result.checkpoint_path).exists()

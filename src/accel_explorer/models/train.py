"""Shared training / evaluation loop and checkpoint IO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accel_explorer.config import ARTIFACTS_DIR


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 32
    device: str | None = None
    task: str = "activity"  # activity | anomaly
    source: str = "custom"
    architecture: str | None = None
    hf_model_id: str | None = None
    label_to_idx: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    checkpoint_path: str
    metrics_path: str
    best_val_loss: float
    label_to_idx: dict[str, int]
    task: str
    source: str
    architecture: str | None = None
    hf_model_id: str | None = None


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch_activity(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        if train_mode:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def _run_epoch_anomaly(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        if train_mode:
            optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total += int(x.size(0))
    return {"loss": total_loss / max(total, 1)}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    config: TrainConfig | None = None,
    artifacts_dir: Path | None = None,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    cfg = config or TrainConfig()
    device = resolve_device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss() if cfg.task == "activity" else nn.MSELoss()
    run_epoch = _run_epoch_activity if cfg.task == "activity" else _run_epoch_anomaly

    out_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ckpt_path = out_dir / f"checkpoint_{cfg.task}_{stamp}.pt"
    metrics_path = out_dir / f"metrics_{cfg.task}_{stamp}.json"

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, Any] | None = None

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer=optimizer, criterion=criterion, device=device
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
        }
        if "accuracy" in train_metrics:
            row["train_accuracy"] = train_metrics["accuracy"]

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_metrics = run_epoch(
                model, val_loader, optimizer=None, criterion=criterion, device=device
            )
            row["val_loss"] = val_metrics["loss"]
            if "accuracy" in val_metrics:
                row["val_accuracy"] = val_metrics["accuracy"]
            monitor = val_metrics["loss"]
        else:
            monitor = train_metrics["loss"]

        history.append(row)
        if progress_callback:
            progress_callback(epoch, row)

        if monitor <= best_val:
            best_val = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "label_to_idx": cfg.label_to_idx,
        "history": history,
    }
    torch.save(payload, ckpt_path)
    metrics_path.write_text(json.dumps({"history": history, "best_val_loss": best_val}, indent=2))

    return TrainResult(
        history=history,
        checkpoint_path=str(ckpt_path),
        metrics_path=str(metrics_path),
        best_val_loss=best_val,
        label_to_idx=cfg.label_to_idx,
        task=cfg.task,
        source=cfg.source,
        architecture=cfg.architecture,
        hf_model_id=cfg.hf_model_id,
    )


def load_checkpoint(path: Path | str, model: nn.Module | None = None) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None and "model_state" in data:
        model.load_state_dict(data["model_state"])
    return data


def predict_activity(
    model: nn.Module,
    windows: torch.Tensor | Any,
    *,
    device: str | None = None,
    idx_to_label: dict[int, str] | None = None,
) -> tuple[list[int], list[str | int]]:
    model.eval()
    dev = resolve_device(device)
    model = model.to(dev)
    if not torch.is_tensor(windows):
        windows = torch.as_tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        logits = model(windows.to(dev))
        preds = logits.argmax(dim=1).cpu().tolist()
    if idx_to_label:
        labels = [idx_to_label[i] for i in preds]
    else:
        labels = preds
    return preds, labels


def score_anomaly(
    model: nn.Module,
    windows: torch.Tensor | Any,
    *,
    device: str | None = None,
) -> list[float]:
    model.eval()
    dev = resolve_device(device)
    model = model.to(dev)
    if not torch.is_tensor(windows):
        windows = torch.as_tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        if hasattr(model, "reconstruction_error"):
            scores = model.reconstruction_error(windows.to(dev))
        else:
            recon = model(windows.to(dev))
            scores = torch.mean((recon - windows.to(dev)) ** 2, dim=(1, 2))
    return scores.cpu().tolist()

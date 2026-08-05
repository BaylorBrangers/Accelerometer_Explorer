"""Hugging Face adapters for fine-tuning on windowed accelerometer data."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from accel_explorer.config import HF_CACHE_DIR


class WindowToSequenceAdapter(nn.Module):
    """
    Map accel windows (B, C, L) into token-like embeddings for a HF encoder.

    This is a scaffold adapter: each time step becomes a token whose features
    are the C accelerometer channels, projected to the model hidden size.
    """

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> (B, L, C) -> (B, L, H)
        return self.proj(x.transpose(1, 2))


class HuggingFaceWindowClassifier(nn.Module):
    """Wrap a HF sequence model with a window adapter + classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = WindowToSequenceAdapter(in_channels, hidden_size)
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.adapter(x)
        outputs = self.encoder(inputs_embeds=embeds)
        # Prefer pooler_output when available; else mean-pool last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(self.dropout(pooled))


def _ensure_hf_cache() -> str:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
    return str(HF_CACHE_DIR)


def load_hf_classifier(
    model_id: str,
    num_classes: int,
    *,
    in_channels: int = 3,
    trust_remote_code: bool = False,
) -> HuggingFaceWindowClassifier:
    """
    Load a pretrained HF model and wrap it for accel window classification.

    Works with encoder models that accept ``inputs_embeds`` (e.g. BERT-family).
    For specialized time-series hubs models, extend this factory.
    """
    from transformers import AutoConfig, AutoModel

    cache = _ensure_hf_cache()
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache,
        trust_remote_code=trust_remote_code,
    )
    encoder = AutoModel.from_pretrained(
        model_id,
        cache_dir=cache,
        trust_remote_code=trust_remote_code,
    )
    hidden = int(getattr(config, "hidden_size", None) or getattr(config, "d_model", 768))
    return HuggingFaceWindowClassifier(
        encoder,
        hidden_size=hidden,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def describe_hf_model(model_id: str) -> dict[str, Any]:
    """Lightweight metadata for UI display (may hit the hub)."""
    from transformers import AutoConfig

    cache = _ensure_hf_cache()
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache)
    return {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "d_model", None)),
    }

"""Custom PyTorch classifiers for windowed accelerometer signals."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DClassifier(nn.Module):
    """1D CNN over (batch, channels, length) accelerometer windows."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class LSTMClassifier(nn.Module):
    """LSTM classifier; input (batch, channels, length) → permute to time-major."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])


CUSTOM_ARCHITECTURES = {
    "Conv1D": Conv1DClassifier,
    "LSTM": LSTMClassifier,
}


def build_custom_model(
    architecture: str,
    num_classes: int,
    *,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    if architecture not in CUSTOM_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Choose from {sorted(CUSTOM_ARCHITECTURES)}"
        )
    cls = CUSTOM_ARCHITECTURES[architecture]
    return cls(num_classes, in_channels=in_channels, **kwargs)

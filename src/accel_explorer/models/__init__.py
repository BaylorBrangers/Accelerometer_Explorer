"""Public model APIs."""

from accel_explorer.models.anomaly import AccelAutoencoder
from accel_explorer.models.custom import Conv1DClassifier, LSTMClassifier, build_custom_model
from accel_explorer.models.huggingface import HuggingFaceWindowClassifier, load_hf_classifier
from accel_explorer.models.registry import (
    build_model,
    list_custom_architectures,
    list_hf_presets,
)
from accel_explorer.models.train import (
    TrainConfig,
    TrainResult,
    load_checkpoint,
    predict_activity,
    score_anomaly,
    train_model,
)

__all__ = [
    "AccelAutoencoder",
    "Conv1DClassifier",
    "LSTMClassifier",
    "HuggingFaceWindowClassifier",
    "TrainConfig",
    "TrainResult",
    "build_custom_model",
    "build_model",
    "list_custom_architectures",
    "list_hf_presets",
    "load_checkpoint",
    "load_hf_classifier",
    "predict_activity",
    "score_anomaly",
    "train_model",
]

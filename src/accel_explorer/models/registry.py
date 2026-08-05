"""Model registry: custom architectures and Hugging Face presets."""

from __future__ import annotations

from accel_explorer.models.anomaly import AccelAutoencoder
from accel_explorer.models.custom import CUSTOM_ARCHITECTURES, build_custom_model
from accel_explorer.models.huggingface import load_hf_classifier

# Curated Hub IDs suitable as encoder backbones for the window adapter.
HF_PRESETS: dict[str, str] = {
    "bert-tiny (prajjwal1/bert-tiny)": "prajjwal1/bert-tiny",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "bert-base-uncased": "bert-base-uncased",
}

CUSTOM_ARCH_NAMES = sorted(CUSTOM_ARCHITECTURES.keys())


def list_custom_architectures() -> list[str]:
    return list(CUSTOM_ARCH_NAMES)


def list_hf_presets() -> dict[str, str]:
    return dict(HF_PRESETS)


def build_model(
    *,
    source: str,
    task: str,
    num_classes: int | None = None,
    architecture: str | None = None,
    hf_model_id: str | None = None,
    in_channels: int = 3,
    window_length: int = 64,
):
    """
    Factory used by the Train page and tests.

    source: "custom" | "huggingface"
    task: "activity" | "anomaly"
    """
    source = source.lower()
    task = task.lower()

    if task == "anomaly":
        return AccelAutoencoder(in_channels=in_channels, length=window_length)

    if num_classes is None or num_classes < 2:
        raise ValueError("Activity classification requires num_classes >= 2")

    if source == "custom":
        if not architecture:
            raise ValueError("Custom source requires an architecture name")
        return build_custom_model(architecture, num_classes, in_channels=in_channels)

    if source in {"huggingface", "hf", "hugging_face"}:
        if not hf_model_id:
            raise ValueError("Hugging Face source requires hf_model_id")
        return load_hf_classifier(hf_model_id, num_classes, in_channels=in_channels)

    raise ValueError(f"Unknown model source: {source!r}")

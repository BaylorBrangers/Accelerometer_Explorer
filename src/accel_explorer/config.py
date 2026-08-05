"""Default paths and preprocessing settings."""

from __future__ import annotations

import os
from pathlib import Path

# Prefer container mounts when present; fall back to repo-relative dirs.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dir(env_key: str, container_path: str, local_name: str) -> Path:
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    container = Path(container_path)
    if container.is_dir():
        return container
    return _REPO_ROOT / local_name


DATA_DIR = _resolve_dir("ACCEL_DATA_DIR", "/data", "data")
ARTIFACTS_DIR = _resolve_dir("ACCEL_ARTIFACTS_DIR", "/artifacts", "artifacts")
HF_CACHE_DIR = ARTIFACTS_DIR / "hf_cache"

# Column aliases mapped to canonical names.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "t", "datetime", "date"),
    "x": ("x", "acc_x", "accel_x", "ax"),
    "y": ("y", "acc_y", "accel_y", "ay"),
    "z": ("z", "acc_z", "accel_z", "az"),
    "label": ("label", "activity", "class", "y_true"),
}

DEFAULT_WINDOW_SIZE = 64
DEFAULT_WINDOW_STRIDE = 32
DEFAULT_SAMPLE_RATE_HZ: float | None = None  # keep native rate unless set
CHANNELS = ("x", "y", "z")

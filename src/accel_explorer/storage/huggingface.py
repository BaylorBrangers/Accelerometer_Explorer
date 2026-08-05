"""Hugging Face Hub storage for large open accelerometer datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from accel_explorer.config import HF_CACHE_DIR


@dataclass(frozen=True)
class HubFile:
    path: str
    size: int | None = None


def ensure_hf_cache(cache_dir: Path | None = None) -> Path:
    cache = Path(cache_dir or HF_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    return cache


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def login(token: str) -> None:
    """Persist a Hub token for this environment (optional; env HF_TOKEN also works)."""
    from huggingface_hub import login as hf_login

    hf_login(token=token, add_to_git_credential=False)


def list_repo_files(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    suffix: str | None = ".csv",
) -> list[HubFile]:
    """List files in a Hub dataset/model repo (metadata only — no download)."""
    api = _api(token)
    paths = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
    )
    files: list[HubFile] = []
    for path in paths:
        if suffix and not path.lower().endswith(suffix.lower()):
            continue
        files.append(HubFile(path=path))
    return files


def list_repo_tree(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    suffix: str | None = ".csv",
    path_in_repo: str | None = None,
) -> list[HubFile]:
    """List files with sizes when available (Hub tree API)."""
    api = _api(token)
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        recursive=True,
    )
    out: list[HubFile] = []
    for entry in entries:
        path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
        if not path:
            continue
        if suffix and not str(path).lower().endswith(suffix.lower()):
            continue
        size = getattr(entry, "size", None)
        out.append(HubFile(path=str(path), size=int(size) if size is not None else None))
    return out


def resolve_to_local_path(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """
    Ensure a Hub file is available on local disk via the HF cache.

    Downloads once; subsequent calls reuse the cache. Does not load file into RAM.
    """
    from huggingface_hub import hf_hub_download

    cache = ensure_hf_cache(cache_dir)
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        cache_dir=str(cache),
    )
    return Path(local)


def open_hub_stream(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
):
    """
    Open a binary stream from the Hub without materializing the full object in RAM.

    Prefer ``resolve_to_local_path`` for multi-epoch training (disk cache).
    Use this for one-pass previews or single-pass streaming jobs.
    """
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(
            token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if repo_type == "model":
            uri = f"hf://{repo_id}/{filename}"
        elif repo_type == "dataset":
            uri = f"hf://datasets/{repo_id}/{filename}"
        else:
            uri = f"hf://spaces/{repo_id}/{filename}"
        return fs.open(uri, "rb")
    except Exception:
        # Fallback: download to cache (still disk-backed, not RAM).
        return resolve_to_local_path(
            repo_id,
            filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
        ).open("rb")

def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def iter_selected_local_paths(
    repo_id: str,
    filenames: Iterable[str],
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
):
    """Yield (repo_path, local_path) for each Hub file, caching to disk."""
    for name in filenames:
        local = resolve_to_local_path(
            repo_id,
            name,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        yield name, local

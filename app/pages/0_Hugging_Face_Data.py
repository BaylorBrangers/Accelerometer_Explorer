"""Connect to a Hugging Face Hub dataset without loading it into RAM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from accel_explorer.config import HF_CACHE_DIR  # noqa: E402
from accel_explorer.storage.huggingface import (  # noqa: E402
    ensure_hf_cache,
    format_size,
    list_repo_tree,
    login,
    resolve_to_local_path,
)
from accel_explorer.streaming import detect_kind, preview_rows  # noqa: E402
from session_state import (  # noqa: E402
    SESSION_DF_KEY,
    init_defaults,
    set_hf_data_source,
)

st.set_page_config(page_title="Hugging Face Data", layout="wide")
init_defaults()

st.title("Hugging Face Data")
st.write(
    "Connect an open **dataset repo** on the Hub. Large files (~60GB) stay on disk in the "
    "HF cache — only a small preview is loaded into memory. Training streams windows from cache."
)

ensure_hf_cache()
st.caption(f"Cache directory: `{HF_CACHE_DIR}`")

token = st.text_input(
    "HF token (optional for public repos)",
    type="password",
    value=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "",
    help="Also read from env HF_TOKEN / secrets. Needed for private datasets.",
)
if token and st.checkbox("Save token for this session via huggingface_hub.login"):
    try:
        login(token)
        st.success("Token registered for this environment.")
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

repo_id = st.text_input("Dataset repo ID", placeholder="username/my-accel-dataset")
repo_type = st.selectbox("Repo type", ["dataset", "model"], index=0)
revision = st.text_input("Revision (optional)", value="") or None
suffix = st.text_input("File filter suffix", value=".csv")

col_a, col_b = st.columns(2)
list_clicked = col_a.button("List files", type="primary")
if "hf_file_listing" not in st.session_state:
    st.session_state["hf_file_listing"] = []

if list_clicked:
    if not repo_id.strip():
        st.error("Enter a dataset repo ID.")
    else:
        try:
            files = list_repo_tree(
                repo_id.strip(),
                repo_type=repo_type,
                revision=revision,
                token=token or None,
                suffix=suffix or None,
            )
            st.session_state["hf_file_listing"] = files
            st.session_state["hf_repo_id"] = repo_id.strip()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to list repo: {exc}")

files = st.session_state.get("hf_file_listing") or []
if files:
    st.subheader("Files in repo")
    table = pd.DataFrame(
        [{"path": f.path, "size": format_size(f.size), "bytes": f.size or 0} for f in files]
    )
    st.dataframe(table[["path", "size"]], use_container_width=True)
    total = sum(f.size or 0 for f in files)
    st.caption(f"{len(files)} matching file(s), total listed size ≈ {format_size(total)}")

    options = [f.path for f in files]
    selected = st.multiselect("Select files to use for training", options=options, default=options[:1])
    prepare = st.button("Prepare cache (download to disk, not RAM)", type="primary")

    if prepare and selected:
        local_paths: list[str] = []
        progress = st.progress(0.0)
        status = st.empty()
        rid = st.session_state.get("hf_repo_id", repo_id.strip())
        for i, name in enumerate(selected):
            status.write(f"Caching `{name}` …")
            try:
                local = resolve_to_local_path(
                    rid,
                    name,
                    repo_type=repo_type,
                    revision=revision,
                    token=token or None,
                )
                local_paths.append(str(local))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{name}: {exc}")
                break
            progress.progress((i + 1) / len(selected))
        if local_paths:
            kind = detect_kind(Path(local_paths[0]))
            set_hf_data_source(
                repo_id=rid,
                files=selected,
                local_paths=local_paths,
                repo_type=repo_type,
                revision=revision,
                kind=kind,
            )
            # Lightweight preview only — never the full corpus
            preview = preview_rows(local_paths[0], n=200, kind=kind)
            preview_df = pd.DataFrame(preview)
            st.session_state[SESSION_DF_KEY] = preview_df  # explore preview only
            st.success(
                f"Ready: {len(local_paths)} file(s) cached on disk as `{kind}`. "
                "Explore shows a 200-row preview; Train streams from the full cached files."
            )
            st.dataframe(preview_df.head(50), use_container_width=True)

src = st.session_state.get("data_source")
if src and src.get("type") == "hf_hub":
    st.info(
        f"Active Hub source: `{src['repo_id']}` — {len(src['local_paths'])} cached file(s), "
        f"kind=`{src.get('kind')}`"
    )

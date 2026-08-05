"""Behavior-aware accelerometer profile visualization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from accel_explorer.preprocess import add_magnitude, create_windows

BEHAVIOR_COLORS = px.colors.qualitative.Set2


def _hex_to_rgba(color: str, alpha: float = 0.2) -> str:
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    return f"rgba(0,0,0,{alpha})"


def _label_col(df: pd.DataFrame) -> str:
    for name in ("label", "behavior", "behaviour", "activity"):
        if name in df.columns and df[name].notna().any():
            return name
    raise ValueError(
        "No behavior/label column found. Load annotated data or merge a behavior annotation file."
    )


def behavior_counts(df: pd.DataFrame) -> pd.DataFrame:
    col = _label_col(df)
    counts = (
        df[col]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("behavior")
        .reset_index(name="samples")
    )
    return counts


def behavior_timeline_figure(
    df: pd.DataFrame,
    *,
    max_points: int = 8000,
) -> go.Figure:
    """Color the magnitude (or z) trace by behavior over time."""
    col = _label_col(df)
    work = add_magnitude(df).copy()
    work[col] = work[col].astype(str)
    if len(work) > max_points:
        step = max(1, len(work) // max_points)
        work = work.iloc[::step]
    x_axis = work["timestamp"] if "timestamp" in work.columns else work.index

    fig = go.Figure()
    behaviors = sorted(work[col].dropna().unique())
    for i, beh in enumerate(behaviors):
        mask = work[col] == beh
        fig.add_trace(
            go.Scatter(
                x=x_axis[mask],
                y=work.loc[mask, "magnitude"],
                mode="markers",
                name=str(beh),
                marker=dict(size=4, color=BEHAVIOR_COLORS[i % len(BEHAVIOR_COLORS)]),
            )
        )
    fig.update_layout(
        title="Behavior timeline (magnitude)",
        xaxis_title="timestamp" if "timestamp" in df.columns else "sample",
        yaxis_title="magnitude",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def behavior_segment_overlay_figure(
    df: pd.DataFrame,
    *,
    channel: str = "magnitude",
    max_segments_per_behavior: int = 8,
    min_segment_len: int = 16,
) -> go.Figure:
    """
    Overlay contiguous behavior segments (normalized to sample index within segment).

    Helps compare shape of accelerometer profiles across behaviors.
    """
    col = _label_col(df)
    work = add_magnitude(df).copy()
    work[col] = work[col].astype(str)
    if channel not in work.columns:
        raise ValueError(f"Channel {channel!r} not in dataframe")

    labels = work[col].to_numpy()
    values = work[channel].to_numpy(dtype=float)
    # Find contiguous runs
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((str(labels[start]), start, i))
            start = i

    by_beh: dict[str, list[np.ndarray]] = {}
    for beh, a, b in runs:
        if b - a < min_segment_len:
            continue
        by_beh.setdefault(beh, []).append(values[a:b])

    fig = go.Figure()
    behaviors = sorted(by_beh)
    for i, beh in enumerate(behaviors):
        segs = by_beh[beh][:max_segments_per_behavior]
        color = BEHAVIOR_COLORS[i % len(BEHAVIOR_COLORS)]
        for j, seg in enumerate(segs):
            # resample each segment to 100 points for overlay
            xp = np.linspace(0, 1, num=len(seg))
            x_new = np.linspace(0, 1, num=100)
            y_new = np.interp(x_new, xp, seg)
            fig.add_trace(
                go.Scatter(
                    x=x_new,
                    y=y_new,
                    mode="lines",
                    name=beh if j == 0 else None,
                    legendgroup=beh,
                    showlegend=j == 0,
                    line=dict(color=color, width=1.5),
                    opacity=0.55,
                )
            )
    fig.update_layout(
        title=f"Segment overlays by behavior ({channel})",
        xaxis_title="normalized segment progress",
        yaxis_title=channel,
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def mean_behavior_profile_figure(
    df: pd.DataFrame,
    *,
    window_size: int = 64,
    stride: int | None = None,
    channel: str = "magnitude",
    max_windows_per_behavior: int = 500,
) -> go.Figure:
    """
    Mean ± std window profile per behavior for a chosen channel.

    Windows are majority-labeled via ``create_windows``.
    """
    col = _label_col(df)
    work = add_magnitude(df)
    stride = stride or max(1, window_size // 2)
    X, y, _ = create_windows(work, window_size=window_size, stride=stride)
    if y is None:
        raise ValueError("Window labels unavailable")

    # Map channel to axis index for raw x/y/z; magnitude computed from window
    channel_idx = {"x": 0, "y": 1, "z": 2}

    fig = go.Figure()
    behaviors = sorted({str(v) for v in y})
    t = np.arange(window_size)
    for i, beh in enumerate(behaviors):
        idxs = np.where(y.astype(str) == beh)[0][:max_windows_per_behavior]
        if len(idxs) == 0:
            continue
        if channel == "magnitude":
            mats = np.sqrt(np.sum(np.square(X[idxs]), axis=1))  # (n, L)
        else:
            mats = X[idxs, channel_idx[channel], :]
        mean = mats.mean(axis=0)
        std = mats.std(axis=0)
        color = BEHAVIOR_COLORS[i % len(BEHAVIOR_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mean + std,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=beh,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mean - std,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.2),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=beh,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mean,
                mode="lines",
                name=f"{beh} (n={len(idxs)})",
                line=dict(color=color, width=2),
                legendgroup=beh,
            )
        )

    fig.update_layout(
        title=f"Mean window profile ± std by behavior ({channel})",
        xaxis_title="sample within window",
        yaxis_title=channel,
        template="plotly_white",
        height=440,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def behavior_channel_boxplot_figure(df: pd.DataFrame) -> go.Figure:
    """Box plots of x/y/z/magnitude distributions per behavior."""
    col = _label_col(df)
    work = add_magnitude(df)
    long = work.melt(
        id_vars=[col],
        value_vars=["x", "y", "z", "magnitude"],
        var_name="channel",
        value_name="value",
    )
    long[col] = long[col].astype(str)
    # downsample for plotting if huge
    if len(long) > 80_000:
        long = long.sample(80_000, random_state=42)
    fig = px.box(
        long,
        x=col,
        y="value",
        color="channel",
        points=False,
        template="plotly_white",
        title="Channel value distributions by behavior",
    )
    fig.update_layout(height=440, margin=dict(l=40, r=20, t=60, b=40), xaxis_title="behavior")
    return fig


def behavior_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-behavior summary statistics for magnitude and axes."""
    col = _label_col(df)
    work = add_magnitude(df)
    rows = []
    for beh, grp in work.groupby(work[col].astype(str)):
        row = {"behavior": beh, "samples": int(len(grp))}
        for ch in ("x", "y", "z", "magnitude"):
            s = grp[ch]
            row[f"{ch}_mean"] = float(s.mean())
            row[f"{ch}_std"] = float(s.std())
            row[f"{ch}_rms"] = float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float)))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("behavior").reset_index(drop=True)


def multi_axis_behavior_figure(
    df: pd.DataFrame,
    *,
    behaviors: Sequence[str] | None = None,
    max_points: int = 4000,
) -> go.Figure:
    """Small-multiples style: x/y/z for selected behaviors."""
    col = _label_col(df)
    work = add_magnitude(df).copy()
    work[col] = work[col].astype(str)
    available = sorted(work[col].unique())
    selected = list(behaviors) if behaviors else available[:4]
    selected = [b for b in selected if b in available]
    if not selected:
        raise ValueError("No matching behaviors to plot")

    fig = make_subplots(
        rows=len(selected),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[str(b) for b in selected],
        vertical_spacing=0.06,
    )
    for r, beh in enumerate(selected, start=1):
        sub = work[work[col] == beh]
        if len(sub) > max_points:
            step = max(1, len(sub) // max_points)
            sub = sub.iloc[::step]
        x_axis = sub["timestamp"] if "timestamp" in sub.columns else sub.index
        for ch, color in (("x", "#1f77b4"), ("y", "#ff7f0e"), ("z", "#2ca02c")):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=sub[ch],
                    mode="lines",
                    name=ch,
                    legendgroup=ch,
                    showlegend=(r == 1),
                    line=dict(color=color, width=1),
                ),
                row=r,
                col=1,
            )
    fig.update_layout(
        title="Accelerometer profiles by behavior",
        template="plotly_white",
        height=220 * len(selected) + 80,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="time", row=len(selected), col=1)
    return fig

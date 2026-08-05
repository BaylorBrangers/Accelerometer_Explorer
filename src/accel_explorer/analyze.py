"""Analysis helpers: metrics and Plotly figures for Streamlit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix

from accel_explorer.preprocess import add_magnitude


def signal_figure(df: pd.DataFrame, *, max_points: int = 5000) -> go.Figure:
    """Plot x/y/z (+ magnitude) against timestamp or sample index."""
    work = add_magnitude(df)
    if len(work) > max_points:
        step = max(1, len(work) // max_points)
        work = work.iloc[::step]
    x_axis = work["timestamp"] if "timestamp" in work.columns else work.index
    fig = go.Figure()
    for col, color in (
        ("x", "#1f77b4"),
        ("y", "#ff7f0e"),
        ("z", "#2ca02c"),
        ("magnitude", "#9467bd"),
    ):
        fig.add_trace(
            go.Scatter(x=x_axis, y=work[col], mode="lines", name=col, line=dict(color=color, width=1))
        )
    fig.update_layout(
        title="Accelerometer signal",
        xaxis_title="timestamp" if "timestamp" in df.columns else "sample",
        yaxis_title="acceleration",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )
    return fig


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("x", "y", "z", "magnitude") if c in df.columns or c == "magnitude"]
    work = add_magnitude(df) if "magnitude" not in df.columns else df
    rows = []
    for col in ("x", "y", "z", "magnitude"):
        s = work[col]
        rows.append(
            {
                "channel": col,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "rms": float(np.sqrt(np.mean(np.square(s.to_numpy(dtype=float))))),
            }
        )
    return pd.DataFrame(rows)


def prediction_timeline_figure(
    meta: Sequence[dict],
    predictions: Sequence[str | int],
    *,
    scores: Sequence[float] | None = None,
) -> go.Figure:
    starts = [m.get("start_idx", i) for i, m in enumerate(meta)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=starts,
            y=[str(p) for p in predictions],
            mode="markers+lines",
            name="prediction",
            line=dict(shape="hv"),
        )
    )
    if scores is not None:
        fig.add_trace(
            go.Scatter(
                x=starts,
                y=list(scores),
                mode="lines",
                name="anomaly_score",
                yaxis="y2",
                line=dict(color="#d62728", width=1),
            )
        )
        fig.update_layout(
            yaxis2=dict(title="anomaly score", overlaying="y", side="right"),
        )
    fig.update_layout(
        title="Window predictions",
        xaxis_title="window start index",
        yaxis_title="prediction",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def classification_metrics(
    y_true: Sequence[str | int],
    y_pred: Sequence[str | int],
) -> dict:
    labels = sorted({str(x) for x in list(y_true) + list(y_pred)})
    yt = [str(x) for x in y_true]
    yp = [str(x) for x in y_pred]
    cm = confusion_matrix(yt, yp, labels=labels)
    report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def results_dataframe(
    meta: Sequence[dict],
    predictions: Sequence[str | int],
    *,
    scores: Sequence[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(meta):
        row = dict(m)
        row["prediction"] = predictions[i]
        if scores is not None:
            row["anomaly_score"] = scores[i]
        rows.append(row)
    return pd.DataFrame(rows)


def loss_curve_figure(history: Sequence[dict]) -> go.Figure:
    fig = go.Figure()
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    if history and "train_loss" in history[0]:
        fig.add_trace(go.Scatter(x=epochs, y=[h["train_loss"] for h in history], name="train_loss"))
    if history and "val_loss" in history[0]:
        fig.add_trace(go.Scatter(x=epochs, y=[h.get("val_loss") for h in history], name="val_loss"))
    if history and "train_accuracy" in history[0]:
        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("train_accuracy") for h in history], name="train_accuracy", yaxis="y2")
        )
    fig.update_layout(
        title="Training history",
        xaxis_title="epoch",
        yaxis_title="loss",
        yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]),
        template="plotly_white",
        height=360,
    )
    return fig

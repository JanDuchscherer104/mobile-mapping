"""Streamlit page: Features & ViewSet (Wo6–Wo7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from mob_map.ui.app import PageContext


def render_feature_page(context: "PageContext") -> None:
    """Render the feature extraction & matching mock page."""

    st.title("Features & ViewSet")
    st.caption("Wo6-Wo7: SURF/SIFT-style extraction + reference viewSet bookkeeping.")

    fig = _build_mock_match_figure(context.frame_index)
    st.plotly_chart(fig, width="stretch")
    st.write(
        "Matches are mocked but wired through Plotly so real detector outputs can drop in later via the `FeaturePipeline`."
    )


def _build_mock_match_figure(frame_index: int) -> go.Figure:
    rng = np.random.RandomState(frame_index)
    keypoints = rng.rand(60, 2)
    offsets = rng.normal(scale=0.05, size=(60, 2))
    ref = keypoints
    live = keypoints + offsets
    fig = go.Figure()
    for pt_a, pt_b in zip(ref, live, strict=False):
        fig.add_trace(
            go.Scatter(
                x=[pt_a[0], pt_b[0] + 1.2],
                y=[pt_a[1], pt_b[1]],
                mode="lines",
                line={"color": "#cccccc"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ref[:, 0],
            y=ref[:, 1],
            mode="markers",
            name="Reference",
            marker={"size": 6, "color": "#2f7ed8"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=live[:, 0] + 1.2,
            y=live[:, 1],
            mode="markers",
            name="Frame",
            marker={"size": 6, "color": "#f28f43"},
        )
    )
    fig.update_layout(
        title="Mock feature correspondences",
        xaxis={"showgrid": False, "showticklabels": False, "range": [-0.1, 2.3]},
        yaxis={"showgrid": False, "showticklabels": False, "range": [-0.1, 1.1]},
        height=500,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


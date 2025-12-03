"""Streamlit page: Optimisation Monitor (Wo11–Wo12)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from mob_map.ui.app import PageContext


def render_optimisation_page(context: "PageContext") -> None:
    """Render the optimisation / BA mock dashboard."""

    st.title("Optimisation Monitor")
    st.caption("Wo11-Wo12: Bundle adjustment, sliding windows, and residual QA.")

    iterations = np.arange(0, 15)
    repro_error = 1.2 * np.exp(-0.3 * iterations) + 0.02 * np.random.RandomState(0).randn(iterations.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=repro_error,
            mode="lines+markers",
            name="RMS reprojection (px)",
        )
    )
    fig.update_layout(
        title="Mock residual decay",
        xaxis_title="Iteration",
        yaxis_title="Error (px)",
        yaxis_type="log",
    )
    st.plotly_chart(fig, width="stretch")
    st.success("Plug BA stats from `BundleAdjuster` here (SciPy's `least_squares`).")


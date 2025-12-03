"""Streamlit page: Trajectory Dashboard (Wo9–Wo10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from mob_map.ui.app import PageContext


def render_trajectory_page(context: "PageContext") -> None:
    """Render the trajectory comparison dashboard."""

    st.title("Trajectory Dashboard")
    st.caption("Wo9-Wo10: compare estimated poses against Tsukuba GT.")

    figure = context.visuals.build_trajectory_panel(title="Pose graph preview")
    st.plotly_chart(figure, width="stretch")


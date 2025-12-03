"""Streamlit page: Dense Reconstruction (Wo10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from mob_map.ui.app import PageContext


def render_dense_page(context: "PageContext") -> None:
    """Render the dense reconstruction placeholder page."""

    st.title("Dense Reconstruction")
    st.caption("Wo10: Stereo SGBM + fused Open3D cloud (mocked here).")

    figure = context.visuals.build_pointcloud_panel(title="Point-cloud mock")
    st.plotly_chart(figure, width="stretch")
    st.info("Hook this panel into `PointCloudFuser.fuse` once dense stereo lands.")


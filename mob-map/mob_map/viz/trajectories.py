"""Visualization primitives for Streamlit/Plotly dashboards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        DashboardAssetsConfig,
        PointCloudVisualizerConfig,
        TrajectoryVisualizerConfig,
    )


class TrajectoryVisualizer:
    """Builds pytransform3d/Plotly representations of SE(3) paths."""

    def __init__(self, config: "TrajectoryVisualizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(
        cls, config: "TrajectoryVisualizerConfig"
    ) -> "TrajectoryVisualizer":
        return cls(config=config)

    def build_plot(self, poses) -> object:  # type: ignore[override]
        raise NotImplementedError


class PointCloudVisualizer:
    """Wraps Open3D/Plotly conversions for dense reconstructions."""

    def __init__(self, config: "PointCloudVisualizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(
        cls, config: "PointCloudVisualizerConfig"
    ) -> "PointCloudVisualizer":
        return cls(config=config)

    def build_plot(self, point_cloud) -> object:  # type: ignore[override]
        raise NotImplementedError


class DashboardAssets:
    """Collects visual components for the Streamlit UI."""

    def __init__(self, config: "DashboardAssetsConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.trajectory_viz = config.trajectory_visualizer.setup_target()
        self.pointcloud_viz = config.pointcloud_visualizer.setup_target()

    @classmethod
    def setup_target(cls, config: "DashboardAssetsConfig") -> "DashboardAssets":
        return cls(config=config)

    def render_demo(self) -> None:
        self.console.log(
            "DashboardAssets skeleton initialised – plug Plotly/pytransform3d renders here."
        )

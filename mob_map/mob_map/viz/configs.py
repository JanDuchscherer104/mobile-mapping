"""Visualization configs for Plotly/Streamlit assets."""

from __future__ import annotations

from pydantic import Field

from mob_map.utils.base_config import BaseConfig

from . import trajectories as _trajectories

__all__ = [
    "TrajectoryVisualizerConfig",
    "PointCloudVisualizerConfig",
    "DashboardAssetsConfig",
]


class TrajectoryVisualizerConfig(BaseConfig[_trajectories.TrajectoryVisualizer]):
    target: type[_trajectories.TrajectoryVisualizer] = Field(  # type: ignore[assignment]
        default_factory=lambda: _trajectories.TrajectoryVisualizer,
        exclude=True,
    )
    show_ground_truth: bool = Field(default=True)
    downsample: int = Field(default=1, ge=1)


class PointCloudVisualizerConfig(BaseConfig[_trajectories.PointCloudVisualizer]):
    target: type[_trajectories.PointCloudVisualizer] = Field(  # type: ignore[assignment]
        default_factory=lambda: _trajectories.PointCloudVisualizer,
        exclude=True,
    )
    max_points: int = Field(default=200_000, ge=1)
    color_map: str = Field(default="Viridis")


class DashboardAssetsConfig(BaseConfig[_trajectories.DashboardAssets]):
    target: type[_trajectories.DashboardAssets] = Field(  # type: ignore[assignment]
        default_factory=lambda: _trajectories.DashboardAssets,
        exclude=True,
    )
    trajectory_visualizer: TrajectoryVisualizerConfig = Field(
        default_factory=TrajectoryVisualizerConfig,
    )
    pointcloud_visualizer: PointCloudVisualizerConfig = Field(
        default_factory=PointCloudVisualizerConfig,
    )

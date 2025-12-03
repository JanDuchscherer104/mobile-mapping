"""Visualization primitives for Streamlit/Plotly dashboards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from mob_map.data.types import TrajectoryPose
from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        DashboardAssetsConfig,
        PointCloudVisualizerConfig,
        TrajectoryVisualizerConfig,
    )


@dataclass(slots=True)
class _TrajectorySeries:
    """Container describing one trajectory to be rendered."""

    name: str
    color: str
    samples: np.ndarray


class TrajectoryVisualizer:
    """Build pytransform3d/Plotly representations of SE(3) paths."""

    def __init__(self, config: "TrajectoryVisualizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "TrajectoryVisualizerConfig") -> "TrajectoryVisualizer":
        return cls(config=config)

    def build_plot(
        self,
        estimated: Sequence[TrajectoryPose | np.ndarray] | None = None,
        *,
        ground_truth: Sequence[TrajectoryPose | np.ndarray] | None = None,
        title: str = "Trajectory overview",
    ) -> go.Figure:
        """Return a 3D plotly figure for the provided trajectories.

        Args:
            estimated: Iterable of homogeneous 4x4 poses representing the
                estimated trajectory. When ``None`` a parametric helix is
                generated to keep the UI interactive even before geometry code
                lands.
            ground_truth: Optional iterable with the reference trajectory.
            title: Plot title shown at the top of the figure.

        Returns:
            Plotly figure ready to be fed into Streamlit.
        """

        series: list[_TrajectorySeries] = []
        estimated_positions = self._normalise_positions(estimated)
        series.append(
            _TrajectorySeries(
                name="Estimated",
                color="#2f7ed8",
                samples=estimated_positions,
            )
        )
        if self.config.show_ground_truth:
            gt_positions = self._normalise_positions(
                ground_truth,
                fallback_offset=np.array([0.0, 0.05, 0.0]),
            )
            series.append(
                _TrajectorySeries(
                    name="Ground truth",
                    color="#f28f43",
                    samples=gt_positions,
                )
            )

        figure = go.Figure()
        for traj in series:
            figure.add_trace(
                go.Scatter3d(
                    x=traj.samples[:, 0],
                    y=traj.samples[:, 1],
                    z=traj.samples[:, 2],
                    mode="lines+markers",
                    name=traj.name,
                    line={"color": traj.color, "width": 4},
                    marker={"size": 3, "color": traj.color},
                )
            )

        figure.update_layout(
            title=title,
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Z (m)",
                "aspectmode": "auto",
            },
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            margin={"l": 0, "r": 0, "t": 60, "b": 0},
        )
        return figure

    def _normalise_positions(
        self,
        poses: Sequence[TrajectoryPose | np.ndarray] | None,
        *,
        fallback_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        if poses:
            stacked = np.vstack([self._extract_translation(pose) for pose in poses])
            return stacked
        return self._placeholder_curve(offset=fallback_offset)

    def _extract_translation(self, pose: TrajectoryPose | np.ndarray) -> np.ndarray:
        matrix = getattr(pose, "matrix", pose)
        arr = np.asarray(matrix, dtype=float)
        if arr.shape not in {(4, 4), (3, 4)}:
            raise ValueError("Poses must be 4x4 (or 3x4) homogeneous matrices.")
        if arr.shape == (3, 4):
            homog = np.eye(4)
            homog[:3, :4] = arr
            arr = homog
        return arr[:3, 3]

    def _placeholder_curve(self, *, offset: np.ndarray | None = None) -> np.ndarray:
        theta = np.linspace(0, 3 * np.pi, 80)
        radius = np.linspace(0.2, 0.5, theta.size)
        curve = np.column_stack(
            (
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.linspace(0.0, 0.6, theta.size),
            )
        )
        if offset is not None:
            curve = curve + offset
        return curve


class PointCloudVisualizer:
    """Wrap Open3D/Plotly conversions for dense reconstructions."""

    def __init__(self, config: "PointCloudVisualizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "PointCloudVisualizerConfig") -> "PointCloudVisualizer":
        return cls(config=config)

    def build_plot(
        self,
        point_cloud: np.ndarray | None,
        *,
        title: str = "Point cloud preview",
    ) -> go.Figure:
        """Render a 3D scatter plot for the fused cloud (or a placeholder)."""

        samples = self._prepare_points(point_cloud)
        figure = go.Figure(
            data=[
                go.Scatter3d(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    z=samples[:, 2],
                    mode="markers",
                    marker={
                        "size": 2,
                        "color": samples[:, 2],
                        "colorscale": self.config.color_map,
                        "showscale": True,
                        "colorbar": {"title": "Depth"},
                    },
                )
            ]
        )
        figure.update_layout(
            title=title,
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Z (m)",
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "t": 60, "b": 0},
        )
        return figure

    def _prepare_points(self, cloud: np.ndarray | None) -> np.ndarray:
        if cloud is not None and cloud.size:
            arr = np.asarray(cloud, dtype=float)
            arr = arr.reshape(-1, 3)
            if arr.shape[0] > self.config.max_points:
                arr = arr[: self.config.max_points]
            return arr
        return self._placeholder_cloud()

    def _placeholder_cloud(self) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        radius = 0.5 + 0.2 * np.cos(3 * phi_grid)
        x = radius * np.sin(phi_grid) * np.cos(theta_grid)
        y = radius * np.sin(phi_grid) * np.sin(theta_grid)
        z = radius * np.cos(phi_grid)
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


class DashboardAssets:
    """Collect visual components for the Streamlit UI."""

    def __init__(self, config: "DashboardAssetsConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.trajectory_viz = config.trajectory_visualizer.setup_target()
        self.pointcloud_viz = config.pointcloud_visualizer.setup_target()

    @classmethod
    def setup_target(cls, config: "DashboardAssetsConfig") -> "DashboardAssets":
        return cls(config=config)

    def build_trajectory_panel(
        self,
        *,
        estimated: Sequence[TrajectoryPose | np.ndarray] | None = None,
        ground_truth: Sequence[TrajectoryPose | np.ndarray] | None = None,
        title: str = "Trajectory comparison",
    ) -> go.Figure:
        """Return a Plotly figure that overlays estimated and GT trajectories."""

        return self.trajectory_viz.build_plot(
            estimated,
            ground_truth=ground_truth,
            title=title,
        )

    def build_pointcloud_panel(
        self,
        *,
        points: np.ndarray | None = None,
        title: str = "Dense fusion preview",
    ) -> go.Figure:
        """Return a Plotly figure for dense reconstruction QA."""

        return self.pointcloud_viz.build_plot(points, title=title)

    def render_demo(self) -> None:
        self.console.log("DashboardAssets initialised – ready to feed Plotly figures to Streamlit.")

"""Geometry-layer skeletons: epipolar estimation and pose graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        EpipolarEstimatorConfig,
        PoseEstimatorConfig,
        PoseGraphBuilderConfig,
    )


class EpipolarEstimator:
    """Computes fundamental / essential matrices from matches."""

    def __init__(self, config: "EpipolarEstimatorConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "EpipolarEstimatorConfig") -> "EpipolarEstimator":
        return cls(config=config)

    def estimate(self, points_a, points_b):  # type: ignore[override]
        raise NotImplementedError


class PoseEstimator:
    """Recovers SE(3) relative poses from essential matrices."""

    def __init__(self, config: "PoseEstimatorConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "PoseEstimatorConfig") -> "PoseEstimator":
        return cls(config=config)

    def recover_pose(self, essential_matrix, intrinsics):  # type: ignore[override]
        raise NotImplementedError


class PoseGraphBuilder:
    """Generates Open3D pose graphs ready for optimisation."""

    def __init__(self, config: "PoseGraphBuilderConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "PoseGraphBuilderConfig") -> "PoseGraphBuilder":
        return cls(config=config)

    def add_edge(self, source: int, target: int, measurement) -> None:  # type: ignore[override]
        raise NotImplementedError

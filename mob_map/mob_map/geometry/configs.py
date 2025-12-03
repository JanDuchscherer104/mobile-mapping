"""Configuration objects for geometry estimation."""

from __future__ import annotations

from pydantic import Field

from mob_map.utils.base_config import BaseConfig

from . import estimators as _estimators

__all__ = [
    "EpipolarEstimatorConfig",
    "PoseEstimatorConfig",
    "PoseGraphBuilderConfig",
]


class EpipolarEstimatorConfig(BaseConfig[_estimators.EpipolarEstimator]):
    """RANSAC and normalisation settings for epipolar estimation."""

    target: type[_estimators.EpipolarEstimator] = Field(  # type: ignore[assignment]
        default_factory=lambda: _estimators.EpipolarEstimator,
        exclude=True,
    )
    method: str = Field(default="RANSAC")
    reprojection_threshold: float = Field(default=1.0)
    confidence: float = Field(default=0.999)


class PoseEstimatorConfig(BaseConfig[_estimators.PoseEstimator]):
    """Essential-matrix decomposition settings."""

    target: type[_estimators.PoseEstimator] = Field(  # type: ignore[assignment]
        default_factory=lambda: _estimators.PoseEstimator,
        exclude=True,
    )
    enforce_chirality: bool = Field(default=True)
    scale_resolution: str = Field(
        default="ground_truth_alignment",
        description="Strategy to resolve the translation scale ambiguity.",
    )


class PoseGraphBuilderConfig(BaseConfig[_estimators.PoseGraphBuilder]):
    """Controls pose-graph creation and edge bookkeeping."""

    target: type[_estimators.PoseGraphBuilder] = Field(  # type: ignore[assignment]
        default_factory=lambda: _estimators.PoseGraphBuilder,
        exclude=True,
    )
    max_window: int = Field(default=30, ge=1)
    information_scaling: float = Field(default=1.0, ge=0.0)

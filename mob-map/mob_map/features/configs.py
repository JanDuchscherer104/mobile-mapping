"""Pydantic configs for feature extraction and view-set management."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mob_map.utils.base_config import BaseConfig

from . import detectors as _detectors

__all__ = [
    "FeatureExtractorConfig",
    "FeatureMatcherConfig",
    "ViewSetRepositoryConfig",
    "FeaturePipelineConfig",
]


class DetectorParameters(BaseModel):
    """Miscellaneous hyper-parameters shared across detectors."""

    max_keypoints: int = Field(default=2_000, ge=1)
    contrast_threshold: float = Field(default=0.04, ge=0.0)
    edge_threshold: float = Field(default=10.0, ge=0.0)


class FeatureExtractorConfig(BaseConfig[_detectors.FeatureExtractor]):
    """Configure the OpenCV Feature2D wrapper."""

    target: type[_detectors.FeatureExtractor] = Field(  # type: ignore[assignment]
        default_factory=lambda: _detectors.FeatureExtractor,
        exclude=True,
    )
    algorithm: str = Field(default="SIFT", description="OpenCV detector name")
    parameters: DetectorParameters = Field(default_factory=DetectorParameters)


class FeatureMatcherConfig(BaseConfig[_detectors.FeatureMatcher]):
    """Select the matching strategy (BF, FLANN, cross-check)."""

    target: type[_detectors.FeatureMatcher] = Field(  # type: ignore[assignment]
        default_factory=lambda: _detectors.FeatureMatcher,
        exclude=True,
    )
    matcher_type: str = Field(default="BF", description="e.g. BF, FLANN")
    ratio_test: float = Field(default=0.75, ge=0.0, le=1.0)


class ViewSetRepositoryConfig(BaseConfig[_detectors.ViewSetRepository]):
    """Persistent viewSet equivalent for Python."""

    target: type[_detectors.ViewSetRepository] = Field(  # type: ignore[assignment]
        default_factory=lambda: _detectors.ViewSetRepository,
        exclude=True,
    )
    storage_path: str | None = Field(
        default=None,
        description="Optional JSON/SQLite path for storing view metadata.",
    )


class FeaturePipelineConfig(BaseConfig[_detectors.FeaturePipeline]):
    """Bundle extractor, matcher and repository configs."""

    target: type[_detectors.FeaturePipeline] = Field(  # type: ignore[assignment]
        default_factory=lambda: _detectors.FeaturePipeline,
        exclude=True,
    )
    extractor: FeatureExtractorConfig = Field(default_factory=FeatureExtractorConfig)
    matcher: FeatureMatcherConfig = Field(default_factory=FeatureMatcherConfig)
    view_repository: ViewSetRepositoryConfig = Field(
        default_factory=ViewSetRepositoryConfig,
    )

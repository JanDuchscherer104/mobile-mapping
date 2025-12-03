"""Feature extraction and matching skeletons."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        FeatureExtractorConfig,
        FeatureMatcherConfig,
        FeaturePipelineConfig,
        ViewSetRepositoryConfig,
    )


class FeatureExtractor:
    """Wraps OpenCV Feature2D APIs behind config-as-factory."""

    def __init__(self, config: "FeatureExtractorConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "FeatureExtractorConfig") -> "FeatureExtractor":
        return cls(config=config)

    def extract(self, image) -> tuple[Sequence[float], memoryview]:  # type: ignore[override]
        """Return keypoints/descriptors for downstream steps."""
        raise NotImplementedError


class FeatureMatcher:
    """Produces matches between descriptor sets."""

    def __init__(self, config: "FeatureMatcherConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "FeatureMatcherConfig") -> "FeatureMatcher":
        return cls(config=config)

    def match(self, descriptors_a, descriptors_b) -> Iterable[tuple[int, int]]:  # type: ignore[override]
        """Yield match indices according to configured metric."""
        raise NotImplementedError


class ViewSetRepository:
    """Stores per-view metadata similar to MATLAB's viewSet."""

    def __init__(self, config: "ViewSetRepositoryConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "ViewSetRepositoryConfig") -> "ViewSetRepository":
        return cls(config=config)

    def add_view(self, frame_id: int, **kwargs) -> None:
        raise NotImplementedError


class FeaturePipeline:
    """Small orchestrator bundling extractor/matcher/view repository."""

    def __init__(self, config: "FeaturePipelineConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.extractor = config.extractor.setup_target()
        self.matcher = config.matcher.setup_target()
        self.view_repository = config.view_repository.setup_target()

    @classmethod
    def setup_target(cls, config: "FeaturePipelineConfig") -> "FeaturePipeline":
        return cls(config=config)

    def run(self) -> None:
        self.console.log("FeaturePipeline skeleton initialised – plug real extraction/matching here.")

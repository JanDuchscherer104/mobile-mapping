"""Optimisation skeletons covering BA, sliding windows and anchors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        BundleAdjusterConfig,
        SlidingWindowOptimizerConfig,
        SupportPointOptimizerConfig,
    )


class BundleAdjuster:
    """Global bundle adjustment entry point."""

    def __init__(self, config: "BundleAdjusterConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "BundleAdjusterConfig") -> "BundleAdjuster":
        return cls(config=config)

    def optimise(self, pose_graph, tracks):  # type: ignore[override]
        raise NotImplementedError


class SlidingWindowOptimizer:
    """Runs windowed BA centred on ±W/2 frames."""

    def __init__(self, config: "SlidingWindowOptimizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(
        cls, config: "SlidingWindowOptimizerConfig"
    ) -> "SlidingWindowOptimizer":
        return cls(config=config)

    def refine(self, pose_graph, window_size: int):  # type: ignore[override]
        raise NotImplementedError


class SupportPointOptimizer:
    """Attaches anchor poses (support points) and invokes LM."""

    def __init__(self, config: "SupportPointOptimizerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "SupportPointOptimizerConfig") -> "SupportPointOptimizer":
        return cls(config=config)

    def stabilise(self, pose_graph, anchors):  # type: ignore[override]
        raise NotImplementedError

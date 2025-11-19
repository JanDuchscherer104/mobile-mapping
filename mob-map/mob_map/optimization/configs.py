"""Optimisation configs referencing SciPy/Open3D backends."""

from __future__ import annotations

from pydantic import Field

from mob_map.utils.base_config import BaseConfig

from . import bundle as _bundle

__all__ = [
    "BundleAdjusterConfig",
    "SlidingWindowOptimizerConfig",
    "SupportPointOptimizerConfig",
]


class BundleAdjusterConfig(BaseConfig[_bundle.BundleAdjuster]):
    target: type[_bundle.BundleAdjuster] = Field(  # type: ignore[assignment]
        default_factory=lambda: _bundle.BundleAdjuster,
        exclude=True,
    )
    loss: str = Field(default="soft_l1")
    max_iterations: int = Field(default=50, ge=1)


class SlidingWindowOptimizerConfig(BaseConfig[_bundle.SlidingWindowOptimizer]):
    target: type[_bundle.SlidingWindowOptimizer] = Field(  # type: ignore[assignment]
        default_factory=lambda: _bundle.SlidingWindowOptimizer,
        exclude=True,
    )
    window_size: int = Field(default=10, ge=2)
    stride: int = Field(default=5, ge=1)


class SupportPointOptimizerConfig(BaseConfig[_bundle.SupportPointOptimizer]):
    target: type[_bundle.SupportPointOptimizer] = Field(  # type: ignore[assignment]
        default_factory=lambda: _bundle.SupportPointOptimizer,
        exclude=True,
    )
    optimiser: str = Field(default="lm")
    anchor_weight: float = Field(default=10.0, ge=0.0)

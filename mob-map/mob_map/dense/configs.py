"""Dense stereo configuration objects."""

from __future__ import annotations

from pydantic import Field

from mob_map.utils.base_config import BaseConfig

from . import stereo as _stereo

__all__ = [
    "StereoRectifierConfig",
    "DisparityComputerConfig",
    "PointCloudFuserConfig",
]


class StereoRectifierConfig(BaseConfig[_stereo.StereoRectifier]):
    target: type[_stereo.StereoRectifier] = Field(  # type: ignore[assignment]
        default_factory=lambda: _stereo.StereoRectifier,
        exclude=True,
    )
    zero_disparity: bool = Field(default=True)


class DisparityComputerConfig(BaseConfig[_stereo.DisparityComputer]):
    target: type[_stereo.DisparityComputer] = Field(  # type: ignore[assignment]
        default_factory=lambda: _stereo.DisparityComputer,
        exclude=True,
    )
    matcher: str = Field(default="SGBM")
    min_disparity: int = Field(default=0)
    num_disparities: int = Field(default=64)
    block_size: int = Field(default=5)


class PointCloudFuserConfig(BaseConfig[_stereo.PointCloudFuser]):
    target: type[_stereo.PointCloudFuser] = Field(  # type: ignore[assignment]
        default_factory=lambda: _stereo.PointCloudFuser,
        exclude=True,
    )
    voxel_size: float = Field(default=0.01, gt=0)
    crop_z: tuple[float, float] | None = Field(default=None)

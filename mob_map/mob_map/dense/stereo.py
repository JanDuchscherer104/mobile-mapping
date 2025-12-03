"""Dense stereo + fusion placeholders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import (
        DisparityComputerConfig,
        PointCloudFuserConfig,
        StereoRectifierConfig,
    )


class StereoRectifier:
    """Handles stereo calibration and rectification maps."""

    def __init__(self, config: "StereoRectifierConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "StereoRectifierConfig") -> "StereoRectifier":
        return cls(config=config)

    def rectify(self, left_image, right_image):  # type: ignore[override]
        raise NotImplementedError


class DisparityComputer:
    """Runs OpenCV SGBM/BM according to config."""

    def __init__(self, config: "DisparityComputerConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "DisparityComputerConfig") -> "DisparityComputer":
        return cls(config=config)

    def compute(self, rectified_pair):  # type: ignore[override]
        raise NotImplementedError


class PointCloudFuser:
    """Transforms local depth maps into a global Open3D point cloud."""

    def __init__(self, config: "PointCloudFuserConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)

    @classmethod
    def setup_target(cls, config: "PointCloudFuserConfig") -> "PointCloudFuser":
        return cls(config=config)

    def fuse(self, disparity_map, pose_graph):  # type: ignore[override]
        raise NotImplementedError

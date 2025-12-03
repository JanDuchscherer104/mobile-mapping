"""Data types and constants used by the mob-map data module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class StereoFramePaths:
    """Bundle of left/right/(optional) disparity image paths for a frame."""

    index: int
    left: Path
    right: Path
    disparity: Path | None


@dataclass(frozen=True, slots=True)
class TrajectoryPose:
    """Homogeneous pose loaded from the Tsukuba ground-truth sequence."""

    index: int
    matrix: np.ndarray
    source: str


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Simple pinhole intrinsics used by the Tsukuba datasets."""

    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    skew: float = 0
    width: int = 640
    height: int = 480

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x3 calibration matrix."""

        return np.array(
            [[self.fx, self.skew, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class Images:
    """Container for stereo images and optional disparity map."""

    left: np.ndarray
    right: np.ndarray
    disparity: np.ndarray | None

    def undistort(self, intrinsics: CameraIntrinsics) -> Images:
        """Return a new Images instance with undistorted images."""

        from mob_map.image_processing.undistort import undistort_image

        left_undistorted = undistort_image(self.left, intrinsics)
        right_undistorted = undistort_image(self.right, intrinsics)

        return Images(
            left=left_undistorted,
            right=right_undistorted,
            disparity=self.disparity,
        )

    def to_rgb(self) -> Images:
        """Return a new Images instance with RGB images."""

        def _to_rgb(image: np.ndarray) -> np.ndarray:
            if image is None:
                return image
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            raise ValueError("Unsupported image shape for RGB conversion")

        left_rgb = _to_rgb(self.left)
        right_rgb = _to_rgb(self.right)

        return Images(
            left=left_rgb,
            right=right_rgb,
            disparity=self.disparity,
        )

    def to_gray(self) -> Images:
        """Return a new Images instance with grayscale images."""

        def _to_gray(image: np.ndarray) -> np.ndarray:
            if image is None:
                return image
            if image.ndim == 2:
                return image
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            raise ValueError("Unsupported image shape for grayscale conversion")

        left_gray = _to_gray(self.left)
        right_gray = _to_gray(self.right)

        return Images(
            left=left_gray,
            right=right_gray,
            disparity=self.disparity,
        )


@dataclass(slots=True)
class DataSample:
    """Container returned by :class:`DataModule` for each dataset item."""

    index: int
    paths: StereoFramePaths
    images: Images
    intrinsics: CameraIntrinsics
    pose: TrajectoryPose | None

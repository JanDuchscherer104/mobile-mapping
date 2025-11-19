"""Data types and constants used by the mob-map data module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel


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


class CameraIntrinsics(BaseModel):
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

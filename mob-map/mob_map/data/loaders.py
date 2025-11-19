"""Data-layer runtime objects for the Tsukuba-style datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pytransform3d.transformations import check_transform

from mob_map.utils.console import Console

from .types import StereoFramePaths, TrajectoryPose

if TYPE_CHECKING:  # pragma: no cover - import cycle safe annotations
    from .configs import (
        ImageSequenceLoaderConfig,
        StereoPairIteratorConfig,
        TrajectoryRepositoryConfig,
    )


class ImageSequenceLoader:
    """Load ordered stereo frames plus calibration metadata."""

    def __init__(self, config: "ImageSequenceLoaderConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.sequence_root = config.sequence_root
        self.images_root = (self.sequence_root / config.images_root).resolve()
        self._left_paths = self._collect_paths(config.left_pattern)
        self._right_paths = self._collect_paths(config.right_pattern)
        if len(self._left_paths) != len(self._right_paths):
            raise ValueError("Left/right image counts differ - check dataset integrity.")
        self._disparity_paths = self._prepare_disparities(config.disparity_pattern)

    # ------------------------------------------------------------------ helpers
    def _collect_paths(self, pattern: str, *, required: bool = True) -> tuple[Path, ...]:
        matches = sorted(self.images_root.glob(pattern))
        resolved = tuple(path.resolve() for path in matches)
        if required and not resolved:
            raise FileNotFoundError(f"Pattern '{pattern}' yielded no files in {self.images_root}")
        return resolved

    def _prepare_disparities(self, pattern: str | None) -> tuple[Path | None, ...]:
        if not pattern:
            return tuple([None] * len(self._left_paths))
        disparity_matches = self._collect_paths(pattern, required=False)
        if not disparity_matches:
            self.console.warn("No disparity images found - downstream dense stage will skip them.")
            return tuple([None] * len(self._left_paths))
        lookup = {path.stem: path for path in disparity_matches}
        mapped: list[Path | None] = []
        missing = 0
        for left in self._left_paths:
            disparity = lookup.get(left.stem)
            if disparity is None:
                missing += 1
            mapped.append(disparity)
        if missing:
            self.console.warn(f"Missing {missing} disparity maps. Frames without disparity will yield None.")
        return tuple(mapped)

    def _read_image(self, path: Path, *, color: bool) -> np.ndarray:
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        array = cv2.imread(str(path), flag)
        if array is None:
            raise FileNotFoundError(f"Failed to read image at {path}")
        return array

    # ------------------------------------------------------------------- public
    def list_images(self) -> Sequence[Path]:
        """Return the ordered list of left-image paths."""

        return self._left_paths

    @property
    def frame_count(self) -> int:
        """Total number of frames in the sequence."""

        return len(self._left_paths)

    @property
    def has_disparity(self) -> bool:
        """Whether disparity images are available for every frame."""

        return all(path is not None for path in self._disparity_paths)

    def get_frame_paths(self, index: int) -> StereoFramePaths:
        """Return file paths for the requested frame index."""

        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame index {index} outside [0, {self.frame_count})")
        return StereoFramePaths(
            index=index,
            left=self._left_paths[index],
            right=self._right_paths[index],
            disparity=self._disparity_paths[index],
        )

    def load_frame(
        self,
        index: int,
        *,
        color: bool = False,
        load_disparity: bool = True,
    ) -> dict[str, np.ndarray | None]:
        """Load stereo images (and optionally disparity) as numpy arrays."""

        paths = self.get_frame_paths(index)
        data: dict[str, np.ndarray | None] = {
            "left": self._read_image(paths.left, color=color),
            "right": self._read_image(paths.right, color=color),
            "disparity": None,
        }
        if load_disparity and paths.disparity is not None:
            data["disparity"] = self._read_image(paths.disparity, color=False)
        return data


class StereoPairIterator:
    """Iterates over calibrated left/right/disparity triplets."""

    def __init__(self, config: "StereoPairIteratorConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._loader: ImageSequenceLoader | None = None

    def bind_loader(self, loader: ImageSequenceLoader) -> None:
        """Register the loader that supplies frame paths."""

        self._loader = loader

    def _require_loader(self) -> ImageSequenceLoader:
        if self._loader is None:
            raise RuntimeError("Call bind_loader before iterating over stereo pairs.")
        return self._loader

    def iter_pairs(self) -> Iterator[StereoFramePaths]:
        """Yield per-frame path bundles for downstream dense stereo."""

        loader = self._require_loader()
        for idx in range(loader.frame_count):
            paths = loader.get_frame_paths(idx)
            if self.config.require_disparity and paths.disparity is None:
                raise FileNotFoundError(f"Disparity image required but missing for frame {idx}.")
            yield paths


class TrajectoryRepository:
    """Load Tsukuba ground-truth (and optional viewSet) trajectories."""

    def __init__(self, config: "TrajectoryRepositoryConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._ground_truth: tuple[TrajectoryPose, ...] | None = None

    def load_ground_truth(self) -> tuple[TrajectoryPose, ...]:
        """Return cached ground-truth SE(3) poses parsed from the text file."""

        if self._ground_truth is None:
            self._ground_truth = self._parse_ground_truth()
        return self._ground_truth

    def _parse_ground_truth(self) -> tuple[TrajectoryPose, ...]:
        poses: list[TrajectoryPose] = []
        with self.config.ground_truth_file.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                values = [float(token) for token in stripped.split()]
                if len(values) != 12:
                    raise ValueError("Each ground-truth row must contain 12 values (3x4 matrix).")
                matrix = np.eye(4, dtype=float)
                matrix[:3, :4] = np.array(values, dtype=float).reshape(3, 4)
                matrix[:3, :3] = self._project_rotation(matrix[:3, :3], idx)
                self._validate_pose(matrix, idx)
                poses.append(
                    TrajectoryPose(
                        index=idx,
                        matrix=matrix,
                        source=str(self.config.ground_truth_file),
                    )
                )
        if not poses:
            raise ValueError(f"Ground-truth pose file {self.config.ground_truth_file} is empty.")
        return tuple(poses)

    # ----------------------------------------------------------------- utilities
    def _project_rotation(self, matrix: np.ndarray, idx: int) -> np.ndarray:
        """Project an arbitrary 3x3 matrix onto SO(3) via SVD."""

        u, _, vh = np.linalg.svd(matrix)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
            self.console.warn(
                f"Pose {idx}: detected improper rotation (det<0); flipping last column to enforce right-handedness."
            )
        return rotation

    def _validate_pose(self, pose: np.ndarray, idx: int) -> None:
        """Ensure SE(3) validity with relaxed tolerance, raising on failure."""

        try:
            check_transform(pose, strict_check=False)
        except ValueError as exc:  # pragma: no cover - depends on dataset quality
            msg = f"Pose {idx} remains invalid after orthonormalisation: {exc}"
            raise ValueError(msg) from exc

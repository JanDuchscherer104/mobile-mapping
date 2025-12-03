"""Dataset-style wrapper around the Tsukuba mobile mapping resources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mob_map.utils.console import Console

from .types import CameraIntrinsics, DataSample, Images

if TYPE_CHECKING:  # pragma: no cover - import cycle safe annotations
    from .configs import DataModuleConfig


class DataModule(Sequence[DataSample]):
    """High-level orchestrator bundling loaders, iterators and trajectories."""

    def __init__(self, config: "DataModuleConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.loader = config.sequence_loader.setup_target()
        self.stereo_iterator = config.stereo_iterator.setup_target()
        self.stereo_iterator.bind_loader(self.loader)
        self.trajectories = config.trajectory_repository.setup_target()
        self._ground_truth = self.trajectories.load_ground_truth()

        self._intinsics = CameraIntrinsics()

    # --------------------------------------------------------------------- info
    def summary(self) -> dict[str, object]:
        """Return and log quick metadata about the dataset."""
        sample = self[0]
        info = {
            "sequence": self.loader.config.sequence,
            "has_disparity": self.loader.has_disparity,
            "camera_matrix": str(self._intinsics.matrix.tolist()),
            "ground_truth_poses": len(self._ground_truth),
            "frames": self.loader.frame_count,
            "image_sizes": {
                "left": sample.images.left.shape,
                "right": sample.images.right.shape,
                "disparity": sample.images.disparity.shape if sample.images.disparity is not None else None,
            },
        }
        self.console.plog(info)
        return info

    # ------------------------------------------------------------------ dataset
    def __getitem__(self, index: int | slice) -> DataSample | list[DataSample]:
        """Return one (or multiple) dataset samples."""

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index, int):  # pragma: no cover - sanity guard
            raise TypeError("DataModule indices must be integers or slices.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):  # pragma: no cover - bounds guard
            raise IndexError(f"Frame index {index} is outside dataset bounds")

        images = self.loader.load_frame(index, color=True, load_disparity=True)
        paths = self.loader.get_frame_paths(index)
        pose = self._ground_truth[index] if index < len(self._ground_truth) else None

        return DataSample(
            index=index,
            paths=paths,
            images=Images(left=images["left"], right=images["right"], disparity=images["disparity"]),
            intrinsics=self._intinsics,
            pose=pose,
        )

    def __repr__(self) -> str:
        """Compact textual representation for debugging and CLI output."""

        return (
            f"{self.__class__.__name__}(sequence='{self.loader.config.sequence}', "
            f"frames={len(self)}, has_disparity={self.loader.has_disparity})"
        )

    def __len__(self) -> int:
        return self.loader.frame_count

"""Configurations powering the data-loading layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mob_map.utils.base_config import BaseConfig

from . import datamodule as _datamodule
from . import loaders as _loaders

__all__ = [
    "ImageSequenceLoaderConfig",
    "StereoPairIteratorConfig",
    "TrajectoryRepositoryConfig",
    "DataModuleConfig",
]


def _default_dataset_root() -> Path:
    return (Path(__file__).resolve().parents[3] / ".data/Proj").resolve()


@dataclass(frozen=True)
class _DatasetMetadata:
    sequence_dir: str
    ground_truth_txt: str
    viewset_real: str
    viewset_est: str


_DATASETS: dict[str, _DatasetMetadata] = {
    "tsukuba10": _DatasetMetadata(
        sequence_dir="Bildseq_Tsukuba_lab_10_images",
        ground_truth_txt="Bildseq_Tsukuba_lab_10_images/tsukuba_ground_truth_poses/tsukuba_ground_truth_poses.txt",
        viewset_real="vSet_real.mat",
        viewset_est="vSet_est.mat",
    ),
    "tsukuba40": _DatasetMetadata(
        sequence_dir="Bildseq_Tsukuba_lab_40_images",
        ground_truth_txt="Bildseq_Tsukuba_lab_40_images/tsukuba_ground_truth_poses/tsukuba_ground_truth_poses.txt",
        viewset_real="vSet_real.mat",
        viewset_est="vSet_est.mat",
    ),
}


class ImageSequenceLoaderConfig(BaseConfig[_loaders.ImageSequenceLoader]):
    """Configure how stereo frames are located and normalised."""

    target: type[_loaders.ImageSequenceLoader] = Field(  # type: ignore[assignment]
        default_factory=lambda: _loaders.ImageSequenceLoader,
        exclude=True,
    )
    dataset_root: Path = Field(
        default_factory=_default_dataset_root,
        description="Folder containing Tsukuba-style datasets.",
    )
    sequence: str = Field(
        default=_DATASETS["tsukuba10"].sequence_dir,
        description="Sub-folder name within dataset_root to load.",
    )
    images_root: str = Field(default="mod")
    left_pattern: str = Field(default="left/image*.jpg")
    right_pattern: str = Field(default="right/image*.jpg")
    disparity_pattern: str | None = Field(default="disp/image*.jpg")

    @property
    def sequence_root(self) -> Path:
        return (self.dataset_root / self.sequence).resolve()

    @model_validator(mode="after")
    def _normalise_paths(self) -> "ImageSequenceLoaderConfig":
        normalised_root = self.dataset_root.expanduser().resolve()
        object.__setattr__(self, "dataset_root", normalised_root)
        if not self.sequence_root.exists():
            raise FileNotFoundError(f"Sequence '{self.sequence}' not found under {self.dataset_root}")
        images_root = (self.sequence_root / self.images_root).resolve()
        if not images_root.exists():
            raise FileNotFoundError(f"Images root '{images_root}' is missing; check dataset extraction.")
        return self


class StereoPairIteratorConfig(BaseConfig[_loaders.StereoPairIterator]):
    """Iterate over rectified stereo/disparity triplets."""

    target: type[_loaders.StereoPairIterator] = Field(  # type: ignore[assignment]
        default_factory=lambda: _loaders.StereoPairIterator,
        exclude=True,
    )
    require_disparity: bool = Field(
        default=False,
        description="Raise an error if a disparity image is missing for a frame.",
    )


class TrajectoryRepositoryConfig(BaseConfig[_loaders.TrajectoryRepository]):
    """Access trajectories stored in text files or MATLAB viewSet archives."""

    target: type[_loaders.TrajectoryRepository] = Field(  # type: ignore[assignment]
        default_factory=lambda: _loaders.TrajectoryRepository,
        exclude=True,
    )
    ground_truth_file: Path = Field(
        default_factory=lambda: _default_dataset_root() / _DATASETS["tsukuba10"].ground_truth_txt,
        description="Plain-text file with 3x4 poses (tsukuba_ground_truth_poses.txt).",
    )
    viewset_real: Path | None = Field(
        default_factory=lambda: _default_dataset_root() / _DATASETS["tsukuba10"].viewset_real,
        description="MATLAB viewSet containing the same real poses (optional).",
    )
    viewset_est: Path | None = Field(
        default_factory=lambda: _default_dataset_root() / _DATASETS["tsukuba10"].viewset_est,
        description="MATLAB viewSet containing the estimated poses (optional).",
    )

    @model_validator(mode="after")
    def _normalise_files(self) -> "TrajectoryRepositoryConfig":
        ground_truth = self.ground_truth_file.expanduser().resolve()
        object.__setattr__(self, "ground_truth_file", ground_truth)
        if not self.ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth pose file missing: {self.ground_truth_file}")
        if self.viewset_real is not None:
            object.__setattr__(
                self,
                "viewset_real",
                self.viewset_real.expanduser().resolve(),
            )
        if self.viewset_est is not None:
            object.__setattr__(
                self,
                "viewset_est",
                self.viewset_est.expanduser().resolve(),
            )
        return self


class DataModuleConfig(BaseConfig[_datamodule.DataModule]):
    """Composite config bundling all data-layer components."""

    target: type[_datamodule.DataModule] = Field(  # type: ignore[assignment]
        default_factory=lambda: _datamodule.DataModule,
        exclude=True,
    )
    dataset: Literal["tsukuba10", "tsukuba40"] = "tsukuba10"

    sequence_loader: ImageSequenceLoaderConfig = Field(
        default_factory=ImageSequenceLoaderConfig,
    )
    stereo_iterator: StereoPairIteratorConfig = Field(
        default_factory=StereoPairIteratorConfig,
    )
    trajectory_repository: TrajectoryRepositoryConfig = Field(
        default_factory=TrajectoryRepositoryConfig,
    )

    @model_validator(mode="after")
    def _apply_dataset_defaults(self) -> "DataModuleConfig":
        metadata = _DATASETS[self.dataset]
        self.sequence_loader.sequence = metadata.sequence_dir
        dataset_root = self.sequence_loader.dataset_root
        ground_truth = (dataset_root / metadata.ground_truth_txt).resolve()
        if not ground_truth.exists():
            raise FileNotFoundError(f"Expected ground-truth poses at {ground_truth} (dataset {self.dataset})")
        self.trajectory_repository.ground_truth_file = ground_truth

        viewset_real = (dataset_root / metadata.viewset_real).resolve()
        viewset_est = (dataset_root / metadata.viewset_est).resolve()
        if self.trajectory_repository.viewset_real is None and viewset_real.exists():
            self.trajectory_repository.viewset_real = viewset_real
        if self.trajectory_repository.viewset_est is None and viewset_est.exists():
            self.trajectory_repository.viewset_est = viewset_est
        return self

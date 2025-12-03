"""UI-level configuration tying together all modules."""

from __future__ import annotations

from pydantic import Field

from mob_map.data import DataModuleConfig
from mob_map.dense import PointCloudFuserConfig
from mob_map.features import FeaturePipelineConfig
from mob_map.geometry import PoseGraphBuilderConfig
from mob_map.optimization import BundleAdjusterConfig
from mob_map.utils.base_config import BaseConfig
from mob_map.viz import DashboardAssetsConfig

from . import app as _app

__all__ = ["StreamlitAppConfig"]


class StreamlitAppConfig(BaseConfig[_app.StreamlitApp]):
    target: type[_app.StreamlitApp] = Field(  # type: ignore[assignment]
        default_factory=lambda: _app.StreamlitApp,
        exclude=True,
    )
    data_module: DataModuleConfig = Field(default_factory=DataModuleConfig)
    feature_pipeline: FeaturePipelineConfig = Field(default_factory=FeaturePipelineConfig)
    geometry_stage: PoseGraphBuilderConfig = Field(default_factory=PoseGraphBuilderConfig)
    dense_stage: PointCloudFuserConfig = Field(default_factory=PointCloudFuserConfig)
    optimisation_stage: BundleAdjusterConfig = Field(default_factory=BundleAdjusterConfig)
    dashboard_assets: DashboardAssetsConfig = Field(
        default_factory=DashboardAssetsConfig,
    )

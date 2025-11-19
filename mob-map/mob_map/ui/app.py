"""UI orchestrator hooking up the Streamlit experience."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mob_map.utils.console import Console

if TYPE_CHECKING:  # pragma: no cover
    from .configs import StreamlitAppConfig


class StreamlitApp:
    """Thin wrapper to bootstrap Streamlit pages with cached components."""

    def __init__(self, config: "StreamlitAppConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self.data = config.data_module.setup_target()
        self.features = config.feature_pipeline.setup_target()
        self.geometry = config.geometry_stage.setup_target()
        self.dense = config.dense_stage.setup_target()
        self.optimisation = config.optimisation_stage.setup_target()
        self.visuals = config.dashboard_assets.setup_target()

    @classmethod
    def setup_target(cls, config: "StreamlitAppConfig") -> "StreamlitApp":
        return cls(config=config)

    def run(self) -> None:
        self.console.log(
            "StreamlitApp skeleton initialised – integrate Streamlit pages later."
        )

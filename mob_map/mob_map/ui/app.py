"""UI orchestrator hooking up the Streamlit experience."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import streamlit as st

from mob_map.data.datamodule import DataSample
from mob_map.ui.pages import (
    render_dense_page,
    render_feature_page,
    render_frame_gallery,
    render_optimisation_page,
    render_trajectory_page,
)
from mob_map.utils.console import Console
from mob_map.viz import DashboardAssets

if TYPE_CHECKING:
    from mob_map.dense import PointCloudFuser
    from mob_map.features import FeaturePipeline
    from mob_map.geometry import PoseGraphBuilder
    from mob_map.optimization import BundleAdjuster

    from .configs import StreamlitAppConfig


RenderFunc = Callable[["PageContext"], None]


@dataclass(slots=True)
class PageContext:
    """Shared state passed to every Streamlit page."""

    frame_index: int
    samples: Sequence[DataSample]
    features: "FeaturePipeline"
    geometry: "PoseGraphBuilder"
    dense: "PointCloudFuser"
    optimisation: "BundleAdjuster"
    visuals: DashboardAssets


@dataclass(slots=True)
class PageDefinition:
    """Declarative definition of a Streamlit page."""

    title: str
    icon: str
    render: RenderFunc
    default: bool = False


class StreamlitApp:
    """Bootstrapper that wires configs into a multi-page Streamlit app."""

    def __init__(self, config: "StreamlitAppConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._data: Sequence[DataSample] | None = None
        self._data_cache: dict[str, Sequence[DataSample]] = {}
        self._features: "FeaturePipeline" | None = None
        self._geometry: "PoseGraphBuilder" | None = None
        self._dense: "PointCloudFuser" | None = None
        self._optimisation: "BundleAdjuster" | None = None
        self._visuals: DashboardAssets | None = None

    @property
    def data(self) -> Sequence[DataSample]:
        if self._data is None:
            self._data = self.config.data_module.setup_target()
        return self._data

    @property
    def features(self) -> "FeaturePipeline":
        if self._features is None:
            self._features = self.config.feature_pipeline.setup_target()
        return self._features

    @property
    def geometry(self) -> "PoseGraphBuilder":
        if self._geometry is None:
            self._geometry = self.config.geometry_stage.setup_target()
        return self._geometry

    @property
    def dense(self) -> "PointCloudFuser":
        if self._dense is None:
            self._dense = self.config.dense_stage.setup_target()
        return self._dense

    @property
    def optimisation(self) -> "BundleAdjuster":
        if self._optimisation is None:
            self._optimisation = self.config.optimisation_stage.setup_target()
        return self._optimisation

    @property
    def visuals(self) -> DashboardAssets:
        if self._visuals is None:
            self._visuals = self.config.dashboard_assets.setup_target()
        return self._visuals

    def run(self) -> None:  # pragma: no cover - exercised via Streamlit runtime
        st.set_page_config(
            page_title="Mobile Mapping Workbench",
            page_icon="🛰️",
            layout="wide",
        )

        dataset_key = self._select_dataset(st)
        dataset = self._resolve_dataset(dataset_key, st)
        if dataset is None:
            return

        frame_index = st.sidebar.slider(
            "Frame index",
            min_value=0,
            max_value=max(len(dataset) - 1, 0),
            value=0,
            step=1,
            help="Select which Tsukuba frame to inspect across the dashboards.",
        )
        st.sidebar.caption(f"Sequence: **{dataset.loader.config.sequence}** (frames: {len(dataset)})")

        context = PageContext(
            frame_index=frame_index,
            samples=dataset,
            features=self.features,
            geometry=self.geometry,
            dense=self.dense,
            optimisation=self.optimisation,
            visuals=self.visuals,
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Roadmap Hooks")
        st.sidebar.write("Each page maps to the Wo6-Wo12 milestones described in `ROADMAP.md`.")

        pages = self._build_pages()
        self._render_navigation(pages, context, st)

    # ------------------------------------------------------------------ helpers
    def _select_dataset(self, st_module: Any) -> str:
        options = _dataset_options(self.config)
        default_idx = (
            options.index(self.config.data_module.dataset) if self.config.data_module.dataset in options else 0
        )
        return st_module.sidebar.selectbox(
            "Dataset",
            options=options,
            index=default_idx,
            help="Choose between the 10- or 40-frame Tsukuba bundles. Falls back to a synthetic stub if assets are missing.",
        )

    def _resolve_dataset(self, dataset_key: str, st_module: Any) -> Sequence[DataSample] | None:
        if dataset_key in self._data_cache:
            self._data = self._data_cache[dataset_key]
            return self._data

        data_config = self.config.data_module.model_copy(update={"dataset": dataset_key})
        dataset: Sequence[DataSample] | None = None
        try:
            dataset = _build_data_module_cached(data_config.model_dump_json())
        except FileNotFoundError as exc:
            st_module.exception(exc)

        except Exception as exc:  # pragma: no cover - defensive UI guard
            st_module.exception(exc)

        self._data = dataset
        self._data_cache[dataset_key] = dataset
        return dataset

    def _build_pages(self) -> list[PageDefinition]:
        return [
            PageDefinition(
                title="Init & Frame QA",
                icon=":material/grid_view:",
                render=render_frame_gallery,
                default=True,
            ),
            PageDefinition(
                title="Features & ViewSet",
                icon=":material/hub:",
                render=render_feature_page,
            ),
            PageDefinition(
                title="Trajectory Dashboard",
                icon=":material/radar:",
                render=render_trajectory_page,
            ),
            PageDefinition(
                title="Dense Reconstruction",
                icon=":material/stacks:",
                render=render_dense_page,
            ),
            PageDefinition(
                title="Optimisation Monitor",
                icon=":material/auto_graph:",
                render=render_optimisation_page,
            ),
        ]

    def _render_navigation(
        self,
        pages: Sequence[PageDefinition],
        context: PageContext,
        st_module: Any,
    ) -> None:
        page_callables = [(_make_page_callable(page.render, context), page) for page in pages]
        if hasattr(st_module, "navigation") and hasattr(st_module, "Page"):
            st_pages = []
            for func, definition in page_callables:
                kwargs = {"title": definition.title, "icon": definition.icon}
                if definition.default:
                    kwargs["default"] = True
                st_pages.append(st_module.Page(func, **kwargs))
            st_module.navigation(st_pages).run()
            return

        # Fallback for Streamlit versions lacking st.navigation
        labels = [definition.title for _, definition in page_callables]
        selected = st_module.sidebar.radio("Page", labels, index=0)
        for func, definition in page_callables:
            if definition.title == selected:
                func()
                break


@st.cache_resource(show_spinner=False)
def _build_data_module_cached(config_json: str) -> Sequence[DataSample]:
    """Instantiate a data module while keeping one instance per config signature."""

    from mob_map.data import DataModuleConfig

    config = DataModuleConfig.model_validate_json(config_json)
    return config.setup_target()


def _dataset_options(config: "StreamlitAppConfig") -> list[str]:
    """Return the available dataset identifiers declared on the config Literal."""

    field = getattr(config.data_module.__class__, "model_fields", {}).get("dataset")
    literal = getattr(field, "annotation", None)
    choices = [opt for opt in get_args(literal or ()) if isinstance(opt, str)]
    if not choices:
        return [config.data_module.dataset]
    return list(dict.fromkeys(choices))


def _make_page_callable(render: RenderFunc, context: PageContext) -> Callable[[], None]:
    def _callable() -> None:
        render(context)

    _callable.__name__ = render.__name__  # type: ignore[attr-defined]
    return _callable


def main() -> None:  # pragma: no cover - exercised by Streamlit CLI
    """Launch the Streamlit UI using the default configuration factory."""

    from mob_map.ui.configs import StreamlitAppConfig

    app = StreamlitAppConfig().setup_target()
    app.run()


def streamlit_entry() -> None:  # pragma: no cover - script invocation
    import sys

    from streamlit.web.cli import main as st_main

    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path)]
    st_main()


if __name__ == "__main__":  # pragma: no cover - script invocation
    main()

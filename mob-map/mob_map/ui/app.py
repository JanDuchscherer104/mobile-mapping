"""UI orchestrator hooking up the Streamlit experience."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mob_map.data.datamodule import DataSample
from mob_map.data.types import CameraIntrinsics, StereoFramePaths, TrajectoryPose
from mob_map.image_processing.undistort import undistort_image
from mob_map.utils.console import Console
from mob_map.viz import DashboardAssets

if TYPE_CHECKING:  # pragma: no cover
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
    data: Sequence[DataSample]
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

    @classmethod
    def setup_target(cls, config: "StreamlitAppConfig") -> "StreamlitApp":
        return cls(config=config)

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
            data=dataset,
            features=self.features,
            geometry=self.geometry,
            dense=self.dense,
            optimisation=self.optimisation,
            visuals=self.visuals,
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Roadmap Hooks")
        st.sidebar.write("Each page maps to the Wo6–Wo12 milestones described in `ROADMAP.md`.")

        pages = self._build_pages()
        self._render_navigation(pages, context, st)

    # ------------------------------------------------------------------ helpers
    def _select_dataset(self, st_module) -> str:
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

    def _resolve_dataset(self, dataset_key: str, st_module) -> Sequence[DataSample] | None:
        if dataset_key in self._data_cache:
            self._data = self._data_cache[dataset_key]
            return self._data

        data_config = self.config.data_module.model_copy(update={"dataset": dataset_key})
        try:
            dataset = _build_data_module_cached(data_config.model_dump_json())
        except FileNotFoundError as exc:
            st_module.warning(
                "Dataset assets missing. Falling back to a synthetic placeholder so the UI remains usable."
            )
            st_module.exception(exc)
            dataset = _build_stub_dataset(sequence_label=f"{dataset_key} (stub)")
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st_module.error("Failed to initialise the data module. Showing a synthetic fallback dataset.")
            st_module.exception(exc)
            dataset = _build_stub_dataset(sequence_label=f"{dataset_key} (error)")

        self._data = dataset
        self._data_cache[dataset_key] = dataset
        return dataset

    def _build_pages(self) -> list[PageDefinition]:
        return [
            PageDefinition(
                title="Init & Frame QA",
                icon=":material/grid_view:",
                render=_render_frame_gallery,
                default=True,
            ),
            PageDefinition(
                title="Features & ViewSet",
                icon=":material/hub:",
                render=_render_feature_page,
            ),
            PageDefinition(
                title="Trajectory Dashboard",
                icon=":material/radar:",
                render=_render_trajectory_page,
            ),
            PageDefinition(
                title="Dense Reconstruction",
                icon=":material/stacks:",
                render=_render_dense_page,
            ),
            PageDefinition(
                title="Optimisation Monitor",
                icon=":material/auto_graph:",
                render=_render_optimisation_page,
            ),
        ]

    def _render_navigation(
        self,
        pages: Sequence[PageDefinition],
        context: PageContext,
        st_module,
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
def _build_data_module_cached(config_json: str):
    """Instantiate a data module while keeping one instance per config signature."""

    from mob_map.data import DataModuleConfig

    config = DataModuleConfig.model_validate_json(config_json)
    return config.setup_target()


def _dataset_options(config) -> list[str]:
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


def _render_frame_gallery(context: PageContext) -> None:
    st.title("Init & Frame QA")
    st.caption("Wo6 deliverable: load Tsukuba stereo frames, undistort them, and inspect ground-truth metadata.")

    sample = context.data[context.frame_index]
    dataset_is_stub = getattr(context.data, "is_stub", False)
    if dataset_is_stub:
        st.info("Synthetic placeholder frames are shown because the dataset assets were not found.")
    if dataset_is_stub:
        payload = _build_gallery_payload_from_arrays(
            images=sample.images,
            intrinsics=sample.intrinsics,
        )
    else:
        payload = _load_gallery_payload(
            left_path=str(sample.paths.left),
            right_path=str(sample.paths.right),
            disparity_path=str(sample.paths.disparity) if sample.paths.disparity else None,
            intrinsics_json=sample.intrinsics.model_dump_json(),
        )

    st.subheader("2×3 Frame Grid")
    st.write("Upper row → raw sensors, lower row → undistorted using the mandated intrinsics (f=615px).")
    labels = ["Left", "Right", "Disparity"]
    top = st.columns(len(labels))
    bottom = st.columns(len(labels))
    for idx, key in enumerate(["left", "right", "disparity"]):
        raw = payload.get(f"{key}_raw")
        undistorted = payload.get(f"{key}_undistorted")
        caption = f"{labels[idx]} (raw)"
        _display_image_panel(top[idx], raw, caption)
        _display_image_panel(bottom[idx], undistorted, f"{labels[idx]} (undistorted)")

    metadata_col, pose_col = st.columns(2)
    metadata_col.metric("Frame", sample.index)
    metadata_col.write(f"Paths: `{sample.paths.left.name}` / `{sample.paths.right.name}`")
    if sample.paths.disparity:
        metadata_col.write(f"Disparity: `{sample.paths.disparity.name}`")
    pose_col.subheader("Ground truth pose")
    if sample.pose is None:
        pose_col.info("Ground-truth pose unavailable for this index.")
    else:
        pose_col.write(np.array2string(sample.pose.matrix, precision=3))


def _render_feature_page(context: PageContext) -> None:
    st.title("Features & ViewSet")
    st.caption("Wo6-Wo7: SURF/SIFT-style extraction + reference viewSet bookkeeping.")

    fig = _build_mock_match_figure(context.frame_index)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "Matches are mocked but wired through Plotly so real detector outputs can drop in later via the `FeaturePipeline`."
    )


def _render_trajectory_page(context: PageContext) -> None:
    st.title("Trajectory Dashboard")
    st.caption("Wo9–Wo10: compare estimated poses against Tsukuba GT.")

    figure = context.visuals.build_trajectory_panel(title="Pose graph preview")
    st.plotly_chart(figure, use_container_width=True)


def _render_dense_page(context: PageContext) -> None:
    st.title("Dense Reconstruction")
    st.caption("Wo10: Stereo SGBM + fused Open3D cloud (mocked here).")

    figure = context.visuals.build_pointcloud_panel(title="Point-cloud mock")
    st.plotly_chart(figure, use_container_width=True)
    st.info("Hook this panel into `PointCloudFuser.fuse` once dense stereo lands.")


def _render_optimisation_page(context: PageContext) -> None:
    st.title("Optimisation Monitor")
    st.caption("Wo11–Wo12: Bundle adjustment, sliding windows, and residual QA.")

    iterations = np.arange(0, 15)
    repro_error = 1.2 * np.exp(-0.3 * iterations) + 0.02 * np.random.RandomState(0).randn(iterations.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=repro_error,
            mode="lines+markers",
            name="RMS reprojection (px)",
        )
    )
    fig.update_layout(
        title="Mock residual decay",
        xaxis_title="Iteration",
        yaxis_title="Error (px)",
        yaxis_type="log",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success("Plug BA stats from `BundleAdjuster` here (SciPy's `least_squares`).")


def _build_gallery_payload_from_arrays(
    *,
    images: dict[str, np.ndarray | None],
    intrinsics: CameraIntrinsics,
) -> dict[str, np.ndarray | None]:
    """Prepare gallery payload when frames are already loaded in memory."""

    payload: dict[str, np.ndarray | None] = {}
    left = images.get("left")
    right = images.get("right")
    disparity = images.get("disparity")

    payload["left_raw"] = _bgr_to_rgb(left)
    payload["right_raw"] = _bgr_to_rgb(right)
    payload["disparity_raw"] = disparity

    payload["left_undistorted"] = _bgr_to_rgb(undistort_image(left, intrinsics) if left is not None else None)
    payload["right_undistorted"] = _bgr_to_rgb(undistort_image(right, intrinsics) if right is not None else None)
    payload["disparity_undistorted"] = undistort_image(disparity, intrinsics) if disparity is not None else None
    return payload


@st.cache_data(show_spinner=False)
def _load_gallery_payload(
    *,
    left_path: str,
    right_path: str,
    disparity_path: str | None,
    intrinsics_json: str,
) -> dict[str, np.ndarray | None]:
    """Load and undistort raw frames once per unique path signature."""

    intrinsics = CameraIntrinsics.model_validate_json(intrinsics_json)
    payload: dict[str, np.ndarray | None] = {}
    payload["left_raw"] = _read_color_image(left_path)
    payload["right_raw"] = _read_color_image(right_path)
    payload["disparity_raw"] = _read_gray_image(disparity_path)
    payload["left_undistorted"] = _bgr_to_rgb(
        undistort_image(payload["left_raw"], intrinsics) if payload["left_raw"] is not None else None
    )
    payload["right_undistorted"] = _bgr_to_rgb(
        undistort_image(payload["right_raw"], intrinsics) if payload["right_raw"] is not None else None
    )
    disparity = payload["disparity_raw"]
    payload["disparity_undistorted"] = undistort_image(disparity, intrinsics) if disparity is not None else None
    payload["left_raw"] = _bgr_to_rgb(payload["left_raw"])
    payload["right_raw"] = _bgr_to_rgb(payload["right_raw"])
    return payload


def _read_color_image(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    import cv2

    array = cv2.imread(path, cv2.IMREAD_COLOR)
    if array is None:
        raise FileNotFoundError(path)
    return array


def _read_gray_image(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    import cv2

    array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(path)
    return array


def _display_image_panel(container, image: np.ndarray | None, caption: str) -> None:
    if image is None:
        container.warning("Missing input")
        return
    container.image(image, channels="RGB" if image.ndim == 3 else "L", caption=caption, use_container_width=True)


def _bgr_to_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 3:
        return image[..., ::-1]
    return image


@dataclass(slots=True)
class _StubLoaderConfig:
    sequence: str


@dataclass(slots=True)
class _StubLoader:
    config: _StubLoaderConfig
    frame_count: int
    has_disparity: bool = True


class _StubDataModule(Sequence[DataSample]):
    """Lightweight dataset used when the Tsukuba assets are unavailable."""

    is_stub = True

    def __init__(self, samples: Sequence[DataSample], sequence_label: str) -> None:
        self._samples = list(samples)
        self.loader = _StubLoader(_StubLoaderConfig(sequence_label), frame_count=len(samples))

    def __getitem__(self, index: int) -> DataSample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)


def _build_stub_dataset(sequence_label: str, frames: int = 3) -> _StubDataModule:
    """Construct a synthetic dataset so the UI remains interactive without files."""

    intrinsics = CameraIntrinsics()
    samples: list[DataSample] = []
    for idx in range(frames):
        left = _synthetic_rgb_frame(seed=idx)
        right = _synthetic_rgb_frame(seed=idx + 7)
        disparity = _synthetic_disparity(seed=idx)
        paths = StereoFramePaths(
            index=idx,
            left=Path(f"{sequence_label.replace(' ', '_')}_left_{idx}.png"),
            right=Path(f"{sequence_label.replace(' ', '_')}_right_{idx}.png"),
            disparity=Path(f"{sequence_label.replace(' ', '_')}_disp_{idx}.png"),
        )
        pose = TrajectoryPose(index=idx, matrix=np.eye(4, dtype=float), source="synthetic placeholder")
        samples.append(
            DataSample(
                index=idx,
                paths=paths,
                images={"left": left, "right": right, "disparity": disparity},
                intrinsics=intrinsics,
                pose=pose,
            )
        )
    return _StubDataModule(samples, sequence_label)


def _synthetic_rgb_frame(*, seed: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a smooth gradient RGB frame for placeholder rendering."""

    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    grid_x, grid_y = np.meshgrid(x, y)
    base = np.stack([grid_x, grid_y, 0.5 * np.ones_like(grid_x)], axis=2)
    noise = rng.normal(scale=0.03, size=base.shape)
    image = np.clip(base + noise, 0.0, 1.0)
    return (255 * image).astype(np.uint8)


def _synthetic_disparity(*, seed: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a pseudo disparity map with gentle sinusoidal parallax."""

    x = np.linspace(0, 1, width)
    base = np.tile(x, (height, 1))
    phase = np.sin(2 * np.pi * (np.linspace(0, 1, height).reshape(-1, 1) + 0.1 * seed))
    disparity = np.clip(base + 0.1 * phase, 0.0, 1.0)
    return (255 * disparity).astype(np.uint8)


def _build_mock_match_figure(frame_index: int) -> go.Figure:
    rng = np.random.RandomState(frame_index)
    keypoints = rng.rand(60, 2)
    offsets = rng.normal(scale=0.05, size=(60, 2))
    ref = keypoints
    live = keypoints + offsets
    fig = go.Figure()
    for pt_a, pt_b in zip(ref, live, strict=False):
        fig.add_trace(
            go.Scatter(
                x=[pt_a[0], pt_b[0] + 1.2],
                y=[pt_a[1], pt_b[1]],
                mode="lines",
                line={"color": "#cccccc"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ref[:, 0],
            y=ref[:, 1],
            mode="markers",
            name="Reference",
            marker={"size": 6, "color": "#2f7ed8"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=live[:, 0] + 1.2,
            y=live[:, 1],
            mode="markers",
            name="Frame",
            marker={"size": 6, "color": "#f28f43"},
        )
    )
    fig.update_layout(
        title="Mock feature correspondences",
        xaxis={"showgrid": False, "showticklabels": False, "range": [-0.1, 2.3]},
        yaxis={"showgrid": False, "showticklabels": False, "range": [-0.1, 1.1]},
        height=500,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def main() -> None:  # pragma: no cover - exercised by Streamlit CLI
    """Launch the Streamlit UI using the default configuration factory."""

    from .configs import StreamlitAppConfig

    config = StreamlitAppConfig()
    app = StreamlitApp.setup_target(config)
    app.run()


if __name__ == "__main__":  # pragma: no cover - script invocation
    main()

"""Streamlit page: Init & QA (Wo6 deliverable)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import streamlit as st

from mob_map.data.types import CameraIntrinsics, Images

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from mob_map.ui.app import PageContext


def render_frame_gallery(context: "PageContext") -> None:
    """Render the frame gallery page with raw/undistorted frames and GT pose."""

    st.title("Init & Frame QA")
    st.caption("Wo6 deliverable: load Tsukuba stereo frames, undistort them, and inspect ground-truth metadata.")

    sample = context.samples[context.frame_index]
    show_rgb = st.checkbox("Show RGB", value=True, help="Toggle between RGB and grayscale display.")
    payload = _build_gallery_payload_from_arrays(
        images=sample.images,
        intrinsics=sample.intrinsics,
        use_rgb=show_rgb,
    )
    st.subheader("Frame Grid")
    st.write("Upper row → raw, lower row → undistorted.")
    labels = ["Left", "Right", "Disparity"]
    top = st.columns(len(labels))
    bottom = st.columns(len(labels))
    raw_images = payload["raw"]
    undistorted_images = payload["undistorted"]
    for idx, key in enumerate(["left", "right", "disparity"]):
        raw = getattr(raw_images, key)
        undistorted = getattr(undistorted_images, key)
        caption = f"{labels[idx]} (raw)"
        _display_image_panel(top[idx], raw, caption)
        _display_image_panel(bottom[idx], undistorted, f"{labels[idx]} (undistorted)")

    metadata_col, pose_col, intrinsics_col = st.columns(3)
    metadata_col.metric("Frame", sample.index)
    metadata_col.write(f"Paths: `{sample.paths.left.name}` / `{sample.paths.right.name}`")
    if sample.paths.disparity:
        metadata_col.write(f"Disparity: `{sample.paths.disparity.name}`")
    pose_col.subheader("Ground truth pose")
    if sample.pose is None:
        pose_col.info("Ground-truth pose unavailable for this index.")
    else:
        pose_col.latex(_mat_to_latex(sample.pose.matrix))
    if sample.intrinsics is not None:
        intrinsics_col.subheader("Camera intrinsics")
        intrinsics_col.latex(_mat_to_latex(sample.intrinsics.matrix, precision=1))


def _mat_to_latex(matrix: np.ndarray, precision: int = 4) -> str:
    """Convert a homogeneous transform into a LaTeX ``bmatrix``."""

    row_strings = " \\\\ ".join(" & ".join(f"{value:.{precision}f}" for value in row) for row in matrix)
    return rf"\begin{{bmatrix}} {row_strings} \end{{bmatrix}}"


def _build_gallery_payload_from_arrays(
    *,
    images: "Images",
    intrinsics: CameraIntrinsics,
    use_rgb: bool,
) -> dict[Literal["raw", "undistorted"], Images]:
    """Prepare gallery payload when frames are already loaded in memory.

    Args:
        images: Source images held in memory.
        intrinsics: Calibration for undistortion.
        use_rgb: If False, convert images to grayscale before display.

    Returns:
        Mapping of panel labels to numpy arrays ready for `st.image`.
    """

    working = images.to_rgb() if use_rgb else images.to_gray()
    undistorted = working.undistort(intrinsics)

    return {
        "raw": working,
        "undistorted": undistorted,
    }


def _display_image_panel(container: Any, image: np.ndarray | None, caption: str) -> None:
    if image is None:
        container.warning("Missing input")
        return
    container.image(image, channels="RGB" if image.ndim == 3 else "L", caption=caption, width="stretch")

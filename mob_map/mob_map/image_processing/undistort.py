"""Image undistortion helpers built on top of OpenCV."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from mob_map.data.types import CameraIntrinsics

__all__ = ["undistort_image"]


def undistort_image(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: Sequence[float] | np.ndarray | None = None,
    *,
    alpha: float = 0.0,
    target_size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Remove lens distortion from an image using a pinhole camera model.

    Args:
        image: ``HxW`` (grayscale) or ``HxWxC`` numpy array.
        intrinsics: Calibrated focal lengths / principal point provided by the data module.
        distortion: Optional OpenCV-compatible distortion coefficients (``k1, k2, p1, p2, k3, ...``).
        alpha: Free-scaling parameter passed to :func:`cv2.getOptimalNewCameraMatrix` (``0`` crops fully, ``1`` keeps all pixels).
        target_size: Optional ``(width, height)`` for the undistorted view. Defaults to the source size.
        interpolation: OpenCV interpolation flag (e.g. ``cv2.INTER_LINEAR``).

    Returns:
        Undistorted image with the requested target size.

    Raises:
        ValueError: If the image has unsupported dimensions or the target size is invalid.
    """

    if image.ndim not in (2, 3):
        msg = "Image must be HxW or HxWxC numpy array."
        raise ValueError(msg)

    height, width = image.shape[:2]
    if target_size is None:
        target_size = (width, height)
    if target_size[0] <= 0 or target_size[1] <= 0:
        msg = "target_size must contain positive integers."
        raise ValueError(msg)

    camera_matrix = intrinsics.matrix.astype(np.float64, copy=True)
    dist_coeffs = _normalise_distortion(distortion)

    # Compute undistortion and rectification transformation map
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        alpha,
        target_size,
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
    )

    # Crop the image based on the ROI (if applicable) and resize to target size
    if roi is not None and alpha == 0.0:
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y : y + h_roi, x : x + w_roi]

    if undistorted.shape[1] != target_size[0] or undistorted.shape[0] != target_size[1]:
        undistorted = cv2.resize(undistorted, target_size, interpolation=interpolation)

    return undistorted


def _normalise_distortion(distortion: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if distortion is None:
        return np.zeros(5, dtype=np.float64)
    coeffs = np.asarray(distortion, dtype=np.float64)
    if coeffs.ndim != 1:
        msg = "Distortion coefficients must form a 1-D sequence."
        raise ValueError(msg)
    return coeffs

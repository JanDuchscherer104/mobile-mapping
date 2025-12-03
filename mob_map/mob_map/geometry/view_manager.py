from dataclasses import dataclass, field

import numpy as np
from pytransform3d.transform_manager import TransformManager


@dataclass(slots=True, frozen=True)
class View:
    """Represents a camera view with its intrinsics, pose, and features.

    Attributes:
        view_id: Unique identifier for this view
        frame_name: Name of the coordinate frame (for TransformManager)
        K: 3x3 camera intrinsic matrix
        image: Optional image data (H x W x 3)
        keypoints: List of cv2.KeyPoint objects
        descriptors: Feature descriptors (N x descriptor_dim)
        width: Image width in pixels
        height: Image height in pixels
    """

    view_id: int
    frame_name: str
    K: np.ndarray  # 3x3 intrinsic matrix
    image: np.ndarray | None = None
    keypoints: list = field(default_factory=list)
    descriptors: np.ndarray | None = None
    width: int = 0
    height: int = 0

    @property
    def focal_length(self) -> float:
        """Return focal length (assumes fx = fy)"""
        ...

    @property
    def principal_point(self) -> tuple[float, float]:
        """Return principal point (cx, cy)"""
        ...

    @property
    def num_features(self) -> int:
        """Return number of detected features"""
        ...


class ViewManager:
    """Manages multiple camera views and 3D points using pytransform3d.

    Similar to MATLAB's viewSet, this class handles:
    - Camera poses via TransformManager
    - Feature correspondences between views
    - 3D point cloud with visibility information
    """

    transform_manager: TransformManager

    def __init__(self, world_frame: str = "world"):
        """Initialize ViewManager with a world coordinate frame.

        Args:
            world_frame: Name of the world coordinate frame (default: "world")
        """
        ...

    def add_view(
        self,
        view_id: int,
        K: np.ndarray,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
        image: np.ndarray | None = None,
        keypoints: list | None = None,
        descriptors: np.ndarray | None = None,
    ) -> "ViewManager":
        """Add a camera view with its pose and intrinsics.

        Args:
            view_id: Unique identifier for this view
            K: 3x3 camera intrinsic matrix
            R: 3x3 rotation matrix (world-to-camera), default: identity
            t: 3x1 translation vector (world-to-camera), default: zero
            image: Optional image data (H x W x 3)
            keypoints: Optional list of cv2.KeyPoint objects
            descriptors: Optional feature descriptors (N x descriptor_dim)

        Returns:
            self (for method chaining)
        """
        ...

    def add_connection(self, from_view_id: int, to_view_id: int, num_matches: int) -> "ViewManager":
        """Add a connection between two views (feature correspondences).

        Args:
            from_view_id: Source view ID
            to_view_id: Target view ID
            num_matches: Number of matched features

        Returns:
            self (for method chaining)
        """
        ...

    def add_points(
        self,
        points_3d: np.ndarray,
        colors: np.ndarray | None = None,
        visibility: dict[int, list[int]] | None = None,
    ) -> "ViewManager":
        """Add 3D points to the reconstruction.

        Args:
            points_3d: N x 3 array of 3D point coordinates (in world frame)
            colors: Optional N x 3 array of RGB colors (0-255)
            visibility: Optional dict mapping point_idx -> [view_ids]

        Returns:
            self (for method chaining)
        """
        ...

    def get_transform(self, from_frame: str, to_frame: str) -> np.ndarray:
        """Get 4x4 transformation matrix between two frames.

        Args:
            from_frame: Source frame name (e.g., "cam1", "world")
            to_frame: Target frame name

        Returns:
            4x4 transformation matrix
        """
        ...

    def get_view(self, view_id: int) -> View:
        """Get a View object by ID.

        Args:
            view_id: View identifier

        Returns:
            View object
        """
        ...

    def get_camera_pose(self, view_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Get camera pose (R, t) in world coordinates.

        Args:
            view_id: View identifier

        Returns:
            R: 3x3 rotation matrix (camera-to-world)
            t: 3x1 translation vector (camera position in world)
        """
        ...

    def triangulate_points(self, view1_id: int, view2_id: int, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from corresponding 2D points in two views.

        Args:
            view1_id: First view ID
            view2_id: Second view ID
            points1: N x 2 array of 2D points in first view
            points2: N x 2 array of 2D points in second view

        Returns:
            N x 3 array of 3D points in world coordinates
        """
        ...

    def plot(self, ax=None, point_size=1, camera_scale=1.0, show_frames=True):
        """Plot the reconstruction with cameras and 3D points.

        Args:
            ax: matplotlib 3D axis (creates new if None)
            point_size: Size of 3D point markers
            camera_scale: Scale factor for camera visualization
            show_frames: Whether to show coordinate frames

        Returns:
            ax: matplotlib 3D axis
        """
        ...

    def __repr__(self) -> str:
        """String representation of the ViewManager."""
        ...

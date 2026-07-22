"""Coordinate and calibration helpers for CARLA <-> Alpamayo integration.

CARLA uses an ego frame with ``x`` forward, ``y`` right, and ``z`` up.
Alpamayo uses ``x`` forward, ``y`` left, and ``z`` up.  Keeping the basis
conversion here prevents the input history, output trajectory, controller, and
visualizer from silently using different handedness conventions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


MODEL_FROM_CARLA_BASIS = np.diag([1.0, -1.0, 1.0])


def _finite_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def carla_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return CARLA's local-to-world 3x3 rotation matrix."""

    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def pose_matrix_from_components(
    x: float,
    y: float,
    z: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Build a homogeneous CARLA local-to-world pose matrix."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = carla_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    matrix[:3, 3] = [float(x), float(y), float(z)]
    if not np.isfinite(matrix).all():
        raise ValueError("pose components must be finite")
    return matrix


def pose_matrix_from_transform(transform: Any) -> np.ndarray:
    """Copy a ``carla.Transform`` (or compatible fake) into a NumPy matrix."""

    get_matrix = getattr(transform, "get_matrix", None)
    if callable(get_matrix):
        matrix = np.asarray(get_matrix(), dtype=np.float64)
        if matrix.shape == (4, 4) and np.isfinite(matrix).all():
            return matrix.copy()

    location = transform.location
    rotation = transform.rotation
    return pose_matrix_from_components(
        location.x,
        location.y,
        location.z,
        getattr(rotation, "roll", 0.0),
        getattr(rotation, "pitch", 0.0),
        getattr(rotation, "yaw", 0.0),
    )


def pose_matrix_from_state(state: dict[str, Any]) -> np.ndarray:
    """Build a pose matrix from a CARLAInterface ego-state mapping."""

    return pose_matrix_from_components(
        state["x"],
        state["y"],
        state["z"],
        state.get("roll", 0.0),
        state.get("pitch", 0.0),
        state.get("yaw", 0.0),
    )


def transform_points(transform_matrix: Any, points: Any) -> np.ndarray:
    """Apply a homogeneous transform to an ``Nx3`` point array."""

    matrix = _finite_array(transform_matrix, name="transform_matrix", shape=(4, 4))
    xyz = _finite_array(points, name="points")
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {xyz.shape}")
    homogeneous = np.concatenate([xyz[:, :3], np.ones((len(xyz), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def carla_ego_points_to_model(points: Any) -> np.ndarray:
    """Convert CARLA ego points (y right) to model ego points (y left)."""

    xyz = _finite_array(points, name="points").copy()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {xyz.shape}")
    xyz[:, :3] = (MODEL_FROM_CARLA_BASIS @ xyz[:, :3].T).T
    return xyz


def model_ego_points_to_carla(points: Any) -> np.ndarray:
    """Convert model ego points (y left) to CARLA ego points (y right)."""

    # The reflection matrix is its own inverse.
    return carla_ego_points_to_model(points)


def model_ego_points_to_world(capture_pose_world: Any, points: Any) -> np.ndarray:
    """Anchor model points to the capture pose exactly once."""

    return transform_points(capture_pose_world, model_ego_points_to_carla(points))


def world_points_to_model_ego(capture_pose_world: Any, points: Any) -> np.ndarray:
    """Transform CARLA world points into the model frame of a capture pose."""

    pose = _finite_array(capture_pose_world, name="capture_pose_world", shape=(4, 4))
    carla_local = transform_points(np.linalg.inv(pose), points)
    return carla_ego_points_to_model(carla_local)


def meaningful_path_tangent_xy(
    points: Any,
    index: int,
    *,
    lookahead_m: float,
    minimum_displacement_m: float,
) -> np.ndarray | None:
    """Return a robust forward XY tangent around one path point.

    The tangent uses a spatial baseline instead of the immediately adjacent
    segment.  This prevents sub-centimetre start/stop jitter from looking like
    a genuine reverse path.  Future points are preferred; near the end of a
    path, earlier points are used with their direction reversed.
    """

    path = _finite_array(points, name="points")
    if path.ndim != 2 or path.shape[1] < 2 or len(path) == 0:
        raise ValueError(f"points must have shape (N, >=2), got {path.shape}")
    point_index = int(index)
    if point_index < 0 or point_index >= len(path):
        raise IndexError("path index is out of range")

    lookahead = float(lookahead_m)
    minimum = float(minimum_displacement_m)
    if not math.isfinite(lookahead) or lookahead <= 0.0:
        raise ValueError("lookahead_m must be a positive finite value")
    if not math.isfinite(minimum) or minimum <= 0.0 or minimum > lookahead:
        raise ValueError(
            "minimum_displacement_m must be positive and no greater than lookahead_m"
        )

    anchor = path[point_index, :2]

    def search(indices: range, *, reverse: bool) -> np.ndarray | None:
        travelled = 0.0
        previous = anchor
        fallback_delta = None
        fallback_norm = 0.0
        for other_index in indices:
            point = path[other_index, :2]
            travelled += float(np.linalg.norm(point - previous))
            previous = point
            delta = anchor - point if reverse else point - anchor
            norm = float(np.linalg.norm(delta))
            if norm > fallback_norm:
                fallback_delta = delta
                fallback_norm = norm
            if travelled >= lookahead and norm >= minimum:
                return delta / norm
        if fallback_delta is not None and fallback_norm >= minimum:
            return fallback_delta / fallback_norm
        return None

    future = search(range(point_index + 1, len(path)), reverse=False)
    if future is not None:
        return future
    return search(range(point_index - 1, -1, -1), reverse=True)


def carla_relative_rotation_to_model(rotation: Any) -> np.ndarray:
    """Change basis for a relative rotation from CARLA ego to model ego."""

    matrix = _finite_array(rotation, name="rotation", shape=(3, 3))
    return MODEL_FROM_CARLA_BASIS @ matrix @ MODEL_FROM_CARLA_BASIS


def camera_intrinsic_matrix(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Return a pinhole intrinsic matrix for a CARLA RGB camera."""

    width = int(width)
    height = int(height)
    fov = float(horizontal_fov_deg)
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

"""Camera calibration and deterministic projection alignment for CarlaMayo.

The model-facing projection convention is OpenCV: ``x`` right, ``y`` down,
``z`` forward.  PhysicalAI rig poses use a rear-axle ground frame with
``x`` forward, ``y`` left, ``z`` up.  CARLA actor poses use ``x`` forward,
``y`` right, ``z`` up.

This module deliberately has no CARLA or PhysicalAI-AV dependency.  Runtime
alignment consumes a small, local JSON profile so gated calibration never
needs to enter source control or telemetry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import camera_intrinsic_matrix, carla_rotation_matrix


CAMERA_PROFILE_SCHEMA_VERSION = 1
CAMERA_PROFILE_RIG_FRAME = "rear_axle_ground_x_forward_y_left_z_up"
CAMERA_ALIGNMENT_MODES = (
    "baseline",
    "pose-only",
    "projection-only",
    "pose-projection",
)
EXPECTED_CAMERA_IDS = (0, 1, 2, 6)
EXPECTED_CAMERA_NAMES = (
    "cam_front_left",
    "cam_front_wide",
    "cam_front_right",
    "cam_front_tele",
)
OUTPUT_RESOLUTION = (1920, 1080)
NVIDIA_TO_CARLA_RIG = np.diag([1.0, -1.0, 1.0])
CARLA_CAMERA_TO_OPENCV = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


class CameraCalibrationError(ValueError):
    """Raised when a camera profile or derived projection is unsafe to use."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CameraCalibrationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise CameraCalibrationError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CameraCalibrationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"{name} must be an integer") from exc
    if result != value or result <= 0:
        raise CameraCalibrationError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CameraCalibrationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"{name} must be an integer") from exc
    if result != value or result < 0:
        raise CameraCalibrationError(f"{name} must be a non-negative integer")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraCalibrationError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"{name} must be numeric") from exc
    if shape is not None and array.shape != shape:
        raise CameraCalibrationError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise CameraCalibrationError(f"{name} must contain only finite values")
    return array


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PinholeProjection:
    """Ideal pinhole projection in the OpenCV optical frame."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    projection_type: str = field(default="pinhole", init=False)

    def __post_init__(self) -> None:
        width = _positive_int(self.width, "pinhole width")
        height = _positive_int(self.height, "pinhole height")
        fx = _finite_float(self.fx, "pinhole fx")
        fy = _finite_float(self.fy, "pinhole fy")
        cx = _finite_float(self.cx, "pinhole cx")
        cy = _finite_float(self.cy, "pinhole cy")
        if fx <= 0.0 or fy <= 0.0:
            raise CameraCalibrationError("pinhole focal lengths must be positive")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)

    @classmethod
    def from_horizontal_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_deg: float,
    ) -> "PinholeProjection":
        matrix = camera_intrinsic_matrix(width, height, horizontal_fov_deg)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def pixel_to_ray(self, pixels: Any) -> tuple[np.ndarray, np.ndarray]:
        pixels_array = _finite_array(pixels, "pixels")
        if pixels_array.shape[-1:] != (2,):
            raise CameraCalibrationError("pixels must have shape (..., 2)")
        rays = np.empty(pixels_array.shape[:-1] + (3,), dtype=np.float64)
        rays[..., 0] = (pixels_array[..., 0] - self.cx) / self.fx
        rays[..., 1] = (pixels_array[..., 1] - self.cy) / self.fy
        rays[..., 2] = 1.0
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        tolerance = 1e-7
        valid = (
            (pixels_array[..., 0] >= -tolerance)
            & (pixels_array[..., 0] <= self.width - 1 + tolerance)
            & (pixels_array[..., 1] >= -tolerance)
            & (pixels_array[..., 1] <= self.height - 1 + tolerance)
        )
        return rays, valid

    def ray_to_pixel(self, rays: Any) -> tuple[np.ndarray, np.ndarray]:
        rays_array = _finite_array(rays, "rays")
        if rays_array.shape[-1:] != (3,):
            raise CameraCalibrationError("rays must have shape (..., 3)")
        z = rays_array[..., 2]
        forward = z > 0.0
        safe_z = np.where(forward, z, 1.0)
        pixels = np.empty(rays_array.shape[:-1] + (2,), dtype=np.float64)
        pixels[..., 0] = self.fx * rays_array[..., 0] / safe_z + self.cx
        pixels[..., 1] = self.fy * rays_array[..., 1] / safe_z + self.cy
        tolerance = 1e-7
        valid = (
            forward
            & (pixels[..., 0] >= -tolerance)
            & (pixels[..., 0] <= self.width - 1 + tolerance)
            & (pixels[..., 1] >= -tolerance)
            & (pixels[..., 1] <= self.height - 1 + tolerance)
        )
        return pixels, valid

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "type": self.projection_type,
            "resolution": [self.width, self.height],
        }


@dataclass(frozen=True)
class FThetaProjection:
    """Polynomial angle-to-radius projection with a dense monotonic inverse."""

    width: int
    height: int
    cx: float
    cy: float
    angle_to_radius_coefficients: tuple[float, ...]
    radius_to_angle_coefficients: tuple[float, ...]
    projection_type: str = field(default="ftheta", init=False)
    _lookup_theta: np.ndarray = field(init=False, repr=False, compare=False)
    _lookup_radius: np.ndarray = field(init=False, repr=False, compare=False)
    _maximum_theta: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        width = _positive_int(self.width, "F-theta width")
        height = _positive_int(self.height, "F-theta height")
        cx = _finite_float(self.cx, "F-theta cx")
        cy = _finite_float(self.cy, "F-theta cy")
        angle_to_radius = tuple(
            _finite_float(value, "angle_to_radius coefficient")
            for value in self.angle_to_radius_coefficients
        )
        radius_to_angle = tuple(
            _finite_float(value, "radius_to_angle coefficient")
            for value in self.radius_to_angle_coefficients
        )
        if len(angle_to_radius) < 2:
            raise CameraCalibrationError(
                "angle_to_radius_coefficients must contain at least two values"
            )
        if not radius_to_angle:
            raise CameraCalibrationError(
                "radius_to_angle_coefficients must contain at least one value"
            )

        # Polynomial coefficients use NumPy's ascending-power convention.  The
        # dense inverse is built only over forward-facing rays.  This avoids
        # relying on a separately fitted, lower-accuracy inverse polynomial.
        theta = np.linspace(0.0, math.nextafter(math.pi / 2.0, 0.0), 65536)
        radius = np.polynomial.polynomial.polyval(theta, angle_to_radius)
        if not np.isfinite(radius).all():
            raise CameraCalibrationError("F-theta polynomial is non-finite")
        differences = np.diff(radius)
        tolerance = max(1e-12, float(np.max(np.abs(radius))) * 1e-12)
        if radius[0] < -tolerance or np.any(differences <= tolerance):
            raise CameraCalibrationError(
                "F-theta angle-to-radius polynomial must be strictly monotonic"
            )

        maximum_pixel_radius = max(
            math.hypot(x - cx, y - cy)
            for x in (0.0, float(width - 1))
            for y in (0.0, float(height - 1))
        )
        if maximum_pixel_radius > float(radius[-1]) + 1e-6:
            raise CameraCalibrationError(
                "F-theta polynomial does not cover every output pixel"
            )
        maximum_theta = float(np.interp(maximum_pixel_radius, radius, theta))

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)
        object.__setattr__(self, "angle_to_radius_coefficients", angle_to_radius)
        object.__setattr__(self, "radius_to_angle_coefficients", radius_to_angle)
        object.__setattr__(self, "_lookup_theta", _readonly(theta))
        object.__setattr__(self, "_lookup_radius", _readonly(radius))
        object.__setattr__(self, "_maximum_theta", maximum_theta)

    def pixel_to_ray(self, pixels: Any) -> tuple[np.ndarray, np.ndarray]:
        pixels_array = _finite_array(pixels, "pixels")
        if pixels_array.shape[-1:] != (2,):
            raise CameraCalibrationError("pixels must have shape (..., 2)")
        delta_x = pixels_array[..., 0] - self.cx
        delta_y = pixels_array[..., 1] - self.cy
        radius = np.hypot(delta_x, delta_y)
        tolerance = 1e-7
        inside = (
            (pixels_array[..., 0] >= -tolerance)
            & (pixels_array[..., 0] <= self.width - 1 + tolerance)
            & (pixels_array[..., 1] >= -tolerance)
            & (pixels_array[..., 1] <= self.height - 1 + tolerance)
            & (radius <= self._lookup_radius[-1])
        )
        theta = np.interp(
            np.minimum(radius, self._lookup_radius[-1]),
            self._lookup_radius,
            self._lookup_theta,
        )
        sin_theta = np.sin(theta)
        radial_scale = np.divide(
            sin_theta,
            radius,
            out=np.zeros_like(radius, dtype=np.float64),
            where=radius > 0.0,
        )
        rays = np.empty(pixels_array.shape[:-1] + (3,), dtype=np.float64)
        rays[..., 0] = delta_x * radial_scale
        rays[..., 1] = delta_y * radial_scale
        rays[..., 2] = np.cos(theta)
        rays = np.where(np.expand_dims(radius == 0.0, -1), [0.0, 0.0, 1.0], rays)
        return rays, inside

    def ray_to_pixel(self, rays: Any) -> tuple[np.ndarray, np.ndarray]:
        rays_array = _finite_array(rays, "rays")
        if rays_array.shape[-1:] != (3,):
            raise CameraCalibrationError("rays must have shape (..., 3)")
        norm = np.linalg.norm(rays_array, axis=-1)
        finite_direction = norm > 0.0
        normalized = np.divide(
            rays_array,
            np.expand_dims(np.where(finite_direction, norm, 1.0), -1),
        )
        xy_norm = np.hypot(normalized[..., 0], normalized[..., 1])
        theta = np.arctan2(xy_norm, normalized[..., 2])
        radius = np.polynomial.polynomial.polyval(
            theta,
            self.angle_to_radius_coefficients,
        )
        radial_scale = np.divide(
            radius,
            xy_norm,
            out=np.zeros_like(radius, dtype=np.float64),
            where=xy_norm > 0.0,
        )
        pixels = np.empty(rays_array.shape[:-1] + (2,), dtype=np.float64)
        pixels[..., 0] = self.cx + normalized[..., 0] * radial_scale
        pixels[..., 1] = self.cy + normalized[..., 1] * radial_scale
        center = xy_norm == 0.0
        pixels[..., 0] = np.where(center, self.cx, pixels[..., 0])
        pixels[..., 1] = np.where(center, self.cy, pixels[..., 1])
        tolerance = 1e-7
        valid = (
            finite_direction
            & (normalized[..., 2] > 0.0)
            & (theta <= self._maximum_theta + 1e-12)
            & (pixels[..., 0] >= -tolerance)
            & (pixels[..., 0] <= self.width - 1 + tolerance)
            & (pixels[..., 1] >= -tolerance)
            & (pixels[..., 1] <= self.height - 1 + tolerance)
        )
        return pixels, valid

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "type": self.projection_type,
            "resolution": [self.width, self.height],
        }


@dataclass(frozen=True)
class CameraProfileEntry:
    name: str
    alpamayo_id: int
    resolution: tuple[int, int]
    sensor_to_rig_matrix: np.ndarray = field(repr=False, compare=False)
    projection: FThetaProjection = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        name = _nonempty_string(self.name, "camera name")
        camera_id = _nonnegative_int(self.alpamayo_id, "alpamayo_id")
        try:
            resolution = tuple(int(item) for item in self.resolution)
        except (TypeError, ValueError) as exc:
            raise CameraCalibrationError("camera resolution must contain two integers") from exc
        if resolution != OUTPUT_RESOLUTION:
            raise CameraCalibrationError(
                f"camera resolution must be {OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]}"
            )
        matrix = _finite_array(
            self.sensor_to_rig_matrix,
            f"{name} sensor_to_rig_matrix",
            (4, 4),
        )
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise CameraCalibrationError(f"{name} transform must be homogeneous")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise CameraCalibrationError(f"{name} rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise CameraCalibrationError(f"{name} rotation determinant must be +1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "alpamayo_id", camera_id)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "sensor_to_rig_matrix", _readonly(matrix.copy()))


@dataclass(frozen=True)
class CameraRigProfile:
    schema_version: int
    profile_id: str
    dataset_revision: str
    source_clip_id: str
    platform_class: str
    rig_frame: str
    vehicle_length: float
    vehicle_width: float
    vehicle_height: float
    rear_axle_to_bbox_center: float
    cameras: tuple[CameraProfileEntry, ...]
    sha256: str
    source_path: str

    def __post_init__(self) -> None:
        if self.schema_version != CAMERA_PROFILE_SCHEMA_VERSION:
            raise CameraCalibrationError(
                f"unsupported camera profile schema_version {self.schema_version}"
            )
        if self.rig_frame != CAMERA_PROFILE_RIG_FRAME:
            raise CameraCalibrationError(
                f"rig_frame must be {CAMERA_PROFILE_RIG_FRAME}"
            )
        for field_name in (
            "profile_id",
            "dataset_revision",
            "source_clip_id",
            "platform_class",
            "sha256",
            "source_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "vehicle_length",
            "vehicle_width",
            "vehicle_height",
            "rear_axle_to_bbox_center",
        ):
            value = _finite_float(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise CameraCalibrationError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        names = tuple(camera.name for camera in self.cameras)
        camera_ids = tuple(camera.alpamayo_id for camera in self.cameras)
        if names != EXPECTED_CAMERA_NAMES or camera_ids != EXPECTED_CAMERA_IDS:
            raise CameraCalibrationError(
                "profile cameras must be [0, 1, 2, 6] in left/wide/right/tele order"
            )

    def camera(self, name: str) -> CameraProfileEntry:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(name)

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "dataset_revision": self.dataset_revision,
            "platform_class": self.platform_class,
            "profile_sha256": self.sha256,
        }


def _projection_from_json(camera_payload: dict[str, Any]) -> FThetaProjection:
    projection = camera_payload.get("projection")
    if not isinstance(projection, dict):
        raise CameraCalibrationError("camera projection must be an object")
    if projection.get("type") != "ftheta":
        raise CameraCalibrationError("camera projection type must be ftheta")
    principal_point = _finite_array(
        projection.get("principal_point"),
        "principal_point",
        (2,),
    )
    resolution = camera_payload.get("resolution")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise CameraCalibrationError("camera resolution must contain width and height")
    return FThetaProjection(
        width=resolution[0],
        height=resolution[1],
        cx=principal_point[0],
        cy=principal_point[1],
        angle_to_radius_coefficients=tuple(
            projection.get("angle_to_radius_coefficients", ())
        ),
        radius_to_angle_coefficients=tuple(
            projection.get("radius_to_angle_coefficients", ())
        ),
    )


def load_camera_rig_profile(path: str | os.PathLike[str]) -> CameraRigProfile:
    """Load and fully validate a local gated camera profile."""

    profile_path = Path(path).expanduser().resolve()
    try:
        raw = profile_path.read_bytes()
    except OSError as exc:
        raise CameraCalibrationError(
            f"cannot read camera profile {profile_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CameraCalibrationError("camera profile is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise CameraCalibrationError("camera profile root must be an object")
    vehicle = payload.get("vehicle")
    if not isinstance(vehicle, dict):
        raise CameraCalibrationError("camera profile vehicle must be an object")
    camera_payloads = payload.get("cameras")
    if not isinstance(camera_payloads, list):
        raise CameraCalibrationError("camera profile cameras must be an array")
    cameras = []
    for camera_payload in camera_payloads:
        if not isinstance(camera_payload, dict):
            raise CameraCalibrationError("each camera profile entry must be an object")
        resolution = camera_payload.get("resolution")
        cameras.append(
            CameraProfileEntry(
                name=camera_payload.get("name"),
                alpamayo_id=camera_payload.get("alpamayo_id"),
                resolution=resolution,
                sensor_to_rig_matrix=camera_payload.get("sensor_to_rig_matrix"),
                projection=_projection_from_json(camera_payload),
            )
        )
    return CameraRigProfile(
        schema_version=payload.get("schema_version"),
        profile_id=payload.get("profile_id"),
        dataset_revision=payload.get("dataset_revision"),
        source_clip_id=payload.get("source_clip_id"),
        platform_class=payload.get("platform_class"),
        rig_frame=payload.get("rig_frame"),
        vehicle_length=vehicle.get("length"),
        vehicle_width=vehicle.get("width"),
        vehicle_height=vehicle.get("height"),
        rear_axle_to_bbox_center=vehicle.get("rear_axle_to_bbox_center"),
        cameras=tuple(cameras),
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(profile_path),
    )


def vehicle_dimension_errors(
    profile: CameraRigProfile,
    bounding_box: Any,
) -> dict[str, float]:
    """Return relative dimension errors against a CARLA ego bounding box."""

    extent = bounding_box.extent
    carla_dimensions = {
        "length": 2.0 * _finite_float(extent.x, "CARLA bbox extent.x"),
        "width": 2.0 * _finite_float(extent.y, "CARLA bbox extent.y"),
        "height": 2.0 * _finite_float(extent.z, "CARLA bbox extent.z"),
    }
    profile_dimensions = {
        "length": profile.vehicle_length,
        "width": profile.vehicle_width,
        "height": profile.vehicle_height,
    }
    errors = {
        name: abs(carla_dimensions[name] - profile_dimensions[name])
        / profile_dimensions[name]
        for name in profile_dimensions
    }
    return errors


def validate_vehicle_dimensions(
    profile: CameraRigProfile,
    bounding_box: Any,
    *,
    maximum_relative_error: float = 0.10,
) -> dict[str, float]:
    errors = vehicle_dimension_errors(profile, bounding_box)
    failures = {
        name: error
        for name, error in errors.items()
        if error > float(maximum_relative_error)
    }
    if failures:
        details = ", ".join(f"{name}={error:.1%}" for name, error in failures.items())
        raise CameraCalibrationError(
            f"CARLA/profile vehicle dimension error exceeds 10%: {details}"
        )
    return errors


def _carla_euler_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = _finite_array(rotation, "CARLA rotation", (3, 3))
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise CameraCalibrationError("converted CARLA rotation is not orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not math.isclose(determinant, 1.0, abs_tol=1e-6):
        raise CameraCalibrationError("converted CARLA rotation determinant must be +1")
    pitch = math.asin(float(np.clip(matrix[2, 0], -1.0, 1.0)))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-8:
        # CARLA's Euler representation is non-unique at +/-90 degree pitch.
        # Choosing roll=0 gives a deterministic representation and preserves
        # the exact matrix through the validation round trip below.
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    else:
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        roll = math.atan2(-matrix[2, 1], matrix[2, 2])
    result = tuple(math.degrees(value) for value in (roll, pitch, yaw))
    reconstructed = carla_rotation_matrix(*result)
    if not np.allclose(reconstructed, matrix, atol=1e-6):
        raise CameraCalibrationError("CARLA Euler conversion round-trip failed")
    return result


def camera_pose_in_carla_actor(
    profile: CameraRigProfile,
    camera: CameraProfileEntry,
    bounding_box: Any,
) -> dict[str, float]:
    """Convert a PhysicalAI sensor pose to a CARLA actor-local pose."""

    matrix = camera.sensor_to_rig_matrix
    tx, ty, tz = matrix[:3, 3]
    bbox_location = bounding_box.location
    bbox_extent = bounding_box.extent
    position = np.array(
        [
            _finite_float(bbox_location.x, "bbox.location.x")
            + tx
            - profile.rear_axle_to_bbox_center,
            _finite_float(bbox_location.y, "bbox.location.y") - ty,
            _finite_float(bbox_location.z, "bbox.location.z")
            - _finite_float(bbox_extent.z, "bbox.extent.z")
            + tz,
        ],
        dtype=np.float64,
    )
    rotation = (
        NVIDIA_TO_CARLA_RIG
        @ matrix[:3, :3]
        @ CARLA_CAMERA_TO_OPENCV
    )
    roll, pitch, yaw = _carla_euler_from_rotation_matrix(rotation)
    if not np.isfinite(position).all() or np.linalg.norm(position) > 20.0:
        raise CameraCalibrationError("converted camera position failed sanity checks")
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


@dataclass(frozen=True)
class CameraRemap:
    source_projection: PinholeProjection
    output_projection: FThetaProjection
    map_x: np.ndarray = field(repr=False, compare=False)
    map_y: np.ndarray = field(repr=False, compare=False)
    valid_mask: np.ndarray = field(repr=False, compare=False)
    source_fov_deg: float
    valid_ratio: float
    has_invalid_pixels: bool

    def __post_init__(self) -> None:
        expected_shape = (self.output_projection.height, self.output_projection.width)
        if self.map_x.shape != expected_shape or self.map_y.shape != expected_shape:
            raise CameraCalibrationError("remap arrays do not match output resolution")
        if self.valid_mask.shape != expected_shape:
            raise CameraCalibrationError("remap valid mask does not match output resolution")
        object.__setattr__(self, "map_x", _readonly(np.asarray(self.map_x, np.float32)))
        object.__setattr__(self, "map_y", _readonly(np.asarray(self.map_y, np.float32)))
        object.__setattr__(self, "valid_mask", _readonly(np.asarray(self.valid_mask, bool)))


def build_ftheta_remap(
    output_projection: FThetaProjection,
    *,
    fov_margin_deg: float = 2.0,
    maximum_source_fov_deg: float = 160.0,
    maximum_invalid_ratio: float = 0.005,
) -> CameraRemap:
    """Precompute one F-theta output to CARLA pinhole source remap."""

    width = output_projection.width
    height = output_projection.height
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    output_pixels = np.stack([grid_x, grid_y], axis=-1)
    target_rays, target_valid = output_projection.pixel_to_ray(output_pixels)
    if np.any(target_valid & (target_rays[..., 2] <= 0.0)):
        raise CameraCalibrationError("F-theta output requires rear-facing source rays")

    z = target_rays[..., 2]
    valid_z = target_valid & (z > 0.0)
    ray_x_over_z = np.abs(
        np.divide(target_rays[..., 0], z, out=np.zeros_like(z), where=valid_z)
    )
    ray_y_over_z = np.abs(
        np.divide(target_rays[..., 1], z, out=np.zeros_like(z), where=valid_z)
    )
    half_tan = max(
        float(np.max(ray_x_over_z[valid_z])),
        float(width / height * np.max(ray_y_over_z[valid_z])),
    )
    source_fov_deg = math.degrees(2.0 * math.atan(half_tan)) + 2.0 * float(
        fov_margin_deg
    )
    if not 0.0 < source_fov_deg <= maximum_source_fov_deg:
        raise CameraCalibrationError(
            f"required source FOV {source_fov_deg:.2f} exceeds "
            f"{maximum_source_fov_deg:.2f} degrees"
        )
    source_projection = PinholeProjection.from_horizontal_fov(
        width,
        height,
        source_fov_deg,
    )
    source_pixels, source_valid = source_projection.ray_to_pixel(target_rays)
    valid_mask = target_valid & source_valid
    valid_ratio = float(np.mean(valid_mask))
    if 1.0 - valid_ratio > maximum_invalid_ratio:
        raise CameraCalibrationError(
            f"predicted invalid output ratio {1.0 - valid_ratio:.3%} exceeds "
            f"{maximum_invalid_ratio:.3%}"
        )
    map_x = np.where(valid_mask, source_pixels[..., 0], -1.0).astype(np.float32)
    map_y = np.where(valid_mask, source_pixels[..., 1], -1.0).astype(np.float32)
    return CameraRemap(
        source_projection=source_projection,
        output_projection=output_projection,
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid_mask,
        source_fov_deg=source_fov_deg,
        valid_ratio=valid_ratio,
        has_invalid_pixels=bool(not np.all(valid_mask)),
    )


def remap_rgb_image(image: np.ndarray, remap: CameraRemap) -> np.ndarray:
    """Apply one precomputed remap and black-fill invalid residual pixels."""

    array = np.asarray(image)
    expected = (
        remap.source_projection.height,
        remap.source_projection.width,
        3,
    )
    if array.shape != expected or array.dtype != np.uint8:
        raise CameraCalibrationError(
            f"source image must be uint8 RGB with shape {expected}, got "
            f"{array.dtype} {array.shape}"
        )
    warped = cv2.remap(
        array,
        remap.map_x,
        remap.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if remap.has_invalid_pixels:
        warped[~remap.valid_mask] = 0
    return warped

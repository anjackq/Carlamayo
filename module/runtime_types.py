"""Dependency-free runtime data contracts for the closed-loop pipeline.

The records in this module deliberately avoid CARLA, NumPy, and PyTorch imports.
Large runtime payloads remain opaque and are summarized, rather than copied, when
records are serialized for telemetry. Constructing a record transfers read-only
ownership of those payload references to the record: producers must publish a new
record instead of mutating an array, image, list, or mapping after publication.
This avoids copying multi-camera tensors while making the ownership rule explicit.
"""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(normalized)


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _optional_finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name, minimum=minimum)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    normalized = value.strip()
    return normalized or None


def _camera_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("camera_ids must be an iterable of integers")
    try:
        normalized = tuple(_nonnegative_int(item, "camera_id") for item in value)
    except TypeError as exc:
        raise TypeError("camera_ids must be an iterable of integers") from exc
    if not normalized:
        raise ValueError("camera_ids must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("camera_ids must be unique")
    return normalized


def _nonnegative_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    try:
        normalized = tuple(_nonnegative_int(item, f"{name} item") for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return normalized


def _nonnegative_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers")
    try:
        normalized = tuple(_finite_float(item, f"{name} item", minimum=0.0) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numbers") from exc
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return normalized


def _waypoint_times(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("waypoint_times_s must be an iterable of numbers")
    try:
        normalized = tuple(_finite_float(item, "waypoint_time", minimum=0.0) for item in value)
    except TypeError as exc:
        raise TypeError("waypoint_times_s must be an iterable of numbers") from exc
    if not normalized:
        raise ValueError("waypoint_times_s must not be empty")
    if normalized[0] <= 0.0:
        raise ValueError("the first waypoint time must be greater than zero")
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError("waypoint_times_s must be strictly increasing")
    expected = tuple(0.1 * index for index in range(1, 65))
    if len(normalized) != len(expected) or any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-6)
        for actual, target in zip(normalized, expected)
    ):
        raise ValueError("waypoint_times_s must use the 64-point [0.1, ..., 6.4] grid")
    return normalized


def _control_value(value: Any, name: str, lower: float, upper: float) -> float:
    normalized = _finite_float(value, name)
    if not lower <= normalized <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return normalized


def _control_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain steering, throttle, and brake")
    try:
        steering, throttle, brake = value
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain steering, throttle, and brake") from exc
    return (
        _control_value(steering, f"{name}.steering", -1.0, 1.0),
        _control_value(throttle, f"{name}.throttle", 0.0, 1.0),
        _control_value(brake, f"{name}.brake", 0.0, 1.0),
    )


def _optional_point3(value: Any, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly three coordinates")
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain exactly three coordinates") from exc
    return (
        _finite_float(x, f"{name}.x"),
        _finite_float(y, f"{name}.y"),
        _finite_float(z, f"{name}.z"),
    )


def _payload_length(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def summarize_payload(value: Any) -> dict[str, Any] | None:
    """Return a compact JSON-safe description of an opaque runtime payload."""

    if value is None:
        return None

    summary: dict[str, Any] = {"type": type(value).__name__}
    try:
        shape = getattr(value, "shape")
    except (AttributeError, RuntimeError):
        shape = None
    if shape is not None:
        try:
            summary["shape"] = [int(item) for item in shape]
        except (TypeError, ValueError):
            summary["shape"] = str(shape)

    try:
        dtype = getattr(value, "dtype")
    except (AttributeError, RuntimeError):
        dtype = None
    if dtype is not None:
        summary["dtype"] = str(dtype)

    length = _payload_length(value)
    if length is not None:
        summary["length"] = int(length)
    if isinstance(value, Mapping):
        summary["keys"] = sorted(str(key) for key in value)
    return summary


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON telemetry cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    to_json = getattr(value, "to_json_dict", None)
    if callable(to_json):
        return to_json()
    return summarize_payload(value)


def to_json_dict(record: Any) -> dict[str, Any]:
    """Serialize a runtime record or mapping to a JSON-safe dictionary."""

    to_json = getattr(record, "to_json_dict", None)
    if callable(to_json):
        result = to_json()
    elif isinstance(record, Mapping):
        result = _json_safe(record)
    else:
        raise TypeError("record must provide to_json_dict() or be a mapping")
    if not isinstance(result, dict):
        raise TypeError("serialized runtime record must be a dictionary")
    validated = _json_safe(result)
    if not isinstance(validated, dict):
        raise TypeError("serialized runtime record must be a dictionary")
    return validated


def to_json_line(record: Any) -> str:
    """Serialize a runtime record as one deterministic compact JSON line."""

    return json.dumps(
        to_json_dict(record),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class SynchronizedObservation:
    """One exact-frame CARLA observation and its opaque sensor payloads."""

    frame_id: int
    simulation_time_s: float
    ego_pose_world: Any = field(repr=False, compare=False)
    ego_velocity_world: Any = field(repr=False, compare=False)
    camera_images: Any = field(repr=False, compare=False)
    camera_ids: tuple[int, ...]
    camera_intrinsics: Any = field(repr=False, compare=False)
    camera_extrinsics: Any = field(repr=False, compare=False)
    ego_history: Any = field(repr=False, compare=False)
    ego_history_frame_ids: tuple[int, ...]
    ego_history_simulation_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "simulation_time_s",
            _finite_float(self.simulation_time_s, "simulation_time_s", minimum=0.0),
        )
        object.__setattr__(self, "camera_ids", _camera_ids(self.camera_ids))
        object.__setattr__(
            self,
            "ego_history_frame_ids",
            _nonnegative_int_tuple(self.ego_history_frame_ids, "ego_history_frame_ids"),
        )
        object.__setattr__(
            self,
            "ego_history_simulation_times_s",
            _nonnegative_float_tuple(
                self.ego_history_simulation_times_s,
                "ego_history_simulation_times_s",
            ),
        )
        if self.ego_pose_world is None:
            raise ValueError("ego_pose_world must not be None")
        if self.ego_velocity_world is None:
            raise ValueError("ego_velocity_world must not be None")
        if self.camera_images is None:
            raise ValueError("camera_images must not be None")
        if self.camera_intrinsics is None:
            raise ValueError("camera_intrinsics must not be None")
        if self.camera_extrinsics is None:
            raise ValueError("camera_extrinsics must not be None")
        if self.ego_history is None:
            raise ValueError("ego_history must not be None")
        camera_count = _payload_length(self.camera_images)
        if camera_count is not None and camera_count != len(self.camera_ids):
            raise ValueError("camera_images length must match camera_ids")
        history_count = _payload_length(self.ego_history)
        if history_count is not None and history_count != len(self.ego_history_frame_ids):
            raise ValueError("ego_history length must match ego_history_frame_ids")
        if len(self.ego_history_frame_ids) != len(self.ego_history_simulation_times_s):
            raise ValueError("ego history frame IDs and timestamps must have equal length")
        if self.ego_history_frame_ids[-1] > self.frame_id:
            raise ValueError("ego history cannot end after the observation frame")
        if self.ego_history_simulation_times_s[-1] > self.simulation_time_s:
            raise ValueError("ego history cannot end after the observation time")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "frame_id": self.frame_id,
            "simulation_time_s": self.simulation_time_s,
            "camera_ids": list(self.camera_ids),
            "ego_history_frame_ids": list(self.ego_history_frame_ids),
            "ego_history_simulation_times_s": list(self.ego_history_simulation_times_s),
            "ego_pose_world": summarize_payload(self.ego_pose_world),
            "ego_velocity_world": summarize_payload(self.ego_velocity_world),
            "camera_images": summarize_payload(self.camera_images),
            "camera_intrinsics": summarize_payload(self.camera_intrinsics),
            "camera_extrinsics": summarize_payload(self.camera_extrinsics),
            "ego_history": summarize_payload(self.ego_history),
            "payloads_summarized": True,
        }


@dataclass(frozen=True)
class InferenceRequest:
    """Model request tied to one synchronized source observation."""

    source_frame_id: int
    source_simulation_time_s: float
    capture_pose_world: Any = field(repr=False, compare=False)
    image_frames: Any = field(repr=False, compare=False)
    ego_history_xyz: Any = field(repr=False, compare=False)
    ego_history_rot: Any = field(repr=False, compare=False)
    camera_ids: tuple[int, ...]
    submission_wall_time_s: float
    navigation_prompt: str = ""
    navigation_weight: float = 1.0
    prompt_revision: int = 0
    respawn_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_frame_id",
            _nonnegative_int(self.source_frame_id, "source_frame_id"),
        )
        object.__setattr__(
            self,
            "source_simulation_time_s",
            _finite_float(
                self.source_simulation_time_s,
                "source_simulation_time_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "submission_wall_time_s",
            _finite_float(self.submission_wall_time_s, "submission_wall_time_s", minimum=0.0),
        )
        object.__setattr__(self, "camera_ids", _camera_ids(self.camera_ids))
        object.__setattr__(
            self,
            "navigation_weight",
            _finite_float(self.navigation_weight, "navigation_weight", minimum=0.0),
        )
        object.__setattr__(
            self,
            "prompt_revision",
            _nonnegative_int(self.prompt_revision, "prompt_revision"),
        )
        object.__setattr__(
            self,
            "respawn_revision",
            _nonnegative_int(self.respawn_revision, "respawn_revision"),
        )
        if not isinstance(self.navigation_prompt, str):
            raise TypeError("navigation_prompt must be a string")
        for name in ("capture_pose_world", "image_frames", "ego_history_xyz", "ego_history_rot"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be None")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "source_frame_id": self.source_frame_id,
            "source_simulation_time_s": self.source_simulation_time_s,
            "submission_wall_time_s": self.submission_wall_time_s,
            "camera_ids": list(self.camera_ids),
            "navigation_prompt": self.navigation_prompt,
            "navigation_weight": self.navigation_weight,
            "prompt_revision": self.prompt_revision,
            "respawn_revision": self.respawn_revision,
            "capture_pose_world": summarize_payload(self.capture_pose_world),
            "image_frames": summarize_payload(self.image_frames),
            "ego_history_xyz": summarize_payload(self.ego_history_xyz),
            "ego_history_rot": summarize_payload(self.ego_history_rot),
            "payloads_summarized": True,
        }


@dataclass(frozen=True)
class TrajectoryPlan:
    """One selected, timestamped model trajectory fixed to its capture pose."""

    plan_id: str
    source_frame_id: int
    source_simulation_time_s: float
    capture_pose_world: Any = field(repr=False, compare=False)
    selected_ego_frame_points: Any = field(repr=False, compare=False)
    world_frame_points: Any = field(repr=False, compare=False)
    waypoint_times_s: tuple[float, ...]
    coc_text: str
    candidate_metadata: Any = field(repr=False, compare=False)
    inference_wall_latency_s: float
    prompt_revision: int
    respawn_revision: int
    selected_candidate_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _nonempty_string(self.plan_id, "plan_id"))
        object.__setattr__(
            self,
            "source_frame_id",
            _nonnegative_int(self.source_frame_id, "source_frame_id"),
        )
        object.__setattr__(
            self,
            "source_simulation_time_s",
            _finite_float(
                self.source_simulation_time_s,
                "source_simulation_time_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(self, "waypoint_times_s", _waypoint_times(self.waypoint_times_s))
        object.__setattr__(
            self,
            "inference_wall_latency_s",
            _finite_float(
                self.inference_wall_latency_s,
                "inference_wall_latency_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "prompt_revision",
            _nonnegative_int(self.prompt_revision, "prompt_revision"),
        )
        object.__setattr__(
            self,
            "respawn_revision",
            _nonnegative_int(self.respawn_revision, "respawn_revision"),
        )
        object.__setattr__(
            self,
            "selected_candidate_index",
            _nonnegative_int(self.selected_candidate_index, "selected_candidate_index"),
        )
        if not isinstance(self.coc_text, str):
            raise TypeError("coc_text must be a string")
        for name in ("capture_pose_world", "selected_ego_frame_points", "world_frame_points"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be None")

        expected_points = len(self.waypoint_times_s)
        for name in ("selected_ego_frame_points", "world_frame_points"):
            payload_length = _payload_length(getattr(self, name))
            if payload_length is not None and payload_length != expected_points:
                raise ValueError(
                    f"{name} length {payload_length} does not match "
                    f"{expected_points} waypoint times"
                )
        if self.candidate_metadata is None:
            raise ValueError("candidate_metadata must not be None")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "plan_id": self.plan_id,
            "source_frame_id": self.source_frame_id,
            "source_simulation_time_s": self.source_simulation_time_s,
            "waypoint_times_s": list(self.waypoint_times_s),
            "coc_text": self.coc_text,
            "inference_wall_latency_s": self.inference_wall_latency_s,
            "prompt_revision": self.prompt_revision,
            "respawn_revision": self.respawn_revision,
            "selected_candidate_index": self.selected_candidate_index,
            "capture_pose_world": summarize_payload(self.capture_pose_world),
            "selected_ego_frame_points": summarize_payload(self.selected_ego_frame_points),
            "world_frame_points": summarize_payload(self.world_frame_points),
            "candidate_metadata": _json_safe(self.candidate_metadata),
            "payloads_summarized": True,
        }


@dataclass(frozen=True)
class PlanValidation:
    """Validity decision for a trajectory at one simulation instant."""

    valid: bool
    rejection_reason: str | None
    source_age_s: float
    remaining_horizon_s: float
    lateral_drift_m: float
    heading_drift_deg: float
    first_usable_waypoint_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        object.__setattr__(
            self,
            "rejection_reason",
            _optional_string(self.rejection_reason, "rejection_reason"),
        )
        for name in (
            "source_age_s",
            "remaining_horizon_s",
            "lateral_drift_m",
            "heading_drift_deg",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name, minimum=0.0))
        object.__setattr__(
            self,
            "first_usable_waypoint_index",
            _optional_nonnegative_int(
                self.first_usable_waypoint_index,
                "first_usable_waypoint_index",
            ),
        )
        if self.valid and self.rejection_reason is not None:
            raise ValueError("a valid plan cannot have a rejection_reason")
        if self.valid and self.first_usable_waypoint_index is None:
            raise ValueError("a valid plan must have a first_usable_waypoint_index")
        if not self.valid and self.rejection_reason is None:
            raise ValueError("an invalid plan must have a rejection_reason")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
            "source_age_s": self.source_age_s,
            "remaining_horizon_s": self.remaining_horizon_s,
            "lateral_drift_m": self.lateral_drift_m,
            "heading_drift_deg": self.heading_drift_deg,
            "first_usable_waypoint_index": self.first_usable_waypoint_index,
        }


@dataclass(frozen=True)
class ControlDecision:
    """Controller request and final applied control, kept separately."""

    frame_id: int
    simulation_time_s: float
    controller_state: str
    target_point_world: Any = field(repr=False, compare=False)
    target_speed_mps: float
    requested_steering: float
    requested_throttle: float
    requested_brake: float
    applied_steering: float
    applied_throttle: float
    applied_brake: float
    fallback_state: str = "NONE"
    fallback_reason: str | None = None
    safety_override_applied: bool = False
    safety_override_type: str | None = None
    safety_override_reason: str | None = None
    source_plan_id: str | None = None
    source_plan_frame_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "simulation_time_s",
            _finite_float(self.simulation_time_s, "simulation_time_s", minimum=0.0),
        )
        object.__setattr__(
            self,
            "controller_state",
            _nonempty_string(self.controller_state, "controller_state"),
        )
        object.__setattr__(
            self,
            "target_point_world",
            _optional_point3(self.target_point_world, "target_point_world"),
        )
        object.__setattr__(
            self,
            "target_speed_mps",
            _finite_float(self.target_speed_mps, "target_speed_mps", minimum=0.0),
        )
        for prefix in ("requested", "applied"):
            object.__setattr__(
                self,
                f"{prefix}_steering",
                _control_value(getattr(self, f"{prefix}_steering"), f"{prefix}_steering", -1, 1),
            )
            for suffix in ("throttle", "brake"):
                name = f"{prefix}_{suffix}"
                object.__setattr__(self, name, _control_value(getattr(self, name), name, 0, 1))

        object.__setattr__(
            self,
            "fallback_state",
            _nonempty_string(self.fallback_state, "fallback_state"),
        )
        object.__setattr__(
            self,
            "fallback_reason",
            _optional_string(self.fallback_reason, "fallback_reason"),
        )
        if not isinstance(self.safety_override_applied, bool):
            raise TypeError("safety_override_applied must be bool")
        object.__setattr__(
            self,
            "safety_override_type",
            _optional_string(self.safety_override_type, "safety_override_type"),
        )
        object.__setattr__(
            self,
            "safety_override_reason",
            _optional_string(self.safety_override_reason, "safety_override_reason"),
        )
        object.__setattr__(
            self,
            "source_plan_id",
            _optional_string(self.source_plan_id, "source_plan_id"),
        )
        object.__setattr__(
            self,
            "source_plan_frame_id",
            _optional_nonnegative_int(self.source_plan_frame_id, "source_plan_frame_id"),
        )
        if self.safety_override_applied:
            if self.safety_override_type is None or self.safety_override_reason is None:
                raise ValueError("an applied safety override requires type and reason")
        elif self.safety_override_type is not None or self.safety_override_reason is not None:
            raise ValueError("inactive safety override cannot have type or reason")
        if (self.source_plan_id is None) != (self.source_plan_frame_id is None):
            raise ValueError("source plan ID and frame ID must both be set or both be None")
        if self.controller_state.upper() == "TRACKING":
            if self.target_point_world is None:
                raise ValueError("TRACKING requires a target_point_world")
            if self.source_plan_id is None:
                raise ValueError("TRACKING requires source plan identity")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "frame_id": self.frame_id,
            "simulation_time_s": self.simulation_time_s,
            "controller_state": self.controller_state,
            "target_point_world": (
                list(self.target_point_world) if self.target_point_world is not None else None
            ),
            "target_speed_mps": self.target_speed_mps,
            "requested_control": {
                "steering": self.requested_steering,
                "throttle": self.requested_throttle,
                "brake": self.requested_brake,
            },
            "applied_control": {
                "steering": self.applied_steering,
                "throttle": self.applied_throttle,
                "brake": self.applied_brake,
            },
            "fallback_state": self.fallback_state,
            "fallback_reason": self.fallback_reason,
            "safety_override_applied": self.safety_override_applied,
            "safety_override_type": self.safety_override_type,
            "safety_override_reason": self.safety_override_reason,
            "source_plan_id": self.source_plan_id,
            "source_plan_frame_id": self.source_plan_frame_id,
            "payloads_summarized": True,
        }


@dataclass(frozen=True)
class VisualizationSnapshot:
    """Immutable semantic input for rendering under the module ownership policy.

    Scalar metadata and control tuples are normalized by the constructor. Opaque
    image/trajectory payloads are ownership-transferred and must not be mutated by
    either producer or renderer after the snapshot is published.
    """

    current_frame_id: int
    current_simulation_time_s: float
    source_observation_frame_id: int | None = None
    source_observation_simulation_time_s: float | None = None
    source_camera_bundle: Any = field(default=None, repr=False, compare=False)
    current_display_camera: Any = field(default=None, repr=False, compare=False)
    candidate_trajectories: Any = field(default=None, repr=False, compare=False)
    selected_candidate_index: int | None = None
    selected_proposal: Any = field(default=None, repr=False, compare=False)
    controller_reference_trajectory: Any = field(default=None, repr=False, compare=False)
    controller_target_point: Any = field(default=None, repr=False, compare=False)
    requested_control: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_trail: Any = field(default=None, repr=False, compare=False)
    applied_control: tuple[float, float, float] = (0.0, 0.0, 0.0)
    safety_zones: Any = field(default=None, repr=False, compare=False)
    conflict_zones: Any = field(default=None, repr=False, compare=False)
    safety_override_applied: bool = False
    safety_override_type: str | None = None
    safety_override_reason: str | None = None
    coc_text: str = ""
    navigation_prompt: str = ""
    navigation_weight: float = 1.0
    inference_backend: str = "local"
    inference_state: str = "idle"
    inference_latency_s: float | None = None
    plan_source_age_s: float | None = None
    remaining_horizon_s: float | None = None
    fallback_state: str = "NONE"
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_frame_id",
            _nonnegative_int(self.current_frame_id, "current_frame_id"),
        )
        object.__setattr__(
            self,
            "current_simulation_time_s",
            _finite_float(
                self.current_simulation_time_s,
                "current_simulation_time_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "source_observation_frame_id",
            _optional_nonnegative_int(
                self.source_observation_frame_id,
                "source_observation_frame_id",
            ),
        )
        object.__setattr__(
            self,
            "source_observation_simulation_time_s",
            _optional_finite_float(
                self.source_observation_simulation_time_s,
                "source_observation_simulation_time_s",
                minimum=0.0,
            ),
        )
        if (self.source_observation_frame_id is None) != (
            self.source_observation_simulation_time_s is None
        ):
            raise ValueError("source observation frame and time must both be set or both be None")

        object.__setattr__(
            self,
            "selected_candidate_index",
            _optional_nonnegative_int(
                self.selected_candidate_index,
                "selected_candidate_index",
            ),
        )
        object.__setattr__(
            self,
            "requested_control",
            _control_triplet(self.requested_control, "requested_control"),
        )
        object.__setattr__(
            self,
            "applied_control",
            _control_triplet(self.applied_control, "applied_control"),
        )
        if not isinstance(self.safety_override_applied, bool):
            raise TypeError("safety_override_applied must be bool")
        object.__setattr__(
            self,
            "safety_override_type",
            _optional_string(self.safety_override_type, "safety_override_type"),
        )
        object.__setattr__(
            self,
            "safety_override_reason",
            _optional_string(self.safety_override_reason, "safety_override_reason"),
        )
        if self.safety_override_applied:
            if self.safety_override_type is None or self.safety_override_reason is None:
                raise ValueError("an applied safety override requires type and reason")
        elif self.safety_override_type is not None or self.safety_override_reason is not None:
            raise ValueError("inactive safety override cannot have type or reason")

        for name in ("coc_text", "navigation_prompt"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        object.__setattr__(
            self,
            "navigation_weight",
            _finite_float(self.navigation_weight, "navigation_weight", minimum=0.0),
        )
        backend = _nonempty_string(self.inference_backend, "inference_backend").lower()
        if backend not in {"local", "remote"}:
            raise ValueError("inference_backend must be 'local' or 'remote'")
        object.__setattr__(self, "inference_backend", backend)
        object.__setattr__(
            self,
            "inference_state",
            _nonempty_string(self.inference_state, "inference_state"),
        )
        for name in ("inference_latency_s", "plan_source_age_s", "remaining_horizon_s"):
            object.__setattr__(
                self,
                name,
                _optional_finite_float(getattr(self, name), name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "fallback_state",
            _nonempty_string(self.fallback_state, "fallback_state"),
        )
        object.__setattr__(
            self,
            "rejection_reason",
            _optional_string(self.rejection_reason, "rejection_reason"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record_type": type(self).__name__,
            "current_frame_id": self.current_frame_id,
            "current_simulation_time_s": self.current_simulation_time_s,
            "source_observation_frame_id": self.source_observation_frame_id,
            "source_observation_simulation_time_s": (self.source_observation_simulation_time_s),
            "source_camera_bundle": summarize_payload(self.source_camera_bundle),
            "current_display_camera": summarize_payload(self.current_display_camera),
            "candidate_trajectories": summarize_payload(self.candidate_trajectories),
            "selected_candidate_index": self.selected_candidate_index,
            "selected_proposal": summarize_payload(self.selected_proposal),
            "controller_reference_trajectory": summarize_payload(
                self.controller_reference_trajectory
            ),
            "controller_target_point": summarize_payload(self.controller_target_point),
            "requested_control": list(self.requested_control),
            "ego_trail": summarize_payload(self.ego_trail),
            "applied_control": list(self.applied_control),
            "safety_zones": summarize_payload(self.safety_zones),
            "conflict_zones": summarize_payload(self.conflict_zones),
            "safety_override_applied": self.safety_override_applied,
            "safety_override_type": self.safety_override_type,
            "safety_override_reason": self.safety_override_reason,
            "coc_text": self.coc_text,
            "navigation_prompt": self.navigation_prompt,
            "navigation_weight": self.navigation_weight,
            "inference_backend": self.inference_backend,
            "inference_state": self.inference_state,
            "inference_latency_s": self.inference_latency_s,
            "plan_source_age_s": self.plan_source_age_s,
            "remaining_horizon_s": self.remaining_horizon_s,
            "fallback_state": self.fallback_state,
            "rejection_reason": self.rejection_reason,
            "payloads_summarized": True,
        }


__all__ = [
    "ControlDecision",
    "InferenceRequest",
    "PlanValidation",
    "SynchronizedObservation",
    "TrajectoryPlan",
    "VisualizationSnapshot",
    "summarize_payload",
    "to_json_dict",
    "to_json_line",
]

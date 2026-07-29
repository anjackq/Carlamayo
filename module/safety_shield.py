"""Pure, fail-closed safety checks for the CARLA closed-loop prototype.

The module intentionally has no CARLA dependency.  A simulator adapter is
responsible for copying one exact-frame world snapshot into the immutable
records below and for performing exact map/lane queries.  This module then
evaluates obstacle conflicts and arbitrates the final control command.

The shield is deliberately stop-only: it may preserve a previously safe or
road-validated steering command, but it never invents a corrective path.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


_EPSILON = 1e-9


class AssessmentStatus(str, Enum):
    """Tri-state result used by every safety input."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


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


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


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


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


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


def _point2(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain x and y")
    try:
        x, y = value
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain x and y") from exc
    return _finite_float(x, f"{name}.x"), _finite_float(y, f"{name}.y")


def _point3(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain x, y, and optionally z")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain x, y, and optionally z") from exc
    if len(items) == 2:
        items = (*items, 0.0)
    if len(items) != 3:
        raise ValueError(f"{name} must contain two or three coordinates")
    return tuple(_finite_float(item, name) for item in items)  # type: ignore[return-value]


def _reason_tuple(values: Iterable[Any], *, allow_empty: bool = True) -> tuple[str, ...]:
    normalized = []
    seen = set()
    for value in values:
        reason = _nonempty_string(value, "reason_code")
        if reason not in seen:
            normalized.append(reason)
            seen.add(reason)
    if not allow_empty and not normalized:
        raise ValueError("at least one reason code is required")
    return tuple(normalized)


@dataclass(frozen=True)
class ControlCommand:
    """One normalized steering/throttle/brake command."""

    steering: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        steering = _finite_float(self.steering, "steering")
        throttle = _finite_float(self.throttle, "throttle")
        brake = _finite_float(self.brake, "brake")
        if not -1.0 <= steering <= 1.0:
            raise ValueError("steering must be between -1 and 1")
        if not 0.0 <= throttle <= 1.0:
            raise ValueError("throttle must be between 0 and 1")
        if not 0.0 <= brake <= 1.0:
            raise ValueError("brake must be between 0 and 1")
        object.__setattr__(self, "steering", steering)
        object.__setattr__(self, "throttle", throttle)
        object.__setattr__(self, "brake", brake)

    @classmethod
    def full_brake(cls, steering: float = 0.0) -> ControlCommand:
        return cls(steering=steering, throttle=0.0, brake=1.0)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.steering, self.throttle, self.brake

    @property
    def is_longitudinally_exclusive(self) -> bool:
        """Whether throttle and brake are not requested simultaneously."""

        return self.throttle == 0.0 or self.brake == 0.0

    def to_json_dict(self) -> dict[str, float]:
        return {
            "steering": self.steering,
            "throttle": self.throttle,
            "brake": self.brake,
        }


@dataclass(frozen=True)
class RoadContainmentSample:
    """One exact-map footprint/path containment query made by an adapter."""

    sample_index: int
    position_xyz: tuple[float, float, float]
    contained: bool | None
    margin_m: float | None = None
    road_id: int | None = None
    lane_id: int | None = None
    is_junction: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_index",
            _nonnegative_int(self.sample_index, "sample_index"),
        )
        object.__setattr__(self, "position_xyz", _point3(self.position_xyz, "position_xyz"))
        if self.contained is not None and not isinstance(self.contained, bool):
            raise TypeError("contained must be bool or None")
        object.__setattr__(self, "margin_m", _optional_finite_float(self.margin_m, "margin_m"))
        object.__setattr__(self, "road_id", _optional_int(self.road_id, "road_id"))
        object.__setattr__(self, "lane_id", _optional_int(self.lane_id, "lane_id"))
        if self.is_junction is not None and not isinstance(self.is_junction, bool):
            raise TypeError("is_junction must be bool or None")
        object.__setattr__(self, "reason", _optional_string(self.reason, "reason"))


@dataclass(frozen=True)
class RoadContainmentAssessment:
    """Adapter-produced road/lane authorization consumed by the shield."""

    status: AssessmentStatus
    reason_codes: tuple[str, ...]
    sample_count: int = 0
    min_margin_m: float | None = None
    first_bad_sample_index: int | None = None
    quality: str = "unspecified"

    def __post_init__(self) -> None:
        try:
            status = AssessmentStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"invalid road-containment status: {self.status}") from exc
        reasons = _reason_tuple(self.reason_codes)
        if status is AssessmentStatus.SAFE and reasons:
            raise ValueError("a SAFE road assessment cannot have reason codes")
        if status is not AssessmentStatus.SAFE and not reasons:
            raise ValueError("an UNSAFE/UNKNOWN road assessment requires a reason code")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "sample_count",
            _nonnegative_int(self.sample_count, "sample_count"),
        )
        object.__setattr__(
            self,
            "min_margin_m",
            _optional_finite_float(self.min_margin_m, "min_margin_m"),
        )
        if self.first_bad_sample_index is not None:
            object.__setattr__(
                self,
                "first_bad_sample_index",
                _nonnegative_int(self.first_bad_sample_index, "first_bad_sample_index"),
            )
        object.__setattr__(self, "quality", _nonempty_string(self.quality, "quality"))

    @classmethod
    def safe(
        cls,
        *,
        sample_count: int = 0,
        min_margin_m: float | None = None,
        quality: str = "exact",
    ) -> RoadContainmentAssessment:
        return cls(AssessmentStatus.SAFE, (), sample_count, min_margin_m, None, quality)

    @classmethod
    def unsafe(
        cls,
        reason_codes: Iterable[str] = ("road_not_contained",),
        *,
        sample_count: int = 0,
        min_margin_m: float | None = None,
        first_bad_sample_index: int | None = None,
        quality: str = "exact",
    ) -> RoadContainmentAssessment:
        return cls(
            AssessmentStatus.UNSAFE,
            tuple(reason_codes),
            sample_count,
            min_margin_m,
            first_bad_sample_index,
            quality,
        )

    @classmethod
    def unknown(
        cls,
        reason_codes: Iterable[str] = ("road_containment_unavailable",),
        *,
        sample_count: int = 0,
        first_bad_sample_index: int | None = None,
        quality: str = "unknown",
    ) -> RoadContainmentAssessment:
        return cls(
            AssessmentStatus.UNKNOWN,
            tuple(reason_codes),
            sample_count,
            None,
            first_bad_sample_index,
            quality,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "sample_count": self.sample_count,
            "min_margin_m": self.min_margin_m,
            "first_bad_sample_index": self.first_bad_sample_index,
            "quality": self.quality,
        }


def assess_road_containment(
    samples: Iterable[RoadContainmentSample] | None,
    *,
    quality: str = "exact",
) -> RoadContainmentAssessment:
    """Reduce exact adapter map queries into one fail-closed assessment."""

    if samples is None:
        return RoadContainmentAssessment.unknown(quality=quality)
    try:
        normalized = tuple(samples)
    except TypeError:
        return RoadContainmentAssessment.unknown(("invalid_road_samples",), quality=quality)
    if not normalized:
        return RoadContainmentAssessment.unknown(("road_samples_missing",), quality=quality)
    if any(not isinstance(sample, RoadContainmentSample) for sample in normalized):
        return RoadContainmentAssessment.unknown(
            ("invalid_road_sample_type",),
            sample_count=len(normalized),
            quality=quality,
        )

    margins = [sample.margin_m for sample in normalized if sample.margin_m is not None]
    min_margin = min(margins) if margins else None
    unsafe_samples = [
        sample
        for sample in normalized
        if sample.contained is False
        or (sample.margin_m is not None and sample.margin_m < 0.0)
    ]
    if unsafe_samples:
        reasons = ["road_not_contained"]
        if any(sample.margin_m is not None and sample.margin_m < 0.0 for sample in unsafe_samples):
            reasons.append("negative_road_margin")
        reasons.extend(sample.reason for sample in unsafe_samples if sample.reason)
        return RoadContainmentAssessment.unsafe(
            _reason_tuple(reasons),
            sample_count=len(normalized),
            min_margin_m=min_margin,
            first_bad_sample_index=min(sample.sample_index for sample in unsafe_samples),
            quality=quality,
        )

    unknown_samples = [sample for sample in normalized if sample.contained is None]
    if unknown_samples:
        reasons = ["road_containment_query_unknown"]
        reasons.extend(sample.reason for sample in unknown_samples if sample.reason)
        return RoadContainmentAssessment.unknown(
            _reason_tuple(reasons),
            sample_count=len(normalized),
            first_bad_sample_index=min(sample.sample_index for sample in unknown_samples),
            quality=quality,
        )

    return RoadContainmentAssessment.safe(
        sample_count=len(normalized),
        min_margin_m=min_margin,
        quality=quality,
    )


@dataclass(frozen=True)
class EgoKinematics:
    """Exact-frame ego pose, velocity, and 2D oriented footprint."""

    frame_id: int
    center_xy: tuple[float, float]
    yaw_rad: float
    velocity_xy: tuple[float, float]
    half_length_m: float
    half_width_m: float
    actor_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(self, "center_xy", _point2(self.center_xy, "center_xy"))
        object.__setattr__(self, "yaw_rad", _finite_float(self.yaw_rad, "yaw_rad"))
        object.__setattr__(self, "velocity_xy", _point2(self.velocity_xy, "velocity_xy"))
        object.__setattr__(
            self,
            "half_length_m",
            _finite_float(self.half_length_m, "half_length_m", minimum=_EPSILON),
        )
        object.__setattr__(
            self,
            "half_width_m",
            _finite_float(self.half_width_m, "half_width_m", minimum=_EPSILON),
        )
        if self.actor_id is not None:
            object.__setattr__(self, "actor_id", _nonnegative_int(self.actor_id, "actor_id"))


@dataclass(frozen=True)
class ActorObstacle:
    """Exact-frame dynamic actor OBB copied from CARLA by an adapter."""

    frame_id: int
    actor_id: int
    type_id: str
    center_xy: tuple[float, float]
    yaw_rad: float
    velocity_xy: tuple[float, float]
    half_length_m: float
    half_width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(self, "actor_id", _nonnegative_int(self.actor_id, "actor_id"))
        object.__setattr__(self, "type_id", _nonempty_string(self.type_id, "type_id"))
        object.__setattr__(self, "center_xy", _point2(self.center_xy, "center_xy"))
        object.__setattr__(self, "yaw_rad", _finite_float(self.yaw_rad, "yaw_rad"))
        object.__setattr__(self, "velocity_xy", _point2(self.velocity_xy, "velocity_xy"))
        object.__setattr__(
            self,
            "half_length_m",
            _finite_float(self.half_length_m, "half_length_m", minimum=_EPSILON),
        )
        object.__setattr__(
            self,
            "half_width_m",
            _finite_float(self.half_width_m, "half_width_m", minimum=_EPSILON),
        )


@dataclass(frozen=True)
class ObstacleThreat:
    """Most relevant geometric/kinematic measurements for one actor."""

    actor_id: int
    type_id: str
    reason_codes: tuple[str, ...]
    surface_gap_m: float | None
    lateral_clearance_m: float | None
    closing_speed_mps: float | None
    stopping_distance_m: float
    ttc_s: float | None = None
    predicted_conflict_time_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _nonnegative_int(self.actor_id, "actor_id"))
        object.__setattr__(self, "type_id", _nonempty_string(self.type_id, "type_id"))
        object.__setattr__(
            self,
            "reason_codes",
            _reason_tuple(self.reason_codes, allow_empty=False),
        )
        for name in (
            "surface_gap_m",
            "lateral_clearance_m",
            "closing_speed_mps",
            "ttc_s",
            "predicted_conflict_time_s",
        ):
            object.__setattr__(self, name, _optional_finite_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "stopping_distance_m",
            _finite_float(self.stopping_distance_m, "stopping_distance_m", minimum=0.0),
        )
        if self.ttc_s is not None and self.ttc_s < 0.0:
            raise ValueError("ttc_s must be non-negative")
        if self.predicted_conflict_time_s is not None and self.predicted_conflict_time_s < 0.0:
            raise ValueError("predicted_conflict_time_s must be non-negative")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "type_id": self.type_id,
            "reason_codes": list(self.reason_codes),
            "surface_gap_m": self.surface_gap_m,
            "lateral_clearance_m": self.lateral_clearance_m,
            "closing_speed_mps": self.closing_speed_mps,
            "stopping_distance_m": self.stopping_distance_m,
            "ttc_s": self.ttc_s,
            "predicted_conflict_time_s": self.predicted_conflict_time_s,
        }


@dataclass(frozen=True)
class ObstacleAssessment:
    """Aggregate dynamic-obstacle safety result for one exact frame."""

    status: AssessmentStatus
    reason_codes: tuple[str, ...]
    primary_threat: ObstacleThreat | None = None
    evaluated_actor_count: int = 0
    prediction_performed: bool = False

    def __post_init__(self) -> None:
        try:
            status = AssessmentStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"invalid obstacle status: {self.status}") from exc
        reasons = _reason_tuple(self.reason_codes)
        if status is AssessmentStatus.SAFE and (reasons or self.primary_threat is not None):
            raise ValueError("a SAFE obstacle assessment cannot contain hazards")
        if status is AssessmentStatus.UNSAFE and self.primary_threat is None:
            raise ValueError("an UNSAFE obstacle assessment requires a primary threat")
        if status is AssessmentStatus.UNKNOWN and not reasons:
            raise ValueError("an UNKNOWN obstacle assessment requires a reason code")
        if status is not AssessmentStatus.UNSAFE and self.primary_threat is not None:
            raise ValueError("only an UNSAFE obstacle assessment may contain a threat")
        if not isinstance(self.prediction_performed, bool):
            raise TypeError("prediction_performed must be bool")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "evaluated_actor_count",
            _nonnegative_int(self.evaluated_actor_count, "evaluated_actor_count"),
        )

    @classmethod
    def safe(
        cls,
        *,
        evaluated_actor_count: int = 0,
        prediction_performed: bool = False,
    ) -> ObstacleAssessment:
        return cls(
            AssessmentStatus.SAFE,
            (),
            None,
            evaluated_actor_count,
            prediction_performed,
        )

    @classmethod
    def unsafe(
        cls,
        primary_threat: ObstacleThreat,
        *,
        reason_codes: Iterable[str] | None = None,
        evaluated_actor_count: int = 0,
        prediction_performed: bool = False,
    ) -> ObstacleAssessment:
        reasons = tuple(reason_codes) if reason_codes is not None else primary_threat.reason_codes
        return cls(
            AssessmentStatus.UNSAFE,
            reasons,
            primary_threat,
            evaluated_actor_count,
            prediction_performed,
        )

    @classmethod
    def unknown(
        cls,
        reason_codes: Iterable[str] = ("obstacle_data_unavailable",),
        *,
        evaluated_actor_count: int = 0,
        prediction_performed: bool = False,
    ) -> ObstacleAssessment:
        return cls(
            AssessmentStatus.UNKNOWN,
            tuple(reason_codes),
            None,
            evaluated_actor_count,
            prediction_performed,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "primary_threat": (
                self.primary_threat.to_json_dict()
                if self.primary_threat is not None
                else None
            ),
            "evaluated_actor_count": self.evaluated_actor_count,
            "prediction_performed": self.prediction_performed,
        }


@dataclass(frozen=True)
class SafetyPolicy:
    """Conservative initial policy; values require scenario calibration."""

    path_sample_spacing_m: float = 0.5
    lateral_clearance_m: float = 0.25
    longitudinal_clearance_m: float = 0.25
    reaction_time_s: float = 0.5
    assumed_deceleration_mps2: float = 4.0
    stop_buffer_m: float = 2.0
    hard_gap_m: float = 1.0
    ttc_threshold_s: float = 2.0
    minimum_closing_speed_mps: float = 0.1
    prediction_horizon_s: float = 3.0
    prediction_time_step_s: float = 0.1
    emergency_hold_ticks: int = 5
    clear_ticks_to_release: int = 3

    def __post_init__(self) -> None:
        for name in (
            "path_sample_spacing_m",
            "lateral_clearance_m",
            "longitudinal_clearance_m",
            "reaction_time_s",
            "assumed_deceleration_mps2",
            "stop_buffer_m",
            "hard_gap_m",
            "ttc_threshold_s",
            "minimum_closing_speed_mps",
            "prediction_horizon_s",
            "prediction_time_step_s",
        ):
            minimum = _EPSILON if name in {
                "path_sample_spacing_m",
                "assumed_deceleration_mps2",
                "ttc_threshold_s",
                "prediction_horizon_s",
                "prediction_time_step_s",
            } else 0.0
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=minimum),
            )
        if self.path_sample_spacing_m > 0.5:
            raise ValueError("path_sample_spacing_m must be no greater than 0.5 m")
        object.__setattr__(
            self,
            "emergency_hold_ticks",
            _nonnegative_int(self.emergency_hold_ticks, "emergency_hold_ticks"),
        )
        object.__setattr__(
            self,
            "clear_ticks_to_release",
            _nonnegative_int(self.clear_ticks_to_release, "clear_ticks_to_release"),
        )
        if self.emergency_hold_ticks < 1 or self.clear_ticks_to_release < 1:
            raise ValueError("emergency_hold_ticks and clear_ticks_to_release must be at least 1")


def _normalize_path(points: Iterable[Sequence[Any]]) -> tuple[tuple[float, ...], ...]:
    if points is None:
        raise ValueError("path points are required")
    try:
        raw_points = tuple(points)
    except TypeError as exc:
        raise TypeError("path points must be iterable") from exc
    if not raw_points:
        raise ValueError("path must contain at least one point")
    normalized = []
    dimension = None
    for index, point in enumerate(raw_points):
        if isinstance(point, (str, bytes)):
            raise TypeError(f"path point {index} must be a coordinate sequence")
        try:
            values = tuple(_finite_float(value, f"path[{index}]") for value in point)
        except TypeError as exc:
            raise TypeError(f"path point {index} must be a coordinate sequence") from exc
        if len(values) < 2:
            raise ValueError(f"path point {index} must have at least x and y")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise ValueError("all path points must have the same dimension")
        normalized.append(values)
    return tuple(normalized)


def densify_path(
    points: Iterable[Sequence[Any]],
    *,
    max_spacing_m: float = 0.5,
) -> tuple[tuple[float, ...], ...]:
    """Interpolate a path so every XY segment is at most ``max_spacing_m``.

    All input dimensions are preserved, including Z for a CARLA map adapter.
    The safety helper intentionally rejects spacing greater than 0.5 metres.
    """

    spacing = _finite_float(max_spacing_m, "max_spacing_m", minimum=_EPSILON)
    if spacing > 0.5:
        raise ValueError("max_spacing_m must be no greater than 0.5 m")
    normalized = _normalize_path(points)
    dense = [normalized[0]]
    for start, end in zip(normalized, normalized[1:]):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(math.ceil(distance / spacing)))
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append(
                tuple(
                    start[axis] + fraction * (end[axis] - start[axis])
                    for axis in range(len(start))
                )
            )
    return tuple(dense)


def stopping_distance_m(speed_mps: float, policy: SafetyPolicy | None = None) -> float:
    """Return the policy's auditable reaction-plus-braking envelope."""

    active_policy = policy or SafetyPolicy()
    speed = _finite_float(speed_mps, "speed_mps", minimum=0.0)
    return (
        active_policy.stop_buffer_m
        + speed * active_policy.reaction_time_s
        + speed * speed / (2.0 * active_policy.assumed_deceleration_mps2)
    )


def _add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def _subtract(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _scale(a: tuple[float, float], scalar: float) -> tuple[float, float]:
    return a[0] * scalar, a[1] * scalar


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm(a: tuple[float, float]) -> float:
    return math.hypot(a[0], a[1])


def _unit_from_yaw(yaw_rad: float) -> tuple[float, float]:
    return math.cos(yaw_rad), math.sin(yaw_rad)


def _normal(tangent: tuple[float, float]) -> tuple[float, float]:
    return -tangent[1], tangent[0]


def _box_support(
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
    axis: tuple[float, float],
) -> float:
    forward = _unit_from_yaw(yaw_rad)
    right = _normal(forward)
    return (
        abs(_dot(forward, axis)) * half_length_m
        + abs(_dot(right, axis)) * half_width_m
    )


def _path_xy(points: Iterable[Sequence[Any]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in points)


def _last_tangent(
    points: Sequence[tuple[float, float]],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    for start, end in reversed(tuple(zip(points, points[1:]))):
        delta = _subtract(end, start)
        length = _norm(delta)
        if length > _EPSILON:
            return _scale(delta, 1.0 / length)
    return fallback


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(_norm(_subtract(end, start)) for start, end in zip(points, points[1:]))


def _project_to_polyline(
    points: Sequence[tuple[float, float]],
    point: tuple[float, float],
    fallback_tangent: tuple[float, float],
) -> tuple[float, float, tuple[float, float]]:
    """Return along-path distance, signed lateral offset, and local tangent."""

    best = None
    prefix = 0.0
    segments = []
    for start, end in zip(points, points[1:]):
        delta = _subtract(end, start)
        length = _norm(delta)
        if length <= _EPSILON:
            continue
        tangent = _scale(delta, 1.0 / length)
        raw_fraction = _dot(_subtract(point, start), tangent) / length
        fraction = min(1.0, max(0.0, raw_fraction))
        closest = _add(start, _scale(delta, fraction))
        distance_sq = _dot(_subtract(point, closest), _subtract(point, closest))
        candidate = (
            distance_sq,
            prefix + fraction * length,
            raw_fraction,
            start,
            length,
            tangent,
            len(segments),
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        segments.append((start, length, tangent, prefix))
        prefix += length

    if best is None:
        delta = _subtract(point, points[0])
        return (
            _dot(delta, fallback_tangent),
            _dot(delta, _normal(fallback_tangent)),
            fallback_tangent,
        )

    _, along, raw_fraction, start, length, tangent, segment_index = best
    if segment_index == 0 and raw_fraction < 0.0:
        along = raw_fraction * length
        closest = _add(start, _scale(tangent, along))
    elif segment_index == len(segments) - 1 and raw_fraction > 1.0:
        along = segments[-1][3] + raw_fraction * length
        closest = _add(start, _scale(tangent, raw_fraction * length))
    else:
        closest = _add(start, _scale(tangent, (along - segments[segment_index][3])))
    lateral = _dot(_subtract(point, closest), _normal(tangent))
    return along, lateral, tangent


def _prepare_corridor_path(
    ego: EgoKinematics,
    points: Iterable[Sequence[Any]],
    policy: SafetyPolicy,
    required_distance_m: float,
) -> tuple[tuple[float, float], ...]:
    normalized = densify_path(points, max_spacing_m=policy.path_sample_spacing_m)
    path = list(_path_xy(normalized))
    if _norm(_subtract(path[0], ego.center_xy)) > _EPSILON:
        path.insert(0, ego.center_xy)
    fallback = _unit_from_yaw(ego.yaw_rad)
    if len(path) == 1:
        path.append(_add(path[0], _scale(fallback, required_distance_m)))
    length = _polyline_length(path)
    if length < required_distance_m:
        tangent = _last_tangent(path, fallback)
        path.append(_add(path[-1], _scale(tangent, required_distance_m - length)))
    return tuple(path)


def _prepare_timed_path(
    ego: EgoKinematics,
    points: Iterable[Sequence[Any]],
    times_s: Iterable[Any],
    policy: SafetyPolicy,
) -> tuple[tuple[tuple[float, float], float], ...]:
    normalized = _normalize_path(points)
    times = tuple(_finite_float(value, "path_time_s", minimum=0.0) for value in times_s)
    if len(times) != len(normalized):
        raise ValueError("path_times_s length must match path_points")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("path_times_s must be strictly increasing")

    timed = [((point[0], point[1]), timestamp) for point, timestamp in zip(normalized, times)]
    if times[0] > 0.0:
        timed.insert(0, (ego.center_xy, 0.0))
    elif _norm(_subtract(timed[0][0], ego.center_xy)) > _EPSILON:
        raise ValueError("a path point at time zero must equal the current ego center")

    dense = [timed[0]]
    for (start, start_time), (end, end_time) in zip(timed, timed[1:]):
        distance = _norm(_subtract(end, start))
        duration = end_time - start_time
        steps = max(
            1,
            int(math.ceil(distance / policy.path_sample_spacing_m)),
            int(math.ceil(duration / policy.prediction_time_step_s)),
        )
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append(
                (
                    _add(start, _scale(_subtract(end, start), fraction)),
                    start_time + fraction * (end_time - start_time),
                )
            )
    return tuple(dense)


def _tangent_at(
    timed_path: Sequence[tuple[tuple[float, float], float]],
    index: int,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    for offset in range(1, len(timed_path)):
        for candidate in (index + offset, index - offset):
            if not 0 <= candidate < len(timed_path) or candidate == index:
                continue
            delta = _subtract(timed_path[candidate][0], timed_path[index][0])
            if candidate < index:
                delta = _scale(delta, -1.0)
            length = _norm(delta)
            if length > _EPSILON:
                return _scale(delta, 1.0 / length)
    return fallback


_OBSTACLE_REASON_PRIORITY = {
    "actor_overlap": 0,
    "actor_within_hard_gap": 1,
    "predicted_actor_conflict": 2,
    "actor_within_stopping_envelope": 3,
    "actor_ttc_below_threshold": 4,
}


def _sort_obstacle_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(reasons),
            key=lambda reason: (_OBSTACLE_REASON_PRIORITY.get(reason, 100), reason),
        )
    )


def _predicted_conflict_time(
    ego: EgoKinematics,
    actor: ActorObstacle,
    timed_path: Sequence[tuple[tuple[float, float], float]],
    policy: SafetyPolicy,
) -> float | None:
    fallback = _unit_from_yaw(ego.yaw_rad)
    for index, (ego_center, timestamp) in enumerate(timed_path):
        if timestamp <= 0.0 or timestamp > policy.prediction_horizon_s:
            continue
        tangent = _tangent_at(timed_path, index, fallback)
        normal = _normal(tangent)
        actor_center = _add(actor.center_xy, _scale(actor.velocity_xy, timestamp))
        relative = _subtract(actor_center, ego_center)
        actor_long = _box_support(
            actor.yaw_rad,
            actor.half_length_m,
            actor.half_width_m,
            tangent,
        )
        actor_lat = _box_support(
            actor.yaw_rad,
            actor.half_length_m,
            actor.half_width_m,
            normal,
        )
        longitudinal_overlap = abs(_dot(relative, tangent)) <= (
            ego.half_length_m + actor_long + policy.longitudinal_clearance_m
        )
        lateral_overlap = abs(_dot(relative, normal)) <= (
            ego.half_width_m + actor_lat + policy.lateral_clearance_m
        )
        if longitudinal_overlap and lateral_overlap:
            return timestamp
    return None


def assess_obstacles(
    *,
    ego: EgoKinematics,
    path_points: Iterable[Sequence[Any]],
    actors: Iterable[ActorObstacle] | None,
    policy: SafetyPolicy | None = None,
    path_times_s: Iterable[Any] | None = None,
) -> ObstacleAssessment:
    """Assess actor OBB conflicts against a path and stopping envelope.

    ``path_times_s`` are future seconds relative to the current ego snapshot.
    When supplied, constant-velocity actor prediction detects practical crossing
    conflicts.  Missing actor data or frame mismatches return UNKNOWN.
    """

    active_policy = policy or SafetyPolicy()
    if not isinstance(ego, EgoKinematics):
        return ObstacleAssessment.unknown(("invalid_ego_kinematics",))
    try:
        path_point_tuple = tuple(path_points)
    except TypeError:
        return ObstacleAssessment.unknown(("invalid_safety_path",))
    if actors is None:
        return ObstacleAssessment.unknown(("obstacle_data_unavailable",))
    try:
        actor_tuple = tuple(actors)
    except TypeError:
        return ObstacleAssessment.unknown(("invalid_obstacle_collection",))
    if any(not isinstance(actor, ActorObstacle) for actor in actor_tuple):
        return ObstacleAssessment.unknown(
            ("invalid_obstacle_type",),
            evaluated_actor_count=len(actor_tuple),
        )
    actor_tuple = tuple(
        actor
        for actor in actor_tuple
        if ego.actor_id is None or actor.actor_id != ego.actor_id
    )
    if any(actor.frame_id != ego.frame_id for actor in actor_tuple):
        return ObstacleAssessment.unknown(
            ("actor_snapshot_frame_mismatch",),
            evaluated_actor_count=len(actor_tuple),
        )
    actor_ids = [actor.actor_id for actor in actor_tuple]
    if len(actor_ids) != len(set(actor_ids)):
        return ObstacleAssessment.unknown(
            ("duplicate_actor_id",),
            evaluated_actor_count=len(actor_tuple),
        )

    ego_speed = _norm(ego.velocity_xy)
    stop_distance = stopping_distance_m(ego_speed, active_policy)
    required_distance = stop_distance + ego.half_length_m + active_policy.hard_gap_m
    try:
        corridor_path = _prepare_corridor_path(
            ego,
            path_point_tuple,
            active_policy,
            required_distance,
        )
    except (TypeError, ValueError, OverflowError):
        return ObstacleAssessment.unknown(
            ("invalid_safety_path",),
            evaluated_actor_count=len(actor_tuple),
        )

    prediction_performed = path_times_s is not None
    timed_path = None
    if path_times_s is not None:
        try:
            timed_path = _prepare_timed_path(
                ego,
                path_point_tuple,
                path_times_s,
                active_policy,
            )
        except (TypeError, ValueError, OverflowError):
            return ObstacleAssessment.unknown(
                ("invalid_path_times",),
                evaluated_actor_count=len(actor_tuple),
            )

    fallback_tangent = _unit_from_yaw(ego.yaw_rad)
    threats = []
    all_reasons = []
    for actor in sorted(actor_tuple, key=lambda item: item.actor_id):
        along, lateral, tangent = _project_to_polyline(
            corridor_path,
            actor.center_xy,
            fallback_tangent,
        )
        normal = _normal(tangent)
        ego_long = _box_support(
            ego.yaw_rad,
            ego.half_length_m,
            ego.half_width_m,
            tangent,
        )
        ego_lat = _box_support(
            ego.yaw_rad,
            ego.half_length_m,
            ego.half_width_m,
            normal,
        )
        actor_long = _box_support(
            actor.yaw_rad,
            actor.half_length_m,
            actor.half_width_m,
            tangent,
        )
        actor_lat = _box_support(
            actor.yaw_rad,
            actor.half_length_m,
            actor.half_width_m,
            normal,
        )
        surface_gap = along - ego_long - actor_long
        lateral_clearance = (
            abs(lateral) - ego_lat - actor_lat - active_policy.lateral_clearance_m
        )
        closing_speed = _dot(_subtract(ego.velocity_xy, actor.velocity_xy), tangent)
        ttc = None
        if closing_speed > active_policy.minimum_closing_speed_mps and surface_gap > 0.0:
            ttc = surface_gap / closing_speed

        reasons = []
        actor_front = along + actor_long
        ahead_or_overlapping = actor_front >= -ego_long
        if lateral_clearance <= 0.0 and ahead_or_overlapping:
            if surface_gap <= 0.0:
                reasons.append("actor_overlap")
            elif surface_gap <= active_policy.hard_gap_m:
                reasons.append("actor_within_hard_gap")
            if 0.0 < surface_gap <= stop_distance:
                reasons.append("actor_within_stopping_envelope")
            if ttc is not None and ttc <= active_policy.ttc_threshold_s:
                reasons.append("actor_ttc_below_threshold")

        predicted_conflict_time = None
        if timed_path is not None:
            predicted_conflict_time = _predicted_conflict_time(
                ego,
                actor,
                timed_path,
                active_policy,
            )
            if predicted_conflict_time is not None:
                reasons.append("predicted_actor_conflict")

        reasons = list(_sort_obstacle_reasons(reasons))
        if not reasons:
            continue
        all_reasons.extend(reasons)
        threats.append(
            ObstacleThreat(
                actor_id=actor.actor_id,
                type_id=actor.type_id,
                reason_codes=tuple(reasons),
                surface_gap_m=surface_gap,
                lateral_clearance_m=lateral_clearance,
                closing_speed_mps=closing_speed,
                stopping_distance_m=stop_distance,
                ttc_s=ttc,
                predicted_conflict_time_s=predicted_conflict_time,
            )
        )

    if not threats:
        return ObstacleAssessment.safe(
            evaluated_actor_count=len(actor_tuple),
            prediction_performed=prediction_performed,
        )

    def threat_key(threat: ObstacleThreat) -> tuple[float, float, int]:
        reason_priority = min(_OBSTACLE_REASON_PRIORITY[reason] for reason in threat.reason_codes)
        time_to_hazard = min(
            value
            for value in (
                threat.predicted_conflict_time_s,
                threat.ttc_s,
                max(0.0, threat.surface_gap_m or 0.0) / max(ego_speed, 0.1),
            )
            if value is not None
        )
        return float(reason_priority), float(time_to_hazard), threat.actor_id

    primary = min(threats, key=threat_key)
    return ObstacleAssessment.unsafe(
        primary,
        reason_codes=_sort_obstacle_reasons(all_reasons),
        evaluated_actor_count=len(actor_tuple),
        prediction_performed=prediction_performed,
    )


_SAFETY_REASON_PRIORITY = {
    "road_containment_unknown": 0,
    "obstacle_assessment_unknown": 1,
    "nominal_control_unavailable": 2,
    "controller_request_unavailable": 3,
    "invalid_nominal_control": 4,
    "actor_overlap": 10,
    "actor_within_hard_gap": 11,
    "predicted_actor_conflict": 12,
    "actor_within_stopping_envelope": 13,
    "actor_ttc_below_threshold": 14,
    "road_containment_failed": 20,
    "emergency_brake_latched": 90,
}


def _sort_safety_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(reasons),
            key=lambda reason: (_SAFETY_REASON_PRIORITY.get(reason, 50), reason),
        )
    )


@dataclass(frozen=True)
class SafetyDecision:
    """Auditable separation of controller request, nominal, and applied control."""

    controller_requested_control: ControlCommand | None
    nominal_control: ControlCommand | None
    applied_control: ControlCommand
    applied_control_source: str
    safety_override_applied: bool
    override_type: str | None
    primary_reason: str | None
    reason_codes: tuple[str, ...]
    latched: bool
    road_containment: RoadContainmentAssessment
    obstacle_assessment: ObstacleAssessment

    def __post_init__(self) -> None:
        source = _nonempty_string(self.applied_control_source, "applied_control_source")
        if source not in {"CONTROLLER_EXECUTION", "SAFETY_OVERRIDE", "FALLBACK"}:
            raise ValueError("invalid applied_control_source")
        if not isinstance(self.safety_override_applied, bool) or not isinstance(self.latched, bool):
            raise TypeError("override and latch flags must be bool")
        reasons = _reason_tuple(self.reason_codes)
        override_type = _optional_string(self.override_type, "override_type")
        primary_reason = _optional_string(self.primary_reason, "primary_reason")
        if self.safety_override_applied:
            if source != "SAFETY_OVERRIDE":
                raise ValueError("an override must identify SAFETY_OVERRIDE as applied source")
            if override_type != "EMERGENCY_BRAKE" or primary_reason is None or not reasons:
                raise ValueError("an override requires emergency-brake type and reasons")
            if primary_reason not in reasons:
                raise ValueError("primary_reason must be present in reason_codes")
            if self.applied_control.throttle != 0.0 or self.applied_control.brake != 1.0:
                raise ValueError("the stop-only override must apply full brake and zero throttle")
        else:
            if source == "SAFETY_OVERRIDE":
                raise ValueError("a non-override decision cannot identify SAFETY_OVERRIDE")
            if override_type is not None or primary_reason is not None or reasons:
                raise ValueError("a non-override decision cannot contain override metadata")
            if self.nominal_control is None or self.applied_control != self.nominal_control:
                raise ValueError("a non-override decision must apply nominal control unchanged")
            if not self.nominal_control.is_longitudinally_exclusive:
                raise ValueError("a non-override decision cannot apply throttle and brake together")
            if source == "CONTROLLER_EXECUTION" and self.latched:
                raise ValueError("controller execution cannot occur while the shield is latched")
            if source == "FALLBACK" and (
                self.applied_control.throttle != 0.0
                or self.applied_control.brake != 1.0
            ):
                raise ValueError("fallback decisions must command zero throttle and full brake")
        object.__setattr__(self, "applied_control_source", source)
        object.__setattr__(self, "override_type", override_type)
        object.__setattr__(self, "primary_reason", primary_reason)
        object.__setattr__(self, "reason_codes", reasons)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "controller_requested_control": (
                self.controller_requested_control.to_json_dict()
                if self.controller_requested_control is not None
                else None
            ),
            "nominal_control": (
                self.nominal_control.to_json_dict() if self.nominal_control is not None else None
            ),
            "applied_control": self.applied_control.to_json_dict(),
            "applied_control_source": self.applied_control_source,
            "safety_override_applied": self.safety_override_applied,
            "safety_override_type": self.override_type,
            "safety_override_reason": self.primary_reason,
            "safety_override_reasons": list(self.reason_codes),
            "safety_override_latched": self.latched,
            "road_containment_status": self.road_containment.status.value,
            "road_containment_reasons": list(self.road_containment.reason_codes),
            "obstacle_status": self.obstacle_assessment.status.value,
            "obstacle_reasons": list(self.obstacle_assessment.reason_codes),
            "road_containment": self.road_containment.to_json_dict(),
            "obstacle_assessment": self.obstacle_assessment.to_json_dict(),
        }


class StopOnlySafetyShield:
    """Stateful emergency-brake arbiter with hold and clear hysteresis."""

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()
        self.reset()

    @property
    def latched(self) -> bool:
        return self._latched

    def reset(self) -> None:
        self._latched = False
        self._latch_age_ticks = 0
        self._clear_ticks = 0
        self._latched_reasons: tuple[str, ...] = ()
        self._last_safe_steering = 0.0

    def decide_fallback(
        self,
        *,
        controller_requested_control: ControlCommand | None,
        nominal_control: ControlCommand,
        reason: str = "no_executable_plan_context",
    ) -> SafetyDecision:
        """Apply an explicit stop fallback without mislabeling it as an override.

        No safety-clear evidence exists on these ticks, so an existing emergency
        latch is retained.  It can only clear after subsequent fully assessed
        safe plan ticks.
        """

        if nominal_control.throttle != 0.0 or nominal_control.brake != 1.0:
            raise ValueError("fallback nominal control must be full brake")
        if self._latched:
            self._latch_age_ticks += 1
        return SafetyDecision(
            controller_requested_control=controller_requested_control,
            nominal_control=nominal_control,
            applied_control=nominal_control,
            applied_control_source="FALLBACK",
            safety_override_applied=False,
            override_type=None,
            primary_reason=None,
            reason_codes=(),
            latched=self._latched,
            road_containment=RoadContainmentAssessment.unknown(
                (reason,),
                quality="not_evaluated_stop_fallback",
            ),
            obstacle_assessment=ObstacleAssessment.unknown((reason,)),
        )

    def decide(
        self,
        *,
        road: RoadContainmentAssessment | None,
        obstacles: ObstacleAssessment | None,
        controller_requested_control: ControlCommand | None,
        nominal_control: ControlCommand | None,
    ) -> SafetyDecision:
        current_reasons = []
        if road is None or road.status is AssessmentStatus.UNKNOWN:
            current_reasons.append("road_containment_unknown")
            if road is not None:
                current_reasons.extend(road.reason_codes)
        elif road.status is AssessmentStatus.UNSAFE:
            current_reasons.append("road_containment_failed")
            current_reasons.extend(road.reason_codes)

        if obstacles is None or obstacles.status is AssessmentStatus.UNKNOWN:
            current_reasons.append("obstacle_assessment_unknown")
            if obstacles is not None:
                current_reasons.extend(obstacles.reason_codes)
        elif obstacles.status is AssessmentStatus.UNSAFE:
            current_reasons.extend(obstacles.reason_codes)

        if nominal_control is None:
            current_reasons.append("nominal_control_unavailable")
        elif not nominal_control.is_longitudinally_exclusive:
            current_reasons.append("invalid_nominal_control")
        if controller_requested_control is None:
            current_reasons.append("controller_request_unavailable")

        current_reasons_tuple = _sort_safety_reasons(current_reasons)
        has_current_trigger = bool(current_reasons_tuple)
        if has_current_trigger:
            if not self._latched:
                self._latch_age_ticks = 0
            self._latched = True
            self._latch_age_ticks += 1
            self._clear_ticks = 0
            self._latched_reasons = current_reasons_tuple
        elif self._latched:
            self._latch_age_ticks += 1
            if self._latch_age_ticks >= self.policy.emergency_hold_ticks:
                self._clear_ticks += 1
            if self._clear_ticks >= self.policy.clear_ticks_to_release:
                self._latched = False
                self._latch_age_ticks = 0
                self._clear_ticks = 0
                self._latched_reasons = ()

        if self._latched:
            reasons = current_reasons_tuple or _sort_safety_reasons(
                (*self._latched_reasons, "emergency_brake_latched")
            )
            road_is_safe = road is not None and road.status is AssessmentStatus.SAFE
            obstacle_is_known_hazard = (
                obstacles is not None and obstacles.status is AssessmentStatus.UNSAFE
            )
            if road_is_safe and obstacle_is_known_hazard and nominal_control is not None:
                steering = nominal_control.steering
            else:
                steering = self._last_safe_steering
            return SafetyDecision(
                controller_requested_control=controller_requested_control,
                nominal_control=nominal_control,
                applied_control=ControlCommand.full_brake(steering),
                applied_control_source="SAFETY_OVERRIDE",
                safety_override_applied=True,
                override_type="EMERGENCY_BRAKE",
                primary_reason=reasons[0],
                reason_codes=reasons,
                latched=True,
                road_containment=road or RoadContainmentAssessment.unknown(),
                obstacle_assessment=obstacles or ObstacleAssessment.unknown(),
            )

        assert nominal_control is not None
        assert controller_requested_control is not None
        self._last_safe_steering = nominal_control.steering
        return SafetyDecision(
            controller_requested_control=controller_requested_control,
            nominal_control=nominal_control,
            applied_control=nominal_control,
            applied_control_source="CONTROLLER_EXECUTION",
            safety_override_applied=False,
            override_type=None,
            primary_reason=None,
            reason_codes=(),
            latched=False,
            road_containment=road,
            obstacle_assessment=obstacles,
        )


__all__ = [
    "ActorObstacle",
    "AssessmentStatus",
    "ControlCommand",
    "EgoKinematics",
    "ObstacleAssessment",
    "ObstacleThreat",
    "RoadContainmentAssessment",
    "RoadContainmentSample",
    "SafetyDecision",
    "SafetyPolicy",
    "StopOnlySafetyShield",
    "assess_obstacles",
    "assess_road_containment",
    "densify_path",
    "stopping_distance_m",
]

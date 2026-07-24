"""CARLA ground-truth adapter for the pure stop-only safety shield.

This is an integration safety layer, not an onboard perception claim.  Every
pose, velocity, lane query, and actor footprint comes from the exact CARLA
snapshot associated with the current control tick.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import carla
import numpy as np

from . import config as cfg
from .geometry import meaningful_path_tangent_xy
from .safety_shield import (
    ActorObstacle,
    AssessmentStatus,
    EgoKinematics,
    ObstacleAssessment,
    RoadContainmentAssessment,
    RoadContainmentSample,
    SafetyPolicy,
    assess_obstacles,
    assess_road_containment,
    densify_path,
)


@dataclass(frozen=True)
class CarlaSafetyAssessment:
    """Road and obstacle facts copied from one exact simulator tick."""

    road: RoadContainmentAssessment
    obstacles: ObstacleAssessment
    safety_source: str = "carla_ground_truth"
    current_ego_road: RoadContainmentAssessment | None = None
    proposed_path_road: RoadContainmentAssessment | None = None
    road_envelope: RoadExecutionEnvelope | None = None


class PlanAdmissionStatus(str, Enum):
    """Safety-gated handoff decision for one Alpamayo proposal."""

    ACCEPT_FULLY_SAFE = "ACCEPT_FULLY_SAFE"
    ACCEPT_SAFE_PREFIX = "ACCEPT_SAFE_PREFIX"
    ACCEPT_RECOVERY_PREFIX = "ACCEPT_RECOVERY_PREFIX"
    REJECT_RETAIN_ACTIVE = "REJECT_RETAIN_ACTIVE"
    REJECT_FALLBACK_STOP = "REJECT_FALLBACK_STOP"


class StoppingReserveStatus(str, Enum):
    """Guarded distance reserve before the first non-executable road pose."""

    UNBOUNDED = "UNBOUNDED"
    ROBUST = "ROBUST"
    FRAGILE = "FRAGILE"
    RECOVERY = "RECOVERY"
    UNAVAILABLE = "UNAVAILABLE"


class RoadRecoveryMode(str, Enum):
    """Why buffered lane clearance may be regained under a physical-surface cap."""

    NONE = "NONE"
    JUNCTION_CLEARANCE = "JUNCTION_CLEARANCE"
    BOUNDARY_CLEARANCE = "BOUNDARY_CLEARANCE"


@dataclass(frozen=True)
class StoppingReserveProfile:
    """Physical cap and one-control-step guarded stopping-distance facts."""

    raw_physical_stopping_cap_mps: float | None
    target_speed_cap_mps: float | None
    guard_speed_mps: float | None
    required_stopping_distance_m: float | None
    stopping_reserve_m: float | None
    status: StoppingReserveStatus

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw_physical_stopping_cap_mps": self.raw_physical_stopping_cap_mps,
            "target_speed_cap_mps": self.target_speed_cap_mps,
            "guard_speed_mps": self.guard_speed_mps,
            "required_stopping_distance_m": self.required_stopping_distance_m,
            "stopping_reserve_m": self.stopping_reserve_m,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class RoadExecutionEnvelope:
    """Timed road facts used separately for admission, control, and shielding."""

    current_ego_road: RoadContainmentAssessment
    near_term_path_road: RoadContainmentAssessment
    full_path_road: RoadContainmentAssessment
    last_safe_waypoint_index: int | None
    time_to_first_bad_s: float | None
    distance_to_first_bad_m: float | None
    target_speed_cap_mps: float | None
    emergency_required: bool
    current_ego_clearance_road: RoadContainmentAssessment | None = None
    near_term_path_surface: RoadContainmentAssessment | None = None
    full_path_surface: RoadContainmentAssessment | None = None
    junction_context: bool = False
    recovery_required: bool = False
    recovery_mode: RoadRecoveryMode = RoadRecoveryMode.NONE
    near_term_end_clearance_road: RoadContainmentAssessment | None = None
    recovery_path_length_m: float | None = None
    recovery_displacement_m: float | None = None
    stopping_reserve_profile: StoppingReserveProfile | None = None
    stopping_reserve_compute_ms: float | None = None
    execution_cursor_index: int | None = None
    ego_path_progress_m: float | None = None
    ego_path_cross_track_m: float | None = None
    ego_path_heading_error_deg: float | None = None
    first_bad_path_progress_m: float | None = None
    effective_stopping_boundary_progress_m: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        reserve = self.stopping_reserve_profile
        return {
            "current_ego_road": self.current_ego_road.to_json_dict(),
            "current_ego_clearance_road": (
                self.current_ego_clearance_road.to_json_dict()
                if self.current_ego_clearance_road is not None
                else None
            ),
            "near_term_path_road": self.near_term_path_road.to_json_dict(),
            "full_path_road": self.full_path_road.to_json_dict(),
            "near_term_path_surface": (
                self.near_term_path_surface.to_json_dict()
                if self.near_term_path_surface is not None
                else None
            ),
            "full_path_surface": (
                self.full_path_surface.to_json_dict()
                if self.full_path_surface is not None
                else None
            ),
            "last_safe_waypoint_index": self.last_safe_waypoint_index,
            "time_to_first_bad_s": self.time_to_first_bad_s,
            "distance_to_first_bad_m": self.distance_to_first_bad_m,
            "target_speed_cap_mps": self.target_speed_cap_mps,
            "emergency_required": bool(self.emergency_required),
            "junction_context": bool(self.junction_context),
            "recovery_required": bool(self.recovery_required),
            "recovery_mode": self.recovery_mode.value,
            "near_term_end_clearance_road": (
                self.near_term_end_clearance_road.to_json_dict()
                if self.near_term_end_clearance_road is not None
                else None
            ),
            "recovery_path_length_m": self.recovery_path_length_m,
            "recovery_displacement_m": self.recovery_displacement_m,
            "raw_physical_stopping_cap_mps": (
                reserve.raw_physical_stopping_cap_mps if reserve is not None else None
            ),
            "guard_speed_mps": (
                reserve.guard_speed_mps if reserve is not None else None
            ),
            "required_stopping_distance_m": (
                reserve.required_stopping_distance_m if reserve is not None else None
            ),
            "stopping_reserve_m": (
                reserve.stopping_reserve_m if reserve is not None else None
            ),
            "stopping_reserve_status": (
                reserve.status.value
                if reserve is not None
                else StoppingReserveStatus.UNAVAILABLE.value
            ),
            "stopping_reserve_profile": (
                reserve.to_json_dict() if reserve is not None else None
            ),
            "stopping_reserve_compute_ms": self.stopping_reserve_compute_ms,
            "execution_cursor": (
                "ego_geometric_progress"
                if self.execution_cursor_index is not None
                else None
            ),
            "execution_cursor_index": self.execution_cursor_index,
            "ego_path_progress_m": self.ego_path_progress_m,
            "ego_path_cross_track_m": self.ego_path_cross_track_m,
            "ego_path_heading_error_deg": self.ego_path_heading_error_deg,
            "first_bad_path_progress_m": self.first_bad_path_progress_m,
            "effective_stopping_boundary_progress_m": (
                self.effective_stopping_boundary_progress_m
            ),
            "time_to_first_bad_quality": (
                "scheduled_model_time"
                if self.time_to_first_bad_s is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _TimedRoadProfile:
    """Cached exact-map samples for one immutable fixed-world plan."""

    plan_id: str
    points: np.ndarray
    times_s: np.ndarray
    cumulative_distance_m: np.ndarray
    upper_waypoint_indices: np.ndarray
    samples_by_pose: tuple[tuple[RoadContainmentSample, ...], ...]
    quality: str


@dataclass(frozen=True)
class _PathProgressProjection:
    """Ego projection used as the physical road-execution cursor."""

    progress_m: float
    first_path_index: int
    cross_track_m: float
    heading_error_deg: float


def decide_plan_admission(
    candidate: RoadExecutionEnvelope,
    active: RoadExecutionEnvelope | None = None,
) -> PlanAdmissionStatus:
    """Choose a plan handoff without letting an unsafe prefix replace a safe plan."""

    current_clearance = (
        candidate.current_ego_clearance_road or candidate.current_ego_road
    )
    candidate_admissible = (
        candidate.current_ego_road.status is AssessmentStatus.SAFE
        and current_clearance.status is AssessmentStatus.SAFE
        and candidate.near_term_path_road.status is AssessmentStatus.SAFE
        and candidate.last_safe_waypoint_index is not None
        and not candidate.emergency_required
    )
    if candidate_admissible:
        if candidate.full_path_road.status is AssessmentStatus.SAFE:
            return PlanAdmissionStatus.ACCEPT_FULLY_SAFE
        return PlanAdmissionStatus.ACCEPT_SAFE_PREFIX

    near_surface = candidate.near_term_path_surface
    recovery_admissible = (
        candidate.current_ego_road.status is AssessmentStatus.SAFE
        and current_clearance.status is AssessmentStatus.UNSAFE
        and near_surface is not None
        and near_surface.status is AssessmentStatus.SAFE
        and candidate.last_safe_waypoint_index is not None
        and candidate.recovery_required
        and not candidate.emergency_required
    )
    if recovery_admissible:
        return PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX

    active_clearance = None
    if active is not None:
        active_clearance = active.current_ego_clearance_road or active.current_ego_road
    active_executable = (
        active is not None
        and active.current_ego_road.status is AssessmentStatus.SAFE
        and (
            active_clearance.status is AssessmentStatus.SAFE
            or active.recovery_required
        )
        and active.last_safe_waypoint_index is not None
        and not active.emergency_required
    )
    if active_executable:
        return PlanAdmissionStatus.REJECT_RETAIN_ACTIVE
    return PlanAdmissionStatus.REJECT_FALLBACK_STOP


def road_stopping_speed_cap_mps(
    distance_to_bad_m: float,
    policy: SafetyPolicy,
) -> float:
    """Controller cap preserving reaction and braking distance before a bad point."""

    return min(
        raw_physical_stopping_speed_cap_mps(distance_to_bad_m, policy),
        float(cfg.TRAJECTORY_MAX_SPEED_MPS),
    )


def raw_physical_stopping_speed_cap_mps(
    distance_to_bad_m: float,
    policy: SafetyPolicy,
) -> float:
    """Unclipped physical cap used only to decide stopping-envelope exhaustion."""

    distance = max(0.0, float(distance_to_bad_m))
    deceleration = float(policy.assumed_deceleration_mps2)
    reaction_time = float(policy.reaction_time_s)
    available_distance = max(0.0, distance - float(policy.stop_buffer_m))
    speed_cap = max(
        0.0,
        -deceleration * reaction_time
        + math.sqrt(
            (deceleration * reaction_time) ** 2
            + 2.0 * deceleration * available_distance
        ),
    )
    return speed_cap


def _near_term_plan_peak_speed_mps(
    plan: Any,
    current_time_s: float,
    *,
    execution_horizon_s: float,
) -> float:
    """Peak timestamped segment speed after interpolating the elapsed prefix."""

    points = np.asarray(plan.world_points, dtype=np.float64)
    times = np.asarray(plan.waypoint_times_s, dtype=np.float64)
    current_time = float(current_time_s)
    horizon = float(execution_horizon_s)
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or times.shape != (len(points),)
        or len(points) == 0
        or not np.isfinite(points[:, :2]).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
        or not math.isfinite(current_time)
        or not math.isfinite(horizon)
        or horizon <= 0.0
    ):
        raise ValueError("invalid timed trajectory for stopping reserve")

    source_time = float(getattr(plan, "source_simulation_time_s", times[0]))
    capture_pose = getattr(plan, "capture_pose_world", None)
    if capture_pose is not None:
        pose = np.asarray(capture_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid capture pose for stopping reserve")
        if source_time < float(times[0]):
            points = np.vstack([pose[:3, 3], points])
            times = np.concatenate([[source_time], times])

    if current_time >= float(times[-1]):
        return 0.0
    if current_time <= float(times[0]):
        start_index = 0
        clipped_points = points
        clipped_times = times
    else:
        upper = int(np.searchsorted(times, current_time, side="right"))
        lower = upper - 1
        fraction = (current_time - float(times[lower])) / float(
            times[upper] - times[lower]
        )
        current_point = points[lower] + fraction * (points[upper] - points[lower])
        clipped_points = np.vstack([current_point, points[upper:]])
        clipped_times = np.concatenate([[current_time], times[upper:]])
        start_index = 0

    execution_end = current_time + horizon
    peak = 0.0
    for index in range(start_index, len(clipped_points) - 1):
        segment_start = float(clipped_times[index])
        segment_end = float(clipped_times[index + 1])
        if segment_start >= execution_end:
            break
        if segment_end <= current_time:
            continue
        segment_dt = segment_end - segment_start
        if segment_dt <= 1e-9:
            continue
        segment_distance = float(
            np.linalg.norm(
                clipped_points[index + 1, :2] - clipped_points[index, :2]
            )
        )
        peak = max(peak, segment_distance / segment_dt)
        if segment_end >= execution_end:
            break
    return peak


def _unavailable_stopping_reserve(
    *,
    target_speed_cap_mps: float | None,
) -> StoppingReserveProfile:
    return StoppingReserveProfile(
        raw_physical_stopping_cap_mps=None,
        target_speed_cap_mps=target_speed_cap_mps,
        guard_speed_mps=None,
        required_stopping_distance_m=None,
        stopping_reserve_m=None,
        status=StoppingReserveStatus.UNAVAILABLE,
    )


def _remove_lateral_clearance(
    samples: tuple[RoadContainmentSample, ...],
    lateral_clearance_m: float,
) -> tuple[RoadContainmentSample, ...]:
    """Convert buffered lane samples into physical Driving-surface samples.

    Exact center/corner Driving queries are unchanged.  Only the analytic
    non-junction lane margin has the planning clearance removed, so a vehicle
    that merely enters the clearance buffer is not mislabeled physically
    off-road.
    """

    clearance = max(0.0, float(lateral_clearance_m))
    physical_samples = []
    for sample in samples:
        if sample.margin_m is None:
            physical_samples.append(sample)
            continue
        physical_margin = float(sample.margin_m) + clearance
        contained = sample.contained
        reason = sample.reason
        if reason == "ego_footprint_exceeds_lane":
            contained = physical_margin >= 0.0
            reason = None if contained else reason
        physical_samples.append(
            replace(
                sample,
                contained=contained,
                margin_m=physical_margin,
                reason=reason,
            )
        )
    return tuple(physical_samples)


def _yaw_rad(transform: Any, bounding_box: Any | None = None) -> float:
    yaw_deg = float(transform.rotation.yaw)
    if bounding_box is not None:
        yaw_deg += float(getattr(getattr(bounding_box, "rotation", None), "yaw", 0.0))
    return math.radians(yaw_deg)


def _bbox_center_xy(transform: Any, bounding_box: Any) -> tuple[float, float]:
    offset = getattr(bounding_box, "location", None)
    offset_x = float(getattr(offset, "x", 0.0))
    offset_y = float(getattr(offset, "y", 0.0))
    yaw = math.radians(float(transform.rotation.yaw))
    return (
        float(transform.location.x) + math.cos(yaw) * offset_x - math.sin(yaw) * offset_y,
        float(transform.location.y) + math.sin(yaw) * offset_x + math.cos(yaw) * offset_y,
    )


def _footprint_points(
    center_xyz: np.ndarray,
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
) -> tuple[np.ndarray, ...]:
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
    right = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
    points = [center_xyz]
    for longitudinal in (-half_length_m, half_length_m):
        for lateral in (-half_width_m, half_width_m):
            points.append(center_xyz + longitudinal * forward + lateral * right)
    return tuple(points)


def _path_yaw(points: np.ndarray, index: int, fallback_yaw_rad: float) -> float:
    tangent = meaningful_path_tangent_xy(
        points,
        index,
        lookahead_m=float(cfg.TRAJECTORY_HEADING_LOOKAHEAD_M),
        minimum_displacement_m=float(cfg.TRAJECTORY_HEADING_MIN_DISPLACEMENT_M),
    )
    if tangent is None:
        return fallback_yaw_rad
    return math.atan2(float(tangent[1]), float(tangent[0]))


def _wrapped_angle_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _densify_timed_path(
    path_points: Any,
    waypoint_times_s: Any,
    *,
    max_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Densify a path while retaining absolute time, distance, and source indices."""

    points = np.asarray(path_points, dtype=np.float64)
    times = np.asarray(waypoint_times_s, dtype=np.float64)
    spacing = float(max_spacing_m)
    if (
        points.ndim != 2
        or points.shape[1] < 3
        or len(points) == 0
        or times.ndim != 1
        or len(times) != len(points)
        or not np.isfinite(points[:, :3]).all()
        or not np.isfinite(times).all()
        or (len(times) > 1 and not np.all(np.diff(times) > 0.0))
        or not math.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError("invalid timed world path")

    dense_points = [points[0, :3].copy()]
    dense_times = [float(times[0])]
    dense_distances = [0.0]
    upper_indices = [0]
    cumulative = 0.0
    for index in range(len(points) - 1):
        start = points[index, :3]
        end = points[index + 1, :3]
        delta = end - start
        distance = float(np.linalg.norm(delta[:2]))
        subdivisions = max(1, int(math.ceil(distance / spacing)))
        for step in range(1, subdivisions + 1):
            fraction = float(step) / float(subdivisions)
            dense_points.append(start + fraction * delta)
            dense_times.append(
                float(times[index] + fraction * (times[index + 1] - times[index]))
            )
            dense_distances.append(cumulative + fraction * distance)
            upper_indices.append(index + 1)
        cumulative += distance
    return (
        np.asarray(dense_points, dtype=np.float64),
        np.asarray(dense_times, dtype=np.float64),
        np.asarray(dense_distances, dtype=np.float64),
        np.asarray(upper_indices, dtype=np.int64),
    )


def _project_path_progress(
    *,
    points: np.ndarray,
    cumulative_distance_m: np.ndarray,
    ego_xy: tuple[float, float],
    ego_yaw_rad: float,
) -> _PathProgressProjection:
    """Project ego onto a path without granting ambiguous longitudinal progress.

    Model timestamps remain the source for freshness and scheduled-time
    telemetry.  Physical stopping distance instead starts at the ego's measured
    path progress, so longitudinal prediction error cannot add fictitious road
    headroom.  A self-intersection with multiple separated progress solutions
    is fail-closed rather than guessed.
    """

    path = np.asarray(points, dtype=np.float64)
    cumulative = np.asarray(cumulative_distance_m, dtype=np.float64)
    position = np.asarray(ego_xy, dtype=np.float64)
    yaw = float(ego_yaw_rad)
    if (
        path.ndim != 2
        or path.shape[1] < 2
        or len(path) == 0
        or cumulative.shape != (len(path),)
        or position.shape != (2,)
        or not np.isfinite(path[:, :2]).all()
        or not np.isfinite(cumulative).all()
        or not np.isfinite(position).all()
        or not math.isfinite(yaw)
        or np.any(np.diff(cumulative) < -1e-9)
    ):
        raise ValueError("invalid path progress projection input")

    if len(path) == 1:
        cross_track = float(np.linalg.norm(position - path[0, :2]))
        if cross_track > float(cfg.TRAJECTORY_MAX_TRACKING_ERROR_M):
            raise ValueError("path progress projection exceeds tracking corridor")
        return _PathProgressProjection(
            progress_m=0.0,
            first_path_index=0,
            cross_track_m=cross_track,
            heading_error_deg=0.0,
        )

    deltas = np.diff(path[:, :2], axis=0)
    length_sq = np.einsum("ij,ij->i", deltas, deltas)
    moving_indices = np.flatnonzero(length_sq > 1e-12)
    if len(moving_indices) == 0:
        cross_track = float(np.linalg.norm(position - path[0, :2]))
        if cross_track > float(cfg.TRAJECTORY_MAX_TRACKING_ERROR_M):
            raise ValueError("path progress projection exceeds tracking corridor")
        return _PathProgressProjection(
            progress_m=0.0,
            first_path_index=0,
            cross_track_m=cross_track,
            heading_error_deg=0.0,
        )

    starts = path[moving_indices, :2]
    moving_deltas = deltas[moving_indices]
    moving_length_sq = length_sq[moving_indices]
    fractions = np.clip(
        np.einsum("ij,ij->i", position - starts, moving_deltas)
        / moving_length_sq,
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * moving_deltas
    errors = np.linalg.norm(projections - position, axis=1)
    progress = (
        cumulative[moving_indices]
        + fractions
        * (cumulative[moving_indices + 1] - cumulative[moving_indices])
    )
    best_error = float(np.min(errors))
    if best_error > float(cfg.TRAJECTORY_MAX_TRACKING_ERROR_M):
        raise ValueError("path progress projection exceeds tracking corridor")

    nearby = np.flatnonzero(
        (
            errors
            <= best_error
            + float(cfg.TRAJECTORY_PROJECTION_AMBIGUITY_DISTANCE_M)
        )
        & (errors <= float(cfg.TRAJECTORY_MAX_TRACKING_ERROR_M))
    )
    ego_forward = np.array(
        [math.cos(yaw), math.sin(yaw)],
        dtype=np.float64,
    )
    heading_errors: dict[int, float] = {}
    compatible: list[int] = []
    for candidate in nearby:
        candidate_index = int(candidate)
        segment_index = int(moving_indices[candidate_index])
        heading_path = np.insert(
            path[:, :2],
            segment_index + 1,
            projections[candidate_index],
            axis=0,
        )
        tangent = meaningful_path_tangent_xy(
            heading_path,
            segment_index + 1,
            lookahead_m=float(cfg.TRAJECTORY_HEADING_LOOKAHEAD_M),
            minimum_displacement_m=float(
                cfg.TRAJECTORY_HEADING_MIN_DISPLACEMENT_M
            ),
        )
        heading_error = 0.0
        if tangent is not None:
            heading_error = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(ego_forward, tangent),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        heading_errors[candidate_index] = heading_error
        if heading_error <= float(cfg.TRAJECTORY_MAX_HEADING_ERROR_DEG):
            compatible.append(candidate_index)

    if not compatible:
        raise ValueError("path progress projection heading mismatch")

    compatible_progress = progress[compatible]
    if (
        len(compatible_progress) > 1
        and float(
            np.max(compatible_progress) - np.min(compatible_progress)
        )
        > float(cfg.TRAJECTORY_PROJECTION_LOCAL_PROGRESS_SPAN_M)
    ):
        raise ValueError("ambiguous path progress projection")

    best = min(compatible, key=lambda index: (float(errors[index]), index))
    selected_error = float(errors[best])
    if selected_error > float(cfg.TRAJECTORY_MAX_TRACKING_ERROR_M):
        raise ValueError("path progress projection exceeds tracking corridor")
    best_progress = float(progress[best])
    heading_error = float(heading_errors[best])
    first_path_index = int(
        min(
            np.searchsorted(cumulative, best_progress, side="left"),
            len(path) - 1,
        )
    )
    return _PathProgressProjection(
        progress_m=best_progress,
        first_path_index=first_path_index,
        cross_track_m=selected_error,
        heading_error_deg=heading_error,
    )


class CarlaGroundTruthSafetyAdapter:
    """Translate exact CARLA snapshots into pure shield assessments."""

    def __init__(self, world: Any, ego_vehicle: Any, policy: SafetyPolicy | None = None):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.policy = policy or SafetyPolicy()
        self._map = world.get_map()
        self._road_profile_cache: OrderedDict[str, _TimedRoadProfile] = OrderedDict()

    def reset(self, ego_vehicle: Any | None = None) -> None:
        if ego_vehicle is not None:
            self.ego_vehicle = ego_vehicle
        self._road_profile_cache.clear()

    def _query_footprint(
        self,
        *,
        center_xyz: np.ndarray,
        yaw_rad: float,
        half_length_m: float,
        half_width_m: float,
        sample_index_start: int,
    ) -> tuple[RoadContainmentSample, ...]:
        try:
            points = _footprint_points(
                center_xyz,
                yaw_rad,
                half_length_m,
                half_width_m,
            )
            waypoints = []
            for point in points:
                waypoint = self._map.get_waypoint(
                    carla.Location(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                    ),
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                waypoints.append(waypoint)
        except Exception as exc:
            return (
                RoadContainmentSample(
                    sample_index=sample_index_start,
                    position_xyz=tuple(float(value) for value in center_xyz),
                    contained=None,
                    reason=f"carla_map_query_error:{type(exc).__name__}",
                ),
            )

        center_waypoint = waypoints[0]
        if center_waypoint is None:
            return (
                RoadContainmentSample(
                    sample_index=sample_index_start,
                    position_xyz=tuple(float(value) for value in center_xyz),
                    contained=False,
                    reason="center_off_driving_lane",
                ),
            )

        is_junction = bool(getattr(center_waypoint, "is_junction", False))
        if is_junction:
            # Junction lane IDs and nominal widths are not a stable footprint
            # boundary: an exact Driving waypoint for every footprint point is
            # the authoritative CARLA drivable-area check here.
            margin_m = None
            center_contained = True
            center_reason = None
        else:
            lane_yaw_rad = math.radians(
                float(center_waypoint.transform.rotation.yaw)
            )
            lane_right = np.array(
                [-math.sin(lane_yaw_rad), math.cos(lane_yaw_rad)],
                dtype=np.float64,
            )
            lane_center = np.array(
                [
                    float(center_waypoint.transform.location.x),
                    float(center_waypoint.transform.location.y),
                ],
                dtype=np.float64,
            )
            lateral_offset = float(np.dot(center_xyz[:2] - lane_center, lane_right))
            path_forward = np.array(
                [math.cos(yaw_rad), math.sin(yaw_rad)],
                dtype=np.float64,
            )
            path_right = np.array(
                [-math.sin(yaw_rad), math.cos(yaw_rad)],
                dtype=np.float64,
            )
            lateral_support = (
                abs(float(np.dot(path_forward, lane_right))) * half_length_m
                + abs(float(np.dot(path_right, lane_right))) * half_width_m
            )
            margin_m = (
                float(center_waypoint.lane_width) / 2.0
                - abs(lateral_offset)
                - lateral_support
                - float(self.policy.lateral_clearance_m)
            )
            heading_error = abs(
                _wrapped_angle_degrees(
                    math.degrees(yaw_rad) - math.degrees(lane_yaw_rad)
                )
            )
            center_contained = margin_m >= 0.0 and heading_error <= 45.0
            center_reason = None
            if margin_m < 0.0:
                center_reason = "ego_footprint_exceeds_lane"
            elif heading_error > 45.0:
                center_reason = "path_heading_opposes_lane"

        samples = [
            RoadContainmentSample(
                sample_index=sample_index_start,
                position_xyz=tuple(float(value) for value in center_xyz),
                contained=center_contained,
                margin_m=margin_m,
                road_id=int(center_waypoint.road_id),
                lane_id=int(center_waypoint.lane_id),
                is_junction=is_junction,
                reason=center_reason,
            )
        ]
        for corner_offset, (point, waypoint) in enumerate(
            zip(points[1:], waypoints[1:]),
            start=1,
        ):
            corner_is_junction = (
                bool(getattr(waypoint, "is_junction", False))
                if waypoint is not None
                else False
            )
            same_lane = (
                waypoint is not None
                and (
                    is_junction
                    or corner_is_junction
                    or (
                        int(waypoint.road_id) == int(center_waypoint.road_id)
                        and int(waypoint.lane_id) == int(center_waypoint.lane_id)
                    )
                )
            )
            samples.append(
                RoadContainmentSample(
                    sample_index=sample_index_start + corner_offset,
                    position_xyz=tuple(float(value) for value in point),
                    contained=bool(same_lane),
                    road_id=(int(waypoint.road_id) if waypoint is not None else None),
                    lane_id=(int(waypoint.lane_id) if waypoint is not None else None),
                    is_junction=(
                        corner_is_junction if waypoint is not None else None
                    ),
                    reason=None if same_lane else "footprint_corner_off_driving_lane",
                )
            )
        return tuple(samples)

    def _assess_path_road(
        self,
        path_points: Any,
        ego: EgoKinematics,
    ) -> RoadContainmentAssessment:
        try:
            dense = np.asarray(
                densify_path(
                    path_points,
                    max_spacing_m=self.policy.path_sample_spacing_m,
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            return RoadContainmentAssessment.unknown(
                (f"invalid_world_path:{type(exc).__name__}",),
                quality="carla_ground_truth",
            )
        samples = []
        for index, point in enumerate(dense):
            yaw = _path_yaw(dense, index, ego.yaw_rad)
            samples.extend(
                self._query_footprint(
                    center_xyz=point[:3],
                    yaw_rad=yaw,
                    half_length_m=ego.half_length_m,
                    half_width_m=ego.half_width_m,
                    sample_index_start=index * 5,
                )
            )
        quality = "carla_ground_truth_exact_lane_and_footprint"
        if any(sample.is_junction for sample in samples):
            quality = "carla_ground_truth_drivable_only_at_junction"
        return assess_road_containment(samples, quality=quality)

    def _build_timed_road_profile(
        self,
        plan: Any,
        ego: EgoKinematics,
    ) -> _TimedRoadProfile:
        points, times, distances, upper_indices = _densify_timed_path(
            plan.world_points,
            plan.waypoint_times_s,
            max_spacing_m=self.policy.path_sample_spacing_m,
        )
        samples_by_pose = []
        for index, point in enumerate(points):
            yaw = _path_yaw(points, index, ego.yaw_rad)
            samples_by_pose.append(
                self._query_footprint(
                    center_xyz=point[:3],
                    yaw_rad=yaw,
                    half_length_m=ego.half_length_m,
                    half_width_m=ego.half_width_m,
                    sample_index_start=index * 5,
                )
            )
        flattened = tuple(sample for pose in samples_by_pose for sample in pose)
        quality = "carla_ground_truth_exact_lane_and_footprint"
        if any(sample.is_junction for sample in flattened):
            quality = "carla_ground_truth_drivable_only_at_junction"
        return _TimedRoadProfile(
            plan_id=str(plan.plan_id),
            points=points,
            times_s=times,
            cumulative_distance_m=distances,
            upper_waypoint_indices=upper_indices,
            samples_by_pose=tuple(samples_by_pose),
            quality=quality,
        )

    def _profile_for_plan(self, plan: Any, ego: EgoKinematics) -> _TimedRoadProfile:
        plan_id = str(plan.plan_id)
        profile = self._road_profile_cache.get(plan_id)
        if profile is not None:
            self._road_profile_cache.move_to_end(plan_id)
            return profile

        profile = self._build_timed_road_profile(plan, ego)
        self._road_profile_cache[plan_id] = profile
        while len(self._road_profile_cache) > int(cfg.SAFETY_ROAD_PROFILE_CACHE_SIZE):
            self._road_profile_cache.popitem(last=False)
        return profile

    @staticmethod
    def _assess_profile_indices(
        profile: _TimedRoadProfile,
        indices: np.ndarray,
        *,
        empty_reason: str,
        physical_surface: bool = False,
        lateral_clearance_m: float = 0.0,
    ) -> RoadContainmentAssessment:
        if len(indices) == 0:
            return RoadContainmentAssessment.unknown(
                (empty_reason,),
                quality=profile.quality,
            )
        samples = tuple(
            sample
            for index in indices
            for sample in profile.samples_by_pose[int(index)]
        )
        quality = profile.quality
        if physical_surface:
            samples = _remove_lateral_clearance(samples, lateral_clearance_m)
            quality = f"{quality}_physical_surface"
        return assess_road_containment(samples, quality=quality)

    @staticmethod
    def _first_bad_profile_index(
        profile: _TimedRoadProfile,
        remaining_indices: np.ndarray,
        *,
        physical_surface: bool = False,
        lateral_clearance_m: float = 0.0,
    ) -> int | None:
        for index in remaining_indices:
            samples = profile.samples_by_pose[int(index)]
            quality = profile.quality
            if physical_surface:
                samples = _remove_lateral_clearance(samples, lateral_clearance_m)
                quality = f"{quality}_physical_surface"
            pose = assess_road_containment(
                samples,
                quality=quality,
            )
            if pose.status is not AssessmentStatus.SAFE:
                return int(index)
        return None

    @staticmethod
    def _indices_have_junction(
        profile: _TimedRoadProfile,
        indices: np.ndarray,
    ) -> bool:
        return any(
            sample.is_junction is True
            for index in indices
            for sample in profile.samples_by_pose[int(index)]
        )

    def _unknown_envelope(
        self,
        reason: str,
        *,
        current_road: RoadContainmentAssessment | None = None,
    ) -> RoadExecutionEnvelope:
        unknown = RoadContainmentAssessment.unknown(
            (reason,),
            quality="carla_ground_truth",
        )
        return RoadExecutionEnvelope(
            current_ego_road=current_road or unknown,
            near_term_path_road=unknown,
            full_path_road=unknown,
            last_safe_waypoint_index=None,
            time_to_first_bad_s=None,
            distance_to_first_bad_m=None,
            target_speed_cap_mps=0.0,
            emergency_required=True,
            current_ego_clearance_road=current_road or unknown,
            near_term_path_surface=unknown,
            full_path_surface=unknown,
            stopping_reserve_profile=_unavailable_stopping_reserve(
                target_speed_cap_mps=0.0
            ),
        )

    def _road_execution_envelope(
        self,
        *,
        profile: _TimedRoadProfile,
        plan: Any,
        ego: EgoKinematics,
        current_time_s: float,
        current_road: RoadContainmentAssessment,
        current_clearance_road: RoadContainmentAssessment,
        current_junction_context: bool,
    ) -> RoadExecutionEnvelope:
        scheduled_remaining_indices = np.flatnonzero(
            profile.times_s > float(current_time_s) + float(cfg.TRAJECTORY_TIME_EPSILON_S)
        )
        if len(scheduled_remaining_indices) == 0:
            exhausted = RoadContainmentAssessment.unknown(
                ("safety_path_exhausted",),
                quality=profile.quality,
            )
            return RoadExecutionEnvelope(
                current_ego_road=current_road,
                near_term_path_road=exhausted,
                full_path_road=exhausted,
                last_safe_waypoint_index=None,
                time_to_first_bad_s=None,
                distance_to_first_bad_m=None,
                target_speed_cap_mps=0.0,
                emergency_required=True,
                current_ego_clearance_road=current_clearance_road,
                near_term_path_surface=exhausted,
                full_path_surface=exhausted,
                junction_context=current_junction_context,
                stopping_reserve_profile=_unavailable_stopping_reserve(
                    target_speed_cap_mps=0.0
                ),
            )

        terminal_profile_index: int | None = None
        terminal_stop_index = getattr(plan, "terminal_stop_index", None)
        projection_points = profile.points
        projection_cumulative_distance_m = profile.cumulative_distance_m
        if terminal_stop_index is not None:
            terminal_stop_index = int(terminal_stop_index)
            if (
                terminal_stop_index < 0
                or terminal_stop_index >= len(plan.world_points)
            ):
                raise ValueError("invalid terminal stop index")
            terminal_profile_indices = np.flatnonzero(
                profile.upper_waypoint_indices <= terminal_stop_index
            )
            if len(terminal_profile_indices) == 0:
                raise ValueError("terminal stop is absent from road profile")
            terminal_profile_index = int(terminal_profile_indices[-1])
            projection_points = profile.points[
                : terminal_profile_index + 1
            ]
            projection_cumulative_distance_m = (
                profile.cumulative_distance_m[
                    : terminal_profile_index + 1
                ]
            )

        projection = _project_path_progress(
            points=projection_points,
            cumulative_distance_m=projection_cumulative_distance_m,
            ego_xy=ego.center_xy,
            ego_yaw_rad=ego.yaw_rad,
        )
        remaining_indices = np.arange(
            projection.first_path_index,
            len(profile.points),
            dtype=np.int64,
        )
        # Rebase the fixed execution horizon at the ego's geometric cursor.
        # Wall time remains useful for freshness and scheduled-time telemetry,
        # but using it here would make an ego behind schedule assess several
        # seconds of physically future path as "near term".
        near_term_origin_s = float(
            profile.times_s[projection.first_path_index]
        )
        near_term_indices = remaining_indices[
            profile.times_s[remaining_indices]
            <= near_term_origin_s + float(cfg.SAFETY_EXECUTION_HORIZON_S)
        ]
        source_waypoint_times_s = np.asarray(
            plan.waypoint_times_s,
            dtype=np.float64,
        )
        recovery_horizon_end_s = (
            near_term_origin_s + float(cfg.SAFETY_EXECUTION_HORIZON_S)
        )
        recovery_endpoint_waypoint_index = int(
            np.searchsorted(
                source_waypoint_times_s,
                recovery_horizon_end_s
                + float(cfg.TRAJECTORY_TIME_EPSILON_S),
                side="right",
            )
            - 1
        )
        recovery_endpoint_profile_index: int | None = None
        if recovery_endpoint_waypoint_index >= 0:
            endpoint_time_s = float(
                source_waypoint_times_s[recovery_endpoint_waypoint_index]
            )
            endpoint_profile_candidates = np.flatnonzero(
                (
                    profile.upper_waypoint_indices
                    == recovery_endpoint_waypoint_index
                )
                & (
                    np.abs(profile.times_s - endpoint_time_s)
                    <= float(cfg.TRAJECTORY_TIME_EPSILON_S)
                )
            )
            if len(endpoint_profile_candidates) > 0:
                recovery_endpoint_profile_index = int(
                    endpoint_profile_candidates[-1]
                )
        full_path = self._assess_profile_indices(
            profile,
            remaining_indices,
            empty_reason="safety_path_exhausted",
        )
        near_term = self._assess_profile_indices(
            profile,
            near_term_indices,
            empty_reason="near_term_path_exhausted",
        )
        full_surface = self._assess_profile_indices(
            profile,
            remaining_indices,
            empty_reason="safety_surface_path_exhausted",
            physical_surface=True,
            lateral_clearance_m=self.policy.lateral_clearance_m,
        )
        near_surface = self._assess_profile_indices(
            profile,
            near_term_indices,
            empty_reason="near_term_surface_path_exhausted",
            physical_surface=True,
            lateral_clearance_m=self.policy.lateral_clearance_m,
        )
        near_term_end = self._assess_profile_indices(
            profile,
            (
                np.asarray(
                    [recovery_endpoint_profile_index],
                    dtype=np.int64,
                )
                if recovery_endpoint_profile_index is not None
                else np.empty(0, dtype=np.int64)
            ),
            empty_reason="near_term_end_path_exhausted",
        )
        junction_context = bool(
            current_junction_context
            or self._indices_have_junction(profile, near_term_indices)
        )
        recovery_path_length_m = (
            max(
                0.0,
                float(
                    profile.cumulative_distance_m[
                        recovery_endpoint_profile_index
                    ]
                )
                - projection.progress_m,
            )
            if recovery_endpoint_profile_index is not None
            else 0.0
        )
        recovery_displacement_m = (
            float(
                np.linalg.norm(
                    profile.points[recovery_endpoint_profile_index, :2]
                    - np.asarray(ego.center_xy, dtype=np.float64)
                )
            )
            if recovery_endpoint_profile_index is not None
            else 0.0
        )
        junction_recovery_required = bool(
            current_road.status is AssessmentStatus.SAFE
            and junction_context
            and near_surface.status is AssessmentStatus.SAFE
            and (
                current_clearance_road.status is AssessmentStatus.UNSAFE
                or near_term.status is AssessmentStatus.UNSAFE
            )
        )
        boundary_recovery_required = bool(
            current_road.status is AssessmentStatus.SAFE
            and not junction_context
            and current_clearance_road.status is AssessmentStatus.UNSAFE
            and near_surface.status is AssessmentStatus.SAFE
            and near_term_end.status is AssessmentStatus.SAFE
            and terminal_stop_index is None
            and recovery_displacement_m
            >= float(cfg.SAFETY_BOUNDARY_RECOVERY_MIN_DISPLACEMENT_M)
        )
        recovery_required = bool(
            junction_recovery_required or boundary_recovery_required
        )
        recovery_mode = (
            RoadRecoveryMode.JUNCTION_CLEARANCE
            if junction_recovery_required
            else (
                RoadRecoveryMode.BOUNDARY_CLEARANCE
                if boundary_recovery_required
                else RoadRecoveryMode.NONE
            )
        )
        first_bad = self._first_bad_profile_index(
            profile,
            remaining_indices,
            # Buffered clearance is already unavailable during recovery, so
            # only a genuine physical-surface violation is an emergency
            # boundary.  Non-junction recovery authority is bounded
            # independently to the near-term endpoint below.
            physical_surface=recovery_required,
            lateral_clearance_m=self.policy.lateral_clearance_m,
        )
        boundary_recovery_authorized_waypoint_index = (
            recovery_endpoint_waypoint_index
            if boundary_recovery_required
            else None
        )
        reserve_compute_started_s = time.perf_counter()
        current_speed = float(math.hypot(*ego.velocity_xy))
        near_term_peak_speed = _near_term_plan_peak_speed_mps(
            plan,
            current_time_s,
            execution_horizon_s=float(cfg.SAFETY_EXECUTION_HORIZON_S),
        )
        guard_speed = (
            max(
                current_speed,
                min(
                    near_term_peak_speed,
                    float(cfg.TRAJECTORY_MAX_SPEED_MPS),
                ),
            )
            + float(cfg.SAFETY_GUARDED_ACCELERATION_MPS2)
            * float(cfg.CONTROL_DT)
        )

        if first_bad is None:
            target_speed_cap = (
                float(cfg.SAFETY_JUNCTION_RECOVERY_SPEED_CAP_MPS)
                if recovery_required
                else None
            )
            reserve_status = (
                StoppingReserveStatus.RECOVERY
                if recovery_required
                else StoppingReserveStatus.UNBOUNDED
            )
            if current_road.status is not AssessmentStatus.SAFE:
                reserve_status = StoppingReserveStatus.UNAVAILABLE
            return RoadExecutionEnvelope(
                current_ego_road=current_road,
                near_term_path_road=near_term,
                full_path_road=full_path,
                last_safe_waypoint_index=(
                    boundary_recovery_authorized_waypoint_index
                    if boundary_recovery_authorized_waypoint_index is not None
                    else int(len(plan.world_points) - 1)
                ),
                time_to_first_bad_s=None,
                distance_to_first_bad_m=None,
                target_speed_cap_mps=target_speed_cap,
                emergency_required=False,
                current_ego_clearance_road=current_clearance_road,
                near_term_path_surface=near_surface,
                full_path_surface=full_surface,
                junction_context=junction_context,
                recovery_required=recovery_required,
                recovery_mode=recovery_mode,
                near_term_end_clearance_road=near_term_end,
                recovery_path_length_m=recovery_path_length_m,
                recovery_displacement_m=recovery_displacement_m,
                stopping_reserve_profile=StoppingReserveProfile(
                    raw_physical_stopping_cap_mps=None,
                    target_speed_cap_mps=target_speed_cap,
                    guard_speed_mps=guard_speed,
                    required_stopping_distance_m=None,
                    stopping_reserve_m=None,
                    status=reserve_status,
                ),
                stopping_reserve_compute_ms=(
                    (time.perf_counter() - reserve_compute_started_s) * 1000.0
                ),
                execution_cursor_index=projection.first_path_index,
                ego_path_progress_m=projection.progress_m,
                ego_path_cross_track_m=projection.cross_track_m,
                ego_path_heading_error_deg=projection.heading_error_deg,
            )

        first_bad_path_progress = float(
            profile.cumulative_distance_m[first_bad]
        )
        effective_stopping_boundary_progress = first_bad_path_progress
        if terminal_profile_index is not None:
            # A validated terminal stop tail is a stationary cluster.  Its
            # numerical back-and-forth jitter is still road-assessed point by
            # point, but cannot manufacture longitudinal stopping headroom.
            effective_stopping_boundary_progress = min(
                effective_stopping_boundary_progress,
                float(
                    profile.cumulative_distance_m[terminal_profile_index]
                ),
            )
        distance_to_bad = max(
            0.0,
            effective_stopping_boundary_progress - projection.progress_m,
        )
        time_to_bad = max(
            0.0,
            float(profile.times_s[first_bad]) - float(current_time_s),
        )
        first_bad_upper_index = int(profile.upper_waypoint_indices[first_bad])
        last_safe_index = first_bad_upper_index - 1
        first_execution_waypoint = int(
            profile.upper_waypoint_indices[projection.first_path_index]
        )
        if last_safe_index < first_execution_waypoint:
            last_safe_index = None
        if (
            last_safe_index is not None
            and boundary_recovery_authorized_waypoint_index is not None
        ):
            last_safe_index = min(
                last_safe_index,
                boundary_recovery_authorized_waypoint_index,
            )

        raw_physical_cap = raw_physical_stopping_speed_cap_mps(
            distance_to_bad,
            self.policy,
        )
        acceleration_guard_mps = (
            float(cfg.SAFETY_GUARDED_ACCELERATION_MPS2)
            * float(cfg.CONTROL_DT)
        )
        target_speed_cap = min(
            max(0.0, raw_physical_cap - acceleration_guard_mps),
            float(cfg.TRAJECTORY_MAX_SPEED_MPS),
        )
        if recovery_required:
            target_speed_cap = min(
                target_speed_cap,
                float(cfg.SAFETY_JUNCTION_RECOVERY_SPEED_CAP_MPS),
            )
        emergency_required = (
            current_speed
            > raw_physical_cap + float(cfg.SAFETY_SPEED_CAP_EPSILON_MPS)
        )
        required_stopping_distance = (
            float(self.policy.stop_buffer_m)
            + guard_speed * float(self.policy.reaction_time_s)
            + guard_speed * guard_speed
            / (2.0 * float(self.policy.assumed_deceleration_mps2))
        )
        stopping_reserve = distance_to_bad - required_stopping_distance
        reserve_status = (
            StoppingReserveStatus.RECOVERY
            if recovery_required
            else (
                StoppingReserveStatus.ROBUST
                if stopping_reserve >= 0.0
                else StoppingReserveStatus.FRAGILE
            )
        )
        if current_road.status is not AssessmentStatus.SAFE:
            reserve_status = StoppingReserveStatus.UNAVAILABLE
        # The raw physical cap remains the emergency boundary.  The controller
        # receives a one-control-step lower setpoint so asymptotic PID tracking
        # cannot consume the acceleration guard used by the reserve profile.
        return RoadExecutionEnvelope(
            current_ego_road=current_road,
            near_term_path_road=near_term,
            full_path_road=full_path,
            last_safe_waypoint_index=last_safe_index,
            time_to_first_bad_s=time_to_bad,
            distance_to_first_bad_m=distance_to_bad,
            target_speed_cap_mps=target_speed_cap,
            emergency_required=emergency_required,
            current_ego_clearance_road=current_clearance_road,
            near_term_path_surface=near_surface,
            full_path_surface=full_surface,
            junction_context=junction_context,
            recovery_required=recovery_required,
            recovery_mode=recovery_mode,
            near_term_end_clearance_road=near_term_end,
            recovery_path_length_m=recovery_path_length_m,
            recovery_displacement_m=recovery_displacement_m,
            stopping_reserve_profile=StoppingReserveProfile(
                raw_physical_stopping_cap_mps=raw_physical_cap,
                target_speed_cap_mps=target_speed_cap,
                guard_speed_mps=guard_speed,
                required_stopping_distance_m=required_stopping_distance,
                stopping_reserve_m=stopping_reserve,
                status=reserve_status,
            ),
            stopping_reserve_compute_ms=(
                (time.perf_counter() - reserve_compute_started_s) * 1000.0
            ),
            execution_cursor_index=projection.first_path_index,
            ego_path_progress_m=projection.progress_m,
            ego_path_cross_track_m=projection.cross_track_m,
            ego_path_heading_error_deg=projection.heading_error_deg,
            first_bad_path_progress_m=first_bad_path_progress,
            effective_stopping_boundary_progress_m=(
                effective_stopping_boundary_progress
            ),
        )

    def assess_ego_transform(self, transform: Any) -> RoadContainmentAssessment:
        """Assess the ego footprint at one candidate transform."""

        bounding_box = self.ego_vehicle.bounding_box
        center_xy = _bbox_center_xy(transform, bounding_box)
        return assess_road_containment(
            self._query_footprint(
                center_xyz=np.array(
                    [
                        *center_xy,
                        float(transform.location.z),
                    ],
                    dtype=np.float64,
                ),
                yaw_rad=_yaw_rad(transform, bounding_box),
                half_length_m=float(bounding_box.extent.x),
                half_width_m=float(bounding_box.extent.y),
                sample_index_start=0,
            ),
            quality="carla_ground_truth_spawn_preflight",
        )

    @staticmethod
    def _combine_road(
        plan_road: RoadContainmentAssessment,
        current_road: RoadContainmentAssessment,
    ) -> RoadContainmentAssessment:
        assessments = (plan_road, current_road)
        reasons = tuple(
            dict.fromkeys(
                reason
                for assessment in assessments
                for reason in assessment.reason_codes
            )
        )
        sample_count = sum(assessment.sample_count for assessment in assessments)
        margins = [
            assessment.min_margin_m
            for assessment in assessments
            if assessment.min_margin_m is not None
        ]
        min_margin = min(margins) if margins else None
        if any(assessment.status is AssessmentStatus.UNKNOWN for assessment in assessments):
            return RoadContainmentAssessment.unknown(
                reasons or ("road_containment_unavailable",),
                sample_count=sample_count,
                quality="carla_ground_truth",
            )
        if any(assessment.status is AssessmentStatus.UNSAFE for assessment in assessments):
            return RoadContainmentAssessment.unsafe(
                reasons or ("road_not_contained",),
                sample_count=sample_count,
                min_margin_m=min_margin,
                quality="carla_ground_truth",
            )
        return RoadContainmentAssessment.safe(
            sample_count=sample_count,
            min_margin_m=min_margin,
            quality="carla_ground_truth",
        )

    def _ego_from_context(self, tick_context: Any) -> EgoKinematics:
        transform = tick_context.ego_transform
        velocity = tick_context.ego_velocity
        bounding_box = self.ego_vehicle.bounding_box
        center_xy = _bbox_center_xy(transform, bounding_box)
        return EgoKinematics(
            frame_id=int(tick_context.frame_id),
            actor_id=int(self.ego_vehicle.id),
            center_xy=center_xy,
            yaw_rad=_yaw_rad(transform, bounding_box),
            velocity_xy=(float(velocity.x), float(velocity.y)),
            half_length_m=float(bounding_box.extent.x),
            half_width_m=float(bounding_box.extent.y),
        )

    def _actors_from_context(self, tick_context: Any) -> tuple[ActorObstacle, ...] | None:
        try:
            actor_list = self.world.get_actors()
            actors = list(actor_list.filter("vehicle.*"))
            actors.extend(actor_list.filter("walker.pedestrian.*"))
            obstacles = []
            for actor in actors:
                if int(actor.id) == int(self.ego_vehicle.id):
                    continue
                actor_snapshot = tick_context.snapshot.find(actor.id)
                if actor_snapshot is None:
                    return None
                transform = actor_snapshot.get_transform()
                velocity = actor_snapshot.get_velocity()
                bounding_box = actor.bounding_box
                obstacles.append(
                    ActorObstacle(
                        frame_id=int(tick_context.frame_id),
                        actor_id=int(actor.id),
                        type_id=str(actor.type_id),
                        center_xy=_bbox_center_xy(transform, bounding_box),
                        yaw_rad=_yaw_rad(transform, bounding_box),
                        velocity_xy=(float(velocity.x), float(velocity.y)),
                        half_length_m=float(bounding_box.extent.x),
                        half_width_m=float(bounding_box.extent.y),
                    )
                )
            return tuple(obstacles)
        except Exception:
            return None

    def assess_plan_road(
        self,
        *,
        tick_context: Any,
        plan: Any,
    ) -> RoadExecutionEnvelope:
        """Return the timed road envelope without querying dynamic actors."""

        try:
            if int(tick_context.snapshot.frame) != int(tick_context.frame_id):
                raise ValueError("snapshot frame mismatch")
            ego = self._ego_from_context(tick_context)
        except Exception as exc:
            return self._unknown_envelope(
                f"invalid_ego_snapshot:{type(exc).__name__}"
            )

        ego_center_z = float(tick_context.ego_transform.location.z)
        current_samples = self._query_footprint(
            center_xyz=np.array([*ego.center_xy, ego_center_z], dtype=np.float64),
            yaw_rad=ego.yaw_rad,
            half_length_m=ego.half_length_m,
            half_width_m=ego.half_width_m,
            sample_index_start=0,
        )
        current_clearance_road = assess_road_containment(
            current_samples,
            quality="carla_ground_truth_current_ego_clearance",
        )
        current_road = assess_road_containment(
            _remove_lateral_clearance(
                current_samples,
                self.policy.lateral_clearance_m,
            ),
            quality="carla_ground_truth_current_ego_surface",
        )
        current_junction_context = any(
            sample.is_junction is True for sample in current_samples
        )
        try:
            profile = self._profile_for_plan(plan, ego)
            return self._road_execution_envelope(
                profile=profile,
                plan=plan,
                ego=ego,
                current_time_s=float(tick_context.simulation_time_s),
                current_road=current_road,
                current_clearance_road=current_clearance_road,
                current_junction_context=current_junction_context,
            )
        except Exception as exc:
            return self._unknown_envelope(
                f"invalid_world_path:{type(exc).__name__}",
                current_road=current_road,
            )

    @staticmethod
    def _shield_road_from_envelope(
        envelope: RoadExecutionEnvelope,
    ) -> RoadContainmentAssessment:
        """Expose only immediate/stopping-envelope failures to the stop-only shield."""

        current = envelope.current_ego_road
        if current.status is not AssessmentStatus.SAFE:
            return current
        if not envelope.emergency_required:
            return current

        future = envelope.full_path_road
        reasons = tuple(
            dict.fromkeys(
                (
                    "road_stopping_envelope_exhausted",
                    *future.reason_codes,
                )
            )
        )
        if future.status is AssessmentStatus.UNKNOWN:
            return RoadContainmentAssessment.unknown(
                reasons,
                sample_count=future.sample_count,
                first_bad_sample_index=future.first_bad_sample_index,
                quality=future.quality,
            )
        return RoadContainmentAssessment.unsafe(
            reasons,
            sample_count=future.sample_count,
            min_margin_m=future.min_margin_m,
            first_bad_sample_index=future.first_bad_sample_index,
            quality=future.quality,
        )

    def assess(self, *, tick_context: Any, plan: Any) -> CarlaSafetyAssessment:
        """Assess the active fixed-world plan against one exact CARLA tick."""

        try:
            if int(tick_context.snapshot.frame) != int(tick_context.frame_id):
                raise ValueError("snapshot frame mismatch")
            ego = self._ego_from_context(tick_context)
        except Exception as exc:
            return CarlaSafetyAssessment(
                road=RoadContainmentAssessment.unknown(
                    (f"invalid_ego_snapshot:{type(exc).__name__}",),
                    quality="carla_ground_truth",
                ),
                obstacles=ObstacleAssessment.unknown(("invalid_ego_snapshot",)),
            )

        envelope = self.assess_plan_road(
            tick_context=tick_context,
            plan=plan,
        )
        road = self._shield_road_from_envelope(envelope)

        current_time = float(tick_context.simulation_time_s)
        first_future = int(
            np.searchsorted(plan.waypoint_times_s, current_time, side="right")
        )
        remaining_points = np.asarray(plan.world_points[first_future:], dtype=np.float64)
        remaining_times = np.asarray(
            plan.waypoint_times_s[first_future:] - current_time,
            dtype=np.float64,
        )
        if len(remaining_points) == 0:
            obstacles = ObstacleAssessment.unknown(("safety_path_exhausted",))
        else:
            obstacles = assess_obstacles(
                ego=ego,
                path_points=remaining_points,
                path_times_s=remaining_times,
                actors=self._actors_from_context(tick_context),
                policy=self.policy,
            )
        return CarlaSafetyAssessment(
            road=road,
            obstacles=obstacles,
            current_ego_road=envelope.current_ego_road,
            proposed_path_road=envelope.full_path_road,
            road_envelope=envelope,
        )


__all__ = [
    "CarlaGroundTruthSafetyAdapter",
    "CarlaSafetyAssessment",
    "PlanAdmissionStatus",
    "RoadRecoveryMode",
    "RoadExecutionEnvelope",
    "StoppingReserveProfile",
    "StoppingReserveStatus",
    "decide_plan_admission",
    "raw_physical_stopping_speed_cap_mps",
    "road_stopping_speed_cap_mps",
]

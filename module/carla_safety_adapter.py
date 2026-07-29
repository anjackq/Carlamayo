"""CARLA ground-truth adapter for the pure stop-only safety shield.

This is an integration safety layer, not an onboard perception claim.  Every
pose, velocity, lane query, and actor footprint comes from the exact CARLA
snapshot associated with the current control tick.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

import carla
import numpy as np

from . import config as cfg
from .geometry import meaningful_path_tangent_xy
from .road_assessment_backend import (
    ExactRoadProcessBackend,
    FootprintQuery,
    FootprintQueryResult,
    RoadProcessBackendCrashed,
    RoadProcessBackendError,
    RoadProcessBackendTimeout,
    WaypointPrimitive,
    validate_footprint_query_result,
)
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
class RoadTickFacts:
    """Immutable exact-map facts shared by every plan in one serial batch."""

    frame_id: int
    simulation_time_s: float
    ego: EgoKinematics
    ego_center_z: float
    current_ego_samples: tuple[RoadContainmentSample, ...]
    current_ego_clearance_road: RoadContainmentAssessment
    current_ego_road: RoadContainmentAssessment
    current_junction_context: bool
    map_digest: str
    policy_digest: str
    bounding_box_digest: str
    fallback_yaw_rad: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "simulation_time_s": self.simulation_time_s,
            "current_ego_clearance_road": (
                self.current_ego_clearance_road.to_json_dict()
            ),
            "current_ego_road": self.current_ego_road.to_json_dict(),
            "current_junction_context": bool(self.current_junction_context),
            "map_digest": self.map_digest,
            "policy_digest": self.policy_digest,
            "bounding_box_digest": self.bounding_box_digest,
            "fallback_yaw_rad": self.fallback_yaw_rad,
        }


@dataclass(frozen=True)
class RoadAssessmentBatchStats:
    """Timing and exact-query workload for one ordered road-assessment batch.

    ``map_query_ms`` and ``map_query_count`` describe the primary backend plus
    the current-ego query.  First-batch exact serial shadow work is reported
    separately so process performance remains directly observable.
    """

    backend_status: str = "serial"
    road_batch_wall_ms: float = 0.0
    validation_ms: float = 0.0
    densify_heading_ms: float = 0.0
    map_query_ms: float = 0.0
    aggregate_ms: float = 0.0
    plan_count: int = 0
    pose_count: int = 0
    map_query_count: int = 0
    worker_count: int = 1
    chunk_count: int = 0
    profile_cache_hits: int = 0
    profile_cache_misses: int = 0
    error_count: int = 0
    shadow_parity_status: str = "not_run"
    fallback_reason: str | None = None
    serial_fallback_ms: float = 0.0
    worker_query_sum_ms: float = 0.0
    commissioning_batch: bool = False
    query_deadline_ms: float = 0.0
    process_attempt_ms: float = 0.0
    shadow_serial_ms: float = 0.0
    shadow_serial_query_count: int = 0
    shadow_parity_ms: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "backend_status": self.backend_status,
            "road_batch_wall_ms": self.road_batch_wall_ms,
            "validation_ms": self.validation_ms,
            "densify_heading_ms": self.densify_heading_ms,
            "map_query_ms": self.map_query_ms,
            "aggregate_ms": self.aggregate_ms,
            "plan_count": self.plan_count,
            "pose_count": self.pose_count,
            "map_query_count": self.map_query_count,
            "worker_count": self.worker_count,
            "chunk_count": self.chunk_count,
            "profile_cache_hits": self.profile_cache_hits,
            "profile_cache_misses": self.profile_cache_misses,
            "error_count": self.error_count,
            "shadow_parity_status": self.shadow_parity_status,
            "fallback_reason": self.fallback_reason,
            "serial_fallback_ms": self.serial_fallback_ms,
            "worker_query_sum_ms": self.worker_query_sum_ms,
            "commissioning_batch": bool(self.commissioning_batch),
            "query_deadline_ms": self.query_deadline_ms,
            "process_attempt_ms": self.process_attempt_ms,
            "shadow_serial_ms": self.shadow_serial_ms,
            "shadow_serial_query_count": self.shadow_serial_query_count,
            "shadow_parity_ms": self.shadow_parity_ms,
        }


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

    cache_key: str
    points: np.ndarray
    times_s: np.ndarray
    cumulative_distance_m: np.ndarray
    upper_waypoint_indices: np.ndarray
    samples_by_pose: tuple[tuple[RoadContainmentSample, ...], ...]
    quality: str


@dataclass(frozen=True)
class _PreparedTimedRoadProfile:
    """Main-process geometry and numeric footprint queries for one cache miss."""

    cache_key: str
    points: np.ndarray
    times_s: np.ndarray
    cumulative_distance_m: np.ndarray
    upper_waypoint_indices: np.ndarray
    yaws_rad: tuple[float, ...]
    half_length_m: float
    half_width_m: float
    queries: tuple[FootprintQuery, ...]


@dataclass(frozen=True)
class _PathProgressProjection:
    """Ego projection used as the physical road-execution cursor."""

    progress_m: float
    first_path_index: int
    cross_track_m: float
    heading_error_deg: float


@dataclass
class _RoadBatchAccumulator:
    """Mutable timing scratchpad converted to a frozen public contract."""

    validation_s: float = 0.0
    densify_heading_s: float = 0.0
    map_query_s: float = 0.0
    aggregate_s: float = 0.0
    pose_count: int = 0
    map_query_count: int = 0
    profile_cache_hits: int = 0
    profile_cache_misses: int = 0
    error_count: int = 0
    worker_count: int = 1
    chunk_count: int = 0
    worker_query_sum_s: float = 0.0
    shadow_parity_status: str = "not_run"
    fallback_reason: str | None = None
    serial_fallback_s: float = 0.0
    commissioning_batch: bool = False
    query_deadline_ms: float = 0.0
    process_attempt_s: float = 0.0
    shadow_serial_s: float = 0.0
    shadow_serial_query_count: int = 0
    shadow_parity_s: float = 0.0
    backend_status: str | None = None


class _PlanIdGeometryMismatch(ValueError):
    """The same logical plan identity was reused for different trajectory data."""


def _timed_road_profile_cacheable(profile: _TimedRoadProfile) -> bool:
    """Return false when transient map-query failures contaminated a profile."""

    return all(
        sample.contained is not None
        and not (
            sample.reason is not None
            and "carla_map_query_error" in sample.reason
        )
        for pose_samples in profile.samples_by_pose
        for sample in pose_samples
    )


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


def _footprint_query(
    *,
    query_id: int,
    center_xyz: np.ndarray,
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
) -> FootprintQuery:
    """Build the numeric-only process/serial query for one footprint pose."""

    return FootprintQuery(
        query_id=int(query_id),
        points_xyz=tuple(
            tuple(float(value) for value in point)
            for point in _footprint_points(
                np.asarray(center_xyz, dtype=np.float64),
                float(yaw_rad),
                float(half_length_m),
                float(half_width_m),
            )
        ),
    )


def _waypoint_primitive(waypoint: Any) -> WaypointPrimitive:
    """Copy the exact CARLA waypoint fields consumed by containment semantics."""

    if waypoint is None:
        return WaypointPrimitive(found=False)
    transform = waypoint.transform
    return WaypointPrimitive(
        found=True,
        road_id=int(waypoint.road_id),
        lane_id=int(waypoint.lane_id),
        is_junction=bool(getattr(waypoint, "is_junction", False)),
        lane_center_x=float(transform.location.x),
        lane_center_y=float(transform.location.y),
        lane_yaw_deg=float(transform.rotation.yaw),
        lane_width=float(waypoint.lane_width),
    )


def _aggregate_footprint_query(
    *,
    query: FootprintQuery,
    result: FootprintQueryResult,
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
    sample_index_start: int,
    lateral_clearance_m: float,
) -> tuple[RoadContainmentSample, ...]:
    """Apply the one shared exact containment reducer to serial/process data."""

    center_xyz = np.asarray(query.points_xyz[0], dtype=np.float64)
    validate_footprint_query_result(
        result,
        expected_query_id=query.query_id,
    )
    if result.error_type is not None:
        return (
            RoadContainmentSample(
                sample_index=sample_index_start,
                position_xyz=tuple(float(value) for value in center_xyz),
                contained=None,
                reason=f"carla_map_query_error:{result.error_type}",
            ),
        )
    if len(result.waypoints) != 5:
        raise ValueError("complete footprint query requires five waypoint results")

    center_waypoint = result.waypoints[0]
    if not center_waypoint.found:
        return (
            RoadContainmentSample(
                sample_index=sample_index_start,
                position_xyz=tuple(float(value) for value in center_xyz),
                contained=False,
                reason="center_off_driving_lane",
            ),
        )

    is_junction = bool(center_waypoint.is_junction)
    if is_junction:
        margin_m = None
        center_contained = True
        center_reason = None
    else:
        lane_yaw_rad = math.radians(float(center_waypoint.lane_yaw_deg))
        lane_right = np.array(
            [-math.sin(lane_yaw_rad), math.cos(lane_yaw_rad)],
            dtype=np.float64,
        )
        lane_center = np.array(
            [
                float(center_waypoint.lane_center_x),
                float(center_waypoint.lane_center_y),
            ],
            dtype=np.float64,
        )
        lateral_offset = float(
            np.dot(center_xyz[:2] - lane_center, lane_right)
        )
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
            - float(lateral_clearance_m)
        )
        heading_error = abs(
            _wrapped_angle_degrees(
                math.degrees(yaw_rad) - float(center_waypoint.lane_yaw_deg)
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
        zip(query.points_xyz[1:], result.waypoints[1:]),
        start=1,
    ):
        corner_is_junction = (
            bool(waypoint.is_junction) if waypoint.found else False
        )
        same_lane = (
            waypoint.found
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
                road_id=int(waypoint.road_id) if waypoint.found else None,
                lane_id=int(waypoint.lane_id) if waypoint.found else None,
                is_junction=(
                    corner_is_junction if waypoint.found else None
                ),
                reason=(
                    None
                    if same_lane
                    else "footprint_corner_off_driving_lane"
                ),
            )
        )
    return tuple(samples)


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


def _sha256_text(label: str, value: str) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _map_content_digest(carla_map: Any) -> str:
    """Hash exact OpenDRIVE once, with a stable map-name fallback."""

    to_opendrive = getattr(carla_map, "to_opendrive", None)
    if callable(to_opendrive):
        try:
            opendrive = to_opendrive()
            if isinstance(opendrive, str) and opendrive:
                return _opendrive_snapshot_digest(opendrive)
        except Exception:
            pass

    map_name = getattr(carla_map, "name", None)
    if map_name is not None:
        return _sha256_text("carla_map_name", str(map_name))
    fallback = (
        f"{type(carla_map).__module__}.{type(carla_map).__qualname__}:"
        f"{id(carla_map)}"
    )
    return _sha256_text("carla_map_object", fallback)


def _opendrive_snapshot_digest(opendrive: str) -> str:
    """Return the exact labeled snapshot identity used by profile caches."""

    if not isinstance(opendrive, str) or not opendrive:
        raise ValueError("OpenDRIVE snapshot must be nonempty")
    return _sha256_text("carla_opendrive", opendrive)


def _policy_content_digest(policy: SafetyPolicy) -> str:
    if is_dataclass(policy):
        values = {
            item.name: getattr(policy, item.name)
            for item in fields(policy)
        }
    else:
        values = dict(vars(policy))
    payload = json.dumps(
        values,
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text("safety_policy", payload)


def _bounding_box_content_digest(bounding_box: Any) -> str:
    extent = getattr(bounding_box, "extent", None)
    location = getattr(bounding_box, "location", None)
    rotation = getattr(bounding_box, "rotation", None)
    values = {
        "extent": [
            float(getattr(extent, axis, 0.0))
            for axis in ("x", "y", "z")
        ],
        "location": [
            float(getattr(location, axis, 0.0))
            for axis in ("x", "y", "z")
        ],
        "rotation": [
            float(getattr(rotation, angle, 0.0))
            for angle in ("roll", "pitch", "yaw")
        ],
    }
    payload = json.dumps(
        values,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text("ego_bounding_box", payload)


def _plan_geometry_digest(plan: Any) -> str:
    """Hash exactly the geometry/timestamps consumed by the road profile."""

    points = np.asarray(plan.world_points, dtype=np.float64)
    times = np.asarray(plan.waypoint_times_s, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] < 3
        or len(points) == 0
        or times.ndim != 1
        or len(times) != len(points)
        or not np.isfinite(points[:, :3]).all()
        or not np.isfinite(times).all()
        or (len(times) > 1 and not np.all(np.diff(times) > 0.0))
    ):
        raise ValueError("invalid timed world path")

    normalized_points = np.ascontiguousarray(points[:, :3], dtype="<f8")
    normalized_times = np.ascontiguousarray(times, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"trajectory_geometry_timestamps_v1\0")
    digest.update(
        json.dumps(
            {
                "points_shape": list(normalized_points.shape),
                "times_shape": list(normalized_times.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(normalized_points.tobytes(order="C"))
    digest.update(b"\0")
    digest.update(normalized_times.tobytes(order="C"))
    return digest.hexdigest()


def _road_profile_cache_key(
    *,
    geometry_digest: str,
    map_digest: str,
    policy_digest: str,
    bounding_box_digest: str,
    fallback_yaw_rad: float,
) -> str:
    fallback_yaw = float(fallback_yaw_rad)
    if not math.isfinite(fallback_yaw):
        raise ValueError("fallback yaw must be finite")
    digest = hashlib.sha256()
    digest.update(b"exact_road_profile_v1\0")
    for value in (
        geometry_digest,
        map_digest,
        policy_digest,
        bounding_box_digest,
        fallback_yaw.hex(),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _effective_profile_fallback_yaw(
    plan: Any,
    fallback_yaw_rad: float,
) -> float:
    """Return zero when trajectory headings never consume the fallback yaw."""

    points = np.asarray(plan.world_points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or len(points) == 0
        or not np.isfinite(points[:, :2]).all()
    ):
        raise ValueError("invalid world path")
    pairwise_delta = points[:, None, :2] - points[None, :, :2]
    maximum_distance_by_point = np.sqrt(
        np.max(np.sum(pairwise_delta * pairwise_delta, axis=2), axis=1)
    )
    if np.any(
        maximum_distance_by_point
        < float(cfg.TRAJECTORY_HEADING_MIN_DISPLACEMENT_M)
    ):
        return float(fallback_yaw_rad)
    return 0.0


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

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        policy: SafetyPolicy | None = None,
        *,
        road_assessment_backend: str = "serial",
        road_assessment_workers: int | None = None,
        process_backend_factory: Any = ExactRoadProcessBackend,
    ):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.policy = policy or SafetyPolicy()
        backend = str(road_assessment_backend)
        if backend not in {"serial", "process"}:
            raise ValueError(
                "road_assessment_backend must be 'serial' or 'process'"
            )
        self._road_assessment_backend = backend
        self._process_backend_factory = process_backend_factory
        self._process_worker_count = (
            self._resolve_process_worker_count(road_assessment_workers)
            if backend == "process"
            else 1
        )
        self._process_backend = None
        self._process_shadow_pending = backend == "process"
        self._process_sticky_disabled_reason: str | None = None
        self._process_generation_disabled_reason: str | None = None
        self._map = None
        self._map_digest = ""
        self._process_opendrive_snapshot: str | None = None
        self._refresh_map_snapshot()
        self._road_profile_cache: OrderedDict[str, _TimedRoadProfile] = OrderedDict()
        self._plan_geometry_by_id: dict[str, str] = {}
        self._road_tick_facts_cache_key: tuple[Any, ...] | None = None
        self._road_tick_facts_cache: RoadTickFacts | None = None
        self._last_road_batch_stats = RoadAssessmentBatchStats()
        if backend == "process":
            self._start_process_backend()

    @staticmethod
    def _resolve_process_worker_count(requested: int | None) -> int:
        if requested is not None:
            count = int(requested)
            if count < 1:
                raise ValueError("road_assessment_workers must be positive")
            return count
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = int(os.cpu_count() or 1)
        count = min(6, int(available) - 2)
        if count < 1:
            raise ValueError(
                "process road assessment requires at least three available CPUs"
            )
        return count

    def _refresh_map_snapshot(self) -> None:
        """Refresh one map generation without duplicate OpenDRIVE exports."""

        self._map = self.world.get_map()
        self._process_opendrive_snapshot = None
        if self._road_assessment_backend == "serial":
            self._map_digest = _map_content_digest(self._map)
            return

        to_opendrive = getattr(self._map, "to_opendrive", None)
        if not callable(to_opendrive):
            raise RuntimeError(
                "process road assessment requires CARLA OpenDRIVE export"
            )
        opendrive = to_opendrive()
        if not isinstance(opendrive, str) or not opendrive:
            raise RuntimeError(
                "process road assessment received empty CARLA OpenDRIVE"
            )
        self._process_opendrive_snapshot = opendrive
        self._map_digest = _opendrive_snapshot_digest(opendrive)

    def _start_process_backend(self) -> None:
        if self._road_assessment_backend != "process":
            return
        if (
            self._process_sticky_disabled_reason is not None
            or self._process_generation_disabled_reason is not None
        ):
            return
        opendrive = self._process_opendrive_snapshot
        if not isinstance(opendrive, str) or not opendrive:
            raise RuntimeError(
                "process road assessment snapshot is unavailable"
            )
        self._process_backend = self._process_backend_factory(
            map_name=str(getattr(self._map, "name", "CARLA")),
            opendrive=opendrive,
            map_digest=self._map_digest,
            worker_count=self._process_worker_count,
            chunk_pose_count=int(
                cfg.ROAD_ASSESSMENT_PROCESS_CHUNK_POSES
            ),
            startup_timeout_s=float(
                cfg.ROAD_ASSESSMENT_PROCESS_STARTUP_TIMEOUT_S
            ),
            commissioning_timeout_s=float(
                cfg.ROAD_ASSESSMENT_PROCESS_COMMISSIONING_TIMEOUT_S
            ),
            batch_timeout_s=float(
                cfg.ROAD_ASSESSMENT_PROCESS_BATCH_TIMEOUT_S
            ),
        )

    def _close_process_backend(self) -> None:
        backend = self._process_backend
        self._process_backend = None
        if backend is not None:
            backend.close()

    def clear_plan_caches(self) -> None:
        """Clear logical plan state without rebuilding immutable map workers."""

        self._road_profile_cache.clear()
        self._plan_geometry_by_id.clear()
        self._last_road_batch_stats = RoadAssessmentBatchStats(
            backend_status=self._road_assessment_backend,
            worker_count=(
                self._process_worker_count
                if self._road_assessment_backend == "process"
                else 1
            ),
        )

    def reset(self, ego_vehicle: Any | None = None) -> None:
        self._close_process_backend()
        if ego_vehicle is not None:
            self.ego_vehicle = ego_vehicle
        self._refresh_map_snapshot()
        self._road_profile_cache.clear()
        self._plan_geometry_by_id.clear()
        self._road_tick_facts_cache_key = None
        self._road_tick_facts_cache = None
        self._last_road_batch_stats = RoadAssessmentBatchStats()
        self._process_generation_disabled_reason = None
        self._process_shadow_pending = (
            self._road_assessment_backend == "process"
            and self._process_sticky_disabled_reason is None
        )
        self._start_process_backend()

    def close(self) -> None:
        """Release process workers idempotently before CARLA world cleanup."""

        self._close_process_backend()

    @property
    def last_road_batch_stats(self) -> RoadAssessmentBatchStats:
        """Stats from the most recent single-plan or batch road assessment."""

        return self._last_road_batch_stats

    def _query_footprint(
        self,
        *,
        center_xyz: np.ndarray,
        yaw_rad: float,
        half_length_m: float,
        half_width_m: float,
        sample_index_start: int,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> tuple[RoadContainmentSample, ...]:
        query = _footprint_query(
            query_id=max(0, int(sample_index_start) // 5),
            center_xyz=center_xyz,
            yaw_rad=yaw_rad,
            half_length_m=half_length_m,
            half_width_m=half_width_m,
        )
        query_started_s = time.perf_counter()
        query_count = 0
        primitives = []
        error_type = None
        for point in query.points_xyz:
            query_count += 1
            try:
                waypoint = self._map.get_waypoint(
                    carla.Location(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                    ),
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                primitives.append(_waypoint_primitive(waypoint))
            except Exception as exc:
                error_type = type(exc).__name__
                break
        query_time_s = time.perf_counter() - query_started_s
        if batch_accumulator is not None:
            batch_accumulator.map_query_count += query_count
            batch_accumulator.map_query_s += query_time_s
            if error_type is not None:
                batch_accumulator.error_count += 1
        result = FootprintQueryResult(
            query_id=query.query_id,
            waypoints=tuple(primitives),
            error_type=error_type,
            map_query_count=query_count,
            worker_query_ms=query_time_s * 1000.0,
            worker_pid=int(os.getpid()),
        )
        return _aggregate_footprint_query(
            query=query,
            result=result,
            yaw_rad=yaw_rad,
            half_length_m=half_length_m,
            half_width_m=half_width_m,
            sample_index_start=sample_index_start,
            lateral_clearance_m=self.policy.lateral_clearance_m,
        )

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

    def _prepare_timed_road_profile(
        self,
        plan: Any,
        ego: EgoKinematics,
        *,
        cache_key: str,
        query_id_start: int = 0,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> _PreparedTimedRoadProfile:
        """Densify and orient one plan without making CARLA map queries."""

        densify_started_s = time.perf_counter()
        points, times, distances, upper_indices = _densify_timed_path(
            plan.world_points,
            plan.waypoint_times_s,
            max_spacing_m=self.policy.path_sample_spacing_m,
        )
        if batch_accumulator is not None:
            batch_accumulator.densify_heading_s += (
                time.perf_counter() - densify_started_s
            )
            batch_accumulator.pose_count += len(points)
        yaws = []
        queries = []
        for index, point in enumerate(points):
            heading_started_s = time.perf_counter()
            yaw = _path_yaw(points, index, ego.yaw_rad)
            if batch_accumulator is not None:
                batch_accumulator.densify_heading_s += (
                    time.perf_counter() - heading_started_s
                )
            yaws.append(float(yaw))
            queries.append(
                _footprint_query(
                    query_id=int(query_id_start) + index,
                    center_xyz=point[:3],
                    yaw_rad=yaw,
                    half_length_m=ego.half_length_m,
                    half_width_m=ego.half_width_m,
                )
            )
        return _PreparedTimedRoadProfile(
            cache_key=cache_key,
            points=points,
            times_s=times,
            cumulative_distance_m=distances,
            upper_waypoint_indices=upper_indices,
            yaws_rad=tuple(yaws),
            half_length_m=float(ego.half_length_m),
            half_width_m=float(ego.half_width_m),
            queries=tuple(queries),
        )

    def _assemble_timed_road_profile(
        self,
        prepared: _PreparedTimedRoadProfile,
        results: tuple[FootprintQueryResult, ...],
        *,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> _TimedRoadProfile:
        """Apply shared footprint semantics and build one immutable profile."""

        if len(results) != len(prepared.queries):
            raise ValueError("prepared profile query/result count mismatch")
        aggregate_started_s = time.perf_counter()
        samples_by_pose = []
        for index, (query, result) in enumerate(
            zip(prepared.queries, results)
        ):
            if result.error_type is not None and batch_accumulator is not None:
                batch_accumulator.error_count += 1
            samples_by_pose.append(
                _aggregate_footprint_query(
                    query=query,
                    result=result,
                    yaw_rad=prepared.yaws_rad[index],
                    half_length_m=prepared.half_length_m,
                    half_width_m=prepared.half_width_m,
                    sample_index_start=index * 5,
                    lateral_clearance_m=self.policy.lateral_clearance_m,
                )
            )
        flattened = tuple(sample for pose in samples_by_pose for sample in pose)
        quality = "carla_ground_truth_exact_lane_and_footprint"
        if any(sample.is_junction for sample in flattened):
            quality = "carla_ground_truth_drivable_only_at_junction"
        profile = _TimedRoadProfile(
            cache_key=prepared.cache_key,
            points=prepared.points,
            times_s=prepared.times_s,
            cumulative_distance_m=prepared.cumulative_distance_m,
            upper_waypoint_indices=prepared.upper_waypoint_indices,
            samples_by_pose=tuple(samples_by_pose),
            quality=quality,
        )
        if batch_accumulator is not None:
            batch_accumulator.aggregate_s += (
                time.perf_counter() - aggregate_started_s
            )
        return profile

    def _query_prepared_profile_serial(
        self,
        prepared: _PreparedTimedRoadProfile,
        *,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> tuple[FootprintQueryResult, ...]:
        """Execute prepared footprint queries on the parent CARLA map."""

        results = []
        for query in prepared.queries:
            query_started_s = time.perf_counter()
            query_count = 0
            primitives = []
            error_type = None
            for point in query.points_xyz:
                query_count += 1
                try:
                    waypoint = self._map.get_waypoint(
                        carla.Location(
                            x=float(point[0]),
                            y=float(point[1]),
                            z=float(point[2]),
                        ),
                        project_to_road=False,
                        lane_type=carla.LaneType.Driving,
                    )
                    primitives.append(_waypoint_primitive(waypoint))
                except Exception as exc:
                    error_type = type(exc).__name__
                    break
            query_time_s = time.perf_counter() - query_started_s
            if batch_accumulator is not None:
                batch_accumulator.map_query_count += query_count
                batch_accumulator.map_query_s += query_time_s
            results.append(
                FootprintQueryResult(
                    query_id=query.query_id,
                    waypoints=tuple(primitives),
                    error_type=error_type,
                    map_query_count=query_count,
                    worker_query_ms=query_time_s * 1000.0,
                    worker_pid=int(os.getpid()),
                )
            )
        return tuple(results)

    def _build_timed_road_profile(
        self,
        plan: Any,
        ego: EgoKinematics,
        *,
        cache_key: str,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> _TimedRoadProfile:
        """Serial compatibility path composed from shared prepare/aggregate."""

        prepared = self._prepare_timed_road_profile(
            plan,
            ego,
            cache_key=cache_key,
            batch_accumulator=batch_accumulator,
        )
        results = self._query_prepared_profile_serial(
            prepared,
            batch_accumulator=batch_accumulator,
        )
        return self._assemble_timed_road_profile(
            prepared,
            results,
            batch_accumulator=batch_accumulator,
        )

    def _profile_cache_key_for_plan(
        self,
        plan: Any,
        ego: EgoKinematics,
        *,
        tick_facts: RoadTickFacts | None = None,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> str:
        validation_started_s = time.perf_counter()
        try:
            plan_id = str(plan.plan_id)
            geometry_digest = _plan_geometry_digest(plan)
            prior_geometry_digest = self._plan_geometry_by_id.get(plan_id)
            if (
                prior_geometry_digest is not None
                and prior_geometry_digest != geometry_digest
            ):
                raise _PlanIdGeometryMismatch(
                    f"plan_id {plan_id!r} was reused for different geometry"
                )
            self._plan_geometry_by_id.setdefault(plan_id, geometry_digest)

            map_digest = (
                tick_facts.map_digest
                if tick_facts is not None
                else self._map_digest
            )
            policy_digest = (
                tick_facts.policy_digest
                if tick_facts is not None
                else _policy_content_digest(self.policy)
            )
            bounding_box_digest = (
                tick_facts.bounding_box_digest
                if tick_facts is not None
                else _bounding_box_content_digest(
                    self.ego_vehicle.bounding_box
                )
            )
            raw_fallback_yaw_rad = (
                tick_facts.fallback_yaw_rad
                if tick_facts is not None
                else ego.yaw_rad
            )
            fallback_yaw_rad = _effective_profile_fallback_yaw(
                plan,
                raw_fallback_yaw_rad,
            )
            cache_key = _road_profile_cache_key(
                geometry_digest=geometry_digest,
                map_digest=map_digest,
                policy_digest=policy_digest,
                bounding_box_digest=bounding_box_digest,
                fallback_yaw_rad=fallback_yaw_rad,
            )
        finally:
            if batch_accumulator is not None:
                batch_accumulator.validation_s += (
                    time.perf_counter() - validation_started_s
                )
        return cache_key

    def _cache_timed_road_profile(
        self,
        profile: _TimedRoadProfile,
    ) -> None:
        if not _timed_road_profile_cacheable(profile):
            return
        self._road_profile_cache[profile.cache_key] = profile
        while len(self._road_profile_cache) > int(
            cfg.SAFETY_ROAD_PROFILE_CACHE_SIZE
        ):
            self._road_profile_cache.popitem(last=False)

    def _profile_for_plan(
        self,
        plan: Any,
        ego: EgoKinematics,
        *,
        tick_facts: RoadTickFacts | None = None,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> _TimedRoadProfile:
        cache_key = self._profile_cache_key_for_plan(
            plan,
            ego,
            tick_facts=tick_facts,
            batch_accumulator=batch_accumulator,
        )
        profile = self._road_profile_cache.get(cache_key)
        if profile is not None:
            self._road_profile_cache.move_to_end(cache_key)
            if batch_accumulator is not None:
                batch_accumulator.profile_cache_hits += 1
            return profile

        if batch_accumulator is not None:
            batch_accumulator.profile_cache_misses += 1
        profile = self._build_timed_road_profile(
            plan,
            ego,
            cache_key=cache_key,
            batch_accumulator=batch_accumulator,
        )
        self._cache_timed_road_profile(profile)
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

    def _road_tick_facts(
        self,
        *,
        tick_context: Any,
        batch_accumulator: _RoadBatchAccumulator,
    ) -> RoadTickFacts:
        """Build the one exact ego-footprint assessment shared by a batch."""

        validation_started_s = time.perf_counter()
        if int(tick_context.snapshot.frame) != int(tick_context.frame_id):
            raise ValueError("snapshot frame mismatch")
        ego = self._ego_from_context(tick_context)
        simulation_time_s = float(tick_context.simulation_time_s)
        ego_center_z = float(tick_context.ego_transform.location.z)
        if not math.isfinite(simulation_time_s) or not math.isfinite(ego_center_z):
            raise ValueError("invalid tick time or ego elevation")
        policy_digest = _policy_content_digest(self.policy)
        bounding_box_digest = _bounding_box_content_digest(
            self.ego_vehicle.bounding_box
        )
        tick_cache_key = (
            int(tick_context.frame_id),
            simulation_time_s.hex(),
            self._map_digest,
            policy_digest,
            bounding_box_digest,
            int(ego.actor_id),
            float(ego.center_xy[0]).hex(),
            float(ego.center_xy[1]).hex(),
            ego_center_z.hex(),
            float(ego.yaw_rad).hex(),
            float(ego.velocity_xy[0]).hex(),
            float(ego.velocity_xy[1]).hex(),
            float(ego.half_length_m).hex(),
            float(ego.half_width_m).hex(),
        )
        batch_accumulator.validation_s += (
            time.perf_counter() - validation_started_s
        )
        if (
            tick_cache_key == self._road_tick_facts_cache_key
            and self._road_tick_facts_cache is not None
        ):
            return self._road_tick_facts_cache

        current_samples = self._query_footprint(
            center_xyz=np.array(
                [*ego.center_xy, ego_center_z],
                dtype=np.float64,
            ),
            yaw_rad=ego.yaw_rad,
            half_length_m=ego.half_length_m,
            half_width_m=ego.half_width_m,
            sample_index_start=0,
            batch_accumulator=batch_accumulator,
        )
        aggregate_started_s = time.perf_counter()
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
        tick_facts = RoadTickFacts(
            frame_id=int(tick_context.frame_id),
            simulation_time_s=simulation_time_s,
            ego=ego,
            ego_center_z=ego_center_z,
            current_ego_samples=current_samples,
            current_ego_clearance_road=current_clearance_road,
            current_ego_road=current_road,
            current_junction_context=current_junction_context,
            map_digest=self._map_digest,
            policy_digest=policy_digest,
            bounding_box_digest=bounding_box_digest,
            fallback_yaw_rad=ego.yaw_rad,
        )
        batch_accumulator.aggregate_s += (
            time.perf_counter() - aggregate_started_s
        )
        self._road_tick_facts_cache_key = tick_cache_key
        self._road_tick_facts_cache = tick_facts
        return tick_facts

    def _envelope_for_profile(
        self,
        *,
        profile: _TimedRoadProfile,
        plan: Any,
        tick_facts: RoadTickFacts,
        batch_accumulator: _RoadBatchAccumulator | None = None,
    ) -> RoadExecutionEnvelope:
        aggregate_started_s = time.perf_counter()
        envelope = self._road_execution_envelope(
            profile=profile,
            plan=plan,
            ego=tick_facts.ego,
            current_time_s=tick_facts.simulation_time_s,
            current_road=tick_facts.current_ego_road,
            current_clearance_road=tick_facts.current_ego_clearance_road,
            current_junction_context=tick_facts.current_junction_context,
        )
        if batch_accumulator is not None:
            batch_accumulator.aggregate_s += (
                time.perf_counter() - aggregate_started_s
            )
        return envelope

    @staticmethod
    def _profiles_semantically_equal(
        process_profile: _TimedRoadProfile,
        serial_profile: _TimedRoadProfile,
    ) -> bool:
        return bool(
            process_profile.cache_key == serial_profile.cache_key
            and process_profile.quality == serial_profile.quality
            and np.array_equal(process_profile.points, serial_profile.points)
            and np.array_equal(process_profile.times_s, serial_profile.times_s)
            and np.array_equal(
                process_profile.cumulative_distance_m,
                serial_profile.cumulative_distance_m,
            )
            and np.array_equal(
                process_profile.upper_waypoint_indices,
                serial_profile.upper_waypoint_indices,
            )
            and process_profile.samples_by_pose
            == serial_profile.samples_by_pose
        )

    @staticmethod
    def _envelopes_semantically_equal(
        process_envelope: RoadExecutionEnvelope,
        serial_envelope: RoadExecutionEnvelope,
    ) -> bool:
        process_payload = process_envelope.to_json_dict()
        serial_payload = serial_envelope.to_json_dict()
        process_payload.pop("stopping_reserve_compute_ms", None)
        serial_payload.pop("stopping_reserve_compute_ms", None)
        return process_payload == serial_payload

    def _serial_profiles_from_prepared(
        self,
        prepared_by_index: dict[int, _PreparedTimedRoadProfile],
        *,
        batch_accumulator: _RoadBatchAccumulator,
    ) -> dict[int, _TimedRoadProfile]:
        profiles = {}
        for plan_index, prepared in prepared_by_index.items():
            results = self._query_prepared_profile_serial(
                prepared,
                batch_accumulator=batch_accumulator,
            )
            profiles[plan_index] = self._assemble_timed_road_profile(
                prepared,
                results,
                batch_accumulator=batch_accumulator,
            )
        return profiles

    def _process_profiles_for_plans(
        self,
        *,
        ordered_plans: tuple[Any, ...],
        tick_facts: RoadTickFacts,
        batch_accumulator: _RoadBatchAccumulator,
    ) -> tuple[list[_TimedRoadProfile | None], dict[int, str]]:
        """Resolve profile cache misses through process query + exact fallback."""

        profiles: list[_TimedRoadProfile | None] = [
            None for _plan in ordered_plans
        ]
        errors: dict[int, str] = {}
        prepared_by_index: dict[int, _PreparedTimedRoadProfile] = {}
        next_query_id = 0
        for plan_index, plan in enumerate(ordered_plans):
            try:
                cache_key = self._profile_cache_key_for_plan(
                    plan,
                    tick_facts.ego,
                    tick_facts=tick_facts,
                    batch_accumulator=batch_accumulator,
                )
                cached = self._road_profile_cache.get(cache_key)
                if cached is not None:
                    self._road_profile_cache.move_to_end(cache_key)
                    batch_accumulator.profile_cache_hits += 1
                    profiles[plan_index] = cached
                    continue
                batch_accumulator.profile_cache_misses += 1
                prepared = self._prepare_timed_road_profile(
                    plan,
                    tick_facts.ego,
                    cache_key=cache_key,
                    query_id_start=next_query_id,
                    batch_accumulator=batch_accumulator,
                )
                next_query_id += len(prepared.queries)
                prepared_by_index[plan_index] = prepared
            except _PlanIdGeometryMismatch:
                errors[plan_index] = "plan_id_geometry_mismatch"
            except Exception as exc:
                errors[plan_index] = (
                    f"invalid_world_path:{type(exc).__name__}"
                )

        if not prepared_by_index:
            if self._process_sticky_disabled_reason is not None:
                batch_accumulator.backend_status = (
                    "process_disabled_shadow_mismatch"
                )
                batch_accumulator.shadow_parity_status = "fail"
                batch_accumulator.fallback_reason = (
                    self._process_sticky_disabled_reason
                )
            elif (
                self._process_generation_disabled_reason is not None
                or self._process_backend is None
            ):
                batch_accumulator.backend_status = (
                    "process_degraded_cache_only"
                )
                batch_accumulator.fallback_reason = (
                    self._process_generation_disabled_reason
                    or "process_unavailable"
                )
            else:
                batch_accumulator.backend_status = (
                    "process_cache_only"
                    if not errors
                    else "process_not_run"
                )
            batch_accumulator.worker_count = self._process_worker_count
            return profiles, errors

        process_profiles: dict[int, _TimedRoadProfile] | None = None
        backend = self._process_backend
        process_attempted = backend is not None
        if backend is not None:
            ordered_queries = tuple(
                query
                for prepared in prepared_by_index.values()
                for query in prepared.queries
            )
            batch_accumulator.commissioning_batch = bool(
                self._process_shadow_pending
            )
            batch_accumulator.query_deadline_ms = (
                float(
                    cfg.ROAD_ASSESSMENT_PROCESS_COMMISSIONING_TIMEOUT_S
                    if self._process_shadow_pending
                    else cfg.ROAD_ASSESSMENT_PROCESS_BATCH_TIMEOUT_S
                )
                * 1000.0
            )
            process_attempt_started_s = time.perf_counter()
            try:
                process_results, process_stats = backend.query(
                    ordered_queries
                )
                results_by_id = {
                    result.query_id: result for result in process_results
                }
                if len(results_by_id) != len(ordered_queries):
                    raise RoadProcessBackendError(
                        "process result query IDs are not unique"
                    )
                process_profiles = {}
                for plan_index, prepared in prepared_by_index.items():
                    results = tuple(
                        results_by_id[query.query_id]
                        for query in prepared.queries
                    )
                    process_profiles[plan_index] = (
                        self._assemble_timed_road_profile(
                            prepared,
                            results,
                            batch_accumulator=batch_accumulator,
                        )
                    )
                batch_accumulator.map_query_s += (
                    process_stats.map_query_wall_ms / 1000.0
                )
                batch_accumulator.worker_query_sum_s += (
                    process_stats.worker_query_sum_ms / 1000.0
                )
                batch_accumulator.map_query_count += (
                    process_stats.map_query_count
                )
                batch_accumulator.worker_count = process_stats.worker_count
                batch_accumulator.chunk_count = process_stats.chunk_count
                batch_accumulator.commissioning_batch = bool(
                    getattr(process_stats, "commissioning_batch", False)
                )
                batch_accumulator.query_deadline_ms = float(
                    getattr(process_stats, "query_deadline_ms", 0.0)
                )
            except RoadProcessBackendTimeout:
                batch_accumulator.fallback_reason = "timeout"
                self._process_generation_disabled_reason = "timeout"
            except RoadProcessBackendCrashed:
                batch_accumulator.fallback_reason = "crash"
                self._process_generation_disabled_reason = "crash"
            except RoadProcessBackendError:
                batch_accumulator.fallback_reason = "backend_error"
                self._process_generation_disabled_reason = "backend_error"
            except Exception as exc:
                batch_accumulator.fallback_reason = (
                    f"protocol_error:{type(exc).__name__}"
                )
                self._process_generation_disabled_reason = (
                    "protocol_error"
                )
            finally:
                batch_accumulator.process_attempt_s += (
                    time.perf_counter() - process_attempt_started_s
                )
            if process_profiles is None:
                self._close_process_backend()

        if process_profiles is None:
            fallback_started_s = time.perf_counter()
            serial_profiles = self._serial_profiles_from_prepared(
                prepared_by_index,
                batch_accumulator=batch_accumulator,
            )
            batch_accumulator.serial_fallback_s += (
                time.perf_counter() - fallback_started_s
            )
            reason = (
                batch_accumulator.fallback_reason
                or self._process_generation_disabled_reason
                or self._process_sticky_disabled_reason
                or "process_unavailable"
            )
            batch_accumulator.fallback_reason = reason
            if process_attempted:
                batch_accumulator.backend_status = (
                    f"process_serial_fallback_{reason}"
                )
            elif self._process_sticky_disabled_reason is not None:
                batch_accumulator.backend_status = (
                    "process_disabled_shadow_mismatch"
                )
                batch_accumulator.shadow_parity_status = "fail"
            else:
                batch_accumulator.backend_status = "process_degraded"
            chosen_profiles = serial_profiles
        elif self._process_shadow_pending:
            shadow_accumulator = _RoadBatchAccumulator()
            shadow_started_s = time.perf_counter()
            serial_profiles = self._serial_profiles_from_prepared(
                prepared_by_index,
                batch_accumulator=shadow_accumulator,
            )
            shadow_duration_s = time.perf_counter() - shadow_started_s
            batch_accumulator.shadow_serial_s += shadow_duration_s
            batch_accumulator.shadow_serial_query_count += (
                shadow_accumulator.map_query_count
            )
            parity_ok = all(
                self._profiles_semantically_equal(
                    process_profiles[plan_index],
                    serial_profiles[plan_index],
                )
                and self._envelopes_semantically_equal(
                    self._envelope_for_profile(
                        profile=process_profiles[plan_index],
                        plan=ordered_plans[plan_index],
                        tick_facts=tick_facts,
                    ),
                    self._envelope_for_profile(
                        profile=serial_profiles[plan_index],
                        plan=ordered_plans[plan_index],
                        tick_facts=tick_facts,
                    ),
                )
                for plan_index in prepared_by_index
            )
            batch_accumulator.shadow_parity_s += (
                time.perf_counter() - shadow_started_s
            )
            self._process_shadow_pending = False
            if parity_ok:
                batch_accumulator.shadow_parity_status = "pass"
                batch_accumulator.backend_status = "process_shadow_pass"
                chosen_profiles = process_profiles
            else:
                batch_accumulator.shadow_parity_status = "fail"
                batch_accumulator.fallback_reason = (
                    "shadow_parity_mismatch"
                )
                batch_accumulator.serial_fallback_s += shadow_duration_s
                batch_accumulator.backend_status = (
                    "process_disabled_shadow_mismatch"
                )
                self._process_sticky_disabled_reason = (
                    "shadow_parity_mismatch"
                )
                self._close_process_backend()
                chosen_profiles = serial_profiles
        else:
            batch_accumulator.backend_status = (
                "process_degraded"
                if batch_accumulator.error_count
                else "process"
            )
            chosen_profiles = process_profiles

        for plan_index, profile in chosen_profiles.items():
            profiles[plan_index] = profile
            self._cache_timed_road_profile(profile)
        return profiles, errors

    def _finish_road_batch_stats(
        self,
        *,
        batch_accumulator: _RoadBatchAccumulator,
        batch_started_s: float,
        plan_count: int,
    ) -> None:
        backend_status = batch_accumulator.backend_status
        if backend_status is None:
            backend_status = (
                "serial_degraded"
                if batch_accumulator.error_count
                else "serial"
            )
        chunk_count = batch_accumulator.chunk_count
        if (
            chunk_count == 0
            and plan_count
            and backend_status.startswith("serial")
        ):
            chunk_count = 1
        self._last_road_batch_stats = RoadAssessmentBatchStats(
            backend_status=backend_status,
            road_batch_wall_ms=(
                (time.perf_counter() - batch_started_s) * 1000.0
            ),
            validation_ms=batch_accumulator.validation_s * 1000.0,
            densify_heading_ms=(
                batch_accumulator.densify_heading_s * 1000.0
            ),
            map_query_ms=batch_accumulator.map_query_s * 1000.0,
            aggregate_ms=batch_accumulator.aggregate_s * 1000.0,
            plan_count=int(plan_count),
            pose_count=int(batch_accumulator.pose_count),
            map_query_count=int(batch_accumulator.map_query_count),
            worker_count=int(batch_accumulator.worker_count),
            chunk_count=int(chunk_count),
            profile_cache_hits=int(batch_accumulator.profile_cache_hits),
            profile_cache_misses=int(batch_accumulator.profile_cache_misses),
            error_count=int(batch_accumulator.error_count),
            shadow_parity_status=batch_accumulator.shadow_parity_status,
            fallback_reason=batch_accumulator.fallback_reason,
            serial_fallback_ms=(
                batch_accumulator.serial_fallback_s * 1000.0
            ),
            worker_query_sum_ms=(
                batch_accumulator.worker_query_sum_s * 1000.0
            ),
            commissioning_batch=bool(
                batch_accumulator.commissioning_batch
            ),
            query_deadline_ms=batch_accumulator.query_deadline_ms,
            process_attempt_ms=(
                batch_accumulator.process_attempt_s * 1000.0
            ),
            shadow_serial_ms=(
                batch_accumulator.shadow_serial_s * 1000.0
            ),
            shadow_serial_query_count=int(
                batch_accumulator.shadow_serial_query_count
            ),
            shadow_parity_ms=(
                batch_accumulator.shadow_parity_s * 1000.0
            ),
        )

    def assess_plans_road(
        self,
        *,
        tick_context: Any,
        plans: Any,
    ) -> tuple[RoadExecutionEnvelope, ...]:
        """Assess an ordered plan batch with one shared exact ego-footprint query."""

        batch_started_s = time.perf_counter()
        batch_accumulator = _RoadBatchAccumulator()
        if self._road_assessment_backend == "process":
            batch_accumulator.worker_count = self._process_worker_count
        ordered_plans = tuple(plans)
        if not ordered_plans:
            self._finish_road_batch_stats(
                batch_accumulator=batch_accumulator,
                batch_started_s=batch_started_s,
                plan_count=0,
            )
            return ()

        try:
            tick_facts = self._road_tick_facts(
                tick_context=tick_context,
                batch_accumulator=batch_accumulator,
            )
        except Exception as exc:
            batch_accumulator.error_count += len(ordered_plans)
            envelopes = tuple(
                self._unknown_envelope(
                    f"invalid_ego_snapshot:{type(exc).__name__}"
                )
                for _plan in ordered_plans
            )
            self._finish_road_batch_stats(
                batch_accumulator=batch_accumulator,
                batch_started_s=batch_started_s,
                plan_count=len(ordered_plans),
            )
            return envelopes

        envelopes = []
        if self._road_assessment_backend == "process":
            profiles, profile_errors = self._process_profiles_for_plans(
                ordered_plans=ordered_plans,
                tick_facts=tick_facts,
                batch_accumulator=batch_accumulator,
            )
            for plan_index, plan in enumerate(ordered_plans):
                profile = profiles[plan_index]
                if profile is None:
                    batch_accumulator.error_count += 1
                    envelope = self._unknown_envelope(
                        profile_errors.get(
                            plan_index,
                            "process_profile_unavailable",
                        ),
                        current_road=tick_facts.current_ego_road,
                    )
                else:
                    try:
                        envelope = self._envelope_for_profile(
                            profile=profile,
                            plan=plan,
                            tick_facts=tick_facts,
                            batch_accumulator=batch_accumulator,
                        )
                    except Exception as exc:
                        batch_accumulator.error_count += 1
                        envelope = self._unknown_envelope(
                            f"invalid_world_path:{type(exc).__name__}",
                            current_road=tick_facts.current_ego_road,
                        )
                envelopes.append(envelope)
        else:
            for plan in ordered_plans:
                try:
                    profile = self._profile_for_plan(
                        plan,
                        tick_facts.ego,
                        tick_facts=tick_facts,
                        batch_accumulator=batch_accumulator,
                    )
                    envelope = self._envelope_for_profile(
                        profile=profile,
                        plan=plan,
                        tick_facts=tick_facts,
                        batch_accumulator=batch_accumulator,
                    )
                except _PlanIdGeometryMismatch:
                    batch_accumulator.error_count += 1
                    envelope = self._unknown_envelope(
                        "plan_id_geometry_mismatch",
                        current_road=tick_facts.current_ego_road,
                    )
                except Exception as exc:
                    batch_accumulator.error_count += 1
                    envelope = self._unknown_envelope(
                        f"invalid_world_path:{type(exc).__name__}",
                        current_road=tick_facts.current_ego_road,
                    )
                envelopes.append(envelope)

        self._finish_road_batch_stats(
            batch_accumulator=batch_accumulator,
            batch_started_s=batch_started_s,
            plan_count=len(ordered_plans),
        )
        return tuple(envelopes)

    def assess_plan_road(
        self,
        *,
        tick_context: Any,
        plan: Any,
    ) -> RoadExecutionEnvelope:
        """Return the timed road envelope without querying dynamic actors."""

        return self.assess_plans_road(
            tick_context=tick_context,
            plans=(plan,),
        )[0]

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
    "RoadAssessmentBatchStats",
    "RoadRecoveryMode",
    "RoadExecutionEnvelope",
    "RoadTickFacts",
    "StoppingReserveProfile",
    "StoppingReserveStatus",
    "decide_plan_admission",
    "raw_physical_stopping_speed_cap_mps",
    "road_stopping_speed_cap_mps",
]

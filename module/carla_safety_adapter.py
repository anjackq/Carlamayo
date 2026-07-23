"""CARLA ground-truth adapter for the pure stop-only safety shield.

This is an integration safety layer, not an onboard perception claim.  Every
pose, velocity, lane query, and actor footprint comes from the exact CARLA
snapshot associated with the current control tick.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
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
    REJECT_RETAIN_ACTIVE = "REJECT_RETAIN_ACTIVE"
    REJECT_FALLBACK_STOP = "REJECT_FALLBACK_STOP"


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

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "current_ego_road": self.current_ego_road.to_json_dict(),
            "near_term_path_road": self.near_term_path_road.to_json_dict(),
            "full_path_road": self.full_path_road.to_json_dict(),
            "last_safe_waypoint_index": self.last_safe_waypoint_index,
            "time_to_first_bad_s": self.time_to_first_bad_s,
            "distance_to_first_bad_m": self.distance_to_first_bad_m,
            "target_speed_cap_mps": self.target_speed_cap_mps,
            "emergency_required": bool(self.emergency_required),
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


def decide_plan_admission(
    candidate: RoadExecutionEnvelope,
    active: RoadExecutionEnvelope | None = None,
) -> PlanAdmissionStatus:
    """Choose a plan handoff without letting an unsafe prefix replace a safe plan."""

    candidate_admissible = (
        candidate.current_ego_road.status is AssessmentStatus.SAFE
        and candidate.near_term_path_road.status is AssessmentStatus.SAFE
        and candidate.last_safe_waypoint_index is not None
        and not candidate.emergency_required
    )
    if candidate_admissible:
        if candidate.full_path_road.status is AssessmentStatus.SAFE:
            return PlanAdmissionStatus.ACCEPT_FULLY_SAFE
        return PlanAdmissionStatus.ACCEPT_SAFE_PREFIX

    active_executable = (
        active is not None
        and active.current_ego_road.status is AssessmentStatus.SAFE
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
    """Maximum speed that preserves reaction and braking distance before a bad point."""

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
    return min(speed_cap, float(cfg.TRAJECTORY_MAX_SPEED_MPS))


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
        return assess_road_containment(samples, quality=profile.quality)

    @staticmethod
    def _first_bad_profile_index(
        profile: _TimedRoadProfile,
        remaining_indices: np.ndarray,
    ) -> int | None:
        for index in remaining_indices:
            pose = assess_road_containment(
                profile.samples_by_pose[int(index)],
                quality=profile.quality,
            )
            if pose.status is not AssessmentStatus.SAFE:
                return int(index)
        return None

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
        )

    def _road_execution_envelope(
        self,
        *,
        profile: _TimedRoadProfile,
        plan: Any,
        ego: EgoKinematics,
        current_time_s: float,
        current_road: RoadContainmentAssessment,
    ) -> RoadExecutionEnvelope:
        remaining_indices = np.flatnonzero(
            profile.times_s > float(current_time_s) + float(cfg.TRAJECTORY_TIME_EPSILON_S)
        )
        near_term_indices = remaining_indices[
            profile.times_s[remaining_indices]
            <= float(current_time_s) + float(cfg.SAFETY_EXECUTION_HORIZON_S)
        ]
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
        if len(remaining_indices) == 0:
            return RoadExecutionEnvelope(
                current_ego_road=current_road,
                near_term_path_road=near_term,
                full_path_road=full_path,
                last_safe_waypoint_index=None,
                time_to_first_bad_s=None,
                distance_to_first_bad_m=None,
                target_speed_cap_mps=0.0,
                emergency_required=True,
            )

        first_bad = self._first_bad_profile_index(profile, remaining_indices)
        if first_bad is None:
            return RoadExecutionEnvelope(
                current_ego_road=current_road,
                near_term_path_road=near_term,
                full_path_road=full_path,
                last_safe_waypoint_index=int(len(plan.world_points) - 1),
                time_to_first_bad_s=None,
                distance_to_first_bad_m=None,
                target_speed_cap_mps=None,
                emergency_required=False,
            )

        first_remaining = int(remaining_indices[0])
        distance_to_path = float(
            np.linalg.norm(profile.points[first_remaining, :2] - np.asarray(ego.center_xy))
        )
        distance_to_bad = max(
            0.0,
            distance_to_path
            + float(
                profile.cumulative_distance_m[first_bad]
                - profile.cumulative_distance_m[first_remaining]
            ),
        )
        time_to_bad = max(
            0.0,
            float(profile.times_s[first_bad]) - float(current_time_s),
        )
        first_bad_upper_index = int(profile.upper_waypoint_indices[first_bad])
        last_safe_index = first_bad_upper_index - 1
        first_future_waypoint = int(
            np.searchsorted(
                np.asarray(plan.waypoint_times_s, dtype=np.float64),
                float(current_time_s),
                side="right",
            )
        )
        if last_safe_index < first_future_waypoint:
            last_safe_index = None

        speed_cap = road_stopping_speed_cap_mps(distance_to_bad, self.policy)
        current_speed = float(math.hypot(*ego.velocity_xy))
        emergency_required = (
            current_speed
            > speed_cap + float(cfg.SAFETY_SPEED_CAP_EPSILON_MPS)
        )
        return RoadExecutionEnvelope(
            current_ego_road=current_road,
            near_term_path_road=near_term,
            full_path_road=full_path,
            last_safe_waypoint_index=last_safe_index,
            time_to_first_bad_s=time_to_bad,
            distance_to_first_bad_m=distance_to_bad,
            target_speed_cap_mps=speed_cap,
            emergency_required=emergency_required,
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
        current_road = assess_road_containment(
            self._query_footprint(
                center_xyz=np.array([*ego.center_xy, ego_center_z], dtype=np.float64),
                yaw_rad=ego.yaw_rad,
                half_length_m=ego.half_length_m,
                half_width_m=ego.half_width_m,
                sample_index_start=0,
            ),
            quality="carla_ground_truth_current_ego",
        )
        try:
            profile = self._profile_for_plan(plan, ego)
            return self._road_execution_envelope(
                profile=profile,
                plan=plan,
                ego=ego,
                current_time_s=float(tick_context.simulation_time_s),
                current_road=current_road,
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
    "RoadExecutionEnvelope",
    "decide_plan_admission",
    "road_stopping_speed_cap_mps",
]

"""CARLA ground-truth adapter for the pure stop-only safety shield.

This is an integration safety layer, not an onboard perception claim.  Every
pose, velocity, lane query, and actor footprint comes from the exact CARLA
snapshot associated with the current control tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


class CarlaGroundTruthSafetyAdapter:
    """Translate exact CARLA snapshots into pure shield assessments."""

    def __init__(self, world: Any, ego_vehicle: Any, policy: SafetyPolicy | None = None):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.policy = policy or SafetyPolicy()
        self._map = world.get_map()
        self._cached_plan_id: str | None = None
        self._cached_plan_road: RoadContainmentAssessment | None = None

    def reset(self, ego_vehicle: Any | None = None) -> None:
        if ego_vehicle is not None:
            self.ego_vehicle = ego_vehicle
        self._cached_plan_id = None
        self._cached_plan_road = None

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

        if self._cached_plan_id != str(plan.plan_id):
            self._cached_plan_id = str(plan.plan_id)
            self._cached_plan_road = self._assess_path_road(plan.world_points, ego)

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
        road = self._combine_road(self._cached_plan_road, current_road)

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
        return CarlaSafetyAssessment(road=road, obstacles=obstacles)


__all__ = ["CarlaGroundTruthSafetyAdapter", "CarlaSafetyAssessment"]

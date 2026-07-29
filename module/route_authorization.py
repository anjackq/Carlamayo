"""Pure strict-route authorization for timestamped trajectory proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from .route_navigation import (
    ROUTE_ASSOCIATION_MAX_DISTANCE_M,
    RoutePlan,
    associate_route_index,
)


ROUTE_EXECUTION_HORIZON_S = 1.5
ROUTE_STOP_BUFFER_M = 1.0
ROUTE_REACTION_TIME_S = 0.35
ROUTE_ASSUMED_DECELERATION_MPS2 = 4.0


class RouteStatus(str, Enum):
    MATCH = "MATCH"
    DEVIATE = "DEVIATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CandidateLaneFact:
    """Exact-map lane identity at one candidate waypoint."""

    road_id: int | None
    section_id: int | None
    lane_id: int | None
    is_junction: bool | None

    @property
    def available(self) -> bool:
        return (
            self.road_id is not None
            and self.section_id is not None
            and self.lane_id is not None
            and self.is_junction is not None
        )

    @property
    def identity(self) -> tuple[int, int, int] | None:
        if not self.available:
            return None
        return int(self.road_id), int(self.section_id), int(self.lane_id)


@dataclass(frozen=True)
class RouteCandidateAssessment:
    current_route_status: RouteStatus
    near_term_route_status: RouteStatus
    full_path_route_status: RouteStatus
    last_authorized_waypoint_index: int | None
    first_deviation_waypoint_index: int | None
    time_to_first_deviation_s: float | None
    distance_to_first_deviation_m: float | None
    route_speed_cap_mps: float | None
    route_progress_m: float
    maximum_cross_track_error_m: float
    lane_change_detected: bool
    branch_match: bool
    reason_codes: tuple[str, ...]
    associated_route_indices: tuple[int | None, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "current_route_status": self.current_route_status.value,
            "near_term_route_status": self.near_term_route_status.value,
            "full_path_route_status": self.full_path_route_status.value,
            "last_authorized_waypoint_index": self.last_authorized_waypoint_index,
            "first_deviation_waypoint_index": self.first_deviation_waypoint_index,
            "time_to_first_deviation_s": self.time_to_first_deviation_s,
            "distance_to_first_deviation_m": self.distance_to_first_deviation_m,
            "route_speed_cap_mps": self.route_speed_cap_mps,
            "route_progress_m": float(self.route_progress_m),
            "maximum_cross_track_error_m": float(self.maximum_cross_track_error_m),
            "lane_change_detected": bool(self.lane_change_detected),
            "branch_match": bool(self.branch_match),
            "reason_codes": list(self.reason_codes),
            "associated_route_indices": list(self.associated_route_indices),
        }


def route_stopping_speed_cap_mps(distance_to_deviation_m: float) -> float:
    usable = max(0.0, float(distance_to_deviation_m) - ROUTE_STOP_BUFFER_M)
    deceleration = ROUTE_ASSUMED_DECELERATION_MPS2
    reaction = ROUTE_REACTION_TIME_S
    return max(
        0.0,
        -deceleration * reaction
        + math.sqrt((deceleration * reaction) ** 2 + 2.0 * deceleration * usable),
    )


def combine_execution_constraints(
    *,
    road_speed_cap_mps: float | None,
    route_speed_cap_mps: float | None,
    road_last_authorized_index: int | None,
    route_last_authorized_index: int | None,
) -> tuple[float | None, int | None]:
    """Return the strict intersection of road and route controller authority."""

    caps = [
        float(value)
        for value in (road_speed_cap_mps, route_speed_cap_mps)
        if value is not None
    ]
    indices = [
        int(value)
        for value in (
            road_last_authorized_index,
            route_last_authorized_index,
        )
        if value is not None
    ]
    return (min(caps) if caps else None, min(indices) if indices else None)


def _aggregate_status(statuses: Sequence[RouteStatus]) -> RouteStatus:
    if any(status is RouteStatus.UNKNOWN for status in statuses):
        return RouteStatus.UNKNOWN
    if any(status is RouteStatus.DEVIATE for status in statuses):
        return RouteStatus.DEVIATE
    return RouteStatus.MATCH


def _associate_monotonic(
    point_xy: np.ndarray,
    start_index: int,
    *,
    route: RoutePlan,
    candidate_identity: tuple[int, int, int] | None,
    candidate_is_junction: bool | None,
) -> tuple[int, float, bool]:
    association = associate_route_index(
        route,
        point_xy,
        start_index=start_index,
        lane_identity=candidate_identity,
        lane_is_junction=candidate_is_junction,
    )
    return (
        association.route_index,
        association.distance_m,
        association.junction_topology_overlap,
    )


def assess_route_candidate(
    *,
    route: RoutePlan,
    current_route_index: int,
    current_route_status: RouteStatus,
    trajectory_world_points: Any,
    waypoint_times_s: Any,
    source_simulation_time_s: float,
    current_simulation_time_s: float | None = None,
    lane_facts: Sequence[CandidateLaneFact],
) -> RouteCandidateAssessment:
    """Assess one trajectory without granting authority from model text/CoC."""

    points = np.asarray(trajectory_world_points, dtype=np.float64)
    times = np.asarray(waypoint_times_s, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or times.shape != (len(points),)
        or len(lane_facts) != len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("route candidate geometry/timing/lane facts are incompatible")
    route_index = min(max(0, int(current_route_index)), len(route.points) - 1)
    cumulative_candidate = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    )
    source_time = float(source_simulation_time_s)
    assessment_time = (
        source_time
        if current_simulation_time_s is None
        else float(current_simulation_time_s)
    )
    first_future = min(
        int(np.searchsorted(times, assessment_time, side="right")),
        len(points) - 1,
    )
    statuses: list[RouteStatus] = [RouteStatus.MATCH] * first_future
    associated: list[int | None] = [None] * first_future
    cross_tracks: list[float] = []
    reasons: set[str] = set()
    first_bad: int | None = None
    lane_change_detected = False

    for index in range(first_future, len(points)):
        point = points[index]
        lane_fact = lane_facts[index]
        if not lane_fact.available:
            status = RouteStatus.UNKNOWN
            associated_index = None
            cross_track = math.inf
            reasons.add("map_query_unavailable")
        else:
            (
                associated_index,
                cross_track,
                junction_topology_overlap,
            ) = _associate_monotonic(
                point[:2],
                route_index,
                route=route,
                candidate_identity=lane_fact.identity,
                candidate_is_junction=lane_fact.is_junction,
            )
            route_index = max(route_index, associated_index)
            route_point = route.points[associated_index]
            route_identity = (
                route_point.road_id,
                route_point.section_id,
                route_point.lane_id,
            )
            if cross_track > ROUTE_ASSOCIATION_MAX_DISTANCE_M:
                status = RouteStatus.DEVIATE
                reasons.add("route_cross_track_exceeded")
            elif junction_topology_overlap:
                status = RouteStatus.MATCH
                reasons.add("junction_topology_overlap_canonicalized")
            elif lane_fact.identity != route_identity:
                status = RouteStatus.DEVIATE
                if lane_fact.road_id == route_point.road_id:
                    lane_change_detected = True
                    reasons.add("unauthorized_lane_change")
                elif bool(lane_fact.is_junction) or route_point.is_junction:
                    reasons.add("unauthorized_junction_branch")
                else:
                    reasons.add("route_lane_identity_mismatch")
            else:
                status = RouteStatus.MATCH
        statuses.append(status)
        associated.append(associated_index)
        cross_tracks.append(cross_track)
        if first_bad is None and status is not RouteStatus.MATCH:
            first_bad = index

    relative_times = times - assessment_time
    near_indices = [
        index
        for index, value in enumerate(relative_times)
        if index >= first_future
        and value <= ROUTE_EXECUTION_HORIZON_S + 1e-9
    ]
    if not near_indices:
        near_indices = [0]
    near_status = _aggregate_status([statuses[index] for index in near_indices])
    full_status = _aggregate_status(statuses[first_future:])
    last_authorized = None if first_bad == first_future else (
        len(points) - 1 if first_bad is None else first_bad - 1
    )
    distance_to_bad = (
        None
        if first_bad is None
        else float(
            cumulative_candidate[first_bad]
            - cumulative_candidate[first_future]
        )
    )
    time_to_bad = (
        None if first_bad is None else max(0.0, float(relative_times[first_bad]))
    )
    finite_cross_tracks = [value for value in cross_tracks if math.isfinite(value)]
    maximum_cross_track = max(finite_cross_tracks, default=math.inf)
    final_associated = next(
        (value for value in reversed(associated) if value is not None),
        current_route_index,
    )
    route_progress = float(
        route.cumulative_distance_m[
            min(max(0, int(final_associated)), len(route.points) - 1)
        ]
    )
    return RouteCandidateAssessment(
        current_route_status=current_route_status,
        near_term_route_status=near_status,
        full_path_route_status=full_status,
        last_authorized_waypoint_index=last_authorized,
        first_deviation_waypoint_index=first_bad,
        time_to_first_deviation_s=time_to_bad,
        distance_to_first_deviation_m=distance_to_bad,
        route_speed_cap_mps=(
            None
            if distance_to_bad is None
            else route_stopping_speed_cap_mps(distance_to_bad)
        ),
        route_progress_m=route_progress,
        maximum_cross_track_error_m=maximum_cross_track,
        lane_change_detected=lane_change_detected,
        branch_match=not any(
            code == "unauthorized_junction_branch" for code in reasons
        ),
        reason_codes=tuple(sorted(reasons)),
        associated_route_indices=tuple(associated),
    )


__all__ = [
    "CandidateLaneFact",
    "RouteCandidateAssessment",
    "RouteStatus",
    "assess_route_candidate",
    "combine_execution_constraints",
    "route_stopping_speed_cap_mps",
]

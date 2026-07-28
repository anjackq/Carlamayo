"""Dependency-light route-derived navigation contracts and tracking.

The runtime-specific CARLA adapter deliberately lives behind helper functions
that accept primitive route points.  Importing this module must not initialize
CARLA, its navigation agents, Torch, or CUDA.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np


ROUTE_ASSOCIATION_MAX_DISTANCE_M = 3.0
ROUTE_ARRIVAL_DISTANCE_M = 1.5
ROUTE_PROMPT_DISTANCE_OMIT_M = 15.0
ROUTE_MANEUVER_COALESCE_DISTANCE_M = 18.0


class NavigationAction(str, Enum):
    FOLLOW_LANE = "FOLLOW_LANE"
    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ARRIVE = "ARRIVE"


class RouteTrackerStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    ARRIVED = "ARRIVED"
    ROUTE_UNAVAILABLE = "ROUTE_UNAVAILABLE"


@dataclass(frozen=True)
class RoutePoint:
    """One immutable point from a CARLA GlobalRoutePlanner trace."""

    xyz: tuple[float, float, float]
    road_id: int
    section_id: int
    lane_id: int
    is_junction: bool
    road_option: str

    def __post_init__(self) -> None:
        xyz = tuple(float(value) for value in self.xyz)
        if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
            raise ValueError("route point xyz must contain three finite values")
        option = str(self.road_option).strip().upper()
        if not option:
            raise ValueError("route point road_option must be nonempty")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "road_id", int(self.road_id))
        object.__setattr__(self, "section_id", int(self.section_id))
        object.__setattr__(self, "lane_id", int(self.lane_id))
        object.__setattr__(self, "is_junction", bool(self.is_junction))
        object.__setattr__(self, "road_option", option)


@dataclass(frozen=True)
class RoutePlan:
    """Validated primitive route and its cumulative arc length."""

    route_id: str
    points: tuple[RoutePoint, ...]
    cumulative_distance_m: tuple[float, ...]
    destination_xyz: tuple[float, float, float]

    @property
    def length_m(self) -> float:
        return float(self.cumulative_distance_m[-1])


@dataclass(frozen=True)
class NavigationContext:
    """Source-time navigation condition attached to an inference request."""

    source: str
    text: str
    weight: float
    conditioning_epoch: int
    route_id: str | None
    maneuver_id: str | None
    action: NavigationAction
    distance_to_maneuver_m: float
    route_progress_m: float
    route_index: int
    target_route_index: int
    lane_change_authorized: bool
    source_frame_id: int
    source_simulation_time_s: float
    tracker_status: RouteTrackerStatus = RouteTrackerStatus.AVAILABLE

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text,
            "weight": float(self.weight),
            "conditioning_epoch": int(self.conditioning_epoch),
            "route_id": self.route_id,
            "maneuver_id": self.maneuver_id,
            "action": self.action.value,
            "distance_to_maneuver_m": float(self.distance_to_maneuver_m),
            "route_progress_m": float(self.route_progress_m),
            "route_index": int(self.route_index),
            "target_route_index": int(self.target_route_index),
            "lane_change_authorized": bool(self.lane_change_authorized),
            "source_frame_id": int(self.source_frame_id),
            "source_simulation_time_s": float(self.source_simulation_time_s),
            "tracker_status": self.tracker_status.value,
        }


@dataclass(frozen=True)
class RouteTrackingUpdate:
    context: NavigationContext
    epoch_changed: bool
    route_distance_m: float


@dataclass(frozen=True)
class RouteAssociation:
    """One monotonic geometric/lane association against an authorized route."""

    route_index: int
    distance_m: float
    lane_identity_matched: bool | None


def associate_route_index(
    route: RoutePlan,
    point_xyz: Sequence[float],
    *,
    start_index: int,
    lane_identity: Sequence[int] | None = None,
) -> RouteAssociation:
    """Associate a point while preserving an exact lane through transitions.

    CARLA's OpenDRIVE lane boundary and the one-metre GRP samples do not switch
    road identity at exactly the same coordinate.  A purely nearest-point
    association can therefore jump to the junction connector while the exact
    map query still reports the incoming lane.  Prefer the nearest matching
    identity anywhere inside the existing 3 m route corridor; identities not
    present in that corridor remain unauthorized.
    """

    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape not in {(2,), (3,)} or not np.isfinite(point).all():
        raise ValueError("route association point must contain two or three finite values")
    start = min(max(0, int(start_index)), len(route.points) - 1)
    route_xy = np.asarray(
        [route_point.xyz[:2] for route_point in route.points],
        dtype=np.float64,
    )
    distances = np.linalg.norm(route_xy[start:] - point[None, :2], axis=1)
    geometric_relative = int(np.argmin(distances))
    geometric_index = start + geometric_relative
    geometric_distance = float(distances[geometric_relative])

    if lane_identity is None:
        return RouteAssociation(
            route_index=geometric_index,
            distance_m=geometric_distance,
            lane_identity_matched=None,
        )
    identity_values = tuple(int(value) for value in lane_identity)
    if len(identity_values) != 3:
        raise ValueError("lane_identity must contain road, section, and lane")
    matching_indices = [
        index
        for index in range(start, len(route.points))
        if (
            route.points[index].road_id,
            route.points[index].section_id,
            route.points[index].lane_id,
        )
        == identity_values
    ]
    if matching_indices:
        identity_index = min(
            matching_indices,
            key=lambda index: (float(np.linalg.norm(route_xy[index] - point[:2])), index),
        )
        identity_distance = float(np.linalg.norm(route_xy[identity_index] - point[:2]))
        if identity_distance <= ROUTE_ASSOCIATION_MAX_DISTANCE_M:
            return RouteAssociation(
                route_index=identity_index,
                distance_m=identity_distance,
                lane_identity_matched=True,
            )
    return RouteAssociation(
        route_index=geometric_index,
        distance_m=geometric_distance,
        lane_identity_matched=False,
    )


def _route_option_action(option: str) -> NavigationAction | None:
    option = str(option).strip().upper()
    return {
        "STRAIGHT": NavigationAction.STRAIGHT,
        "LEFT": NavigationAction.LEFT,
        "RIGHT": NavigationAction.RIGHT,
    }.get(option)


def _route_digest(points: Sequence[RoutePoint]) -> str:
    digest = hashlib.sha256()
    for point in points:
        digest.update(
            (
                f"{point.xyz[0]:.6f},{point.xyz[1]:.6f},{point.xyz[2]:.6f};"
                f"{point.road_id},{point.section_id},{point.lane_id};"
                f"{int(point.is_junction)};{point.road_option}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()[:16]


def build_route_plan(points: Iterable[RoutePoint]) -> RoutePlan:
    """Validate and freeze one primitive route trace."""

    frozen = tuple(points)
    if len(frozen) < 2:
        raise ValueError("route must contain at least two points")
    if any(
        point.road_option in {"CHANGELANELEFT", "CHANGELANERIGHT"}
        for point in frozen
    ):
        raise ValueError("route contains an unauthorized lane-change edge")
    xyz = np.asarray([point.xyz for point in frozen], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    # CARLA GRP intentionally repeats a waypoint at RoadOption boundaries
    # (LANEFOLLOW -> STRAIGHT/RIGHT).  Preserve those zero-length transition
    # markers; dropping them would erase maneuver truth.
    if (
        not np.isfinite(segment_lengths).all()
        or np.any(segment_lengths < 0.0)
        or float(np.sum(segment_lengths)) <= 1e-6
    ):
        raise ValueError("route has no finite forward extent")
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return RoutePlan(
        route_id=_route_digest(frozen),
        points=frozen,
        cumulative_distance_m=tuple(float(value) for value in cumulative),
        destination_xyz=frozen[-1].xyz,
    )


def quantize_prompt_distance(distance_m: float) -> int | None:
    """Return the model-facing distance bucket, or omit at close range."""

    distance = max(0.0, float(distance_m))
    if distance <= ROUTE_PROMPT_DISTANCE_OMIT_M:
        return None
    return max(10, int(math.floor(distance / 10.0 + 0.5)) * 10)


def format_navigation_prompt(
    action: NavigationAction,
    distance_to_maneuver_m: float,
) -> str:
    """Format one route-provable maneuver and no unsupported scene facts."""

    if action is NavigationAction.ARRIVE:
        return "Stop at the destination."
    bucket = quantize_prompt_distance(distance_to_maneuver_m)
    suffix = "" if bucket is None else f" in {bucket}m"
    if action is NavigationAction.STRAIGHT:
        return f"Continue straight at the next junction{suffix}."
    if action is NavigationAction.LEFT:
        return f"Turn left at the next junction{suffix}."
    if action is NavigationAction.RIGHT:
        return f"Turn right at the next junction{suffix}."
    return (
        "Continue in the current lane."
        if bucket is None
        else f"Continue in the current lane for {bucket}m."
    )


def next_maneuver(
    route: RoutePlan,
    route_index: int,
) -> tuple[NavigationAction, int, str]:
    """Return the next maneuver after the monotonic route association."""

    start = min(max(0, int(route_index)), len(route.points) - 1)
    remaining = route.length_m - route.cumulative_distance_m[start]
    if remaining <= ROUTE_ARRIVAL_DISTANCE_M:
        return NavigationAction.ARRIVE, len(route.points) - 1, f"{route.route_id}:arrive"
    def _coalesce_straight_entry(
        action: NavigationAction,
        action_index: int,
    ) -> tuple[NavigationAction, int]:
        if action is not NavigationAction.STRAIGHT:
            return action, action_index
        run_option = route.points[action_index].road_option
        scan = action_index + 1
        while (
            scan < len(route.points)
            and route.points[scan].road_option == run_option
        ):
            scan += 1
        while scan < len(route.points):
            next_action = _route_option_action(
                route.points[scan].road_option
            )
            if next_action in {
                NavigationAction.LEFT,
                NavigationAction.RIGHT,
            }:
                separation = (
                    route.cumulative_distance_m[scan]
                    - route.cumulative_distance_m[action_index]
                )
                if separation <= ROUTE_MANEUVER_COALESCE_DISTANCE_M:
                    return next_action, scan
                break
            scan += 1
        return action, action_index

    active_action = _route_option_action(route.points[start].road_option)
    if active_action is not None:
        run_start = start
        while (
            run_start > 0
            and route.points[run_start - 1].road_option
            == route.points[start].road_option
        ):
            run_start -= 1
        selected_action, selected_index = _coalesce_straight_entry(
            active_action,
            run_start,
        )
        return (
            selected_action,
            selected_index,
            f"{route.route_id}:{selected_action.value}:"
            f"{selected_index if selected_index != run_start else run_start}",
        )
    for index in range(start, len(route.points)):
        action = _route_option_action(route.points[index].road_option)
        if action is not None:
            # Contiguous GRP points often repeat one RoadOption.  A maneuver ID
            # identifies the first point in that run.
            if (
                index > start
                and route.points[index - 1].road_option
                == route.points[index].road_option
            ):
                continue
            selected_action, selected_index = _coalesce_straight_entry(
                action,
                index,
            )
            return (
                selected_action,
                selected_index,
                f"{route.route_id}:{selected_action.value}:{selected_index}",
            )
    return (
        NavigationAction.FOLLOW_LANE,
        len(route.points) - 1,
        f"{route.route_id}:follow:{len(route.points) - 1}",
    )


class RouteNavigationTracker:
    """Monotonic route association and navigation-context epoch management."""

    def __init__(self, route: RoutePlan, *, weight: float = 1.0, epoch: int = 0):
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError("navigation weight must be finite and nonnegative")
        if int(epoch) < 0:
            raise ValueError("conditioning epoch must be nonnegative")
        self.route = route
        self.weight = float(weight)
        self.conditioning_epoch = int(epoch)
        self.route_index = 0
        self._maneuver_id: str | None = None
        self._last_context: NavigationContext | None = None

    @property
    def last_context(self) -> NavigationContext | None:
        return self._last_context

    def replace_route(self, route: RoutePlan) -> None:
        """Install a deterministic replan and invalidate prior conditioning."""

        self.route = route
        self.route_index = 0
        self._maneuver_id = None
        self._last_context = None
        self.conditioning_epoch += 1

    def update(
        self,
        ego_xyz: Sequence[float],
        *,
        source_frame_id: int,
        source_simulation_time_s: float,
        ego_lane_identity: Sequence[int] | None = None,
        require_lane_identity: bool = False,
    ) -> RouteTrackingUpdate:
        ego = np.asarray(ego_xyz, dtype=np.float64)
        if ego.shape != (3,) or not np.isfinite(ego).all():
            raise ValueError("ego_xyz must contain three finite values")
        association = associate_route_index(
            self.route,
            ego,
            start_index=self.route_index,
            lane_identity=ego_lane_identity,
        )
        associated_index = association.route_index
        route_distance = association.distance_m
        identity_unavailable = require_lane_identity and ego_lane_identity is None
        identity_mismatch = (
            require_lane_identity
            and association.lane_identity_matched is not True
        )

        if (
            route_distance > ROUTE_ASSOCIATION_MAX_DISTANCE_M
            or identity_unavailable
            or identity_mismatch
        ):
            context = NavigationContext(
                source="route",
                text="",
                weight=self.weight,
                conditioning_epoch=self.conditioning_epoch,
                route_id=self.route.route_id,
                maneuver_id=self._maneuver_id,
                action=NavigationAction.FOLLOW_LANE,
                distance_to_maneuver_m=0.0,
                route_progress_m=float(
                    self.route.cumulative_distance_m[self.route_index]
                ),
                route_index=self.route_index,
                target_route_index=self.route_index,
                lane_change_authorized=False,
                source_frame_id=int(source_frame_id),
                source_simulation_time_s=float(source_simulation_time_s),
                tracker_status=RouteTrackerStatus.ROUTE_UNAVAILABLE,
            )
            self._last_context = context
            return RouteTrackingUpdate(
                context=context,
                epoch_changed=False,
                route_distance_m=route_distance,
            )

        self.route_index = max(self.route_index, associated_index)
        action, target_index, maneuver_id = next_maneuver(self.route, self.route_index)
        epoch_changed = self._maneuver_id is not None and maneuver_id != self._maneuver_id
        if epoch_changed:
            self.conditioning_epoch += 1
        self._maneuver_id = maneuver_id
        route_progress = float(self.route.cumulative_distance_m[self.route_index])
        distance_to_maneuver = max(
            0.0,
            float(self.route.cumulative_distance_m[target_index]) - route_progress,
        )
        status = (
            RouteTrackerStatus.ARRIVED
            if action is NavigationAction.ARRIVE
            else RouteTrackerStatus.AVAILABLE
        )
        context = NavigationContext(
            source="route",
            text=format_navigation_prompt(action, distance_to_maneuver),
            weight=self.weight,
            conditioning_epoch=self.conditioning_epoch,
            route_id=self.route.route_id,
            maneuver_id=maneuver_id,
            action=action,
            distance_to_maneuver_m=distance_to_maneuver,
            route_progress_m=route_progress,
            route_index=self.route_index,
            target_route_index=target_index,
            lane_change_authorized=False,
            source_frame_id=int(source_frame_id),
            source_simulation_time_s=float(source_simulation_time_s),
            tracker_status=status,
        )
        self._last_context = context
        return RouteTrackingUpdate(
            context=context,
            epoch_changed=epoch_changed,
            route_distance_m=route_distance,
        )


__all__ = [
    "NavigationAction",
    "NavigationContext",
    "ROUTE_ARRIVAL_DISTANCE_M",
    "ROUTE_ASSOCIATION_MAX_DISTANCE_M",
    "ROUTE_MANEUVER_COALESCE_DISTANCE_M",
    "RouteAssociation",
    "RouteNavigationTracker",
    "RoutePlan",
    "RoutePoint",
    "RouteTrackerStatus",
    "RouteTrackingUpdate",
    "associate_route_index",
    "build_route_plan",
    "format_navigation_prompt",
    "next_maneuver",
    "quantize_prompt_distance",
]

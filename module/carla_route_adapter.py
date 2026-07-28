"""Lazy CARLA GlobalRoutePlanner adapter for route-derived navigation."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .route_navigation import RoutePlan, RoutePoint, build_route_plan
from .route_authorization import CandidateLaneFact, RouteStatus


FIXED_TOWN03_ORIGIN_XYZ = (
    -5.532081127166748,
    -79.03245544433594,
    0.0,
)
FIXED_TOWN03_DESTINATION_XYZ = (
    -43.350975036621094,
    -2.8402605056762695,
    0.0,
)
FIXED_TOWN03_ORIGIN_LANE = (28, 0, 3)
FIXED_TOWN03_DESTINATION_LANE = (61, 0, -1)
FIXED_TOWN03_ROUTE_LENGTH_M = 97.075309
FIXED_TOWN03_ROUTE_LENGTH_TOLERANCE_M = 0.25
ROUTE_PROJECTION_TOLERANCE_M = 0.25


class CarlaRouteError(RuntimeError):
    """Route planning is unavailable or violates the runtime contract."""


def parse_destination_xyz(raw: str | Sequence[float] | None) -> tuple[float, float, float]:
    if raw is None:
        raise ValueError("route destination is required")
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",")]
    else:
        pieces = list(raw)
    if len(pieces) != 3:
        raise ValueError("route destination must use X,Y,Z")
    try:
        xyz = tuple(float(value) for value in pieces)
    except (TypeError, ValueError) as exc:
        raise ValueError("route destination must use finite numeric X,Y,Z") from exc
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError("route destination must use finite numeric X,Y,Z")
    return xyz


def resolve_carla_python_api_path(
    explicit_path: str | os.PathLike[str] | None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve the directory that directly contains the ``agents`` package."""

    environment = os.environ if environ is None else environ
    candidates = [
        explicit_path,
        environment.get("CARLAMAYO_CARLA_PYTHONAPI"),
    ]
    carla_root = environment.get("CARLAMAYO_CARLA_ROOT")
    if carla_root:
        candidates.append(Path(carla_root) / "PythonAPI" / "carla")
    attempted: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        base = Path(candidate).expanduser().resolve()
        variants = (
            base,
            base / "carla",
            base / "PythonAPI" / "carla",
        )
        for variant in variants:
            marker = variant / "agents" / "navigation" / "global_route_planner.py"
            attempted.append(str(variant))
            if marker.is_file():
                return variant
    detail = ", ".join(attempted) if attempted else "(no paths configured)"
    raise CarlaRouteError(
        "CARLA GlobalRoutePlanner is unavailable; searched: " + detail
    )


def load_global_route_planner(
    python_api_path: str | os.PathLike[str],
):
    """Import CARLA's official planner only after its path is validated."""

    path = resolve_carla_python_api_path(python_api_path, environ={})
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except Exception as exc:  # pragma: no cover - depends on external CARLA tree
        raise CarlaRouteError(
            f"failed to import CARLA GlobalRoutePlanner from {path}"
        ) from exc
    return GlobalRoutePlanner


def _location_distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _lane_identity(waypoint: Any) -> tuple[int, int, int]:
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
    )


def trace_carla_route(
    world_map: Any,
    *,
    origin_xyz: Sequence[float],
    destination_xyz: Sequence[float],
    planner_type: Any,
    carla_module: Any,
    sampling_resolution_m: float = 1.0,
) -> tuple[RoutePlan, dict[str, Any]]:
    """Project endpoints, run official GRP, and return primitive route facts."""

    origin = parse_destination_xyz(origin_xyz)
    destination = parse_destination_xyz(destination_xyz)
    if not math.isfinite(float(sampling_resolution_m)) or sampling_resolution_m <= 0:
        raise CarlaRouteError("route sampling resolution must be finite and positive")
    origin_location = carla_module.Location(*origin)
    destination_location = carla_module.Location(*destination)
    try:
        lane_type = carla_module.LaneType.Driving
        origin_waypoint = world_map.get_waypoint(
            origin_location,
            project_to_road=True,
            lane_type=lane_type,
        )
        destination_waypoint = world_map.get_waypoint(
            destination_location,
            project_to_road=True,
            lane_type=lane_type,
        )
    except Exception as exc:
        raise CarlaRouteError("failed to project route endpoint to Driving lane") from exc
    if origin_waypoint is None or destination_waypoint is None:
        raise CarlaRouteError("route endpoint does not project to a Driving lane")
    destination_error = _location_distance(
        destination_location,
        destination_waypoint.transform.location,
    )
    if destination_error > ROUTE_PROJECTION_TOLERANCE_M:
        raise CarlaRouteError(
            "route destination projection exceeds "
            f"{ROUTE_PROJECTION_TOLERANCE_M:.2f}m: {destination_error:.3f}m"
        )
    try:
        planner = planner_type(world_map, float(sampling_resolution_m))
        traced = planner.trace_route(
            origin_waypoint.transform.location,
            destination_waypoint.transform.location,
        )
    except Exception as exc:
        raise CarlaRouteError("CARLA GlobalRoutePlanner failed to trace route") from exc
    if not traced:
        raise CarlaRouteError("CARLA GlobalRoutePlanner returned an empty route")
    points = []
    for waypoint, road_option in traced:
        option_name = str(getattr(road_option, "name", road_option)).upper()
        location = waypoint.transform.location
        # Junction topology waypoints returned by GRP may use synthetic lane
        # identities that ``Map.get_waypoint(project_to_road=False)`` does not
        # echo at the exact same coordinate.  Canonicalize the route trace to
        # the same exact-query semantics used for candidate authorization.
        try:
            query_waypoint = world_map.get_waypoint(
                location,
                project_to_road=False,
                lane_type=carla_module.LaneType.Driving,
            )
        except Exception:
            query_waypoint = None
        identity_waypoint = query_waypoint or waypoint
        points.append(
            RoutePoint(
                xyz=(float(location.x), float(location.y), float(location.z)),
                road_id=int(identity_waypoint.road_id),
                section_id=int(identity_waypoint.section_id),
                lane_id=int(identity_waypoint.lane_id),
                is_junction=bool(identity_waypoint.is_junction),
                road_option=option_name,
            )
        )
    try:
        route = build_route_plan(points)
    except ValueError as exc:
        raise CarlaRouteError(str(exc)) from exc
    return route, {
        "origin_projection_error_m": _location_distance(
            origin_location,
            origin_waypoint.transform.location,
        ),
        "destination_projection_error_m": destination_error,
        "origin_lane_identity": _lane_identity(origin_waypoint),
        "destination_lane_identity": _lane_identity(destination_waypoint),
        "route_point_count": len(route.points),
        "route_length_m": route.length_m,
    }


def validate_fixed_town03_route(
    route: RoutePlan,
    facts: dict[str, Any],
    *,
    map_name: str,
) -> None:
    """Fail fast when the research route no longer matches Town03 truth."""

    normalized_map = str(map_name).replace("\\", "/").rsplit("/", 1)[-1]
    if normalized_map != "Town03":
        raise CarlaRouteError(f"fixed route requires Town03, got {map_name!r}")
    if tuple(facts.get("origin_lane_identity", ())) != FIXED_TOWN03_ORIGIN_LANE:
        raise CarlaRouteError(
            "fixed route origin lane mismatch: "
            f"{facts.get('origin_lane_identity')!r}"
        )
    if tuple(facts.get("destination_lane_identity", ())) != FIXED_TOWN03_DESTINATION_LANE:
        raise CarlaRouteError(
            "fixed route destination lane mismatch: "
            f"{facts.get('destination_lane_identity')!r}"
        )
    if float(facts.get("destination_projection_error_m", math.inf)) > ROUTE_PROJECTION_TOLERANCE_M:
        raise CarlaRouteError("fixed route destination projection mismatch")
    length_error = abs(route.length_m - FIXED_TOWN03_ROUTE_LENGTH_M)
    if length_error > FIXED_TOWN03_ROUTE_LENGTH_TOLERANCE_M:
        raise CarlaRouteError(
            "fixed route length mismatch: "
            f"{route.length_m:.6f}m (expected {FIXED_TOWN03_ROUTE_LENGTH_M:.6f}m)"
        )


def query_candidate_lane_facts(
    world_map: Any,
    world_points: Any,
    *,
    carla_module: Any,
) -> tuple[CandidateLaneFact, ...]:
    """Return exact, non-projecting Driving-lane facts for candidate points."""

    facts = []
    for point in world_points:
        try:
            location = carla_module.Location(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
            )
            waypoint = world_map.get_waypoint(
                location,
                project_to_road=False,
                lane_type=carla_module.LaneType.Driving,
            )
        except Exception:
            waypoint = None
        if waypoint is None:
            facts.append(CandidateLaneFact(None, None, None, None))
        else:
            facts.append(
                CandidateLaneFact(
                    int(waypoint.road_id),
                    int(waypoint.section_id),
                    int(waypoint.lane_id),
                    bool(waypoint.is_junction),
                )
            )
    return tuple(facts)


def assess_current_route_status(
    world_map: Any,
    ego_xyz: Sequence[float],
    *,
    route: RoutePlan,
    route_index: int,
    carla_module: Any,
) -> RouteStatus:
    """Verify current ego center against the monotonic authorized route lane."""

    fact = query_candidate_lane_facts(
        world_map,
        (ego_xyz,),
        carla_module=carla_module,
    )[0]
    if not fact.available:
        return RouteStatus.UNKNOWN
    start = min(max(0, int(route_index)), len(route.points) - 1)
    ego_x, ego_y = float(ego_xyz[0]), float(ego_xyz[1])
    distances = [
        math.hypot(point.xyz[0] - ego_x, point.xyz[1] - ego_y)
        for point in route.points[start:]
    ]
    minimum = min(distances)
    if minimum > 3.0:
        return RouteStatus.UNKNOWN
    candidate_indices = [
        start + index
        for index, distance in enumerate(distances)
        if distance <= minimum + 0.25
    ]
    return (
        RouteStatus.MATCH
        if any(
            fact.identity
            == (
                route.points[index].road_id,
                route.points[index].section_id,
                route.points[index].lane_id,
            )
            for index in candidate_indices
        )
        else RouteStatus.DEVIATE
    )
__all__ = [
    "CarlaRouteError",
    "FIXED_TOWN03_DESTINATION_LANE",
    "FIXED_TOWN03_DESTINATION_XYZ",
    "FIXED_TOWN03_ORIGIN_LANE",
    "FIXED_TOWN03_ORIGIN_XYZ",
    "FIXED_TOWN03_ROUTE_LENGTH_M",
    "load_global_route_planner",
    "parse_destination_xyz",
    "resolve_carla_python_api_path",
    "assess_current_route_status",
    "query_candidate_lane_facts",
    "trace_carla_route",
    "validate_fixed_town03_route",
]

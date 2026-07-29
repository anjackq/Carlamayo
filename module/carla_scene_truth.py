"""Capture lightweight source-time CARLA truth for diagnostic CoC audit."""

from __future__ import annotations

import math
from typing import Any

from .coc_semantic_audit import (
    SceneActorTruth,
    SceneTrafficControlTruth,
    SceneTruthSnapshot,
)


SCENE_TRUTH_ACTOR_RADIUS_M = 60.0
ROUTE_CONTROL_CROSS_TRACK_M = 5.0


def _actor_class(type_id: str) -> str | None:
    lowered = str(type_id).lower()
    if lowered.startswith("vehicle."):
        if any(token in lowered for token in ("bike", "bicycle", "motorcycle")):
            return "cyclist"
        return "vehicle"
    if lowered.startswith("walker.pedestrian"):
        return "pedestrian"
    return None


def _distance_xy(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _route_relevant(
    location: Any,
    route_plan: Any | None,
    route_index: int,
) -> bool:
    if route_plan is None:
        return True
    start = min(max(0, int(route_index)), len(route_plan.points) - 1)
    return any(
        math.hypot(
            point.xyz[0] - float(location.x),
            point.xyz[1] - float(location.y),
        )
        <= ROUTE_CONTROL_CROSS_TRACK_M
        for point in route_plan.points[start:]
    )


def capture_scene_truth_snapshot(
    *,
    world: Any,
    ego_vehicle: Any,
    source_frame_id: int,
    source_simulation_time_s: float,
    navigation_context: Any | None,
    route_plan: Any | None,
    route_index: int,
    carla_module: Any,
    strict_lane_policy_enabled: bool = True,
) -> SceneTruthSnapshot:
    """Capture only facts observable at inference submission time."""

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    try:
        ego_waypoint = world.get_map().get_waypoint(
            ego_location,
            project_to_road=False,
            lane_type=carla_module.LaneType.Driving,
        )
    except Exception:
        ego_waypoint = None

    dynamic_actors = []
    traffic_controls = []
    for actor in world.get_actors():
        if int(actor.id) == int(ego_vehicle.id):
            continue
        type_id = str(getattr(actor, "type_id", ""))
        try:
            actor_location = actor.get_transform().location
        except Exception:
            continue
        distance = _distance_xy(ego_location, actor_location)
        actor_class = _actor_class(type_id)
        if actor_class is not None and distance <= SCENE_TRUTH_ACTOR_RADIUS_M:
            dynamic_actors.append(
                SceneActorTruth(
                    actor_id=int(actor.id),
                    actor_class=actor_class,
                    distance_m=distance,
                )
            )
            continue
        control_type = None
        state = None
        if type_id.startswith("traffic.traffic_light"):
            control_type = "traffic_light"
            try:
                state_value = actor.get_state()
                state = str(
                    getattr(state_value, "name", state_value)
                ).lower()
            except Exception:
                state = None
        elif "traffic.stop" in type_id:
            control_type = "stop_sign"
        elif "traffic.yield" in type_id:
            control_type = "yield_sign"
        if control_type is not None and distance <= SCENE_TRUTH_ACTOR_RADIUS_M:
            traffic_controls.append(
                SceneTrafficControlTruth(
                    control_type=control_type,
                    state=state,
                    distance_m=distance,
                    route_relevant=_route_relevant(
                        actor_location,
                        route_plan,
                        route_index,
                    ),
                )
            )

    return SceneTruthSnapshot(
        source_frame_id=int(source_frame_id),
        source_simulation_time_s=float(source_simulation_time_s),
        ego_road_id=(
            None if ego_waypoint is None else int(ego_waypoint.road_id)
        ),
        ego_section_id=(
            None if ego_waypoint is None else int(ego_waypoint.section_id)
        ),
        ego_lane_id=(
            None if ego_waypoint is None else int(ego_waypoint.lane_id)
        ),
        ego_is_junction=(
            None if ego_waypoint is None else bool(ego_waypoint.is_junction)
        ),
        navigation_context=navigation_context,
        dynamic_actors=tuple(
            sorted(dynamic_actors, key=lambda item: (item.distance_m, item.actor_id))
        ),
        traffic_controls=tuple(
            sorted(
                traffic_controls,
                key=lambda item: (
                    item.distance_m,
                    item.control_type,
                    item.state or "",
                ),
            )
        ),
        strict_lane_policy_enabled=bool(strict_lane_policy_enabled),
    )


__all__ = ["capture_scene_truth_snapshot"]

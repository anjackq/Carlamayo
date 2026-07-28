from pathlib import Path

import pytest

from module.carla_route_adapter import (
    assess_current_route_status,
    CarlaRouteError,
    FIXED_TOWN03_DESTINATION_XYZ,
    parse_destination_xyz,
    resolve_carla_python_api_path,
)
from module.route_authorization import RouteStatus
from module.route_navigation import RoutePoint, build_route_plan


class _Location:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _Carla:
    Location = _Location

    class LaneType:
        Driving = "Driving"


class _Waypoint:
    road_id = 28
    section_id = 0
    lane_id = 3
    is_junction = False


class _TransitionMap:
    def get_waypoint(self, location, *, project_to_road, lane_type):
        assert project_to_road is False
        assert lane_type == "Driving"
        return _Waypoint()


def test_parse_destination_xyz():
    assert parse_destination_xyz("-43.0,-2.0,0") == (-43.0, -2.0, 0.0)
    assert parse_destination_xyz(FIXED_TOWN03_DESTINATION_XYZ) == (
        FIXED_TOWN03_DESTINATION_XYZ
    )
    with pytest.raises(ValueError, match="X,Y,Z"):
        parse_destination_xyz("1,2")
    with pytest.raises(ValueError, match="finite"):
        parse_destination_xyz("1,nan,3")


def test_resolve_python_api_priority_and_shapes(tmp_path):
    env_root = tmp_path / "env" / "PythonAPI" / "carla"
    explicit = tmp_path / "explicit"
    for root in (env_root, explicit):
        marker = root / "agents" / "navigation" / "global_route_planner.py"
        marker.parent.mkdir(parents=True)
        marker.write_text("# test marker\n", encoding="utf-8")
    assert resolve_carla_python_api_path(
        explicit,
        environ={"CARLAMAYO_CARLA_ROOT": str(tmp_path / "env")},
    ) == explicit.resolve()
    assert resolve_carla_python_api_path(
        None,
        environ={"CARLAMAYO_CARLA_ROOT": str(tmp_path / "env")},
    ) == env_root.resolve()


def test_resolve_python_api_fails_closed(tmp_path):
    with pytest.raises(CarlaRouteError, match="unavailable"):
        resolve_carla_python_api_path(
            Path(tmp_path) / "missing",
            environ={},
        )


def test_current_route_status_prefers_exact_lane_at_connector_overlap():
    route = build_route_plan(
        [
            RoutePoint((0, 0, 0), 28, 0, 3, False, "LANEFOLLOW"),
            RoutePoint((2, 0, 0), 1691, 0, 3, True, "LANEFOLLOW"),
            RoutePoint((10, 0, 0), 25, 0, 3, False, "RIGHT"),
        ]
    )

    status = assess_current_route_status(
        _TransitionMap(),
        (1.1, 0.0, 0.0),
        route=route,
        route_index=0,
        carla_module=_Carla,
    )

    assert status is RouteStatus.MATCH

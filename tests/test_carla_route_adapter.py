from pathlib import Path

import pytest

from module.carla_route_adapter import (
    CarlaRouteError,
    FIXED_TOWN03_DESTINATION_XYZ,
    parse_destination_xyz,
    resolve_carla_python_api_path,
)


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

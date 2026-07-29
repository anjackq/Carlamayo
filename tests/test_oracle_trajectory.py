import numpy as np
import pytest

from module.oracle_trajectory import (
    build_oracle_route_trajectory,
    interpolate_route_world_points,
    project_route_progress,
)
from module.route_navigation import RoutePoint, build_route_plan


def _route(length=100):
    return build_route_plan(
        [
            RoutePoint(
                xyz=(float(index), 0.0, 0.0),
                road_id=1,
                section_id=0,
                lane_id=1,
                is_junction=False,
                road_option="LANEFOLLOW",
            )
            for index in range(length + 1)
        ]
    )


def test_route_interpolation_uses_arc_length():
    route = _route()

    points = interpolate_route_world_points(route, [0.5, 10.25, 99.5])

    assert points[:, 0] == pytest.approx([0.5, 10.25, 99.5])
    assert points[:, 1:] == pytest.approx(np.zeros((3, 2)))


def test_route_projection_returns_continuous_progress_and_distance():
    progress, distance = project_route_progress(
        _route(),
        (12.25, 0.4, 0.0),
        start_index=10,
    )

    assert progress == pytest.approx(12.25)
    assert distance == pytest.approx(0.4)


def test_oracle_launch_has_bounded_acceleration_and_exact_contract():
    trajectory = build_oracle_route_trajectory(
        _route(),
        start_progress_m=0.0,
        source_simulation_time_s=10.0,
        capture_pose_world=np.eye(4),
        current_speed_mps=0.0,
        target_speed_mps=2.0,
        maximum_acceleration_mps2=2.0,
    )

    assert trajectory.world_points.shape == (64, 3)
    assert trajectory.model_points.shape == (64, 3)
    assert trajectory.waypoint_times_s[0] == pytest.approx(10.1)
    assert trajectory.waypoint_times_s[-1] == pytest.approx(16.4)
    assert trajectory.speed_profile_mps[0] == pytest.approx(0.2)
    assert np.max(np.diff(trajectory.speed_profile_mps)) <= 0.2 + 1e-9
    assert trajectory.speed_profile_mps[10] == pytest.approx(2.0)
    assert np.all(np.diff(trajectory.route_progress_m) >= 0.0)
    assert not trajectory.world_points.flags.writeable


def test_oracle_converts_carla_y_right_to_model_y_left():
    route = build_route_plan(
        [
            RoutePoint(
                xyz=(float(index), float(index), 0.0),
                road_id=1,
                section_id=0,
                lane_id=1,
                is_junction=False,
                road_option="LANEFOLLOW",
            )
            for index in range(101)
        ]
    )

    trajectory = build_oracle_route_trajectory(
        route,
        start_progress_m=0.0,
        source_simulation_time_s=0.0,
        capture_pose_world=np.eye(4),
        current_speed_mps=1.0,
        target_speed_mps=1.0,
    )

    assert np.all(trajectory.world_points[:, 1] > 0.0)
    assert trajectory.model_points[:, 1] == pytest.approx(
        -trajectory.world_points[:, 1]
    )


def test_oracle_destination_profile_stops_and_repeats_terminal_point():
    trajectory = build_oracle_route_trajectory(
        _route(length=5),
        start_progress_m=4.5,
        source_simulation_time_s=0.0,
        capture_pose_world=np.eye(4),
        current_speed_mps=2.0,
        target_speed_mps=4.0,
        stop_at_destination=True,
    )

    assert trajectory.route_progress_m[-1] == pytest.approx(5.0)
    assert trajectory.speed_profile_mps[-1] == pytest.approx(0.0)
    terminal = np.flatnonzero(
        np.isclose(trajectory.route_progress_m, trajectory.route_progress_m[-1])
    )
    assert len(terminal) >= 6


def test_oracle_can_stop_at_an_intermediate_capture_marker():
    trajectory = build_oracle_route_trajectory(
        _route(),
        start_progress_m=8.0,
        source_simulation_time_s=0.0,
        capture_pose_world=np.eye(4),
        current_speed_mps=2.0,
        target_speed_mps=2.0,
        stop_progress_m=10.0,
    )

    assert trajectory.route_progress_m[-1] == pytest.approx(10.0)
    assert trajectory.speed_profile_mps[-1] == pytest.approx(0.0)
    assert trajectory.world_points[-1, 0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_speed_mps": -1.0},
        {"maximum_acceleration_mps2": 0.0},
        {"source_simulation_time_s": float("nan")},
    ],
)
def test_oracle_rejects_invalid_inputs(kwargs):
    values = {
        "start_progress_m": 0.0,
        "source_simulation_time_s": 0.0,
        "capture_pose_world": np.eye(4),
        "current_speed_mps": 0.0,
        "target_speed_mps": 1.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        build_oracle_route_trajectory(_route(), **values)

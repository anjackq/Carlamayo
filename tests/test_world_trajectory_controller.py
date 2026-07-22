import types

import numpy as np
import pytest

from module import pid_controller
from module.trajectory_runtime import detect_terminal_stop_index


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeRotation:
    pass


class FakeTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or FakeLocation()
        self.rotation = rotation or FakeRotation()


class RecordingVehiclePIDController:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def run_step(self, target_speed_kmh, waypoint):
        self.calls.append((float(target_speed_kmh), waypoint))
        if target_speed_kmh <= 0.0:
            return types.SimpleNamespace(steer=0.0, throttle=0.0, brake=1.0)
        return types.SimpleNamespace(steer=0.1, throttle=0.2, brake=0.0)


class MovableVehicle:
    def __init__(self, x=0.0, y=0.0):
        self.transform = FakeTransform(FakeLocation(x, y, 0.0))

    def get_transform(self):
        return self.transform


class FakeWorld:
    pass


@pytest.fixture
def follower(monkeypatch):
    fake_carla = types.SimpleNamespace(
        Location=FakeLocation,
        Rotation=FakeRotation,
        Transform=FakeTransform,
    )
    monkeypatch.setattr(pid_controller, "carla", fake_carla)
    monkeypatch.setattr(
        pid_controller,
        "_resolve_vehicle_pid_controller",
        lambda: RecordingVehiclePIDController,
    )
    vehicle = MovableVehicle()
    return pid_controller.OfficialPIDFollower(FakeWorld(), vehicle)


def _straight_path(step_m):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = step_m * np.arange(1, 65)
    return points


def _times():
    return np.arange(1, 65, dtype=np.float64) * 0.1


def test_timestamp_spacing_sets_speed_without_positive_minimum(follower):
    points = _straight_path(0.5)

    _, throttle, brake, debug = follower.compute_world_control(
        plan_id="constant-speed",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
    )

    assert debug["target_speed_mps"] == pytest.approx(5.0)
    assert follower.pid.calls[-1][0] == pytest.approx(18.0)
    assert (throttle, brake) == pytest.approx((0.2, 0.0))

    slow_points = _straight_path(0.02)
    follower.compute_world_control(
        plan_id="slow-stop",
        wp_world=slow_points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
    )
    assert follower.pid.calls[-1][0] == pytest.approx(0.72)
    assert follower.pid.calls[-1][0] < 10.0


def test_progress_never_moves_backwards_for_same_plan(follower):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 10.0

    *_, first_debug = follower.compute_world_control(
        plan_id="one-plan",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
    )
    follower.vehicle.transform.location.x = 2.0
    *_, second_debug = follower.compute_world_control(
        plan_id="one-plan",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
    )

    assert first_debug["progress_s_m"] > 0.0
    assert second_debug["progress_s_m"] >= first_debug["progress_s_m"]
    assert second_debug["progress_index"] >= first_debug["progress_index"]


def test_new_plan_resets_monotonic_progress(follower):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 10.0
    *_, old_debug = follower.compute_world_control(
        plan_id="old",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
    )
    follower.vehicle.transform.location.x = 2.0
    *_, new_debug = follower.compute_world_control(
        plan_id="new",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
    )

    assert new_debug["progress_s_m"] < old_debug["progress_s_m"]


def test_moving_path_with_repeated_terminal_points_brakes_and_stops(follower):
    points = _straight_path(0.2)
    points[40:] = points[39]
    stop_index = detect_terminal_stop_index(points)
    assert stop_index is not None and stop_index > 0

    *_, initial_debug = follower.compute_world_control(
        plan_id="planned-stop",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
        terminal_stop_index=stop_index,
    )
    assert initial_debug["target_speed_mps"] > 0.0

    follower.vehicle.transform.location.x = points[stop_index, 0] - 0.2
    steer, throttle, brake, decelerating = follower.compute_world_control(
        plan_id="planned-stop",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=3.0,
        speed_mps=1.0,
        terminal_stop_index=stop_index,
    )
    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert decelerating["controller_state"] == "DECELERATING"
    assert decelerating["bypass_smoothing"] is True

    follower.vehicle.transform.location.x = points[stop_index, 0]
    *controls, stopped = follower.compute_world_control(
        plan_id="planned-stop",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=3.0,
        speed_mps=0.1,
        terminal_stop_index=stop_index,
    )
    assert controls == pytest.approx([0.0, 0.0, 1.0])
    assert stopped["controller_state"] == "STOPPED"


def test_all_stationary_trajectory_commands_full_brake(follower):
    points = np.zeros((64, 3), dtype=np.float64)

    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="stationary",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
        stop_requested=True,
        terminal_stop_index=0,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["controller_state"] == "STOPPED"
    assert debug["bypass_smoothing"] is True
    assert follower.pid.calls == []


@pytest.mark.parametrize(
    ("points", "times", "current_time"),
    [
        (np.zeros((64, 2)), _times(), 0.0),
        (np.full((64, 3), np.nan), _times(), 0.0),
        (_straight_path(0.2), np.zeros(64), 0.0),
        (_straight_path(0.2), _times(), np.nan),
    ],
)
def test_invalid_fixed_world_trajectory_fails_closed(follower, points, times, current_time):
    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="bad",
        wp_world=points,
        waypoint_times_s=times,
        current_simulation_time_s=current_time,
        speed_mps=0.0,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["controller_state"] == "FALLBACK"
    assert debug["bypass_smoothing"] is True


def test_exhausted_trajectory_fails_closed(follower):
    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="expired",
        wp_world=_straight_path(0.2),
        waypoint_times_s=_times(),
        current_simulation_time_s=6.4,
        speed_mps=3.0,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["mode"] == "trajectory_exhausted"
    assert debug["controller_state"] == "FALLBACK"

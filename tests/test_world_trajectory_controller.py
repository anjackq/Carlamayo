import json
import types
from pathlib import Path

import numpy as np
import pytest

from module import config as cfg
from module import pid_controller
from module.trajectory_runtime import detect_terminal_stop_index


PID_TIME_GEOMETRY_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22862294_pid_time_geometry.json"
)


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


def _launch_path(prefix_step_m=0.02, cruise_step_m=0.5, prefix_points=6):
    """A moving plan that barely moves for its prefix then accelerates to cruise.

    This is the shape Alpamayo emits for a stopped ego it predicts will launch:
    a near-stationary prefix (~0.2 m/s) followed by real forward motion (~5 m/s).
    Reading the fresh-plan target speed off the prefix is what deadlocks the
    launch under 1 s proposal replacement.
    """

    steps = np.full(64, cruise_step_m, dtype=np.float64)
    steps[:prefix_points] = prefix_step_m
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = np.cumsum(steps)
    return points


def test_launch_floor_pulls_stopped_ego_off_the_line_across_plan_swaps(follower):
    points = _launch_path()

    # Simulate the ~1 s proposal-replacement cadence: a brand-new plan_id every
    # call, each freshly anchored (current_time=0) with the ego still stopped.
    # Without the floor every call would target the ~0.2 m/s prefix and the
    # launch would never complete.  The floor must hold across all swaps.
    for plan_id in ("p1", "p2", "p3"):
        _, throttle, brake, debug = follower.compute_world_control(
            plan_id=plan_id,
            wp_world=points,
            waypoint_times_s=_times(),
            current_simulation_time_s=0.0,
            speed_mps=0.0,
        )
        assert debug["launch_floor_mps"] == pytest.approx(cfg.PID_LAUNCH_SPEED_MPS)
        assert debug["target_speed_mps"] >= cfg.PID_LAUNCH_SPEED_MPS
        assert throttle > 0.0
        assert brake == pytest.approx(0.0)
        assert follower.pid.calls[-1][0] == pytest.approx(cfg.PID_LAUNCH_SPEED_MPS * 3.6)


def test_launch_floor_never_overrides_a_faster_timestamp_profile(follower):
    # A constant 5 m/s plan already exceeds the launch floor, so the floor must
    # not reduce or alter the timestamp-derived target.
    points = _straight_path(0.5)
    _, _, _, debug = follower.compute_world_control(
        plan_id="cruise",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
    )
    assert debug["target_speed_mps"] == pytest.approx(5.0)
    assert debug["launch_floor_mps"] == pytest.approx(cfg.PID_LAUNCH_SPEED_MPS)


def test_launch_floor_exempts_low_intent_creep_plans(follower):
    # A uniform 0.2 m/s creep plan intends less than the minimum launch intent,
    # so no floor is applied and no motion the model did not predict is invented.
    points = _straight_path(0.02)
    _, _, _, debug = follower.compute_world_control(
        plan_id="creep",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
    )
    assert debug["launch_floor_mps"] == pytest.approx(0.0)
    assert debug["target_speed_mps"] == pytest.approx(0.2)


def test_launch_floor_disengages_once_ego_is_rolling(follower):
    # Once the ego is above the engaged speed the gearbox has bitten; the floor
    # must release so the timestamp profile governs cruise and deceleration.
    points = _launch_path()
    _, _, _, debug = follower.compute_world_control(
        plan_id="rolling",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=cfg.PID_LAUNCH_ENGAGE_SPEED_MPS + 0.5,
    )
    assert debug["launch_floor_mps"] == pytest.approx(0.0)


def test_road_speed_cap_overrides_launch_floor(follower):
    points = _launch_path()

    _, throttle, brake, debug = follower.compute_world_control(
        plan_id="road-limited-launch",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
        target_speed_cap_mps=1.0,
        maximum_authorized_waypoint_index=20,
    )

    assert debug["launch_floor_mps"] == pytest.approx(cfg.PID_LAUNCH_SPEED_MPS)
    assert debug["unconstrained_target_speed_mps"] >= cfg.PID_LAUNCH_SPEED_MPS
    assert debug["target_speed_mps"] == pytest.approx(1.0)
    assert debug["road_speed_limited"] is True
    assert debug["controller_state"] == "ROAD_CONSTRAINED_DECELERATING"
    assert follower.pid.calls[-1][0] == pytest.approx(3.6)
    assert throttle > 0.0
    assert brake == pytest.approx(0.0)


def test_controller_target_never_exceeds_last_safe_waypoint(follower):
    points = _straight_path(0.5)

    *_, debug = follower.compute_world_control(
        plan_id="bounded-target",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=0.0,
        maximum_authorized_waypoint_index=4,
    )

    assert debug["target_idx"] <= 4
    assert debug["maximum_authorized_waypoint_index"] == 4


def test_delayed_plan_steering_uses_geometric_progress_not_elapsed_time(
    follower,
):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 5.0

    *_, first_debug = follower.compute_world_control(
        plan_id="delayed-plan",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=3.6,
        speed_mps=2.0,
    )
    *_, later_debug = follower.compute_world_control(
        plan_id="delayed-plan",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=4.0,
        speed_mps=2.0,
    )

    assert first_debug["first_future_index"] == 36
    assert later_debug["first_future_index"] == 40
    assert first_debug["steering_reference"] == "geometric_progress"
    assert later_debug["steering_target_index"] == first_debug["target_idx"]
    assert later_debug["target_idx"] == first_debug["target_idx"]
    assert later_debug["target_distance_m"] == pytest.approx(
        first_debug["target_distance_m"]
    )
    assert later_debug["time_geometry_gap_m"] > first_debug[
        "time_geometry_gap_m"
    ]
    assert first_debug["steering_target_path_distance_m"] <= (
        first_debug["lookahead_m"] + 0.5
    )


def test_geometric_progress_ahead_of_time_still_drives_local_lookahead(
    follower,
):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 10.0

    *_, debug = follower.compute_world_control(
        plan_id="geometry-ahead",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
    )

    assert debug["first_future_index"] == 0
    assert debug["progress_index"] == 19
    assert debug["steering_reference_index"] == 19
    assert debug["time_geometry_gap_m"] < 0.0
    assert debug["target_distance_m"] == pytest.approx(5.0)


def test_geometric_progress_past_authorized_prefix_fails_closed(follower):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 10.0

    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="geometry-past-prefix",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
        maximum_authorized_waypoint_index=4,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["mode"] == "road_safe_prefix_exhausted"
    assert debug["controller_state"] == "ROAD_CONSTRAINED_DECELERATING"
    assert debug["target_idx"] == 4


def test_geometric_progress_past_terminal_stop_remains_fail_closed(follower):
    points = _straight_path(0.5)
    follower.vehicle.transform.location.x = 10.0

    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="geometry-past-stop",
        wp_world=points,
        waypoint_times_s=_times(),
        current_simulation_time_s=0.0,
        speed_mps=2.0,
        terminal_stop_index=4,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["mode"] == "terminal_stop"
    assert debug["controller_state"] == "DECELERATING"
    assert debug["target_idx"] == 4


def test_job_22862294_steering_target_replay_uses_local_curve(follower):
    replay = json.loads(PID_TIME_GEOMETRY_FIXTURE.read_text(encoding="utf-8"))
    points_xy = np.asarray(replay["world_points_xy"], dtype=np.float64)
    points = np.column_stack((points_xy, np.zeros(len(points_xy))))
    source_time = float(replay["source_simulation_time_s"])
    waypoint_dt = float(replay["waypoint_dt_s"])
    times = source_time + waypoint_dt * np.arange(1, len(points) + 1)
    tick = replay["ticks"][0]
    follower.vehicle.transform.location.x = tick["ego_xy"][0]
    follower.vehicle.transform.location.y = tick["ego_xy"][1]

    *_, debug = follower.compute_world_control(
        plan_id=replay["source_plan_id"],
        wp_world=points,
        waypoint_times_s=times,
        current_simulation_time_s=tick["simulation_time_s"],
        speed_mps=tick["speed_mps"],
        maximum_authorized_waypoint_index=len(points) - 1,
    )

    assert debug["progress_index"] == tick[
        "expected_geometric_progress_index"
    ]
    assert debug["target_idx"] == tick["expected_geometric_target_index"]
    assert debug["target_idx"] < tick["legacy_target_index"]
    assert debug["target_distance_m"] == pytest.approx(
        tick["expected_geometric_target_distance_m"],
        abs=0.02,
    )
    assert debug["target_distance_m"] < tick["legacy_target_distance_m"] - 5.0
    assert debug["time_geometry_gap_m"] == pytest.approx(
        tick["expected_time_geometry_gap_m"],
        abs=0.02,
    )


def test_exhausted_safe_prefix_commands_controller_brake(follower):
    steer, throttle, brake, debug = follower.compute_world_control(
        plan_id="safe-prefix-exhausted",
        wp_world=_straight_path(0.5),
        waypoint_times_s=_times(),
        current_simulation_time_s=1.0,
        speed_mps=0.0,
        target_speed_cap_mps=0.0,
        maximum_authorized_waypoint_index=4,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 1.0))
    assert debug["mode"] == "road_safe_prefix_exhausted"
    assert debug["controller_state"] == "ROAD_CONSTRAINED_DECELERATING"


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

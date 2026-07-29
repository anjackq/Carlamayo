import pytest

from carlamayo_closed_loop import smooth_controller_control


def test_smoothing_history_tracks_nominal_intent_not_safety_applied_brake():
    previous_nominal = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    }

    first, _ = smooth_controller_control(
        steering_raw=0.0,
        throttle_raw=0.6,
        brake_raw=0.0,
        previous_nominal=previous_nominal,
        alpha=0.25,
    )
    # A fail-closed shield may apply full brake here. That final command is
    # deliberately absent from the next controller-only EMA update.
    second, _ = smooth_controller_control(
        steering_raw=0.0,
        throttle_raw=0.6,
        brake_raw=0.0,
        previous_nominal=first,
        alpha=0.25,
    )

    assert first == pytest.approx(
        {"steering": 0.0, "throttle": 0.15, "brake": 0.0}
    )
    assert second == pytest.approx(
        {"steering": 0.0, "throttle": 0.2625, "brake": 0.0}
    )


def test_emergency_controller_stop_bypasses_smoothing():
    nominal, postprocessing = smooth_controller_control(
        steering_raw=0.4,
        throttle_raw=0.6,
        brake_raw=0.0,
        previous_nominal={
            "steering": -0.2,
            "throttle": 0.5,
            "brake": 0.0,
        },
        alpha=0.25,
        bypass_smoothing=True,
    )

    assert nominal == {
        "steering": 0.4,
        "throttle": 0.0,
        "brake": 1.0,
    }
    assert postprocessing == ("emergency_brake_bypass_ema",)


def test_road_constrained_brake_cuts_throttle_without_longitudinal_ema():
    nominal, postprocessing = smooth_controller_control(
        steering_raw=0.4,
        throttle_raw=0.0,
        brake_raw=0.079,
        previous_nominal={
            "steering": -0.2,
            "throttle": 0.41,
            "brake": 0.0,
        },
        alpha=0.25,
        constrained_deceleration=True,
    )

    assert nominal == pytest.approx(
        {
            "steering": -0.05,
            "throttle": 0.0,
            "brake": 0.079,
        }
    )
    assert postprocessing == (
        "steering_ema_smoothing",
        "road_deceleration_bypass_longitudinal_ema",
    )


def test_low_speed_governor_bypasses_only_longitudinal_ema():
    nominal, postprocessing = smooth_controller_control(
        steering_raw=0.4,
        throttle_raw=0.25,
        brake_raw=0.0,
        previous_nominal={
            "steering": -0.2,
            "throttle": 0.0,
            "brake": 1.0,
        },
        alpha=0.25,
        direct_longitudinal=True,
    )

    assert nominal == pytest.approx(
        {
            "steering": -0.05,
            "throttle": 0.25,
            "brake": 0.0,
        }
    )
    assert postprocessing == (
        "steering_ema_smoothing",
        "low_speed_direct_longitudinal_control",
    )

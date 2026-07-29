import numpy as np
import pytest

from module.trajectory_reachability import (
    PhysicalReachabilityStatus,
    SourceSpeedPriorStatus,
    compute_trajectory_reachability_profile,
)


def _path(step_m):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = np.arange(1, 65) * step_m
    return points


def _history(speed_mps):
    history = np.zeros((16, 3), dtype=np.float64)
    history[:, 0] = np.arange(-15, 1) * speed_mps * 0.1
    return history


def test_constant_source_speed_prediction_is_reachable_and_consistent():
    profile = compute_trajectory_reachability_profile(
        _path(0.2),
        _history(2.0),
        source_speed_mps=2.0,
    )

    assert profile.path_length_m == pytest.approx(12.8)
    assert profile.required_constant_acceleration_mps2 == pytest.approx(0.0)
    assert profile.physical_status is PhysicalReachabilityStatus.REACHABLE
    assert profile.source_speed_prior_status is SourceSpeedPriorStatus.CONSISTENT
    assert profile.history_terminal_speed_mps == pytest.approx(2.0)


def test_excessive_distance_is_physically_too_long():
    profile = compute_trajectory_reachability_profile(
        _path(1.0),
        _history(2.0),
        source_speed_mps=2.0,
    )

    assert profile.path_length_m == pytest.approx(64.0)
    assert profile.maximum_reachable_distance_m == pytest.approx(53.76)
    assert profile.physical_status is PhysicalReachabilityStatus.TOO_LONG
    assert (
        profile.source_speed_prior_status
        is SourceSpeedPriorStatus.ACCELERATION_PRIOR
    )


def test_instant_stop_from_moving_source_is_too_short_to_stop():
    profile = compute_trajectory_reachability_profile(
        _path(0.0),
        _history(2.0),
        source_speed_mps=2.0,
    )

    assert profile.minimum_reachable_distance_m == pytest.approx(0.8)
    assert (
        profile.physical_status
        is PhysicalReachabilityStatus.TOO_SHORT_TO_STOP
    )
    assert profile.source_speed_prior_status is SourceSpeedPriorStatus.STOP_PRIOR


@pytest.mark.parametrize(
    ("points", "history", "speed"),
    [
        (np.zeros((0, 3)), _history(1.0), 1.0),
        (_path(0.1), np.zeros((1, 3)), 1.0),
        (_path(0.1), _history(1.0), -1.0),
    ],
)
def test_invalid_reachability_input_is_rejected(points, history, speed):
    with pytest.raises(ValueError):
        compute_trajectory_reachability_profile(
            points,
            history,
            source_speed_mps=speed,
        )

from dataclasses import replace

import numpy as np
import pytest

from module.geometry import pose_matrix_from_components
from module.trajectory_runtime import (
    TrajectoryValidationError,
    build_fixed_world_trajectory,
    classify_and_validate_model_trajectory,
    detect_terminal_stop_index,
    validate_model_trajectory,
    validate_plan_alignment,
    validate_plan_for_execution,
)


def _moving_points(step_m=0.2):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = step_m * np.arange(1, 65)
    return points


def _build_plan(*, source_time=10.0, points=None, prompt_revision=2, respawn_revision=3):
    return build_fixed_world_trajectory(
        plan_id="plan-1",
        source_frame_id=100,
        source_simulation_time_s=source_time,
        capture_pose_world=np.eye(4),
        model_points=_moving_points() if points is None else points,
        coc_text="Follow the lane.",
        prompt_revision=prompt_revision,
        respawn_revision=respawn_revision,
    )


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((63, 3)),
        np.zeros((64, 2)),
        np.zeros((64, 4)),
        np.zeros((1, 64, 3)),
    ],
)
def test_model_trajectory_requires_exactly_64_by_3(points):
    with pytest.raises(TrajectoryValidationError, match="invalid_shape"):
        validate_model_trajectory(points)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_model_trajectory_rejects_non_finite_points(bad_value):
    points = _moving_points()
    points[20, 1] = bad_value

    with pytest.raises(TrajectoryValidationError, match="non_finite_points"):
        validate_model_trajectory(points)


def test_terminal_repeated_points_are_classified_as_a_planned_stop():
    points = _moving_points()
    points[40:] = points[39]

    validated, stop_requested = classify_and_validate_model_trajectory(points)
    stop_index = detect_terminal_stop_index(validated)

    assert stop_requested is True
    assert 1 <= stop_index <= 40
    np.testing.assert_allclose(
        validated[stop_index:],
        np.repeat(validated[-1][None, :], len(validated) - stop_index, axis=0),
    )


def test_stationary_plan_is_valid_stop_but_moving_plan_is_not():
    stationary = np.zeros((64, 3), dtype=np.float64)

    stationary_points, stationary_stop = classify_and_validate_model_trajectory(stationary)
    moving_points, moving_stop = classify_and_validate_model_trajectory(_moving_points())

    assert stationary_stop is True
    assert detect_terminal_stop_index(stationary_points) == 0
    assert moving_stop is False
    assert detect_terminal_stop_index(moving_points) is None


def test_stationary_stop_still_validates_jump_from_capture_origin():
    impossible_stop = np.zeros((64, 3), dtype=np.float64)
    impossible_stop[:, 0] = 100.0

    with pytest.raises(TrajectoryValidationError, match="excessive_waypoint_step"):
        validate_model_trajectory(impossible_stop)


def test_plan_is_anchored_once_to_an_owned_capture_pose():
    capture_pose = pose_matrix_from_components(10.0, 20.0, 1.0, yaw_deg=90.0)
    points = _moving_points(step_m=1.0)
    points[:, 1] = 2.0

    plan = build_fixed_world_trajectory(
        plan_id="anchored",
        source_frame_id=7,
        source_simulation_time_s=3.0,
        capture_pose_world=capture_pose,
        model_points=points,
        coc_text="",
        prompt_revision=0,
        respawn_revision=0,
    )
    original_world_points = plan.world_points.copy()
    capture_pose[0, 3] = 999.0
    points[0] = 999.0

    # Model (x=1, y-left=2) becomes CARLA local (1, -2); yaw=90 then
    # translates it to world (12, 21, 1).
    np.testing.assert_allclose(plan.world_points[0], [12.0, 21.0, 1.0], atol=1e-9)
    np.testing.assert_array_equal(plan.world_points, original_world_points)
    assert plan.capture_pose_world[0, 3] == pytest.approx(10.0)
    assert plan.world_points.flags.writeable is False
    assert plan.capture_pose_world.flags.writeable is False
    with pytest.raises(ValueError):
        plan.world_points[0, 0] = 0.0


def test_waypoint_times_are_absolute_simulation_times():
    plan = _build_plan(source_time=10.0)

    assert plan.waypoint_times_s[0] == pytest.approx(10.1)
    assert plan.waypoint_times_s[-1] == pytest.approx(16.4)
    np.testing.assert_allclose(plan.waypoint_offsets_s, np.arange(1, 65) * 0.1)


def test_execution_validation_uses_strictly_future_waypoints_and_boundary_tolerance():
    plan = _build_plan(source_time=10.0)

    at_source = validate_plan_for_execution(plan, 10.0)
    at_first_waypoint = validate_plan_for_execution(plan, 10.1)
    at_maximum_age = validate_plan_for_execution(plan, 14.4)

    assert at_source.valid and at_source.first_future_index == 0
    assert at_first_waypoint.valid and at_first_waypoint.first_future_index == 1
    assert at_maximum_age.valid
    assert at_maximum_age.source_age_s == pytest.approx(4.4)
    assert at_maximum_age.remaining_horizon_s == pytest.approx(2.0)


def test_execution_validation_rejects_future_stale_and_short_horizon_plans():
    plan = _build_plan(source_time=10.0)

    future = validate_plan_for_execution(plan, 9.9)
    stale = validate_plan_for_execution(plan, 14.401)
    short_horizon = validate_plan_for_execution(
        plan,
        14.5,
        maximum_plan_age_s=100.0,
    )

    assert future.rejection_reason == "source_time_in_future"
    assert stale.rejection_reason == "plan_source_age_exceeded"
    assert short_horizon.rejection_reason == "insufficient_remaining_horizon"
    assert all(result.first_future_index is None for result in (future, stale, short_horizon))


def test_execution_validation_rejects_prompt_and_respawn_revisions():
    plan = _build_plan(prompt_revision=2, respawn_revision=3)

    prompt_stale = validate_plan_for_execution(
        plan,
        10.0,
        current_prompt_revision=4,
        current_respawn_revision=3,
    )
    respawn_stale = validate_plan_for_execution(
        plan,
        10.0,
        current_prompt_revision=2,
        current_respawn_revision=4,
    )

    assert prompt_stale.rejection_reason == "prompt_revision_mismatch"
    assert respawn_stale.rejection_reason == "respawn_revision_mismatch"


def test_plan_alignment_ignores_longitudinal_lag_but_rejects_cross_track_error():
    plan = _build_plan(source_time=10.0)
    # At t=14.0 the timestamped point is x=8.0, but an asynchronous result may
    # legitimately arrive while the ego is still on the same path at x=1.0.
    lagging_pose = pose_matrix_from_components(1.0, 0.0, 0.0)
    lateral_pose = pose_matrix_from_components(1.0, 3.0, 0.0)

    lagging = validate_plan_alignment(plan, 14.0, lagging_pose)
    lateral = validate_plan_alignment(plan, 14.0, lateral_pose)

    assert lagging.valid is True
    assert lagging.tracking_error_m == pytest.approx(0.0)
    assert lateral.valid is False
    assert lateral.rejection_reason == "plan_tracking_error_exceeded"
    assert lateral.tracking_error_m == pytest.approx(3.0)


def test_plan_alignment_rejects_heading_divergence_but_allows_stationary_stop():
    moving_plan = _build_plan(source_time=10.0)
    turned_pose = pose_matrix_from_components(2.0, 0.0, 0.0, yaw_deg=90.0)

    turned = validate_plan_alignment(moving_plan, 11.0, turned_pose)
    stopped = validate_plan_alignment(
        _build_plan(source_time=10.0, points=np.zeros((64, 3))),
        10.5,
        pose_matrix_from_components(0.0, 0.0, 0.0, yaw_deg=180.0),
    )

    assert turned.valid is False
    assert turned.rejection_reason == "plan_heading_error_exceeded"
    assert turned.heading_error_deg == pytest.approx(90.0)
    assert stopped.valid is True


def test_plan_alignment_ignores_submillimetre_reverse_start_jitter():
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = np.concatenate(
        [
            np.array([-0.00003, -0.00010, -0.00016, -0.00019, -0.00014]),
            np.linspace(0.0001, 20.0, 59),
        ]
    )
    plan = _build_plan(source_time=10.0, points=points)

    aligned = validate_plan_alignment(
        plan,
        10.0,
        pose_matrix_from_components(0.0, 0.0, 0.0),
    )

    assert aligned.valid is True
    assert aligned.heading_error_deg == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("plan", "current_time", "reason"),
    [
        (
            replace(_build_plan(), world_points=np.zeros((63, 3))),
            10.5,
            "invalid_plan_geometry",
        ),
        (
            replace(
                _build_plan(),
                waypoint_times_s=np.full(64, np.nan),
            ),
            10.5,
            "invalid_plan_geometry",
        ),
        (_build_plan(), 9.0, "source_time_in_future"),
        (_build_plan(), 16.4, "trajectory_exhausted"),
    ],
)
def test_plan_alignment_fails_closed_for_malformed_or_exhausted_plan(
    plan,
    current_time,
    reason,
):
    result = validate_plan_alignment(
        plan,
        current_time,
        pose_matrix_from_components(0.0, 0.0, 0.0),
    )

    assert result.valid is False
    assert result.rejection_reason == reason

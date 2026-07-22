import json
import math
from dataclasses import FrozenInstanceError

import pytest

from module.safety_shield import (
    ActorObstacle,
    AssessmentStatus,
    ControlCommand,
    EgoKinematics,
    ObstacleAssessment,
    RoadContainmentAssessment,
    RoadContainmentSample,
    SafetyPolicy,
    StopOnlySafetyShield,
    assess_obstacles,
    assess_road_containment,
    densify_path,
    stopping_distance_m,
)


def _ego(*, speed=5.0, frame_id=10):
    return EgoKinematics(
        frame_id=frame_id,
        center_xy=(0.0, 0.0),
        yaw_rad=0.0,
        velocity_xy=(speed, 0.0),
        half_length_m=2.0,
        half_width_m=1.0,
        actor_id=1,
    )


def _actor(
    actor_id,
    center_xy,
    *,
    velocity_xy=(0.0, 0.0),
    yaw_rad=0.0,
    half_length_m=2.0,
    half_width_m=1.0,
    frame_id=10,
    type_id="vehicle.test",
):
    return ActorObstacle(
        frame_id=frame_id,
        actor_id=actor_id,
        type_id=type_id,
        center_xy=center_xy,
        yaw_rad=yaw_rad,
        velocity_xy=velocity_xy,
        half_length_m=half_length_m,
        half_width_m=half_width_m,
    )


def _straight_path(length=30.0):
    return ((0.0, 0.0, 0.0), (length, 0.0, 0.0))


def test_densify_path_preserves_dimensions_and_caps_xy_spacing():
    dense = densify_path(
        [(0.0, 0.0, 1.0), (1.2, 0.0, 2.2)],
        max_spacing_m=0.5,
    )

    assert dense[0] == (0.0, 0.0, 1.0)
    assert dense[-1] == pytest.approx((1.2, 0.0, 2.2))
    assert len(dense) == 4
    assert max(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(dense, dense[1:])
    ) <= 0.5

    with pytest.raises(ValueError, match="no greater than 0.5"):
        densify_path([(0.0, 0.0), (1.0, 0.0)], max_spacing_m=0.51)
    with pytest.raises(ValueError, match="finite"):
        densify_path([(0.0, 0.0), (float("nan"), 0.0)])


def test_road_sample_reduction_is_fail_closed_and_reports_first_bad_sample():
    safe = assess_road_containment(
        [
            RoadContainmentSample(0, (0.0, 0.0, 0.0), True, margin_m=0.6),
            RoadContainmentSample(1, (0.5, 0.0, 0.0), True, margin_m=0.2),
        ],
        quality="exact_carla_map",
    )
    assert safe.status is AssessmentStatus.SAFE
    assert safe.min_margin_m == pytest.approx(0.2)
    assert safe.sample_count == 2

    unsafe = assess_road_containment(
        [
            RoadContainmentSample(0, (0.0, 0.0, 0.0), True, margin_m=0.4),
            RoadContainmentSample(
                1,
                (0.5, 0.0, 0.0),
                True,
                margin_m=-0.01,
                reason="footprint_corner_outside_lane",
            ),
        ]
    )
    assert unsafe.status is AssessmentStatus.UNSAFE
    assert unsafe.first_bad_sample_index == 1
    assert unsafe.reason_codes == (
        "road_not_contained",
        "negative_road_margin",
        "footprint_corner_outside_lane",
    )

    unknown = assess_road_containment(
        [RoadContainmentSample(3, (1.5, 0.0, 0.0), None, reason="map_query_failed")]
    )
    assert unknown.status is AssessmentStatus.UNKNOWN
    assert unknown.first_bad_sample_index == 3
    assert assess_road_containment([]).status is AssessmentStatus.UNKNOWN


def test_stopping_envelope_ttc_and_obb_corridor_filtering():
    ego = _ego(speed=5.0)
    policy = SafetyPolicy()
    assert stopping_distance_m(5.0, policy) == pytest.approx(7.625)

    inside_envelope = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(2, (11.0, 0.0))],
        policy=policy,
    )
    assert inside_envelope.status is AssessmentStatus.UNSAFE
    assert inside_envelope.primary_threat.surface_gap_m == pytest.approx(7.0)
    assert "actor_within_stopping_envelope" in inside_envelope.reason_codes

    ttc_only = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(3, (12.0, 0.0), velocity_xy=(0.0, 0.0))],
        policy=policy,
    )
    assert ttc_only.primary_threat.surface_gap_m == pytest.approx(8.0)
    assert ttc_only.primary_threat.ttc_s == pytest.approx(1.6)
    assert ttc_only.reason_codes == ("actor_ttc_below_threshold",)

    adjacent = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(4, (8.0, 3.0))],
        policy=policy,
    )
    assert adjacent.status is AssessmentStatus.SAFE

    behind = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(5, (-10.0, 0.0))],
        policy=policy,
    )
    assert behind.status is AssessmentStatus.SAFE

    receding_beyond_envelope = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(6, (12.0, 0.0), velocity_xy=(6.0, 0.0))],
        policy=policy,
    )
    assert receding_beyond_envelope.status is AssessmentStatus.SAFE

    rotated_wide_projection = assess_obstacles(
        ego=ego,
        path_points=_straight_path(),
        actors=[_actor(7, (8.0, 3.0), yaw_rad=math.pi / 2.0)],
        policy=policy,
    )
    assert rotated_wide_projection.status is AssessmentStatus.UNSAFE


def test_constant_velocity_prediction_detects_a_crossing_actor():
    ego = _ego(speed=2.5)
    crossing_actor = _actor(
        9,
        (5.0, 5.0),
        velocity_xy=(0.0, -2.5),
        half_length_m=0.4,
        half_width_m=0.4,
        type_id="walker.pedestrian.test",
    )
    path = ((2.5, 0.0), (5.0, 0.0), (7.5, 0.0))

    without_prediction = assess_obstacles(
        ego=ego,
        path_points=path,
        actors=[crossing_actor],
    )
    assert without_prediction.status is AssessmentStatus.SAFE
    assert without_prediction.prediction_performed is False

    with_prediction = assess_obstacles(
        ego=ego,
        path_points=(point for point in path),
        path_times_s=(1.0, 2.0, 3.0),
        actors=[crossing_actor],
    )
    assert with_prediction.status is AssessmentStatus.UNSAFE
    assert with_prediction.prediction_performed is True
    assert with_prediction.primary_threat.actor_id == 9
    assert "predicted_actor_conflict" in with_prediction.reason_codes
    assert with_prediction.primary_threat.predicted_conflict_time_s == pytest.approx(
        1.5,
        abs=0.21,
    )


@pytest.mark.parametrize(
    ("actors", "expected_reason"),
    [
        (None, "obstacle_data_unavailable"),
        ([_actor(2, (10.0, 0.0), frame_id=9)], "actor_snapshot_frame_mismatch"),
        ([object()], "invalid_obstacle_type"),
    ],
)
def test_obstacle_inputs_fail_closed_to_unknown(actors, expected_reason):
    assessment = assess_obstacles(
        ego=_ego(frame_id=10),
        path_points=_straight_path(),
        actors=actors,
    )

    assert assessment.status is AssessmentStatus.UNKNOWN
    assert expected_reason in assessment.reason_codes


def test_stop_only_arbiter_preserves_all_three_control_layers_and_steering_policy():
    shield = StopOnlySafetyShield(
        SafetyPolicy(emergency_hold_ticks=1, clear_ticks_to_release=1)
    )
    safe_road = RoadContainmentAssessment.safe(sample_count=10, min_margin_m=0.5)
    safe_obstacles = ObstacleAssessment.safe(evaluated_actor_count=3)
    requested = ControlCommand(0.4, 0.6, 0.0)
    nominal = ControlCommand(0.3, 0.4, 0.0)

    safe = shield.decide(
        road=safe_road,
        obstacles=safe_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )
    assert safe.safety_override_applied is False
    assert safe.controller_requested_control == requested
    assert safe.nominal_control == nominal
    assert safe.applied_control == nominal
    assert safe.applied_control_source == "CONTROLLER_EXECUTION"

    road_failure = shield.decide(
        road=RoadContainmentAssessment.unsafe(("footprint_corner_outside_lane",)),
        obstacles=safe_obstacles,
        controller_requested_control=ControlCommand(-0.8, 0.6, 0.0),
        nominal_control=ControlCommand(-0.7, 0.4, 0.0),
    )
    assert road_failure.applied_control == ControlCommand.full_brake(0.3)
    assert road_failure.applied_control_source == "SAFETY_OVERRIDE"
    assert road_failure.primary_reason == "road_containment_failed"
    assert road_failure.reason_codes == (
        "road_containment_failed",
        "footprint_corner_outside_lane",
    )

    shield.reset()
    obstacle_assessment = assess_obstacles(
        ego=_ego(),
        path_points=_straight_path(),
        actors=[_actor(2, (3.0, 0.0))],
    )
    obstacle_failure = shield.decide(
        road=safe_road,
        obstacles=obstacle_assessment,
        controller_requested_control=ControlCommand(-0.3, 0.5, 0.0),
        nominal_control=ControlCommand(-0.2, 0.3, 0.0),
    )
    assert obstacle_failure.applied_control == ControlCommand.full_brake(-0.2)
    assert obstacle_failure.primary_reason == "actor_overlap"

    payload = obstacle_failure.to_json_dict()
    assert payload["controller_requested_control"]["throttle"] == 0.5
    assert payload["nominal_control"]["throttle"] == 0.3
    assert payload["applied_control"] == {
        "steering": -0.2,
        "throttle": 0.0,
        "brake": 1.0,
    }
    assert payload["road_containment"]["status"] == "SAFE"
    assert payload["obstacle_assessment"]["primary_threat"]["actor_id"] == 2
    assert payload["obstacle_assessment"]["primary_threat"]["surface_gap_m"] < 0.0
    json.dumps(payload, allow_nan=False)


def test_unknown_assessment_and_missing_nominal_control_apply_full_brake():
    shield = StopOnlySafetyShield()
    decision = shield.decide(
        road=RoadContainmentAssessment.safe(),
        obstacles=ObstacleAssessment.unknown(("actor_query_failed",)),
        controller_requested_control=None,
        nominal_control=None,
    )

    assert decision.safety_override_applied is True
    assert decision.applied_control == ControlCommand.full_brake()
    assert decision.primary_reason == "obstacle_assessment_unknown"
    assert decision.reason_codes[:3] == (
        "obstacle_assessment_unknown",
        "nominal_control_unavailable",
        "controller_request_unavailable",
    )
    assert "actor_query_failed" in decision.reason_codes


def test_explicit_stop_fallback_is_not_misattributed_to_safety_override():
    shield = StopOnlySafetyShield()
    fallback = shield.decide_fallback(
        controller_requested_control=None,
        nominal_control=ControlCommand.full_brake(),
        reason="waiting_for_valid_plan",
    )

    assert fallback.applied_control == ControlCommand.full_brake()
    assert fallback.controller_requested_control is None
    assert fallback.applied_control_source == "FALLBACK"
    assert fallback.safety_override_applied is False
    assert fallback.latched is False
    assert fallback.road_containment.quality == "not_evaluated_stop_fallback"


def test_explicit_stop_fallback_retains_an_existing_safety_latch():
    shield = StopOnlySafetyShield()
    shield.decide(
        road=RoadContainmentAssessment.unknown(("map_query_failed",)),
        obstacles=ObstacleAssessment.safe(),
        controller_requested_control=ControlCommand(0.0, 0.2, 0.0),
        nominal_control=ControlCommand(0.0, 0.2, 0.0),
    )

    fallback = shield.decide_fallback(
        controller_requested_control=None,
        nominal_control=ControlCommand.full_brake(),
    )

    assert shield.latched is True
    assert fallback.latched is True
    assert fallback.applied_control_source == "FALLBACK"
    assert fallback.safety_override_applied is False


def test_emergency_brake_latch_requires_hold_and_consecutive_clear_ticks():
    policy = SafetyPolicy(emergency_hold_ticks=3, clear_ticks_to_release=2)
    shield = StopOnlySafetyShield(policy)
    safe_road = RoadContainmentAssessment.safe()
    safe_obstacles = ObstacleAssessment.safe()
    unknown_obstacles = ObstacleAssessment.unknown(("snapshot_missing",))
    requested = ControlCommand(0.1, 0.4, 0.0)
    nominal = ControlCommand(0.1, 0.3, 0.0)

    triggered = shield.decide(
        road=safe_road,
        obstacles=unknown_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )
    assert triggered.latched is True

    clear_1 = shield.decide(
        road=safe_road,
        obstacles=safe_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )
    clear_2 = shield.decide(
        road=safe_road,
        obstacles=safe_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )
    clear_3 = shield.decide(
        road=safe_road,
        obstacles=safe_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )

    assert clear_1.safety_override_applied is True
    assert clear_2.safety_override_applied is True
    assert "emergency_brake_latched" in clear_2.reason_codes
    assert clear_3.safety_override_applied is False
    assert shield.latched is False

    shield.decide(
        road=None,
        obstacles=safe_obstacles,
        controller_requested_control=requested,
        nominal_control=nominal,
    )
    assert shield.latched is True
    shield.reset()
    assert shield.latched is False


def test_safety_records_are_immutable_and_control_values_are_validated():
    assessment = RoadContainmentAssessment.safe()
    with pytest.raises(FrozenInstanceError):
        assessment.status = AssessmentStatus.UNSAFE
    with pytest.raises(ValueError, match="throttle"):
        ControlCommand(0.0, 1.1, 0.0)
    with pytest.raises(ValueError, match="no greater than 0.5"):
        SafetyPolicy(path_sample_spacing_m=0.6)

    invalid_nominal = StopOnlySafetyShield().decide(
        road=RoadContainmentAssessment.safe(),
        obstacles=ObstacleAssessment.safe(),
        controller_requested_control=ControlCommand(0.0, 0.2, 0.2),
        nominal_control=ControlCommand(0.0, 0.1, 0.1),
    )
    assert invalid_nominal.safety_override_applied is True
    assert "invalid_nominal_control" in invalid_nominal.reason_codes
    assert invalid_nominal.applied_control == ControlCommand.full_brake()

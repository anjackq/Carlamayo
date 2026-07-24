import pytest

from module.plan_handoff import PlanHandoffStatus, decide_plan_handoff


def _decision(**overrides):
    values = {
        "candidate_plan_id": "candidate",
        "candidate_admission_status": "ACCEPT_SAFE_PREFIX",
        "candidate_motion_class": "MOVING",
        "candidate_reserve_status": "ROBUST",
        "candidate_distance_to_first_bad_m": 12.0,
        "candidate_time_to_first_bad_s": 3.0,
        "candidate_explicit_stop": False,
        "active_plan_id": "active",
        "active_admission_status": "ACCEPT_SAFE_PREFIX",
        "active_motion_class": "MOVING",
        "active_reserve_status": "ROBUST",
        "active_distance_to_first_bad_m": 12.0,
        "active_time_to_first_bad_s": 3.0,
        "active_remaining_horizon_s": 4.0,
        "active_executable": True,
        "verified_empty_road": True,
    }
    values.update(overrides)
    return decide_plan_handoff(**values)


def test_no_executable_active_plan_activates_candidate():
    decision = _decision(active_plan_id=None, active_executable=False)

    assert decision.status is PlanHandoffStatus.ACTIVATE_NO_ACTIVE
    assert decision.activate_candidate is True


def test_candidate_safety_improvement_activates_immediately():
    decision = _decision(
        candidate_reserve_status="ROBUST",
        active_reserve_status="FRAGILE",
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_SAFETY_IMPROVEMENT


def test_fragile_candidate_cannot_replace_robust_active_plan():
    decision = _decision(
        candidate_reserve_status="FRAGILE",
        active_reserve_status="ROBUST",
    )

    assert decision.status is PlanHandoffStatus.RETAIN_ACTIVE_STOPPING_RESERVE
    assert decision.retain_active is True


def test_material_safe_prefix_headroom_regression_retains_active_plan():
    decision = _decision(
        candidate_distance_to_first_bad_m=10.627130203753559,
        candidate_time_to_first_bad_s=2.4499999999999993,
        active_distance_to_first_bad_m=19.19592915328279,
        active_time_to_first_bad_s=3.649999970197677,
        active_remaining_horizon_s=4.399999970197676,
    )

    assert decision.status is PlanHandoffStatus.RETAIN_ACTIVE_SAFETY_HEADROOM
    assert decision.activate_candidate is False
    assert decision.reason == "candidate_materially_reduces_safety_headroom"


def test_small_safe_prefix_headroom_change_still_activates_fresh_candidate():
    decision = _decision(
        candidate_distance_to_first_bad_m=10.1,
        candidate_time_to_first_bad_s=2.6,
        active_distance_to_first_bad_m=12.0,
        active_time_to_first_bad_s=3.0,
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_FRESH


def test_missing_safe_prefix_headroom_boundedly_retains_active_plan():
    decision = _decision(candidate_distance_to_first_bad_m=None)

    assert decision.status is PlanHandoffStatus.RETAIN_ACTIVE_SAFETY_HEADROOM
    assert decision.reason == "safe_prefix_headroom_unavailable"


def test_worse_admission_tier_cannot_replace_fully_safe_active_plan():
    decision = _decision(
        candidate_admission_status="ACCEPT_SAFE_PREFIX",
        active_admission_status="ACCEPT_FULLY_SAFE",
        active_reserve_status="UNBOUNDED",
        active_distance_to_first_bad_m=None,
        active_time_to_first_bad_s=None,
    )

    assert decision.status is PlanHandoffStatus.RETAIN_ACTIVE_STOPPING_RESERVE


def test_empty_road_worse_motion_is_boundedly_retained():
    decision = _decision(
        candidate_motion_class="CREEP_OR_STALL",
        active_motion_class="MOVING",
    )

    assert decision.status is PlanHandoffStatus.RETAIN_ACTIVE_MOTION_QUALITY


def test_normal_traffic_does_not_apply_motion_retention():
    decision = _decision(
        candidate_motion_class="CREEP_OR_STALL",
        active_motion_class="MOVING",
        verified_empty_road=False,
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_FRESH


@pytest.mark.parametrize("remaining_horizon", [3.0, 2.9])
def test_retention_deadline_forces_handoff(remaining_horizon):
    decision = _decision(
        candidate_motion_class="CREEP_OR_STALL",
        active_motion_class="MOVING",
        active_remaining_horizon_s=remaining_horizon,
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_RETENTION_DEADLINE
    assert decision.activate_candidate is True


def test_retention_deadline_overrides_safety_headroom_regression():
    decision = _decision(
        candidate_distance_to_first_bad_m=8.0,
        candidate_time_to_first_bad_s=2.0,
        active_distance_to_first_bad_m=20.0,
        active_time_to_first_bad_s=5.0,
        active_remaining_horizon_s=3.0,
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_RETENTION_DEADLINE
    assert decision.activate_candidate is True


def test_explicit_stop_bypasses_motion_and_reserve_retention():
    decision = _decision(
        candidate_motion_class="EXPLICIT_STOP",
        candidate_reserve_status="FRAGILE",
        candidate_explicit_stop=True,
        active_motion_class="MOVING",
        active_reserve_status="ROBUST",
    )

    assert decision.status is PlanHandoffStatus.ACTIVATE_FRESH
    assert decision.activate_candidate is True


def test_invalid_candidate_never_replaces_active_plan():
    decision = _decision(candidate_admission_status="REJECT_RETAIN_ACTIVE")

    assert decision.status is PlanHandoffStatus.NO_EXECUTABLE_PLAN
    assert decision.activate_candidate is False

import pytest

from module.active_plan_availability import (
    ActivePlanAvailabilityStatus,
    decide_active_plan_availability,
)
from module.carla_safety_adapter import (
    RoadExecutionEnvelope,
    StoppingReserveProfile,
    StoppingReserveStatus,
)
from module.safety_shield import RoadContainmentAssessment
from module.trajectory_runtime import (
    PlanAlignmentValidity,
    PlanExecutionValidity,
)


def _validity(
    *,
    valid=True,
    reason=None,
    age=4.5,
    remaining=1.9,
    first_future=45,
):
    return PlanExecutionValidity(
        valid=valid,
        rejection_reason=reason,
        source_age_s=age,
        remaining_horizon_s=remaining,
        first_future_index=first_future if valid else None,
    )


def _alignment(*, valid=True, reason=None):
    return PlanAlignmentValidity(
        valid=valid,
        rejection_reason=reason,
        tracking_error_m=0.1,
        heading_error_deg=0.5,
    )


def _envelope(
    *,
    current=None,
    clearance=None,
    near=None,
    full=None,
    reserve=StoppingReserveStatus.UNBOUNDED,
    emergency=False,
    recovery=False,
    last_safe=63,
):
    safe = RoadContainmentAssessment.safe(sample_count=5)
    return RoadExecutionEnvelope(
        current_ego_road=current or safe,
        current_ego_clearance_road=clearance or safe,
        near_term_path_road=near or safe,
        full_path_road=full or safe,
        last_safe_waypoint_index=last_safe,
        time_to_first_bad_s=None,
        distance_to_first_bad_m=None,
        target_speed_cap_mps=None,
        emergency_required=emergency,
        recovery_required=recovery,
        stopping_reserve_profile=StoppingReserveProfile(
            raw_physical_stopping_cap_mps=None,
            target_speed_cap_mps=None,
            guard_speed_mps=2.0,
            required_stopping_distance_m=None,
            stopping_reserve_m=None,
            status=reserve,
        ),
    )


def _decide(**overrides):
    values = {
        "active_plan_present": True,
        "standard_validity": _validity(
            valid=False,
            reason="plan_source_age_exceeded",
        ),
        "bridge_validity": _validity(valid=True),
        "alignment_validity": _alignment(),
        "road_envelope": _envelope(),
        "bridge_deadline_age_s": 4.9,
        "bridge_min_remaining_horizon_s": 1.5,
    }
    values.update(overrides)
    return decide_active_plan_availability(**values)


def test_standard_active_plan_does_not_attempt_bridge():
    decision = _decide(
        standard_validity=_validity(valid=True, age=4.4, remaining=2.0),
    )

    assert decision.status is ActivePlanAvailabilityStatus.STANDARD_EXECUTION
    assert decision.execution_allowed
    assert not decision.bridge_attempted
    assert not decision.bridge_active


@pytest.mark.parametrize(
    ("age", "remaining"),
    [(4.5, 1.9), (4.9, 1.5)],
)
def test_full_safe_unbounded_active_plan_can_use_bounded_bridge(age, remaining):
    decision = _decide(
        standard_validity=_validity(
            valid=False,
            reason="plan_source_age_exceeded",
            age=age,
            remaining=remaining,
        ),
        bridge_validity=_validity(valid=True, age=age, remaining=remaining),
    )

    assert decision.status is ActivePlanAvailabilityStatus.BRIDGED_FULL_SAFE
    assert decision.execution_allowed
    assert decision.bridge_active
    assert decision.original_rejection_reason == "plan_source_age_exceeded"


def test_bridge_deadline_remains_fail_closed():
    decision = _decide(
        standard_validity=_validity(
            valid=False,
            reason="plan_source_age_exceeded",
            age=5.0,
            remaining=1.4,
        ),
        bridge_validity=_validity(
            valid=False,
            reason="plan_source_age_exceeded",
            age=5.0,
            remaining=1.4,
        ),
    )

    assert decision.status is ActivePlanAvailabilityStatus.BRIDGE_DENIED
    assert not decision.execution_allowed
    assert decision.denial_reason == "plan_source_age_exceeded"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "bridge_deadline_age_s": float("nan"),
            },
            "invalid_bridge_policy_bounds",
        ),
        (
            {
                "bridge_min_remaining_horizon_s": -1.0,
            },
            "invalid_bridge_policy_bounds",
        ),
        (
            {
                "bridge_validity": _validity(
                    valid=True,
                    age=99.0,
                    remaining=1.9,
                ),
            },
            "bridge_deadline_exceeded",
        ),
        (
            {
                "bridge_validity": _validity(
                    valid=True,
                    age=4.5,
                    remaining=0.01,
                ),
            },
            "bridge_horizon_below_minimum",
        ),
        (
            {
                "bridge_validity": _validity(
                    valid=True,
                    first_future=None,
                ),
            },
            "invalid_bridge_timing",
        ),
        (
            {
                "bridge_validity": _validity(
                    valid=True,
                    first_future=60,
                ),
                "road_envelope": _envelope(last_safe=59),
            },
            "bridge_authorized_prefix_exhausted",
        ),
    ],
)
def test_bridge_policy_enforces_its_own_bounds(overrides, reason):
    decision = _decide(**overrides)

    assert decision.status is ActivePlanAvailabilityStatus.BRIDGE_DENIED
    assert not decision.execution_allowed
    assert decision.denial_reason == reason


@pytest.mark.parametrize(
    "reason",
    [
        "prompt_revision_mismatch",
        "respawn_revision_mismatch",
        "invalid_plan_geometry",
        "source_time_in_future",
        "trajectory_exhausted",
    ],
)
def test_non_freshness_failures_never_attempt_bridge(reason):
    decision = _decide(
        standard_validity=_validity(valid=False, reason=reason),
    )

    assert decision.status is ActivePlanAvailabilityStatus.BRIDGE_DENIED
    assert not decision.execution_allowed
    assert not decision.bridge_attempted
    assert decision.denial_reason == reason


def test_alignment_failure_denies_bridge():
    decision = _decide(
        alignment_validity=_alignment(
            valid=False,
            reason="trajectory_tracking_error_exceeded",
        ),
    )

    assert not decision.execution_allowed
    assert decision.denial_reason == "trajectory_tracking_error_exceeded"


@pytest.mark.parametrize(
    ("envelope", "reason"),
    [
        (None, "bridge_road_assessment_unavailable"),
        (
            _envelope(
                current=RoadContainmentAssessment.unknown(("map_error",)),
            ),
            "bridge_current_ego_not_safe",
        ),
        (
            _envelope(
                near=RoadContainmentAssessment.unsafe(("road_not_contained",)),
            ),
            "bridge_near_term_not_safe",
        ),
        (
            _envelope(
                full=RoadContainmentAssessment.unsafe(("road_not_contained",)),
            ),
            "bridge_full_path_not_safe",
        ),
        (
            _envelope(reserve=StoppingReserveStatus.FRAGILE),
            "bridge_reserve_not_unbounded",
        ),
        (
            _envelope(reserve=StoppingReserveStatus.RECOVERY, recovery=True),
            "bridge_recovery_not_authorized",
        ),
        (_envelope(emergency=True), "bridge_emergency_required"),
        (_envelope(last_safe=None), "bridge_no_authorized_waypoint"),
    ],
)
def test_road_or_reserve_failure_denies_bridge(envelope, reason):
    decision = _decide(road_envelope=envelope)

    assert not decision.execution_allowed
    assert decision.denial_reason == reason


def test_no_active_plan_has_explicit_status():
    decision = _decide(
        active_plan_present=False,
        standard_validity=None,
        bridge_validity=None,
        alignment_validity=None,
        road_envelope=None,
    )

    assert decision.status is ActivePlanAvailabilityStatus.NO_ACTIVE_PLAN
    assert not decision.execution_allowed
    assert decision.denial_reason == "no_active_plan"

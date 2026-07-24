import json
from pathlib import Path

import pytest

from module.plan_handoff import PlanHandoffStatus, decide_plan_handoff


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22863527_standby_handoff.json"
)


def _handoff(
    replay,
    *,
    active_remaining_horizon_s,
):
    active = replay["active"]
    candidate = replay["retained_candidate"]
    return decide_plan_handoff(
        candidate_plan_id=candidate["plan_id"],
        candidate_admission_status=candidate["admission_status"],
        candidate_motion_class=candidate["motion_class"],
        candidate_reserve_status=candidate["reserve_status"],
        candidate_distance_to_first_bad_m=(
            candidate["distance_to_first_bad_m"]
        ),
        candidate_time_to_first_bad_s=candidate["time_to_first_bad_s"],
        candidate_explicit_stop=candidate["explicit_stop"],
        active_plan_id=active["plan_id"],
        active_admission_status=active["admission_status"],
        active_motion_class=active["motion_class"],
        active_reserve_status=active["reserve_status"],
        active_distance_to_first_bad_m=active["distance_to_first_bad_m"],
        active_time_to_first_bad_s=active["time_to_first_bad_s"],
        active_remaining_horizon_s=active_remaining_horizon_s,
        active_executable=active["executable"],
        verified_empty_road=replay["scenario"]["empty_road"],
        retention_deadline_s=replay["retention_deadline"]["threshold_s"],
    )


def test_job_22863527_retained_candidate_is_activated_at_deadline():
    replay = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed = replay["observed_handoff"]

    arrival_decision = _handoff(
        replay,
        active_remaining_horizon_s=(
            replay["active"]["remaining_horizon_s_at_retention"]
        ),
    )
    assert arrival_decision.status.value == observed["status"]
    assert arrival_decision.reason == observed["reason"]
    assert arrival_decision.activate_candidate is observed["activate_candidate"]
    assert arrival_decision.retain_active is observed["retain_active"]

    deadline = replay["retention_deadline"]
    deadline_decision = _handoff(
        replay,
        active_remaining_horizon_s=deadline["active_remaining_horizon_s"],
    )
    expected = replay["expected_with_standby"]
    assert deadline_decision.status is (
        PlanHandoffStatus.ACTIVATE_RETENTION_DEADLINE
    )
    assert deadline_decision.status.value == expected[
        "activation_handoff_status"
    ]
    assert deadline_decision.activate_candidate is True
    assert deadline_decision.reason == (
        "active_plan_retention_deadline_reached"
    )


def test_job_22863527_deadline_precedes_observed_fallback_gap():
    replay = json.loads(FIXTURE.read_text(encoding="utf-8"))
    deadline = replay["retention_deadline"]
    candidate = replay["retained_candidate"]
    observed = replay["observed_without_standby"]

    source_age = (
        deadline["simulation_time_s"] - candidate["source_simulation_time_s"]
    )
    remaining_horizon = (
        candidate["horizon_end_s"] - deadline["simulation_time_s"]
    )
    assert source_age == pytest.approx(deadline["standby_source_age_s"])
    assert remaining_horizon == pytest.approx(
        deadline["standby_remaining_horizon_s"]
    )
    assert deadline["loop_tick_id"] < observed["first_fallback_tick"]
    assert observed["last_fallback_tick"] - observed["first_fallback_tick"] + 1 == (
        observed["gap_fallback_tick_count"]
    )
    assert observed["gap_applied_control"]["brake"] == 1.0
    assert observed["gap_applied_control"]["throttle"] == 0.0


def test_standby_fixture_is_compact_and_requires_fresh_road_authority():
    replay = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidate = replay["retained_candidate"]
    expected = replay["expected_with_standby"]

    assert candidate["current_ego_road_status"] == "SAFE"
    assert candidate["near_term_road_status"] == "SAFE"
    assert candidate["full_path_road_status"] == "UNSAFE"
    assert candidate["reserve_status"] == "ROBUST"
    assert candidate["emergency_required"] is False
    assert expected["activation_requires_fresh_exact_validation"] is True

    serialized = FIXTURE.read_text(encoding="utf-8")
    assert "trajectory_points" not in serialized
    assert "camera_frames" not in serialized
    assert "road_envelope" not in serialized
    assert "coc_text" not in serialized
    assert len(replay["retention_event"]["coc_sha256"]) == 64

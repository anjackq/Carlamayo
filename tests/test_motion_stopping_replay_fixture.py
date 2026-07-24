import json
from pathlib import Path

import numpy as np
import pytest

from module.candidate_selector import CandidateEvaluation, rank_candidate_evaluations
from module.carla_safety_adapter import raw_physical_stopping_speed_cap_mps
from module.safety_shield import SafetyPolicy
from module.trajectory_runtime import (
    TrajectoryMotionClass,
    TrajectoryValidationError,
    build_fixed_world_trajectory,
    compute_trajectory_motion_profile,
)
from scripts.build_motion_stopping_replay_fixture import validate_fixture


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "motion_stopping_baseline_v1.json"
)


@pytest.fixture(scope="module")
def fixture():
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_fixture(payload)
    return payload


def _case(payload, case_id):
    return next(case for case in payload["cases"] if case["case_id"] == case_id)


def _plan(case, candidate):
    return build_fixed_world_trajectory(
        plan_id=f"{case['case_id']}/{candidate['candidate_index']}",
        source_frame_id=0,
        source_simulation_time_s=case["source_simulation_time_s"],
        capture_pose_world=np.eye(4),
        model_points=candidate["trajectory_model"],
        coc_text="",
        prompt_revision=0,
        respawn_revision=0,
        selected_candidate_index=candidate["candidate_index"],
    )


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("delayed_start", TrajectoryMotionClass.DELAYED_START),
        ("creep_or_stall", TrajectoryMotionClass.CREEP_OR_STALL),
        ("explicit_stop", TrajectoryMotionClass.EXPLICIT_STOP),
    ],
)
def test_real_baseline_motion_cases_classify_as_expected(fixture, case_id, expected):
    case = _case(fixture, case_id)
    candidate = case["candidates"][0]

    profile = compute_trajectory_motion_profile(
        _plan(case, candidate),
        case["source_simulation_time_s"],
    )

    assert profile.motion_class is expected


def test_cap_saturation_ticks_no_longer_require_emergency(fixture):
    policy = SafetyPolicy()
    evidence = [
        tick
        for case in fixture["cases"]
        if case["expected_kind"] == "CAP_SATURATION_FALSE_EMERGENCY"
        for tick in case["tick_evidence"]
    ]

    assert len(evidence) == 3
    for tick in evidence:
        raw_cap = raw_physical_stopping_speed_cap_mps(
            tick["distance_to_first_bad_m"],
            policy,
        )
        assert tick["legacy_emergency_required"] is True
        assert tick["direct_safety_trigger"] is True
        assert tick["speed_mps"] > tick["legacy_target_speed_cap_mps"]
        assert tick["speed_mps"] <= raw_cap + 0.1


def test_robust_alternative_replay_outranks_fragile_candidate(fixture):
    case = _case(fixture, "robust_alternative_batch")
    evaluations = []
    reserves = {}
    for candidate in case["candidates"]:
        points = np.asarray(candidate["trajectory_model"], dtype=np.float64)
        motion = compute_trajectory_motion_profile(
            _plan(case, candidate),
            case["source_simulation_time_s"],
        )
        guard_speed = (
            max(
                case["source_speed_mps"],
                min(motion.near_term_peak_speed_mps, 35.0 / 3.6),
            )
            + 5.0 * 0.1
        )
        required_distance = (
            2.0 + guard_speed * 0.5 + guard_speed * guard_speed / 8.0
        )
        reserve = (
            candidate["road_envelope"]["distance_to_first_bad_m"]
            - required_distance
        )
        status = "ROBUST" if reserve >= 0.0 else "FRAGILE"
        reserves[candidate["candidate_index"]] = reserve
        lateral_index = int(np.argmax(np.abs(points[:, 1])))
        evaluations.append(
            CandidateEvaluation(
                candidate_index=candidate["candidate_index"],
                plan_id=f"candidate-{candidate['candidate_index']}",
                admission_status=candidate["admission_status"],
                rejection_reason=candidate["rejection_reason"],
                stop_requested=candidate["stop_requested"],
                forward_progress_m=float(np.max(points[:, 0])),
                representative_lateral_m=float(points[lateral_index, 1]),
                full_path_margin_m=candidate["road_envelope"][
                    "full_path_margin_m"
                ],
                continuity_m=candidate["continuity_m"],
                motion_class=motion.motion_class.value,
                initial_target_speed_mps=motion.initial_target_speed_mps,
                stopping_reserve_status=status,
                stopping_reserve_m=reserve,
                time_to_first_bad_s=candidate["road_envelope"][
                    "time_to_first_bad_s"
                ],
            )
        )

    selection = rank_candidate_evaluations(
        evaluations,
        navigation_text="Continue in the current lane.",
        prefer_moving=True,
        current_speed_mps=case["source_speed_mps"],
    )

    assert reserves[0] == pytest.approx(-0.41, abs=0.1)
    assert reserves[2] == pytest.approx(5.01, abs=0.1)
    assert selection.selected_index == 2


def test_all_invalid_real_batch_remains_fail_closed(fixture):
    case = _case(fixture, "all_invalid_batch")

    assert len(case["candidates"]) == 3
    assert all(
        candidate["rejection_reason"] == "excessive_lateral_displacement"
        for candidate in case["candidates"]
    )
    for candidate in case["candidates"]:
        with pytest.raises(
            TrajectoryValidationError,
            match="excessive_lateral_displacement",
        ):
            _plan(case, candidate)

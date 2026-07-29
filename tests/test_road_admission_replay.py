import json
from pathlib import Path

import pytest

from module.carla_safety_adapter import (
    PlanAdmissionStatus,
    RoadExecutionEnvelope,
    decide_plan_admission,
    road_stopping_speed_cap_mps,
)
from module.safety_shield import RoadContainmentAssessment, SafetyPolicy


FIXTURE = Path(__file__).parent / "fixtures" / "job_22850036_road_admission.json"


def test_job_22850036_far_future_violations_replay_as_safe_prefixes():
    replay = json.loads(FIXTURE.read_text(encoding="utf-8"))
    policy = SafetyPolicy(
        reaction_time_s=replay["policy"]["reaction_time_s"],
        assumed_deceleration_mps2=replay["policy"]["assumed_deceleration_mps2"],
        stop_buffer_m=replay["policy"]["stop_buffer_m"],
    )

    admissions = []
    caps = {}
    for proposal in replay["proposals"]:
        cap = road_stopping_speed_cap_mps(
            proposal["distance_to_first_bad_m"],
            policy,
        )
        envelope = RoadExecutionEnvelope(
            current_ego_road=RoadContainmentAssessment.safe(sample_count=5),
            near_term_path_road=RoadContainmentAssessment.safe(sample_count=75),
            full_path_road=RoadContainmentAssessment.unsafe(
                ("road_not_contained",),
                min_margin_m=proposal["min_margin_m"],
            ),
            last_safe_waypoint_index=40,
            time_to_first_bad_s=proposal["time_to_first_bad_s"],
            distance_to_first_bad_m=proposal["distance_to_first_bad_m"],
            target_speed_cap_mps=cap,
            emergency_required=False,
        )
        admission = decide_plan_admission(envelope)
        admissions.append(admission.value)
        caps[proposal["proposal_number"]] = cap

        assert proposal["time_to_first_bad_s"] > replay["policy"]["execution_horizon_s"]
        assert admission is PlanAdmissionStatus.ACCEPT_SAFE_PREFIX
        assert admission.value == proposal["expected_admission"]

    assert admissions == ["ACCEPT_SAFE_PREFIX"] * 5
    assert caps[12] == pytest.approx(1.2496, abs=0.01)
    assert caps[12] < caps[2]
    assert caps[12] < caps[3]
    assert caps[12] < caps[7]
    assert caps[12] < caps[16]

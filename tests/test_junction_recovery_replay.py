import json
from pathlib import Path

from module.carla_safety_adapter import (
    PlanAdmissionStatus,
    RoadExecutionEnvelope,
    decide_plan_admission,
)
from module.safety_shield import RoadContainmentAssessment


FIXTURE = Path(__file__).parent / "fixtures" / "job_22850702_junction_recovery.json"


def _safe(margin):
    return RoadContainmentAssessment.safe(
        sample_count=5,
        min_margin_m=margin,
        quality="carla_ground_truth_physical_surface",
    )


def _unsafe(margin):
    return RoadContainmentAssessment.unsafe(
        ("road_not_contained", "negative_road_margin"),
        sample_count=5,
        min_margin_m=margin,
        first_bad_sample_index=0,
        quality="carla_ground_truth_buffered_clearance",
    )


def test_job_22850702_marginal_junction_proposals_are_recoverable():
    replay = json.loads(FIXTURE.read_text(encoding="utf-8"))
    clearance = replay["planning_lateral_clearance_m"]

    for proposal in replay["proposals"]:
        current_surface_margin = proposal["current_buffered_margin_m"] + clearance
        near_surface_margin = proposal["near_buffered_margin_m"] + clearance
        full_surface = (
            _safe(proposal["full_buffered_margin_m"] + clearance)
            if proposal["full_surface_safe"]
            else _unsafe(proposal["full_buffered_margin_m"] + clearance)
        )
        envelope = RoadExecutionEnvelope(
            current_ego_road=_safe(current_surface_margin),
            current_ego_clearance_road=_unsafe(
                proposal["current_buffered_margin_m"]
            ),
            near_term_path_road=_unsafe(proposal["near_buffered_margin_m"]),
            full_path_road=_unsafe(proposal["full_buffered_margin_m"]),
            near_term_path_surface=_safe(near_surface_margin),
            full_path_surface=full_surface,
            last_safe_waypoint_index=24,
            time_to_first_bad_s=None,
            distance_to_first_bad_m=None,
            target_speed_cap_mps=0.75,
            emergency_required=False,
            junction_context=True,
            recovery_required=True,
        )

        admission = decide_plan_admission(envelope)

        assert current_surface_margin > 0.0
        assert near_surface_margin > 0.0
        assert admission is PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX
        assert admission.value == proposal["expected_admission"]

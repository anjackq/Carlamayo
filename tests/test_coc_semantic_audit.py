from types import SimpleNamespace

from module.coc_semantic_audit import (
    ClaimPolarity,
    ClaimVerdict,
    SceneActorTruth,
    SceneTrafficControlTruth,
    SceneTruthSnapshot,
    audit_coc_semantics,
    parse_coc_claims,
)
from module.route_navigation import (
    NavigationAction,
    NavigationContext,
    RouteTrackerStatus,
)


def _context(action=NavigationAction.RIGHT):
    return NavigationContext(
        source="route",
        text="Turn right at the next junction.",
        weight=1.0,
        conditioning_epoch=2,
        route_id="route",
        maneuver_id="right",
        action=action,
        distance_to_maneuver_m=10.0,
        route_progress_m=20.0,
        route_index=2,
        target_route_index=3,
        lane_change_authorized=False,
        source_frame_id=10,
        source_simulation_time_s=1.0,
        tracker_status=RouteTrackerStatus.AVAILABLE,
    )


def _snapshot(*, actors=(), controls=(), context=None):
    return SceneTruthSnapshot(
        source_frame_id=10,
        source_simulation_time_s=1.0,
        ego_road_id=1,
        ego_section_id=0,
        ego_lane_id=-1,
        ego_is_junction=False,
        navigation_context=_context() if context is None else context,
        dynamic_actors=tuple(actors),
        traffic_controls=tuple(controls),
        strict_lane_policy_enabled=True,
    )


def test_parser_preserves_negate_and_uncertain_polarity():
    claims = parse_coc_claims(
        "There is no lead vehicle. A potential pedestrian may approach."
    )
    assert [(claim.value, claim.polarity) for claim in claims] == [
        ("vehicle", ClaimPolarity.NEGATE),
        ("pedestrian", ClaimPolarity.UNCERTAIN),
    ]


def test_actor_hallucination_and_negation_use_source_truth():
    audit = audit_coc_semantics(
        "A vehicle is ahead. No pedestrian is present.",
        _snapshot(),
    )
    assert [claim.verdict for claim in audit.claims] == [
        ClaimVerdict.CONTRADICTED,
        ClaimVerdict.SUPPORTED,
    ]
    assert audit.to_json_dict()["positive_hallucination_count"] == 1


def test_lane_change_is_policy_conflict_and_turn_uses_route_truth():
    audit = audit_coc_semantics(
        "Change lane left, then turn left.",
        _snapshot(),
    )
    assert audit.claims[0].verdict is ClaimVerdict.POLICY_CONFLICT
    assert audit.claims[1].verdict is ClaimVerdict.CONTRADICTED


def test_traffic_light_claim_requires_route_relevant_matching_control():
    audit = audit_coc_semantics(
        "The light is red.",
        _snapshot(
            controls=(
                SceneTrafficControlTruth("traffic_light", "red", 12.0, True),
            )
        ),
    )
    # The parser accepts both "red light" and "red traffic light".
    assert audit.claims[0].verdict is ClaimVerdict.SUPPORTED


def test_motion_mismatch_and_roundabout_not_evaluable():
    audit = audit_coc_semantics(
        "Stop for the roundabout.",
        _snapshot(),
        motion_profile=SimpleNamespace(
            motion_class="MOVING",
            peak_smoothed_deceleration_mps2=0.0,
            near_term_peak_speed_mps=2.0,
            initial_target_speed_mps=2.0,
        ),
    )
    verdicts = {claim.claim.value: claim.verdict for claim in audit.claims}
    assert verdicts["stop"] is ClaimVerdict.TRAJECTORY_MISMATCH
    assert verdicts["roundabout"] is ClaimVerdict.NOT_EVALUABLE

import json
import math

from scripts.summarize_route_rollout import summarize_route_rollout


def _tick(index, x, speed, **overrides):
    event = {
        "event_type": "tick",
        "loop_tick_id": index,
        "ego_position_world": {"x": float(x), "y": 0.0, "z": 0.0},
        "speed_mps": float(speed),
        "collision_count": 0,
        "applied_control_source": "CONTROLLER",
        "direct_safety_trigger": False,
        "latch_only": False,
        "controller_state": "TRACKING",
        "road_execution_envelope": {
            "current_ego_road": {"status": "SAFE"},
        },
    }
    event.update(overrides)
    return event


def _context(progress, *, text="Turn right at the next junction in 20m."):
    return {
        "source": "route",
        "text": text,
        "action": "RIGHT",
        "distance_to_maneuver_m": 20.0,
        "route_progress_m": float(progress),
        "tracker_status": "AVAILABLE",
    }


def test_route_summary_reports_successful_destination_stop():
    events = [
        {
            "event_type": "episode_start",
            "run_id": "route-test",
            "scenario_seed": 0,
            "route_startup_facts": {"route_length_m": 10.0},
            "navigation_context": _context(0.0),
        },
        {
            "event_type": "inference_submitted",
            "navigation_context": _context(9.6),
        },
        _tick(1, 0.0, 1.0),
        _tick(2, 1.0, 0.0),
        {
            "event_type": "candidate_evaluation",
            "selected": True,
            "admitted": True,
            "candidate_route_assessment": {
                "current_route_status": "MATCH",
                "near_term_route_status": "MATCH",
                "lane_change_detected": False,
            },
            "coc_semantic_audit": {
                "audit_error": None,
                "positive_hallucination_count": 0,
                "verdict_counts": {"SUPPORTED": 1},
            },
        },
        {"event_type": "episode_summary", "stop_reason": "destination_arrived"},
    ]

    summary = summarize_route_rollout(
        events,
        destination_xyz=(1.0, 0.0, 0.0),
    )

    assert math.isclose(summary["route"]["completion_ratio"], 0.96)
    assert summary["motion"]["integrated_distance_m"] == 1.0
    assert summary["route"]["prompt_truth_mismatch_count"] == 0
    assert summary["route"]["accepted_unauthorized_prefix_count"] == 0
    assert summary["gates"]["route_completion_95_percent"]
    assert summary["gates"]["destination_stop"]
    assert summary["coc_audit"]["verdict_counts"] == {"SUPPORTED": 1}


def test_route_summary_detects_absorbing_stop_and_policy_failures():
    events = [
        {
            "event_type": "episode_start",
            "route_startup_facts": {"route_length_m": 100.0},
            "navigation_context": _context(0.0, text="Continue straight."),
        },
        {
            "event_type": "inference_submitted",
            "navigation_context": {
                **_context(30.0),
                "tracker_status": "ROUTE_UNAVAILABLE",
                "text": "",
            },
        },
        *[_tick(index, 30.0, 0.0) for index in range(1, 51)],
        {
            "event_type": "candidate_evaluation",
            "selected": True,
            "admitted": True,
            "candidate_route_assessment": {
                "current_route_status": "MATCH",
                "near_term_route_status": "DEVIATE",
                "lane_change_detected": True,
            },
        },
    ]
    events[2]["direct_safety_trigger"] = True
    events[2]["safety_reason_codes"] = ["route_policy_failed"]
    events[2]["road_execution_envelope"]["current_ego_road"]["status"] = "UNSAFE"

    summary = summarize_route_rollout(
        events,
        destination_xyz=(100.0, 0.0, 0.0),
    )

    assert summary["route"]["prompt_truth_mismatch_count"] == 1
    assert summary["route"]["route_unavailable_context_count"] == 1
    assert summary["route"]["accepted_unauthorized_prefix_count"] == 1
    assert summary["route"]["route_only_emergency_override_ticks"] == 1
    assert summary["safety"]["current_ego_road_unsafe_or_unknown_ticks"] == 1
    assert summary["motion"]["stationary_suffix_ticks"] == 50
    assert summary["motion"]["absorbing_stop"]
    assert not summary["gates"]["no_absorbing_stop"]


def test_route_summary_accepts_active_maneuver_phase_prompt():
    context = {
        **_context(50.0),
        "distance_to_maneuver_m": 0.0,
        "maneuver_phase": "ACTIVE",
        "text": "Follow the current lane through the right turn.",
    }
    summary = summarize_route_rollout(
        [
            {
                "event_type": "episode_start",
                "route_startup_facts": {"route_length_m": 100.0},
                "navigation_context": context,
            }
        ]
    )

    assert summary["route"]["prompt_truth_mismatch_count"] == 0
    assert summary["gates"]["prompt_grounded"]


def test_route_summary_can_be_json_serialized():
    summary = summarize_route_rollout(
        [
            {
                "event_type": "episode_start",
                "route_startup_facts": {"route_length_m": 1.0},
            }
        ]
    )
    json.dumps(summary)


def test_route_summary_reports_reachability_selector_and_sync_contract():
    context = {
        **_context(40.0),
        "distance_to_maneuver_m": 0.0,
        "maneuver_phase": "ACTIVE",
        "text": "Follow the current lane through the right turn.",
    }
    events = [
        {
            "event_type": "episode_start",
            "run_id": "reachability-test",
            "scenario_seed": 0,
            "execution": "sync",
            "candidate_ranking_policy": "reachability-first",
            "trajectory_reachability_audit": True,
            "route_startup_facts": {"route_length_m": 100.0},
            "navigation_context": context,
        },
        {
            "event_type": "alpamayo_proposal",
            "proposal_id": "proposal-1",
            "navigation_context": context,
        },
        {
            "event_type": "candidate_selection",
            "proposal_id": "proposal-1",
            "selected_candidate_index": 1,
            "current_shadow_selected_index": 0,
            "reachability_shadow_selected_index": 1,
            "shadow_selection_changed": True,
            "near_executable_candidate_indices": [1],
            "full_turn_executable_candidate_indices": [1],
            "candidate_ranking_fallback_reason": None,
        },
        {
            "event_type": "candidate_evaluation",
            "proposal_id": "proposal-1",
            "candidate_plan_id": "proposal-1/candidate-1",
            "selected": True,
            "admitted": True,
            "motion_class": "MOVING",
            "near_physical_status": "REACHABLE",
            "trajectory_reachability_profile": {
                "near_physical_status": "REACHABLE",
                "near_source_speed_prior_status": "CONSISTENT",
            },
            "trajectory_reachability_compute_ms": 0.2,
            "candidate_route_assessment": {
                "current_route_status": "MATCH",
                "near_term_route_status": "MATCH",
            },
        },
        {
            "event_type": "plan_handoff",
            "candidate_plan_id": "proposal-1/candidate-1",
            "status": "ACTIVATE_FRESH",
            "activate_candidate": True,
            "retain_active": False,
        },
        {
            "event_type": "inference_result",
            "source_simulation_time_s": 4.0,
            "arrival_simulation_time_s": 4.0,
        },
        _tick(
            1,
            40.0,
            0.0,
            applied_control={"throttle": 0.0, "brake": 0.9},
            controller_debug={
                "low_speed_longitudinal_governor": {
                    "mode": "COAST",
                }
            },
        ),
        _tick(
            2,
            40.5,
            0.6,
            applied_control={"throttle": 0.2, "brake": 0.0},
            controller_debug={
                "low_speed_longitudinal_governor": {
                    "mode": "THROTTLE_LIMITED",
                }
            },
        ),
    ]

    summary = summarize_route_rollout(events)
    policy = summary["candidate_policy"]

    assert policy["near_executable_coverage_at_k"] == 1.0
    assert policy["full_turn_executable_coverage_at_k"] == 1.0
    assert policy["current_near_conditional_misses"] == 1
    assert policy["reachability_near_conditional_misses"] == 0
    assert policy["effective_near_conditional_misses"] == 0
    assert policy["phase"]["RIGHT/ACTIVE"]["requests"] == 1
    assert policy["selected_activated_count"] == 1
    assert summary["motion"]["stop_go_restart_count"] == 1
    assert summary["motion"]["hard_brake_ticks"] == 1
    assert summary["controller"]["low_speed_governor_mode_counts"] == {
        "COAST": 1,
        "THROTTLE_LIMITED": 1,
    }
    assert (
        summary["synchronous_contract"][
            "maximum_inference_simulation_duration_s"
        ]
        == 0.0
    )
    assert summary["gates"]["synchronous_inference_paused"]
    assert summary["gates"]["effective_near_selector_recall"]

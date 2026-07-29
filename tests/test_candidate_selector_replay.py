import json

from scripts.replay_candidate_selector import replay_batch, summarize


def _candidate(index, *, motion, route, branch, cross_track):
    return {
        "candidate_index": index,
        "valid": True,
        "error": None,
        "stop_intent": motion == "EXPLICIT_STOP",
        "forward_progress_m": 10.0,
        "representative_lateral_m": -2.0,
        "motion_profile": {
            "motion_class": motion,
            "initial_target_speed_mps": 2.0,
        },
        "route_assessment": {
            "near_term_route_status": route,
            "full_path_route_status": route,
            "branch_match": branch,
            "maximum_cross_track_error_m": cross_track,
        },
        "reachability_profile": {
            "near_physical_status": "REACHABLE",
            "physical_status": "REACHABLE",
            "near_source_speed_prior_status": "CONSISTENT",
            "near_required_constant_acceleration_mps2": 0.0,
        },
    }


def test_route_first_exposes_motion_before_route_regret():
    event = {
        "fixture_label": "junction",
        "fixture_id": "fixture",
        "seed": 0,
        "navigation_text": "Turn right.",
        "actual_speed_mps": 2.0,
        "candidate_audits": [
            _candidate(
                0,
                motion="MOVING",
                route="DEVIATE",
                branch=False,
                cross_track=2.0,
            ),
            _candidate(
                1,
                motion="DELAYED_START",
                route="MATCH",
                branch=True,
                cross_track=0.2,
            ),
        ],
    }

    replay = replay_batch(event)

    assert replay["current_selected_index"] == 0
    assert replay["route_first_selected_index"] == 1
    assert replay["current_selected_wrong_when_match_available"] is True


def test_selector_replay_summary_reports_conditional_accuracy():
    base = {
        "fixture_label": "junction",
        "fixture_id": "fixture",
        "navigation_text": "Turn right.",
        "actual_speed_mps": 2.0,
        "candidate_audits": [
            _candidate(
                0,
                motion="MOVING",
                route="MATCH",
                branch=True,
                cross_track=0.1,
            ),
            _candidate(
                1,
                motion="MOVING",
                route="DEVIATE",
                branch=False,
                cross_track=1.0,
            ),
        ],
    }
    records = [replay_batch({**base, "seed": seed}) for seed in range(3)]

    summary = summarize(records)

    assert summary["batches"] == 3
    assert summary["route_evaluable_batches"] == 3
    assert summary["current_route_first_accuracy"] == 1.0
    assert summary["current_wrong_when_route_match_available"] == 0


def test_selector_replay_serializes_invalid_candidate_diagnostics():
    event = {
        "fixture_label": "invalid",
        "fixture_id": "fixture",
        "seed": 0,
        "navigation_text": "Continue in the current lane.",
        "actual_speed_mps": 1.0,
        "candidate_audits": [
            _candidate(
                0,
                motion="MOVING",
                route="MATCH",
                branch=True,
                cross_track=0.1,
            ),
            {
                "candidate_index": 1,
                "valid": False,
                "error": "invalid trajectory",
                "stop_intent": False,
                "forward_progress_m": 0.0,
                "representative_lateral_m": 0.0,
                "motion_profile": None,
                "route_assessment": None,
            },
        ],
    }

    replay = replay_batch(event)

    invalid = replay["current_selection"]["candidate_evaluations"][1]
    assert invalid["speed_continuity_rank_mps"] is None
    json.dumps(replay, allow_nan=False)


def test_selector_replay_does_not_claim_route_accuracy_without_route_truth():
    candidate = _candidate(
        0,
        motion="MOVING",
        route=None,
        branch=None,
        cross_track=None,
    )
    event = {
        "fixture_label": "synthetic-history",
        "fixture_id": "fixture",
        "seed": 0,
        "navigation_text": "Continue in the current lane.",
        "actual_speed_mps": 1.0,
        "candidate_audits": [candidate],
    }

    replay = replay_batch(event)
    summary = summarize([replay])

    assert replay["route_evaluable"] is False
    assert summary["route_evaluable_batches"] == 0
    assert summary["current_route_first_accuracy"] is None


def test_reachability_ablation_prefers_reachable_route_matching_candidate():
    impossible = _candidate(
        0,
        motion="MOVING",
        route="MATCH",
        branch=True,
        cross_track=0.1,
    )
    impossible["reachability_profile"].update(
        {
            "near_physical_status": "TOO_LONG",
            "physical_status": "TOO_LONG",
            "near_source_speed_prior_status": "ACCELERATION_PRIOR",
            "near_required_constant_acceleration_mps2": 3.0,
        }
    )
    reachable = _candidate(
        1,
        motion="MOVING",
        route="MATCH",
        branch=True,
        cross_track=0.2,
    )
    event = {
        "fixture_label": "junction",
        "fixture_id": "fixture",
        "seed": 0,
        "navigation_text": "Turn right.",
        "actual_speed_mps": 2.0,
        "candidate_audits": [impossible, reachable],
    }

    replay = replay_batch(event)
    summary = summarize([replay])

    assert replay["current_selected_index"] == 0
    assert replay["reachability_first_selected_index"] == 1
    assert replay["reachable_route_prefix_candidate_indices"] == [1]
    assert (
        replay[
            "reachability_selected_wrong_when_reachable_route_prefix_available"
        ]
        is False
    )
    assert summary["reachability_selection_change_count"] == 1
    assert (
        summary["current_wrong_when_reachable_route_prefix_available"] == 1
    )
    assert (
        summary["reachability_wrong_when_reachable_route_prefix_available"] == 0
    )
    assert summary["full_route_match_available_batches"] == 1
    assert summary["reachability_wrong_when_route_match_available"] == 0
    assert summary["full_branch_reachable_available_batches"] == 1
    assert (
        summary["reachability_wrong_when_full_branch_reachable_available"] == 0
    )
    assert summary["fixture_summaries"]["junction"] == {
        "batches": 1,
        "selection_changes": 1,
        "reachable_route_prefix_available": 1,
        "current_wrong_when_reachable_route_prefix_available": 1,
        "reachability_wrong_when_reachable_route_prefix_available": 0,
        "current_selected_near_physical_status_counts": {"TOO_LONG": 1},
        "reachability_selected_near_physical_status_counts": {"REACHABLE": 1},
    }


def test_reachability_ablation_does_not_promote_explicit_stop_on_empty_road():
    moving = _candidate(
        0,
        motion="MOVING",
        route="MATCH",
        branch=True,
        cross_track=0.1,
    )
    moving["reachability_profile"].update(
        {
            "near_physical_status": "TOO_LONG",
            "physical_status": "REACHABLE",
            "near_source_speed_prior_status": "ACCELERATION_PRIOR",
            "near_required_constant_acceleration_mps2": 2.6,
        }
    )
    explicit_stop = _candidate(
        1,
        motion="EXPLICIT_STOP",
        route="MATCH",
        branch=True,
        cross_track=0.2,
    )
    event = {
        "fixture_label": "empty-road",
        "fixture_id": "fixture",
        "seed": 0,
        "navigation_text": "Continue in the current lane.",
        "actual_speed_mps": 1.0,
        "candidate_audits": [moving, explicit_stop],
    }

    replay = replay_batch(event)

    assert replay["reachability_first_selected_index"] == 0
    assert (
        replay["reachability_first_selected_facts"]["motion_class"] == "MOVING"
    )

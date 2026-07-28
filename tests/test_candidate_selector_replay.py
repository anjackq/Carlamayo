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
            "full_path_route_status": route,
            "branch_match": branch,
            "maximum_cross_track_error_m": cross_track,
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
    assert summary["current_route_first_accuracy"] == 1.0
    assert summary["current_wrong_when_route_match_available"] == 0

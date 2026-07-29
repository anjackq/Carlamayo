from scripts.summarize_oracle_route_experiment import summarize


def test_oracle_summary_reports_control_and_capture_facts():
    events = [
        {
            "event_type": "episode_start",
            "execution": "sync",
            "trajectory_source": "oracle_route",
            "scenario_seed": 0,
            "target_speed_mps": 1.0,
            "low_speed_longitudinal_governor": True,
        },
        {
            "event_type": "tick",
            "simulation_time_s": 10.0,
            "speed_mps": 0.0,
            "route_progress_m": 0.0,
            "controller_state": "TRACKING",
            "collision_count": 0,
            "road_envelope": {"current_ego_road": {"status": "SAFE"}},
            "route_candidate_assessment": {"near_term_route_status": "MATCH"},
            "safety_decision": {"safety_override_applied": False},
            "applied_control": {"throttle": 0.25, "brake": 0.0},
            "controller_debug": {
                "low_speed_longitudinal_governor": {"mode": "TRACKING"}
            },
        },
        {
            "event_type": "tick",
            "simulation_time_s": 13.0,
            "speed_mps": 1.0,
            "route_progress_m": 3.0,
            "controller_state": "TRACKING",
            "collision_count": 0,
            "road_envelope": {"current_ego_road": {"status": "SAFE"}},
            "route_candidate_assessment": {"near_term_route_status": "MATCH"},
            "safety_decision": {"safety_override_applied": True},
            "applied_control": {"throttle": 0.0, "brake": 1.0},
            "controller_debug": {
                "low_speed_longitudinal_governor": {"mode": "COAST"}
            },
        },
        {
            "event_type": "episode_summary",
            "stop_reason": "fixture_captured",
            "simulation_duration_s": 3.0,
            "integrated_distance_m": 3.0,
            "maximum_route_progress_m": 3.0,
            "collision_count": 0,
            "camera_fixture": {"fixture_id": "abc"},
        },
    ]

    result = summarize(events, source="runtime.jsonl")

    assert result["execution"] == "sync"
    assert result["mean_speed_mps"] == 0.5
    assert result["settled_speed_mae_mps"] == 0.0
    assert result["maximum_route_progress_m"] == 3.0
    assert result["safety_override_ticks"] == 1
    assert result["fixture_captured"] is True
    assert result["fixture_id"] == "abc"
    assert result["low_speed_longitudinal_governor"] is True
    assert result["low_speed_governor_mode_counts"] == {
        "COAST": 1,
        "TRACKING": 1,
    }
    assert result["applied_throttle_ticks"] == 1
    assert result["applied_brake_ticks"] == 1
    assert result["applied_hard_brake_ticks"] == 1


def test_oracle_summary_counts_a_restart_after_a_full_stop():
    events = [
        {"event_type": "episode_start", "target_speed_mps": 2.0},
        *[
            {
                "event_type": "tick",
                "simulation_time_s": float(index),
                "speed_mps": speed,
                "route_progress_m": float(index),
            }
            for index, speed in enumerate((0.0, 1.0, 0.0, 1.0))
        ],
        {"event_type": "episode_summary"},
    ]

    result = summarize(events)

    assert result["stop_go_cycle_count"] == 1
    assert result["longest_post_launch_stop_ticks"] == 1

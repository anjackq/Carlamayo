from scripts.summarize_reachability_gate import aggregate


def _summary(
    arm,
    *,
    completion=1.0,
    active_available=8,
    active_requests=10,
):
    return {
        "route": {
            "completion_ratio": completion,
            "accepted_unauthorized_prefix_count": 0,
            "route_only_emergency_override_ticks": 0,
            "route_unavailable_context_count": 0,
            "prompt_truth_mismatch_count": 0,
        },
        "motion": {
            "integrated_distance_m": 97.0 * completion,
            "stationary_suffix_ticks": 0,
            "longest_stationary_streak_ticks": 0,
            "absorbing_stop": False,
        },
        "safety": {
            "collisions": 0,
            "current_ego_road_unsafe_or_unknown_ticks": 0,
            "direct_override_ticks": 0,
            "fallback_ticks": 3,
        },
        "candidate_policy": {
            "reachability_profile_error_count": 0,
            "effective_near_conditional_misses": 0,
            "effective_full_conditional_misses": 0,
            "reachability_compute_p95_ms": 0.5,
            "selection_compute_p95_ms": 20.0,
            "selected_motion_class_counts": {"MOVING": 10},
            "phase": {
                "RIGHT/ACTIVE": {
                    "requests": active_requests,
                    "full_turn_executable_available": active_available,
                }
            },
        },
        "synchronous_contract": {
            "execution": "sync",
            "maximum_inference_simulation_duration_s": 0.0,
        },
        "gates": {
            "destination_stop": completion >= 0.95,
        },
        "arm": arm,
    }


def _matrix(**patch_overrides):
    result = {}
    for seed in range(3):
        result[("current", seed)] = _summary("current")
        result[("reachability-first", seed)] = _summary(
            "reachability-first",
            **patch_overrides,
        )
    return result


def test_gate_reports_direct_controller_go_when_all_conditions_pass():
    result = aggregate(_matrix())

    assert result["decision"] == "DIRECT_CONTROLLER_GO_EXPAND_VALIDATION"
    assert all(result["gates"].values())


def test_gate_attributes_low_active_candidate_coverage_to_model_role():
    result = aggregate(
        _matrix(active_available=1, active_requests=10)
    )

    assert (
        result["decision"]
        == "ALPAMAYO_DIRECT_LATERAL_AUTHORITY_NO_GO"
    )
    assert not result["gates"]["active_full_turn_coverage"]


def test_gate_keeps_model_inconclusive_when_coverage_is_high_but_route_fails():
    result = aggregate(_matrix(completion=0.6))

    assert result["decision"] == "INTEGRATION_INCONCLUSIVE"
    assert result["gates"]["active_full_turn_coverage"]
    assert not result["gates"]["direct_controller"]

import pytest

from module.candidate_selector import (
    CandidateEvaluation,
    navigation_turn_direction,
    rank_candidate_evaluations,
)


def _candidate(
    index,
    *,
    admission="ACCEPT_FULLY_SAFE",
    stop=False,
    progress=12.0,
    lateral=0.0,
    margin=0.5,
    continuity=1.0,
    rejection=None,
    motion=None,
    initial_speed=None,
    reserve_status=None,
    reserve=None,
    time_to_bad=None,
    route_status=None,
    branch_match=None,
    route_cross_track=None,
):
    return CandidateEvaluation(
        candidate_index=index,
        plan_id=f"plan/candidate-{index}",
        admission_status=admission,
        rejection_reason=rejection,
        stop_requested=stop,
        forward_progress_m=progress,
        representative_lateral_m=lateral,
        full_path_margin_m=margin,
        continuity_m=continuity,
        motion_class=motion,
        initial_target_speed_mps=initial_speed,
        stopping_reserve_status=reserve_status,
        stopping_reserve_m=reserve,
        time_to_first_bad_s=time_to_bad,
        full_path_route_status=route_status,
        route_branch_match=branch_match,
        route_cross_track_error_m=route_cross_track,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Take the first exit to the right.", "right"),
        ("Turn LEFT in 20 m.", "left"),
        ("Continue in the current lane.", None),
        ("Keep left, then turn right.", None),
    ],
)
def test_navigation_turn_direction_requires_one_unambiguous_direction(text, expected):
    assert navigation_turn_direction(text) == expected


def test_admitted_candidate_always_beats_rejected_continuity_candidate():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="REJECT_RETAIN_ACTIVE",
                continuity=0.01,
                rejection="near_term_unsafe",
            ),
            _candidate(1, admission="ACCEPT_SAFE_PREFIX", continuity=8.0),
        ],
        navigation_text="Continue in the current lane.",
        prefer_moving=False,
    )

    assert selection.selected_index == 1
    assert selection.selected_admitted is True


def test_empty_road_moving_safe_prefix_beats_stationary_full_safe_candidate():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="ACCEPT_FULLY_SAFE",
                stop=True,
                progress=0.1,
                continuity=0.01,
            ),
            _candidate(
                1,
                admission="ACCEPT_SAFE_PREFIX",
                stop=False,
                progress=14.0,
                continuity=2.0,
            ),
        ],
        navigation_text="Continue in the current lane.",
        prefer_moving=True,
    )

    assert selection.selected_index == 1
    assert selection.selected.stop_penalty == 0


def test_normal_traffic_does_not_penalize_stop_intent():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="ACCEPT_FULLY_SAFE",
                stop=True,
                progress=0.1,
            ),
            _candidate(
                1,
                admission="ACCEPT_SAFE_PREFIX",
                stop=False,
                progress=14.0,
            ),
        ],
        navigation_text="Continue in the current lane.",
        prefer_moving=False,
    )

    assert selection.selected_index == 0


def test_full_safe_candidate_beats_safe_prefix_before_continuity():
    selection = rank_candidate_evaluations(
        [
            _candidate(0, admission="ACCEPT_SAFE_PREFIX", continuity=0.01),
            _candidate(1, admission="ACCEPT_FULLY_SAFE", continuity=8.0),
        ],
        navigation_text=None,
        prefer_moving=False,
    )

    assert selection.selected_index == 1


def test_robust_stopping_reserve_beats_fragile_continuity_candidate():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="ACCEPT_SAFE_PREFIX",
                reserve_status="FRAGILE",
                reserve=-0.4,
                time_to_bad=2.0,
                continuity=0.01,
            ),
            _candidate(
                2,
                admission="ACCEPT_SAFE_PREFIX",
                reserve_status="ROBUST",
                reserve=5.0,
                time_to_bad=4.0,
                continuity=4.0,
            ),
        ],
        navigation_text=None,
        prefer_moving=False,
    )

    assert selection.selected_index == 2
    assert selection.selected.reserve_fragility_rank == 0


def test_empty_road_motion_class_precedes_admission_quality():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="ACCEPT_FULLY_SAFE",
                motion="CREEP_OR_STALL",
                reserve_status="UNBOUNDED",
            ),
            _candidate(
                1,
                admission="ACCEPT_SAFE_PREFIX",
                motion="MOVING",
                reserve_status="UNBOUNDED",
            ),
        ],
        navigation_text=None,
        prefer_moving=True,
        current_speed_mps=0.0,
    )

    assert selection.selected_index == 1
    assert selection.selected.motion_quality_rank == 0


def test_normal_traffic_neutralizes_motion_and_speed_continuity():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                motion="EXPLICIT_STOP",
                initial_speed=0.0,
                continuity=0.1,
            ),
            _candidate(
                1,
                motion="MOVING",
                initial_speed=5.0,
                continuity=2.0,
            ),
        ],
        navigation_text=None,
        prefer_moving=False,
        current_speed_mps=5.0,
    )

    assert selection.selected_index == 0
    assert all(item.motion_quality_rank == 0 for item in selection.ranked_candidates)
    assert all(
        item.speed_continuity_rank_mps == 0.0
        for item in selection.ranked_candidates
    )


def test_empty_road_speed_continuity_breaks_otherwise_equal_candidates():
    selection = rank_candidate_evaluations(
        [
            _candidate(0, motion="MOVING", initial_speed=1.0),
            _candidate(1, motion="MOVING", initial_speed=4.5),
        ],
        navigation_text=None,
        prefer_moving=True,
        current_speed_mps=4.0,
    )

    assert selection.selected_index == 1


def test_navigation_direction_breaks_equal_road_quality_before_continuity():
    selection = rank_candidate_evaluations(
        [
            _candidate(0, lateral=2.0, continuity=0.01),
            _candidate(1, lateral=-2.0, continuity=3.0),
        ],
        navigation_text="Take the first exit to the right.",
        prefer_moving=False,
    )

    assert selection.selected_index == 1
    assert selection.desired_turn == "right"


def test_structured_route_match_replaces_prompt_regex_lateral_heuristic():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                lateral=-2.0,
                continuity=0.01,
                route_status="DEVIATE",
                branch_match=False,
                route_cross_track=1.0,
            ),
            _candidate(
                1,
                lateral=2.0,
                continuity=3.0,
                route_status="MATCH",
                branch_match=True,
                route_cross_track=0.2,
            ),
        ],
        navigation_text="Turn right.",
        prefer_moving=False,
    )

    assert selection.selected_index == 1
    assert selection.selected.route_mismatch_rank == 0.0


def test_retain_active_beats_fallback_when_no_candidate_is_admissible():
    selection = rank_candidate_evaluations(
        [
            _candidate(
                0,
                admission="REJECT_FALLBACK_STOP",
                rejection="unsafe",
            ),
            _candidate(
                1,
                admission="REJECT_RETAIN_ACTIVE",
                rejection="unsafe_but_active_executable",
            ),
        ],
        navigation_text=None,
        prefer_moving=False,
    )

    assert selection.selected_index == 1
    assert selection.selection_reason == "no_admissible_candidate_retain_active"


def test_invalid_duplicate_candidate_indices_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        rank_candidate_evaluations(
            [_candidate(0), _candidate(0)],
            navigation_text=None,
            prefer_moving=False,
        )

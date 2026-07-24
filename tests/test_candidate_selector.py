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

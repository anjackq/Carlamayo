import numpy as np
import pytest

from scripts.audit_frozen_policy_ablation import (
    audit_candidate,
    summarize_audits,
)


def _fixture():
    history = np.zeros((16, 3), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
    return {
        "metadata": {
            "actual_speed_mps": 2.0,
            "navigation_context": {
                "action": "RIGHT",
                "route_index": 0,
            },
        },
        "simulation_times_s": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "frame_ids": np.asarray([1, 2, 3, 4]),
        "capture_pose_world": np.eye(4),
        "ego_history_xyz": history,
        "ego_history_rot": rotations,
    }


def _moving_candidate():
    trajectory = np.zeros((64, 3), dtype=np.float64)
    trajectory[:, 0] = np.arange(1, 65) * 0.2
    trajectory[:, 1] = -np.linspace(0.0, 3.0, 64)
    return {
        "trajectory": trajectory.tolist(),
        "candidate_sha256": "abc",
        "coc_text": "turn right",
        "stop_intent": False,
    }


def test_frozen_audit_computes_motion_and_model_frame_turn_direction():
    audit = audit_candidate(
        _moving_candidate(),
        _fixture(),
        candidate_index=0,
    )

    assert audit["valid"] is True
    assert audit["motion_profile"]["motion_class"] == "MOVING"
    assert audit["navigation_direction_match"] is True
    assert audit["route_assessment"] is None
    assert audit["reachability_error"] is None
    assert audit["reachability_profile"]["physical_status"] == "REACHABLE"
    assert (
        audit["reachability_profile"]["source_speed_prior_status"]
        == "CONSISTENT"
    )


def test_frozen_audit_summary_separates_coverage_from_candidate_rate():
    candidate = audit_candidate(
        _moving_candidate(),
        _fixture(),
        candidate_index=0,
    )
    records = [
        {
            "fixture_label": "low-speed",
            "seed": seed,
            "candidate_audits": [candidate],
        }
        for seed in range(3)
    ]

    summary = summarize_audits(records)["low-speed"]

    assert summary["requests"] == 3
    assert summary["moving_candidate_coverage_at_k"] == pytest.approx(1.0)
    assert summary["near_term_route_match_coverage_at_k"] is None
    assert summary["navigation_direction_mismatch_rate"] == pytest.approx(0.0)
    assert summary["physically_reachable_coverage_at_k"] == pytest.approx(1.0)
    assert summary["source_speed_consistent_coverage_at_k"] == pytest.approx(1.0)


def test_frozen_audit_summary_separates_near_prefix_from_full_branch():
    candidate = audit_candidate(
        _moving_candidate(),
        _fixture(),
        candidate_index=0,
    )
    candidate["route_assessment"] = {
        "near_term_route_status": "MATCH",
        "branch_match": False,
    }

    summary = summarize_audits(
        [
            {
                "fixture_label": "junction",
                "seed": 0,
                "candidate_audits": [candidate],
            }
        ]
    )["junction"]

    assert summary["near_term_route_match_coverage_at_k"] == pytest.approx(1.0)
    assert summary["route_match_branch_coverage_at_k"] == pytest.approx(0.0)


def test_frozen_audit_can_mirror_route_safe_prefix_lateral_policy():
    candidate = _moving_candidate()
    trajectory = np.asarray(candidate["trajectory"], dtype=np.float64)
    trajectory[:, 1] = -np.linspace(0.0, 15.0, 64)
    candidate["trajectory"] = trajectory.tolist()

    strict = audit_candidate(candidate, _fixture(), candidate_index=0)
    relaxed = audit_candidate(
        candidate,
        _fixture(),
        candidate_index=0,
        allow_far_lateral_route_prefix=True,
    )

    assert strict["valid"] is False
    assert "excessive_lateral_displacement" in strict["error"]
    assert strict["reachability_profile"] is not None
    assert relaxed["valid"] is True
    assert relaxed["first_lateral_limit_violation_index"] is not None
    assert strict["reachability_profile"] == relaxed["reachability_profile"]

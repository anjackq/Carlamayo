#!/usr/bin/env python3
"""Compare current, route-first, and reachability-first frozen selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.candidate_selector import (  # noqa: E402
    CandidateEvaluation,
    CandidateRankingPolicy,
    rank_candidate_evaluations,
)


_MOTION_RANK = {
    "MOVING": 0,
    "DELAYED_START": 1,
    "CREEP_OR_STALL": 2,
    "EXPLICIT_STOP": 3,
    None: 4,
}
_ROUTE_RANK = {
    "MATCH": 0,
    "DEVIATE": 1,
    "UNKNOWN": 2,
    None: 3,
}
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
    return events


def _route_fields(candidate: dict[str, Any]) -> tuple[str | None, bool | None, float | None]:
    route = candidate.get("route_assessment")
    if not isinstance(route, dict):
        return None, None, None
    status = route.get("full_path_route_status")
    branch = route.get("branch_match")
    cross_track = route.get("maximum_cross_track_error_m")
    return (
        None if status is None else str(status),
        None if branch is None else bool(branch),
        None if cross_track is None else float(cross_track),
    )


def _near_route_status(candidate: dict[str, Any]) -> str | None:
    route = candidate.get("route_assessment")
    if not isinstance(route, dict):
        return None
    status = route.get("near_term_route_status")
    return None if status is None else str(status)


def _reachability_fields(
    candidate: dict[str, Any],
) -> tuple[str | None, str | None, str | None, float | None]:
    profile = candidate.get("reachability_profile")
    if not isinstance(profile, dict):
        return None, None, None, None
    near_physical = profile.get("near_physical_status")
    full_physical = profile.get("physical_status")
    prior = profile.get("near_source_speed_prior_status")
    required_acceleration = profile.get(
        "near_required_constant_acceleration_mps2"
    )
    return (
        None if near_physical is None else str(near_physical),
        None if full_physical is None else str(full_physical),
        None if prior is None else str(prior),
        (
            None
            if required_acceleration is None
            else float(required_acceleration)
        ),
    )


def _motion_fields(candidate: dict[str, Any]) -> tuple[str | None, float | None]:
    motion = candidate.get("motion_profile")
    if not isinstance(motion, dict):
        return None, None
    motion_class = motion.get("motion_class")
    speed = motion.get("initial_target_speed_mps")
    return (
        None if motion_class is None else str(motion_class),
        None if speed is None else float(speed),
    )


def _json_safe(value: Any) -> Any:
    """Replace non-finite diagnostic floats without changing ranking inputs."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _evaluation(candidate: dict[str, Any]) -> CandidateEvaluation:
    route_status, branch_match, cross_track = _route_fields(candidate)
    near_route_status = _near_route_status(candidate)
    motion_class, initial_speed = _motion_fields(candidate)
    near_physical, full_physical, prior, required_acceleration = (
        _reachability_fields(candidate)
    )
    valid = bool(candidate.get("valid"))
    return CandidateEvaluation(
        candidate_index=int(candidate["candidate_index"]),
        plan_id=f"frozen-{candidate['candidate_index']}",
        admission_status=("ACCEPT_FULLY_SAFE" if valid else None),
        rejection_reason=(None if valid else str(candidate.get("error") or "invalid")),
        stop_requested=bool(candidate.get("stop_intent")),
        forward_progress_m=float(candidate.get("forward_progress_m") or 0.0),
        representative_lateral_m=float(
            candidate.get("representative_lateral_m") or 0.0
        ),
        full_path_margin_m=0.0 if valid else None,
        continuity_m=None,
        motion_class=motion_class,
        initial_target_speed_mps=initial_speed,
        stopping_reserve_status="UNBOUNDED" if valid else None,
        stopping_reserve_m=None,
        time_to_first_bad_s=None,
        near_term_route_status=near_route_status,
        full_path_route_status=route_status,
        route_branch_match=branch_match,
        route_cross_track_error_m=cross_track,
        near_physical_status=near_physical,
        full_physical_status=full_physical,
        near_source_speed_prior_status=prior,
        near_required_constant_acceleration_mps2=required_acceleration,
    )


def route_first_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    route_status, branch_match, cross_track = _route_fields(candidate)
    motion_class, _ = _motion_fields(candidate)
    valid_rank = 0 if candidate.get("valid") else 1
    branch_rank = 0 if branch_match is True else 1 if branch_match is False else 2
    return (
        float(valid_rank),
        float(_ROUTE_RANK[route_status]),
        float(branch_rank),
        float(math.inf if cross_track is None else cross_track),
        float(_MOTION_RANK[motion_class]),
        -float(candidate.get("forward_progress_m") or 0.0),
        float(candidate["candidate_index"]),
    )


def _selection_facts(candidate: dict[str, Any]) -> dict[str, Any]:
    full_route_status, branch_match, _ = _route_fields(candidate)
    near_physical, full_physical, prior, required_acceleration = (
        _reachability_fields(candidate)
    )
    motion_class, _ = _motion_fields(candidate)
    return {
        "candidate_index": int(candidate["candidate_index"]),
        "valid": bool(candidate.get("valid")),
        "motion_class": motion_class,
        "near_route_status": _near_route_status(candidate),
        "full_route_status": full_route_status,
        "branch_match": branch_match,
        "near_physical_status": near_physical,
        "full_physical_status": full_physical,
        "near_source_speed_prior_status": prior,
        "near_required_constant_acceleration_mps2": required_acceleration,
    }


def replay_batch(event: dict[str, Any]) -> dict[str, Any]:
    candidates = list(event.get("candidate_audits") or ())
    if not candidates:
        raise ValueError("policy audit batch has no candidates")
    route_evaluable = any(
        _route_fields(candidate)[0] is not None for candidate in candidates
    )
    evaluations = [_evaluation(candidate) for candidate in candidates]
    current = rank_candidate_evaluations(
        evaluations,
        navigation_text=event.get("navigation_text"),
        prefer_moving=True,
        current_speed_mps=event.get("actual_speed_mps"),
        ranking_policy=CandidateRankingPolicy.CURRENT,
    )
    route_ranked = sorted(candidates, key=route_first_key)
    route_first = int(route_ranked[0]["candidate_index"])
    best_without_index = route_first_key(route_ranked[0])[:-1]
    ties = sum(
        route_first_key(candidate)[:-1] == best_without_index
        for candidate in route_ranked
    )
    reachability_selection = rank_candidate_evaluations(
        evaluations,
        navigation_text=event.get("navigation_text"),
        prefer_moving=True,
        current_speed_mps=event.get("actual_speed_mps"),
        ranking_policy=CandidateRankingPolicy.REACHABILITY_FIRST,
    )
    reachability_first = int(reachability_selection.selected_index)
    reachability_ranked = list(
        reachability_selection.ranked_candidates
    )
    reachability_best_without_index = (
        reachability_ranked[0].ranking_key[:-1]
    )
    reachability_ties = sum(
        ranked.ranking_key[:-1] == reachability_best_without_index
        for ranked in reachability_ranked
    )
    reachability_evaluable = any(
        _reachability_fields(candidate)[0] is not None
        for candidate in candidates
    )
    match_candidates = [
        int(candidate["candidate_index"])
        for candidate in candidates
        if _route_fields(candidate)[0] == "MATCH"
        and _route_fields(candidate)[1] is True
        and candidate.get("valid")
    ]
    reachable_route_prefix_candidates = [
        int(candidate["candidate_index"])
        for candidate in candidates
        if candidate.get("valid")
        and _near_route_status(candidate) == "MATCH"
        and _reachability_fields(candidate)[0] == "REACHABLE"
    ]
    full_branch_reachable_candidates = [
        int(candidate["candidate_index"])
        for candidate in candidates
        if candidate.get("valid")
        and _route_fields(candidate)[0] == "MATCH"
        and _route_fields(candidate)[1] is True
        and _reachability_fields(candidate)[0] == "REACHABLE"
        and _reachability_fields(candidate)[1] == "REACHABLE"
    ]
    candidates_by_index = {
        int(candidate["candidate_index"]): candidate for candidate in candidates
    }
    return {
        "event_type": "selector_replay",
        "schema_version": 1,
        "fixture_label": event.get("fixture_label"),
        "fixture_id": event.get("fixture_id"),
        "seed": int(event["seed"]),
        "current_selected_index": int(current.selected_index),
        "route_first_selected_index": route_first,
        "reachability_first_selected_index": reachability_first,
        "model_first_selected_index": min(
            int(candidate["candidate_index"]) for candidate in candidates
        ),
        "route_evaluable": route_evaluable,
        "route_first_unambiguous": ties == 1,
        "reachability_evaluable": reachability_evaluable,
        "reachability_first_unambiguous": reachability_ties == 1,
        "current_matches_route_first": int(current.selected_index) == route_first,
        "current_matches_reachability_first": (
            int(current.selected_index) == reachability_first
        ),
        "route_match_candidate_indices": match_candidates,
        "reachable_route_prefix_candidate_indices": (
            reachable_route_prefix_candidates
        ),
        "full_branch_reachable_candidate_indices": (
            full_branch_reachable_candidates
        ),
        "current_selected_wrong_when_match_available": bool(
            match_candidates and int(current.selected_index) not in match_candidates
        ),
        "reachability_selected_wrong_when_match_available": bool(
            match_candidates and reachability_first not in match_candidates
        ),
        "current_selected_wrong_when_reachable_route_prefix_available": bool(
            reachable_route_prefix_candidates
            and int(current.selected_index)
            not in reachable_route_prefix_candidates
        ),
        "reachability_selected_wrong_when_reachable_route_prefix_available": bool(
            reachable_route_prefix_candidates
            and reachability_first not in reachable_route_prefix_candidates
        ),
        "current_selected_facts": _selection_facts(
            candidates_by_index[int(current.selected_index)]
        ),
        "route_first_selected_facts": _selection_facts(
            candidates_by_index[route_first]
        ),
        "reachability_first_selected_facts": _selection_facts(
            candidates_by_index[reachability_first]
        ),
        "current_selection": _json_safe(current.to_json_dict()),
        "route_first_ranking": [
            {
                "candidate_index": int(candidate["candidate_index"]),
                "ranking_key": [
                    None if not math.isfinite(value) else value
                    for value in route_first_key(candidate)
                ],
            }
            for candidate in route_ranked
        ],
        "reachability_first_ranking": [
            {
                "candidate_index": int(
                    ranked.evaluation.candidate_index
                ),
                "ranking_key": [
                    None if not math.isfinite(value) else value
                    for value in ranked.ranking_key
                ],
            }
            for ranked in reachability_ranked
        ],
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    route_evaluable = [record for record in records if record["route_evaluable"]]
    unambiguous = [
        record
        for record in route_evaluable
        if record["route_first_unambiguous"]
    ]
    wrong = sum(
        record["current_selected_wrong_when_match_available"] for record in records
    )
    reachability_wrong = sum(
        record["reachability_selected_wrong_when_match_available"]
        for record in records
    )
    reachability_evaluable = [
        record for record in records if record["reachability_evaluable"]
    ]
    reachability_unambiguous = [
        record
        for record in reachability_evaluable
        if record["reachability_first_unambiguous"]
    ]
    reachable_prefix_available = [
        record
        for record in records
        if record["reachable_route_prefix_candidate_indices"]
    ]
    route_match_available = [
        record for record in records if record["route_match_candidate_indices"]
    ]
    full_branch_reachable_available = [
        record
        for record in records
        if record["full_branch_reachable_candidate_indices"]
    ]
    current_reachable_wrong = sum(
        record[
            "current_selected_wrong_when_reachable_route_prefix_available"
        ]
        for record in records
    )
    reachability_reachable_wrong = sum(
        record[
            "reachability_selected_wrong_when_reachable_route_prefix_available"
        ]
        for record in records
    )
    selector_fact_counts: dict[str, Counter[str]] = {}
    selector_motion_counts: dict[str, Counter[str]] = {}
    selector_prior_counts: dict[str, Counter[str]] = {}
    for selector in ("current", "route_first", "reachability_first"):
        facts_key = f"{selector}_selected_facts"
        selector_fact_counts[selector] = Counter(
            str(record[facts_key].get("near_physical_status"))
            for record in records
        )
        selector_motion_counts[selector] = Counter(
            str(record[facts_key].get("motion_class")) for record in records
        )
        selector_prior_counts[selector] = Counter(
            str(record[facts_key].get("near_source_speed_prior_status"))
            for record in records
        )
    fixture_counts = Counter(str(record.get("fixture_label")) for record in records)
    fixture_summaries: dict[str, dict[str, Any]] = {}
    for fixture in sorted(fixture_counts):
        subset = [
            record
            for record in records
            if str(record.get("fixture_label")) == fixture
        ]
        fixture_summaries[fixture] = {
            "batches": len(subset),
            "selection_changes": sum(
                not record["current_matches_reachability_first"]
                for record in subset
            ),
            "reachable_route_prefix_available": sum(
                bool(record["reachable_route_prefix_candidate_indices"])
                for record in subset
            ),
            "current_wrong_when_reachable_route_prefix_available": sum(
                record[
                    "current_selected_wrong_when_reachable_route_prefix_available"
                ]
                for record in subset
            ),
            "reachability_wrong_when_reachable_route_prefix_available": sum(
                record[
                    "reachability_selected_wrong_when_reachable_route_prefix_available"
                ]
                for record in subset
            ),
            "current_selected_near_physical_status_counts": dict(
                sorted(
                    Counter(
                        str(
                            record["current_selected_facts"].get(
                                "near_physical_status"
                            )
                        )
                        for record in subset
                    ).items()
                )
            ),
            "reachability_selected_near_physical_status_counts": dict(
                sorted(
                    Counter(
                        str(
                            record["reachability_first_selected_facts"].get(
                                "near_physical_status"
                            )
                        )
                        for record in subset
                    ).items()
                )
            ),
        }
    return {
        "batches": len(records),
        "fixture_batch_counts": dict(sorted(fixture_counts.items())),
        "fixture_summaries": fixture_summaries,
        "route_evaluable_batches": len(route_evaluable),
        "route_first_unambiguous_batches": len(unambiguous),
        "current_route_first_accuracy": (
            sum(record["current_matches_route_first"] for record in unambiguous)
            / len(unambiguous)
            if unambiguous
            else None
        ),
        "full_route_match_available_batches": len(route_match_available),
        "current_wrong_when_route_match_available": wrong,
        "reachability_wrong_when_route_match_available": reachability_wrong,
        "reachability_evaluable_batches": len(reachability_evaluable),
        "reachability_first_unambiguous_batches": len(
            reachability_unambiguous
        ),
        "current_reachability_first_accuracy": (
            sum(
                record["current_matches_reachability_first"]
                for record in reachability_unambiguous
            )
            / len(reachability_unambiguous)
            if reachability_unambiguous
            else None
        ),
        "reachability_selection_change_count": sum(
            not record["current_matches_reachability_first"]
            for record in reachability_evaluable
        ),
        "reachable_route_prefix_available_batches": len(
            reachable_prefix_available
        ),
        "current_wrong_when_reachable_route_prefix_available": (
            current_reachable_wrong
        ),
        "reachability_wrong_when_reachable_route_prefix_available": (
            reachability_reachable_wrong
        ),
        "full_branch_reachable_available_batches": len(
            full_branch_reachable_available
        ),
        "current_wrong_when_full_branch_reachable_available": sum(
            int(record["current_selected_index"])
            not in record["full_branch_reachable_candidate_indices"]
            for record in full_branch_reachable_available
        ),
        "reachability_wrong_when_full_branch_reachable_available": sum(
            int(record["reachability_first_selected_index"])
            not in record["full_branch_reachable_candidate_indices"]
            for record in full_branch_reachable_available
        ),
        "selected_near_physical_status_counts": {
            selector: dict(sorted(counts.items()))
            for selector, counts in selector_fact_counts.items()
        },
        "selected_motion_class_counts": {
            selector: dict(sorted(counts.items()))
            for selector, counts in selector_motion_counts.items()
        },
        "selected_near_source_prior_counts": {
            selector: dict(sorted(counts.items()))
            for selector, counts in selector_prior_counts.items()
        },
    }


def _external_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY_ROOT or REPOSITORY_ROOT in path.parents:
        raise argparse.ArgumentTypeError("replay output must remain outside the repository")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_audit_jsonl", type=Path)
    parser.add_argument("--output", type=_external_path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = [
        replay_batch(event)
        for event in _read_jsonl(args.policy_audit_jsonl)
        if event.get("event_type") == "policy_audit"
    ]
    summary = summarize(records)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        stream.write(
            json.dumps(
                {
                    "event_type": "selector_replay_summary",
                    "schema_version": 1,
                    **summary,
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

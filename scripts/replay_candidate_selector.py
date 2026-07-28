#!/usr/bin/env python3
"""Compare current and route-first selection on frozen candidate audits."""

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


def _evaluation(candidate: dict[str, Any]) -> CandidateEvaluation:
    route_status, branch_match, cross_track = _route_fields(candidate)
    motion_class, initial_speed = _motion_fields(candidate)
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
        full_path_route_status=route_status,
        route_branch_match=branch_match,
        route_cross_track_error_m=cross_track,
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


def replay_batch(event: dict[str, Any]) -> dict[str, Any]:
    candidates = list(event.get("candidate_audits") or ())
    if not candidates:
        raise ValueError("policy audit batch has no candidates")
    current = rank_candidate_evaluations(
        [_evaluation(candidate) for candidate in candidates],
        navigation_text=event.get("navigation_text"),
        prefer_moving=True,
        current_speed_mps=event.get("actual_speed_mps"),
    )
    route_ranked = sorted(candidates, key=route_first_key)
    route_first = int(route_ranked[0]["candidate_index"])
    best_without_index = route_first_key(route_ranked[0])[:-1]
    ties = sum(
        route_first_key(candidate)[:-1] == best_without_index
        for candidate in route_ranked
    )
    match_candidates = [
        int(candidate["candidate_index"])
        for candidate in candidates
        if _route_fields(candidate)[0] == "MATCH"
        and _route_fields(candidate)[1] is True
        and candidate.get("valid")
    ]
    return {
        "event_type": "selector_replay",
        "schema_version": 1,
        "fixture_label": event.get("fixture_label"),
        "fixture_id": event.get("fixture_id"),
        "seed": int(event["seed"]),
        "current_selected_index": int(current.selected_index),
        "route_first_selected_index": route_first,
        "model_first_selected_index": min(
            int(candidate["candidate_index"]) for candidate in candidates
        ),
        "route_first_unambiguous": ties == 1,
        "current_matches_route_first": int(current.selected_index) == route_first,
        "route_match_candidate_indices": match_candidates,
        "current_selected_wrong_when_match_available": bool(
            match_candidates and int(current.selected_index) not in match_candidates
        ),
        "current_selection": current.to_json_dict(),
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
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unambiguous = [record for record in records if record["route_first_unambiguous"]]
    wrong = sum(
        record["current_selected_wrong_when_match_available"] for record in records
    )
    fixture_counts = Counter(str(record.get("fixture_label")) for record in records)
    return {
        "batches": len(records),
        "fixture_batch_counts": dict(sorted(fixture_counts.items())),
        "route_first_unambiguous_batches": len(unambiguous),
        "current_route_first_accuracy": (
            sum(record["current_matches_route_first"] for record in unambiguous)
            / len(unambiguous)
            if unambiguous
            else None
        ),
        "current_wrong_when_route_match_available": wrong,
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

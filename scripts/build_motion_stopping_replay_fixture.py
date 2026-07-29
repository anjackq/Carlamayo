#!/usr/bin/env python3
"""Extract non-visual motion/stopping regression cases from local runtime JSONL.

The generated fixture intentionally excludes camera data, calibration, raw CoC
text, world frames, videos, and model artifacts.  It is safe to commit and can
be regenerated only by users who retain the original ignored runtime logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CASE_MANIFEST = (
    {
        "case_id": "false_cap_22859856_tick55",
        "job_id": 22859856,
        "proposal_ordinal": 5,
        "candidate_indices": (2,),
        "tick_ids": (55,),
        "expected_kind": "CAP_SATURATION_FALSE_EMERGENCY",
    },
    {
        "case_id": "false_cap_22859869_ticks50_51",
        "job_id": 22859869,
        "proposal_ordinal": 4,
        "candidate_indices": (2,),
        "tick_ids": (50, 51),
        "expected_kind": "CAP_SATURATION_FALSE_EMERGENCY",
    },
    {
        "case_id": "fragile_stopping_reserve",
        "job_id": 22859868,
        "proposal_ordinal": 9,
        "candidate_indices": (0,),
        "tick_ids": (),
        "expected_kind": "FRAGILE",
    },
    {
        "case_id": "robust_alternative_batch",
        "job_id": 22859869,
        "proposal_ordinal": 6,
        "candidate_indices": (0, 2),
        "tick_ids": (),
        "expected_kind": "ROBUST_ALTERNATIVE",
    },
    {
        "case_id": "delayed_start",
        "job_id": 22859856,
        "proposal_ordinal": 14,
        "candidate_indices": (0,),
        "tick_ids": (),
        "expected_kind": "DELAYED_START",
    },
    {
        "case_id": "creep_or_stall",
        "job_id": 22859856,
        "proposal_ordinal": 13,
        "candidate_indices": (2,),
        "tick_ids": (),
        "expected_kind": "CREEP_OR_STALL",
    },
    {
        "case_id": "explicit_stop",
        "job_id": 22859868,
        "proposal_ordinal": 11,
        "candidate_indices": (1,),
        "tick_ids": (),
        "expected_kind": "EXPLICIT_STOP",
    },
    {
        "case_id": "all_invalid_batch",
        "job_id": 22859856,
        "proposal_ordinal": 6,
        "candidate_indices": (0, 1, 2),
        "tick_ids": (),
        "expected_kind": "ALL_INVALID",
    },
)
FORBIDDEN_KEYS = frozenset(
    {
        "camera_ids",
        "camera_frames",
        "images",
        "calibration",
        "camera_profile",
        "coc_text",
        "coc_text_full",
        "candidate_coc_texts_full",
        "selected_trajectory_world",
    }
)


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def _one(events: list[dict[str, Any]], event_type: str, predicate) -> dict[str, Any]:
    matches = [
        event
        for event in events
        if event.get("event_type") == event_type and predicate(event)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {event_type}, found {len(matches)}")
    return matches[0]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("fixture source contains a non-finite number")
    return result


def _compact_envelope(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    full_path = value.get("full_path_road") or {}
    near_path = value.get("near_term_path_road") or {}
    current = value.get("current_ego_road") or {}
    return {
        "current_ego_status": current.get("status"),
        "near_term_status": near_path.get("status"),
        "full_path_status": full_path.get("status"),
        "full_path_margin_m": _finite(full_path.get("min_margin_m")),
        "last_safe_waypoint_index": value.get("last_safe_waypoint_index"),
        "time_to_first_bad_s": _finite(value.get("time_to_first_bad_s")),
        "distance_to_first_bad_m": _finite(value.get("distance_to_first_bad_m")),
        "legacy_target_speed_cap_mps": _finite(value.get("target_speed_cap_mps")),
        "legacy_emergency_required": bool(value.get("emergency_required")),
        "reason_codes": list(full_path.get("reason_codes") or ()),
    }


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _integrity_sha256(payload_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload_without_hash)).hexdigest()


def _extract_case(
    *,
    spec: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    suffix = f":{spec['proposal_ordinal']}"
    proposal = _one(
        events,
        "alpamayo_proposal",
        lambda event: str(event.get("proposal_id", "")).endswith(suffix),
    )
    proposal_id = str(proposal["proposal_id"])
    selection = _one(
        events,
        "candidate_selection",
        lambda event: event.get("proposal_id") == proposal_id,
    )
    evaluations = {
        int(event["candidate_index"]): event
        for event in events
        if event.get("event_type") == "candidate_evaluation"
        and event.get("proposal_id") == proposal_id
    }
    trajectories = proposal.get("candidate_trajectories_model")
    coc_hashes = proposal.get("candidate_coc_sha256")
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != 3
        or not isinstance(coc_hashes, list)
        or len(coc_hashes) != 3
    ):
        raise ValueError(f"{proposal_id}: incomplete K=3 proposal")

    source_time = _finite(proposal.get("source_simulation_time_s"))
    source_frame = int(proposal["source_carla_frame_id"])
    source_tick = _one(
        events,
        "tick",
        lambda event: int(event.get("carla_frame_id", -1)) == source_frame,
    )
    candidates = []
    for candidate_index in spec["candidate_indices"]:
        evaluation = evaluations.get(int(candidate_index))
        if evaluation is None:
            raise ValueError(f"{proposal_id}: missing candidate {candidate_index}")
        points = trajectories[int(candidate_index)]
        if (
            not isinstance(points, list)
            or len(points) != 64
            or any(not isinstance(point, list) or len(point) != 3 for point in points)
        ):
            raise ValueError(f"{proposal_id}: candidate trajectory is not 64x3")
        candidates.append(
            {
                "candidate_index": int(candidate_index),
                "trajectory_model": points,
                "waypoint_times_s": [
                    source_time + 0.1 * index for index in range(1, 65)
                ],
                "coc_sha256": str(coc_hashes[int(candidate_index)]),
                "admission_status": evaluation.get("admission_status"),
                "rejection_reason": evaluation.get("rejection_reason"),
                "stop_requested": bool(evaluation.get("stop_requested")),
                "selected": bool(evaluation.get("selected")),
                "continuity_m": _finite(evaluation.get("continuity_m")),
                "road_envelope": _compact_envelope(
                    evaluation.get("candidate_road_envelope")
                ),
            }
        )

    tick_evidence = []
    for tick_id in spec["tick_ids"]:
        tick = _one(
            events,
            "tick",
            lambda event, tick_id=tick_id: int(event.get("loop_tick_id", -1))
            == int(tick_id),
        )
        tick_evidence.append(
            {
                "loop_tick_id": int(tick_id),
                "speed_mps": _finite(tick.get("speed_mps")),
                "active_plan_id": tick.get("active_plan_id"),
                "distance_to_first_bad_m": _finite(
                    (tick.get("road_execution_envelope") or {}).get(
                        "distance_to_first_bad_m"
                    )
                ),
                "legacy_target_speed_cap_mps": _finite(
                    (tick.get("road_execution_envelope") or {}).get(
                        "target_speed_cap_mps"
                    )
                ),
                "legacy_emergency_required": bool(
                    (tick.get("road_execution_envelope") or {}).get(
                        "emergency_required"
                    )
                ),
                "direct_safety_trigger": bool(tick.get("direct_safety_trigger")),
            }
        )

    return {
        "case_id": spec["case_id"],
        "expected_kind": spec["expected_kind"],
        "job_id": int(spec["job_id"]),
        "proposal_ordinal": int(spec["proposal_ordinal"]),
        "source_simulation_time_s": source_time,
        "source_speed_mps": _finite(source_tick.get("speed_mps")),
        "baseline_selected_candidate_index": int(
            selection["selected_candidate_index"]
        ),
        "candidates": candidates,
        "tick_evidence": tick_evidence,
    }


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_fixture(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixture schema")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("fixture cases must be a list")
    expected_ids = {spec["case_id"] for spec in CASE_MANIFEST}
    actual_ids = {case.get("case_id") for case in cases if isinstance(case, dict)}
    if actual_ids != expected_ids:
        raise ValueError("fixture does not contain the complete case manifest")
    forbidden = FORBIDDEN_KEYS.intersection(_walk_keys(payload))
    if forbidden:
        raise ValueError(f"fixture contains forbidden keys: {sorted(forbidden)}")
    for case in cases:
        for candidate in case["candidates"]:
            points = candidate["trajectory_model"]
            if len(points) != 64 or any(len(point) != 3 for point in points):
                raise ValueError("fixture trajectory is not 64x3")
            for point in points:
                if not all(math.isfinite(float(value)) for value in point):
                    raise ValueError("fixture trajectory contains non-finite values")
            digest = candidate["coc_sha256"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid CoC hash")
    integrity = payload.get("integrity_sha256")
    unsigned = dict(payload)
    unsigned.pop("integrity_sha256", None)
    if integrity != _integrity_sha256(unsigned):
        raise ValueError("fixture integrity hash mismatch")


def build_fixture(runs_root: Path) -> dict[str, Any]:
    by_job = {}
    cases = []
    for spec in CASE_MANIFEST:
        job_id = int(spec["job_id"])
        if job_id not in by_job:
            by_job[job_id] = _read_events(runs_root / str(job_id) / "runtime.jsonl")
        cases.append(_extract_case(spec=spec, events=by_job[job_id]))
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": "7ae43f3",
        "baseline_configuration": {
            "camera_alignment": "projection-only",
            "num_traj_samples": 3,
            "diffusion_temperature": 1.0,
            "road_assessment_backend": "serial",
            "navigation_source": "manual",
        },
        "source_jobs": [22859856, 22859868, 22859869],
        "cases": cases,
    }
    return {**unsigned, "integrity_sha256": _integrity_sha256(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/home/aqiu/carlamayo-runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/motion_stopping_baseline_v1.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        with args.output.open("r", encoding="utf-8") as handle:
            validate_fixture(json.load(handle))
        print(f"valid fixture: {args.output}")
        return

    payload = build_fixture(args.runs_root)
    validate_fixture(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.output} ({len(payload['cases'])} cases)")


if __name__ == "__main__":
    main()

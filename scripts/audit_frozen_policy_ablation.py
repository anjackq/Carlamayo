#!/usr/bin/env python3
"""Audit frozen Alpamayo candidates for motion and exact CARLA route quality."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.camera_fixture import load_camera_fixture  # noqa: E402
from module.carla_route_adapter import (  # noqa: E402
    assess_current_route_status,
    query_candidate_lane_facts,
)
from module.route_authorization import assess_route_candidate  # noqa: E402
from module.route_navigation import RoutePoint, build_route_plan  # noqa: E402
from module.trajectory_runtime import (  # noqa: E402
    TrajectoryValidationError,
    build_fixed_world_trajectory,
    compute_trajectory_motion_profile,
)


def _fixture_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("fixture must use LABEL=/path/to/fixture.npz")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("fixture label and path must be nonempty")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _outside_repository(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY_ROOT or REPOSITORY_ROOT in path.parents:
        raise argparse.ArgumentTypeError("audit output must remain outside the repository")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(event)
    return events


def _route_from_metadata(metadata: dict[str, Any]):
    raw_points = metadata.get("route_points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return None
    return build_route_plan(
        RoutePoint(
            xyz=tuple(item["xyz"]),
            road_id=item["road_id"],
            section_id=item["section_id"],
            lane_id=item["lane_id"],
            is_junction=item["is_junction"],
            road_option=item["road_option"],
        )
        for item in raw_points
    )


def load_opendrive_map(path: Path | None):
    if path is None:
        return None, None
    import carla

    text = path.read_text(encoding="utf-8")
    return carla.Map(path.stem, text), carla


def audit_candidate(
    candidate: dict[str, Any],
    fixture: dict[str, Any],
    *,
    candidate_index: int,
    world_map=None,
    carla_module=None,
) -> dict[str, Any]:
    metadata = fixture["metadata"]
    source_time = float(fixture["simulation_times_s"][-1])
    navigation_context = metadata.get("navigation_context")
    plan = None
    motion = None
    route_assessment = None
    error = None
    try:
        plan = build_fixed_world_trajectory(
            plan_id=f"frozen-{candidate_index}",
            source_frame_id=int(fixture["frame_ids"][-1]),
            source_simulation_time_s=source_time,
            capture_pose_world=fixture["capture_pose_world"],
            model_points=candidate["trajectory"],
            coc_text=str(candidate.get("coc_text") or ""),
            prompt_revision=0,
            respawn_revision=0,
            selected_candidate_index=candidate_index,
            navigation_context=navigation_context,
        )
        motion = compute_trajectory_motion_profile(plan, source_time)
        route = _route_from_metadata(metadata)
        if (
            route is not None
            and world_map is not None
            and carla_module is not None
            and isinstance(navigation_context, dict)
        ):
            route_index = int(navigation_context.get("route_index", 0))
            ego_xyz = np.asarray(fixture["capture_pose_world"], dtype=np.float64)[:3, 3]
            current_status = assess_current_route_status(
                world_map,
                ego_xyz,
                route=route,
                route_index=route_index,
                carla_module=carla_module,
            )
            route_assessment = assess_route_candidate(
                route=route,
                current_route_index=route_index,
                current_route_status=current_status,
                trajectory_world_points=plan.world_points,
                waypoint_times_s=plan.waypoint_times_s,
                source_simulation_time_s=source_time,
                current_simulation_time_s=source_time,
                lane_facts=query_candidate_lane_facts(
                    world_map,
                    plan.world_points,
                    carla_module=carla_module,
                ),
            )
    except (TrajectoryValidationError, ValueError, TypeError) as exc:
        error = f"{type(exc).__name__}:{exc}"

    points = np.asarray(candidate.get("trajectory"), dtype=np.float64)
    progress_m = (
        float(np.max(points[:, 0], initial=0.0))
        if points.ndim == 2 and points.shape[1] >= 2 and np.isfinite(points).all()
        else None
    )
    representative_lateral_m = None
    if points.ndim == 2 and points.shape[1] >= 2 and np.isfinite(points).all():
        index = int(np.argmax(np.abs(points[:, 1])))
        representative_lateral_m = float(points[index, 1])
    action = (
        str(navigation_context.get("action"))
        if isinstance(navigation_context, dict)
        else None
    )
    direction_match = None
    if (
        action in {"LEFT", "RIGHT"}
        and representative_lateral_m is not None
        and abs(representative_lateral_m) >= 0.5
    ):
        # Alpamayo model coordinates use y-left.
        direction_match = (
            representative_lateral_m > 0.0
            if action == "LEFT"
            else representative_lateral_m < 0.0
        )
    return {
        "candidate_index": int(candidate_index),
        "candidate_sha256": candidate.get("candidate_sha256"),
        "valid": plan is not None and error is None,
        "error": error,
        "stop_intent": candidate.get("stop_intent"),
        "forward_progress_m": progress_m,
        "representative_lateral_m": representative_lateral_m,
        "navigation_action": action,
        "navigation_direction_match": direction_match,
        "motion_profile": motion.to_json_dict() if motion is not None else None,
        "route_assessment": (
            route_assessment.to_json_dict()
            if route_assessment is not None
            else None
        ),
    }


def summarize_audits(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["fixture_label"])].append(record)
    result = {}
    for label, group in groups.items():
        motion_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        request_moving: dict[int, bool] = defaultdict(bool)
        request_route_match: dict[int, bool] = defaultdict(bool)
        direction_evaluable = 0
        direction_mismatch = 0
        valid = 0
        for record in group:
            request_seed = int(record["seed"])
            for candidate in record["candidate_audits"]:
                if candidate["valid"]:
                    valid += 1
                motion = candidate.get("motion_profile")
                if isinstance(motion, dict):
                    motion_class = str(motion.get("motion_class"))
                    motion_counts[motion_class] += 1
                    if motion_class == "MOVING":
                        request_moving[request_seed] = True
                route = candidate.get("route_assessment")
                if isinstance(route, dict):
                    status = str(route.get("near_term_route_status"))
                    route_counts[status] += 1
                    if status == "MATCH" and route.get("branch_match") is True:
                        request_route_match[request_seed] = True
                match = candidate.get("navigation_direction_match")
                if match is not None:
                    direction_evaluable += 1
                    direction_mismatch += int(match is False)
        request_count = len({int(record["seed"]) for record in group})
        candidate_count = sum(len(record["candidate_audits"]) for record in group)
        result[label] = {
            "requests": request_count,
            "candidates": candidate_count,
            "valid_candidates": valid,
            "motion_class_counts": dict(sorted(motion_counts.items())),
            "moving_candidate_coverage_at_k": (
                sum(request_moving.values()) / request_count
                if request_count
                else None
            ),
            "near_term_route_status_counts": dict(sorted(route_counts.items())),
            "route_match_branch_coverage_at_k": (
                sum(request_route_match.values()) / request_count
                if request_count and route_counts
                else None
            ),
            "navigation_direction_evaluable_candidates": direction_evaluable,
            "navigation_direction_mismatch_rate": (
                direction_mismatch / direction_evaluable
                if direction_evaluable
                else None
            ),
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation_jsonl", type=Path)
    parser.add_argument(
        "--fixture",
        action="append",
        type=_fixture_argument,
        required=True,
    )
    parser.add_argument("--opendrive", type=Path)
    parser.add_argument("--output", type=_outside_repository, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.fixture]
    if len(labels) != len(set(labels)):
        parser.error("fixture labels must be unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixtures = {
        label: load_camera_fixture(path) for label, path in args.fixture
    }
    world_map, carla_module = load_opendrive_map(args.opendrive)
    records = []
    for event in _read_jsonl(args.ablation_jsonl):
        if event.get("event_type") != "inference_result":
            continue
        label = str(event.get("fixture_label"))
        if label not in fixtures:
            raise ValueError(f"missing fixture for ablation label {label!r}")
        records.append(
            {
                "event_type": "policy_audit",
                "schema_version": 1,
                "fixture_label": label,
                "fixture_id": event.get("fixture_id"),
                "seed": int(event["seed"]),
                "navigation_text": str(
                    fixtures[label]["metadata"].get("navigation_text") or ""
                ),
                "actual_speed_mps": fixtures[label]["metadata"].get(
                    "actual_speed_mps"
                ),
                "candidate_audits": [
                    audit_candidate(
                        candidate,
                        fixtures[label],
                        candidate_index=index,
                        world_map=world_map,
                        carla_module=carla_module,
                    )
                    for index, candidate in enumerate(event.get("candidates", ()))
                ],
            }
        )
    summary = summarize_audits(records)
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
                    "event_type": "policy_audit_summary",
                    "schema_version": 1,
                    "groups": summary,
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

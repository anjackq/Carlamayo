#!/usr/bin/env python3
"""Build a compact, simulator-free replay of a CARLA route transition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.carla_route_adapter import (
    FIXED_TOWN03_DESTINATION_XYZ,
    FIXED_TOWN03_ORIGIN_XYZ,
    load_global_route_planner,
    query_candidate_lane_facts,
    trace_carla_route,
)
from module.geometry import model_ego_points_to_world, pose_matrix_from_components
from module.route_navigation import associate_route_index


SCHEMA_VERSION = "carlamayo.route-transition-replay.v1"
SOURCE_JOB_ID = 22921097
SOURCE_REQUEST_ID = 31
FORBIDDEN_KEYS = {
    "camera_frames",
    "camera_images",
    "coc_text",
    "candidate_coc_texts",
    "video",
}


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _single_event(
    events: list[dict[str, Any]],
    event_type: str,
    **matches: Any,
) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if event.get("event_type") == event_type
        and all(event.get(key) == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {event_type} event matching {matches}, got {len(selected)}"
        )
    return selected[0]


def _route_point_dict(point: Any) -> dict[str, Any]:
    return {
        "xyz": [float(value) for value in point.xyz],
        "road_id": int(point.road_id),
        "section_id": int(point.section_id),
        "lane_id": int(point.lane_id),
        "is_junction": bool(point.is_junction),
        "road_option": str(point.road_option),
    }


def _lane_fact_dict(fact: Any) -> dict[str, Any]:
    return {
        "road_id": fact.road_id,
        "section_id": fact.section_id,
        "lane_id": fact.lane_id,
        "is_junction": fact.is_junction,
    }


def validate_fixture(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported route transition fixture schema")
    serialized_keys = set()

    def collect_keys(value: Any) -> None:
        if isinstance(value, dict):
            serialized_keys.update(value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(payload)
    forbidden = serialized_keys.intersection(FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"fixture contains forbidden keys: {sorted(forbidden)}")
    if payload.get("source_job_id") != SOURCE_JOB_ID:
        raise ValueError("unexpected source job")
    if payload.get("source_request_id") != SOURCE_REQUEST_ID:
        raise ValueError("unexpected source request")
    route_points = payload.get("route_points")
    trajectory = payload.get("selected_trajectory_world")
    lane_facts = payload.get("candidate_lane_facts")
    waypoint_times = payload.get("waypoint_times_s")
    if not isinstance(route_points, list) or len(route_points) < 2:
        raise ValueError("fixture route is incomplete")
    if not (
        isinstance(trajectory, list)
        and len(trajectory) == 64
        and all(isinstance(point, list) and len(point) == 3 for point in trajectory)
    ):
        raise ValueError("fixture trajectory must be 64x3")
    if not (
        isinstance(lane_facts, list)
        and isinstance(waypoint_times, list)
        and len(lane_facts) == len(trajectory) == len(waypoint_times)
    ):
        raise ValueError("fixture trajectory metadata is inconsistent")
    numeric_values = [
        value
        for point in trajectory
        for value in point
    ] + list(waypoint_times)
    if not np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all():
        raise ValueError("fixture contains non-finite trajectory data")
    unsigned = dict(payload)
    integrity = unsigned.pop("integrity_sha256", None)
    if integrity != _canonical_digest(unsigned):
        raise ValueError("fixture integrity hash mismatch")


def build_fixture(
    *,
    runtime_jsonl: Path,
    town03_xodr: Path,
    carla_python_api_path: Path,
) -> dict[str, Any]:
    events = _read_events(runtime_jsonl)
    proposal = _single_event(
        events,
        "alpamayo_proposal",
        request_id=SOURCE_REQUEST_ID,
    )
    submitted = _single_event(
        events,
        "inference_submitted",
        request_id=SOURCE_REQUEST_ID,
    )
    source_tick = _single_event(
        events,
        "tick",
        loop_tick_id=int(submitted["source_loop_tick_id"]),
    )

    planner_type = load_global_route_planner(carla_python_api_path)
    import carla

    world_map = carla.Map("Town03", town03_xodr.read_text(encoding="utf-8"))
    route, _ = trace_carla_route(
        world_map,
        origin_xyz=FIXED_TOWN03_ORIGIN_XYZ,
        destination_xyz=FIXED_TOWN03_DESTINATION_XYZ,
        planner_type=planner_type,
        carla_module=carla,
        sampling_resolution_m=1.0,
    )

    position = source_tick["ego_position_world"]
    pose = pose_matrix_from_components(
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
        yaw_deg=float(source_tick["ego_yaw_deg"]),
    )
    selected_index = int(proposal["selected_candidate_index"])
    model_points = np.asarray(
        proposal["candidate_trajectories_model"][selected_index],
        dtype=np.float64,
    )
    world_points = model_ego_points_to_world(pose, model_points)
    candidate_facts = query_candidate_lane_facts(
        world_map,
        world_points,
        carla_module=carla,
    )
    ego_xyz = tuple(float(position[key]) for key in ("x", "y", "z"))
    ego_fact = query_candidate_lane_facts(
        world_map,
        (ego_xyz,),
        carla_module=carla,
    )[0]
    association = associate_route_index(
        route,
        ego_xyz,
        start_index=0,
        lane_identity=ego_fact.identity,
    )
    navigation_context = proposal["navigation_context"]
    source_time = float(proposal["source_simulation_time_s"])
    coc_hashes = proposal.get("candidate_coc_sha256") or []
    selected_coc_hash = (
        str(coc_hashes[selected_index])
        if selected_index < len(coc_hashes)
        else str(proposal["coc_sha256"])
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_job_id": SOURCE_JOB_ID,
        "source_request_id": SOURCE_REQUEST_ID,
        "source_commit": "23dba07",
        "source_configuration": {
            "camera_alignment": "projection-only",
            "num_traj_samples": 3,
            "diffusion_temperature": 1.0,
            "navigation_source": "route",
            "road_assessment_backend": "serial",
        },
        "source_loop_tick_id": int(submitted["source_loop_tick_id"]),
        "source_simulation_time_s": source_time,
        "ego_xyz": list(ego_xyz),
        "ego_yaw_deg": float(source_tick["ego_yaw_deg"]),
        "ego_lane_fact": _lane_fact_dict(ego_fact),
        "telemetry_route_index": int(navigation_context["route_index"]),
        "expected_identity_route_index": int(association.route_index),
        "route_points": [_route_point_dict(point) for point in route.points],
        "selected_candidate_index": selected_index,
        "selected_coc_sha256": selected_coc_hash,
        "selected_trajectory_world": world_points.tolist(),
        "waypoint_times_s": (
            source_time + 0.1 * np.arange(1, len(world_points) + 1)
        ).tolist(),
        "candidate_lane_facts": [
            _lane_fact_dict(fact) for fact in candidate_facts
        ],
        "expected": {
            "tracker_status": "AVAILABLE",
            "current_route_status": "MATCH",
            "near_term_route_status": "MATCH",
            "full_path_route_status": "MATCH",
            "wrong_branch_status": "DEVIATE",
        },
    }
    fixture = {**unsigned, "integrity_sha256": _canonical_digest(unsigned)}
    validate_fixture(fixture)
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-jsonl",
        type=Path,
        default=Path(f"/home/aqiu/carlamayo-runs/{SOURCE_JOB_ID}/runtime.jsonl"),
    )
    parser.add_argument(
        "--town03-xodr",
        type=Path,
        default=Path(
            "/home/aqiu/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr"
        ),
    )
    parser.add_argument(
        "--carla-python-api-path",
        type=Path,
        default=Path("/home/aqiu/carla/PythonAPI/carla"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/job_22921097_route_transition.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        validate_fixture(json.loads(args.output.read_text(encoding="utf-8")))
        print(f"valid fixture: {args.output}")
        return
    payload = build_fixture(
        runtime_jsonl=args.runtime_jsonl,
        town03_xodr=args.town03_xodr,
        carla_python_api_path=args.carla_python_api_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

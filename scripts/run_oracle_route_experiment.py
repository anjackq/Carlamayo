#!/usr/bin/env python3
"""Run a model-free CARLA route-controller experiment and optional moving capture."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from carlamayo_closed_loop import (  # noqa: E402
    build_safety_policy,
    capture_control_observation,
    smooth_controller_control,
)
from module import config as cfg  # noqa: E402
from module.camera_fixture import save_camera_fixture  # noqa: E402
from module.carla_interface import CARLAInterface  # noqa: E402
from module.carla_route_adapter import (  # noqa: E402
    assess_current_route_status,
    FIXED_TOWN03_DESTINATION_XYZ,
    FIXED_TOWN03_ORIGIN_XYZ,
    load_global_route_planner,
    query_candidate_lane_facts,
    resolve_carla_python_api_path,
    trace_carla_route,
    validate_fixed_town03_route,
)
from module.carla_safety_adapter import CarlaGroundTruthSafetyAdapter  # noqa: E402
from module.oracle_trajectory import (  # noqa: E402
    build_oracle_route_trajectory,
    project_route_progress,
)
from module.pid_controller import OfficialPIDFollower  # noqa: E402
from module.route_authorization import (  # noqa: E402
    RouteStatus,
    assess_route_candidate,
    combine_execution_constraints,
)
from module.route_navigation import (  # noqa: E402
    RouteNavigationTracker,
    RouteTrackerStatus,
)
from module.runtime_metrics import JsonlWriter  # noqa: E402
from module.safety_shield import ControlCommand, StopOnlySafetyShield  # noqa: E402
from module.trajectory_runtime import build_fixed_world_trajectory  # noqa: E402


def _destination(value: str) -> tuple[float, float, float]:
    try:
        pieces = tuple(float(piece.strip()) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("destination must use X,Y,Z") from exc
    if len(pieces) != 3 or not all(math.isfinite(piece) for piece in pieces):
        raise argparse.ArgumentTypeError("destination must use finite X,Y,Z")
    return pieces


def _outside_repository(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY_ROOT or REPOSITORY_ROOT in path.parents:
        raise argparse.ArgumentTypeError("experiment artifacts must be outside the repository")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-speed-mps", type=float, required=True)
    parser.add_argument("--maximum-acceleration-mps2", type=float, default=2.0)
    parser.add_argument("--comfortable-deceleration-mps2", type=float, default=2.5)
    parser.add_argument("--max-episode-seconds", type=float, default=20.0)
    parser.add_argument("--plan-refresh-seconds", type=float, default=1.0)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--stop-at-destination", action="store_true")
    parser.add_argument(
        "--route-destination",
        type=_destination,
        default=FIXED_TOWN03_DESTINATION_XYZ,
    )
    parser.add_argument("--carla-python-api-path", default=None)
    parser.add_argument(
        "--camera-alignment",
        choices=("baseline", "pose-only", "projection-only", "pose-projection"),
        default="projection-only",
    )
    parser.add_argument(
        "--camera-profile",
        default=os.environ.get("CARLAMAYO_CAMERA_PROFILE", ""),
    )
    parser.add_argument("--telemetry-jsonl", type=_outside_repository, required=True)
    parser.add_argument("--capture-inference-fixture", type=_outside_repository)
    parser.add_argument("--capture-route-progress-m", type=float)
    parser.add_argument("--capture-speed-tolerance-mps", type=float, default=0.35)
    parser.add_argument(
        "--capture-stationary-at-marker",
        action="store_true",
        help="Use the marker as an oracle stop target and capture after 16 stationary ticks.",
    )
    args = parser.parse_args(argv)
    numeric_positive = {
        "--target-speed-mps": args.target_speed_mps,
        "--maximum-acceleration-mps2": args.maximum_acceleration_mps2,
        "--comfortable-deceleration-mps2": args.comfortable_deceleration_mps2,
        "--max-episode-seconds": args.max_episode_seconds,
        "--plan-refresh-seconds": args.plan_refresh_seconds,
        "--capture-speed-tolerance-mps": args.capture_speed_tolerance_mps,
    }
    for name, value in numeric_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if not 0 <= args.scenario_seed <= cfg.MAX_SCENARIO_SEED:
        parser.error("--scenario-seed is out of range")
    if args.capture_inference_fixture is None and args.capture_route_progress_m is not None:
        parser.error("--capture-route-progress-m requires --capture-inference-fixture")
    if args.capture_inference_fixture is not None and args.capture_route_progress_m is None:
        parser.error("--capture-inference-fixture requires --capture-route-progress-m")
    if args.capture_route_progress_m is not None and (
        not math.isfinite(args.capture_route_progress_m)
        or args.capture_route_progress_m < 0.0
    ):
        parser.error("--capture-route-progress-m must be finite and nonnegative")
    if args.capture_stationary_at_marker and args.capture_inference_fixture is None:
        parser.error(
            "--capture-stationary-at-marker requires --capture-inference-fixture"
        )
    if args.camera_alignment != "baseline" and not args.camera_profile:
        parser.error("non-baseline camera alignment requires --camera-profile")
    try:
        args.carla_python_api_path = str(
            resolve_carla_python_api_path(args.carla_python_api_path)
        )
    except Exception as exc:
        parser.error(str(exc))
    return args


def _append(writer: JsonlWriter, event_type: str, **fields: Any) -> None:
    writer.append(
        {
            "schema_version": 2,
            "event_type": event_type,
            "wall_time_unix_s": time.time(),
            **fields,
        }
    )


def _capture_fixture(
    path: Path,
    *,
    frame_buffer: deque[dict[str, Any]],
    carla_if: CARLAInterface,
    capture_entry: dict[str, Any],
    route,
    navigation_context,
    target_speed_mps: float,
    actual_speed_mps: float,
    route_progress_m: float,
) -> dict[str, str]:
    images_array = np.zeros(
        (
            cfg.NUM_CAMERAS,
            cfg.NUM_FRAMES,
            cfg.IMG_HEIGHT,
            cfg.IMG_WIDTH,
            cfg.IMG_CHANNELS,
        ),
        dtype=np.uint8,
    )
    for temporal_index, entry in enumerate(frame_buffer):
        for camera_index, image in enumerate(entry["images"]):
            images_array[camera_index, temporal_index] = image
    history_xyz, history_rot = carla_if.get_history_in_local_frame()
    alignment = carla_if.get_camera_alignment_metadata()
    profile = alignment.get("profile") or {}
    return save_camera_fixture(
        path,
        images_array=images_array,
        history_xyz=history_xyz,
        history_rot=history_rot,
        camera_ids=capture_entry["camera_ids"],
        frame_ids=tuple(int(entry["frame_id"]) for entry in frame_buffer),
        simulation_times_s=tuple(
            float(entry["simulation_time_s"]) for entry in frame_buffer
        ),
        capture_pose_world=capture_entry["capture_pose_world"],
        metadata={
            "experiment": "oracle_route_moving_capture",
            "camera_alignment_mode": alignment["alignment_mode"],
            "camera_profile_id": profile.get("profile_id"),
            "camera_profile_sha256": profile.get("profile_sha256"),
            "dataset_revision": profile.get("dataset_revision"),
            "navigation_text": navigation_context.text,
            "navigation_context": navigation_context.to_json_dict(),
            "route_id": route.route_id,
            "route_points": [
                {
                    "xyz": list(point.xyz),
                    "road_id": point.road_id,
                    "section_id": point.section_id,
                    "lane_id": point.lane_id,
                    "is_junction": point.is_junction,
                    "road_option": point.road_option,
                }
                for point in route.points
            ],
            "route_length_m": float(route.length_m),
            "route_progress_m": float(route_progress_m),
            "target_speed_mps": float(target_speed_mps),
            "actual_speed_mps": float(actual_speed_mps),
            "map": cfg.CARLA_MAP,
            "spawn_index": cfg.EMPTY_ROAD_EGO_SPAWN_INDEX,
            "scenario_seed": cfg.EMPTY_ROAD_SCENARIO_SEED,
            "synthetic_scene": {
                "empty_road": True,
                "npc_vehicle_count": 0,
                "npc_walker_count": 0,
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    writer = JsonlWriter(args.telemetry_jsonl)
    carla_if = (
        CARLAInterface()
        if args.camera_alignment == "baseline"
        else CARLAInterface(
            camera_alignment=args.camera_alignment,
            camera_profile=args.camera_profile,
        )
    )
    safety_adapter = None
    capture_identity = None
    stop_reason = "episode_limit"
    start_wall = time.monotonic()
    tick_count = 0
    integrated_distance = 0.0
    maximum_route_progress = 0.0
    minimum_capture_speed_error = math.inf
    previous_position = None
    plan = None
    oracle = None
    plan_sequence = 0
    last_plan_time = None
    previous_nominal = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
    frame_buffer: deque[dict[str, Any]] = deque(maxlen=cfg.NUM_FRAMES)
    stationary_capture_ticks = 0
    try:
        carla_if.connect()
        carla_if.load_map(cfg.CARLA_MAP, force_reload=True)
        carla_if.set_scenario_seed(args.scenario_seed)
        carla_if.spawn_ego_vehicle(
            spawn_index=cfg.EMPTY_ROAD_EGO_SPAWN_INDEX,
            center_on_driving_lane=True,
        )
        planner_type = load_global_route_planner(args.carla_python_api_path)
        carla_module = __import__("carla")
        route, route_facts = trace_carla_route(
            carla_if.world.get_map(),
            origin_xyz=FIXED_TOWN03_ORIGIN_XYZ,
            destination_xyz=args.route_destination,
            planner_type=planner_type,
            carla_module=carla_module,
            sampling_resolution_m=1.0,
        )
        validate_fixed_town03_route(
            route,
            route_facts,
            map_name=carla_if.world.get_map().name,
        )
        tracker = RouteNavigationTracker(route)
        carla_if.enable_synchronous_mode()
        if args.capture_inference_fixture is not None:
            carla_if.setup_cameras()
        carla_if.setup_collision_sensor()
        policy = build_safety_policy()
        safety_adapter = CarlaGroundTruthSafetyAdapter(
            carla_if.world,
            carla_if.ego_vehicle,
            policy,
        )
        follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
        shield = StopOnlySafetyShield(policy)
        carla_if.apply_control(0.0, 0.0, 1.0)
        _append(
            writer,
            "episode_start",
            wall_elapsed_s=0.0,
            execution="sync",
            trajectory_source="oracle_route",
            target_speed_mps=args.target_speed_mps,
            maximum_acceleration_mps2=args.maximum_acceleration_mps2,
            comfortable_deceleration_mps2=args.comfortable_deceleration_mps2,
            stop_at_destination=args.stop_at_destination,
            scenario_seed=args.scenario_seed,
            route_startup_facts=route_facts,
            camera_alignment=carla_if.get_camera_alignment_metadata(),
        )

        while True:
            tick_context = carla_if.tick()
            tick_count += 1
            if args.capture_inference_fixture is not None:
                entry = capture_control_observation(
                    carla_if,
                    tick_context,
                    loop_tick_id=tick_count,
                    simulation_tick_seconds=float(cfg.CONTROL_DT),
                )
                frame_buffer.append(entry)
                state = entry["state"]
            else:
                state = carla_if.get_ego_state(tick_context)
            simulation_time = float(tick_context.simulation_time_s)
            position = np.asarray([state["x"], state["y"], state["z"]], dtype=np.float64)
            if previous_position is not None:
                integrated_distance += float(
                    np.linalg.norm(position[:2] - previous_position[:2])
                )
            previous_position = position
            lane_fact = query_candidate_lane_facts(
                carla_if.world.get_map(),
                (position,),
                carla_module=carla_module,
            )[0]
            route_update = tracker.update(
                position,
                source_frame_id=int(tick_context.frame_id),
                source_simulation_time_s=simulation_time,
                ego_lane_identity=(
                    lane_fact.identity if lane_fact.available else None
                ),
                ego_lane_is_junction=(
                    lane_fact.is_junction if lane_fact.available else None
                ),
                require_lane_identity=True,
            )
            navigation_context = route_update.context
            route_progress, route_distance = project_route_progress(
                route,
                position,
                start_index=tracker.route_index,
            )
            maximum_route_progress = max(maximum_route_progress, route_progress)
            capture_speed_error = (
                float(state["speed"])
                if args.capture_stationary_at_marker
                else abs(float(state["speed"]) - args.target_speed_mps)
            )
            minimum_capture_speed_error = min(
                minimum_capture_speed_error,
                capture_speed_error,
            )
            if (
                navigation_context.tracker_status
                is RouteTrackerStatus.ROUTE_UNAVAILABLE
            ):
                stop_reason = "route_unavailable"
                carla_if.apply_control(0.0, 0.0, 1.0)
                break
            if (
                last_plan_time is None
                or simulation_time - last_plan_time
                >= args.plan_refresh_seconds - 1e-6
            ):
                plan_sequence += 1
                oracle = build_oracle_route_trajectory(
                    route,
                    start_progress_m=route_progress,
                    source_simulation_time_s=simulation_time,
                    capture_pose_world=state["pose_world"],
                    current_speed_mps=float(state["speed"]),
                    target_speed_mps=args.target_speed_mps,
                    maximum_acceleration_mps2=args.maximum_acceleration_mps2,
                    comfortable_deceleration_mps2=args.comfortable_deceleration_mps2,
                    stop_at_destination=args.stop_at_destination,
                    stop_progress_m=(
                        max(
                            float(route_progress),
                            float(args.capture_route_progress_m),
                        )
                        if args.capture_stationary_at_marker
                        else None
                    ),
                )
                plan = build_fixed_world_trajectory(
                    plan_id=f"oracle-{plan_sequence}",
                    source_frame_id=int(tick_context.frame_id),
                    source_simulation_time_s=simulation_time,
                    capture_pose_world=state["pose_world"],
                    model_points=oracle.model_points,
                    coc_text="ORACLE_ROUTE_REFERENCE",
                    prompt_revision=int(navigation_context.conditioning_epoch),
                    respawn_revision=0,
                    navigation_context=navigation_context,
                )
                last_plan_time = simulation_time
                _append(
                    writer,
                    "oracle_plan",
                    wall_elapsed_s=time.monotonic() - start_wall,
                    loop_tick_id=tick_count,
                    simulation_time_s=simulation_time,
                    plan_id=plan.plan_id,
                    route_progress_m=route_progress,
                    navigation_context=navigation_context.to_json_dict(),
                    speed_profile_mps=oracle.speed_profile_mps.tolist(),
                    trajectory_world=oracle.world_points.tolist(),
                )

            assert plan is not None
            adapter_assessment = safety_adapter.assess(
                tick_context=tick_context,
                plan=plan,
            )
            envelope = adapter_assessment.road_envelope
            current_route_status = assess_current_route_status(
                carla_if.world.get_map(),
                position,
                route=route,
                route_index=tracker.route_index,
                carla_module=carla_module,
            )
            route_assessment = assess_route_candidate(
                route=route,
                current_route_index=tracker.route_index,
                current_route_status=current_route_status,
                trajectory_world_points=plan.world_points,
                waypoint_times_s=plan.waypoint_times_s,
                source_simulation_time_s=plan.source_simulation_time_s,
                current_simulation_time_s=simulation_time,
                lane_facts=query_candidate_lane_facts(
                    carla_if.world.get_map(),
                    plan.world_points,
                    carla_module=carla_module,
                ),
            )
            speed_cap, authorized_index = combine_execution_constraints(
                road_speed_cap_mps=envelope.target_speed_cap_mps,
                route_speed_cap_mps=route_assessment.route_speed_cap_mps,
                road_last_authorized_index=envelope.last_safe_waypoint_index,
                route_last_authorized_index=route_assessment.last_authorized_waypoint_index,
            )
            route_requires_stop = (
                route_assessment.current_route_status is not RouteStatus.MATCH
                or (
                    route_assessment.near_term_route_status is not RouteStatus.MATCH
                    and route_assessment.last_authorized_waypoint_index is None
                )
            )
            if route_requires_stop:
                steering_raw, throttle_raw, brake_raw = (0.0, 0.0, 1.0)
                controller_debug = {
                    "controller_state": "ROUTE_POLICY_CONSTRAINT",
                    "bypass_smoothing": True,
                }
            else:
                (
                    steering_raw,
                    throttle_raw,
                    brake_raw,
                    controller_debug,
                ) = follower.compute_world_control(
                    plan_id=plan.plan_id,
                    wp_world=plan.world_points,
                    waypoint_times_s=plan.waypoint_times_s,
                    current_simulation_time_s=simulation_time,
                    speed_mps=float(state["speed"]),
                    stop_requested=bool(plan.stop_requested),
                    terminal_stop_index=plan.terminal_stop_index,
                    capture_origin_world=plan.capture_pose_world[:3, 3],
                    target_speed_cap_mps=speed_cap,
                    maximum_authorized_waypoint_index=authorized_index,
                )
            nominal_dict, _ = smooth_controller_control(
                steering_raw=steering_raw,
                throttle_raw=throttle_raw,
                brake_raw=brake_raw,
                previous_nominal=previous_nominal,
                alpha=cfg.CONTROL_SMOOTH_ALPHA,
                bypass_smoothing=bool(controller_debug.get("bypass_smoothing")),
                constrained_deceleration=(
                    controller_debug.get("controller_state")
                    in {"ROAD_CONSTRAINED_DECELERATING", "ROUTE_POLICY_CONSTRAINT"}
                ),
            )
            nominal = ControlCommand(**nominal_dict)
            requested = ControlCommand(
                steering=float(steering_raw),
                throttle=float(throttle_raw),
                brake=float(brake_raw),
            )
            decision = shield.decide(
                road=adapter_assessment.road,
                obstacles=adapter_assessment.obstacles,
                controller_requested_control=requested,
                nominal_control=nominal,
            )
            carla_if.apply_control(*decision.applied_control.as_tuple())
            previous_nominal = nominal_dict
            echo = carla_if.get_applied_control()
            _append(
                writer,
                "tick",
                wall_elapsed_s=time.monotonic() - start_wall,
                loop_tick_id=tick_count,
                carla_frame_id=int(tick_context.frame_id),
                simulation_time_s=simulation_time,
                active_plan_id=plan.plan_id,
                controller_state=controller_debug.get("controller_state", "TRACKING"),
                speed_mps=float(state["speed"]),
                target_speed_mps=controller_debug.get("target_speed_mps"),
                ego_position_world={
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                applied_control=decision.applied_control.to_json_dict(),
                echoed_control=echo,
                navigation_context=navigation_context.to_json_dict(),
                route_progress_m=route_progress,
                route_distance_m=route_distance,
                road_envelope=envelope.to_json_dict(),
                route_candidate_assessment=route_assessment.to_json_dict(),
                safety_decision=decision.to_json_dict(),
                controller_debug=controller_debug,
                collision_count=carla_if.get_episode_collision_count(),
            )

            capture_due = (
                args.capture_inference_fixture is not None
                and capture_identity is None
                and len(frame_buffer) == cfg.NUM_FRAMES
                and carla_if.has_complete_ego_history()
                and route_progress
                >= float(args.capture_route_progress_m)
                - (
                    args.capture_speed_tolerance_mps
                    if args.capture_stationary_at_marker
                    else 0.0
                )
                and (
                    float(state["speed"]) <= args.capture_speed_tolerance_mps
                    if args.capture_stationary_at_marker
                    else abs(float(state["speed"]) - args.target_speed_mps)
                    <= args.capture_speed_tolerance_mps
                )
            )
            if args.capture_stationary_at_marker:
                if capture_due:
                    stationary_capture_ticks += 1
                else:
                    stationary_capture_ticks = 0
                capture_due = stationary_capture_ticks >= cfg.NUM_HISTORY
            if capture_due:
                capture_identity = _capture_fixture(
                    args.capture_inference_fixture,
                    frame_buffer=frame_buffer,
                    carla_if=carla_if,
                    capture_entry=frame_buffer[-1],
                    route=route,
                    navigation_context=navigation_context,
                    target_speed_mps=args.target_speed_mps,
                    actual_speed_mps=float(state["speed"]),
                    route_progress_m=route_progress,
                )
                _append(
                    writer,
                    "camera_fixture_captured",
                    wall_elapsed_s=time.monotonic() - start_wall,
                    simulation_time_s=simulation_time,
                    route_progress_m=route_progress,
                    actual_speed_mps=float(state["speed"]),
                    **capture_identity,
                )
                stop_reason = "fixture_captured"
                break
            if (
                args.stop_at_destination
                and navigation_context.tracker_status is RouteTrackerStatus.ARRIVED
                and float(state["speed"]) <= 0.2
            ):
                stop_reason = "destination_arrived"
                break
            if tick_count * float(cfg.CONTROL_DT) >= args.max_episode_seconds:
                break
        if args.capture_inference_fixture is not None and capture_identity is None:
            stop_reason = "fixture_not_captured"
            return_code = 2
        else:
            return_code = 0
    except Exception as exc:
        stop_reason = f"error:{type(exc).__name__}:{exc}"
        return_code = 1
        raise
    finally:
        try:
            _append(
                writer,
                "episode_summary",
                wall_elapsed_s=time.monotonic() - start_wall,
                stop_reason=stop_reason,
                loop_tick_count=tick_count,
                simulation_duration_s=tick_count * float(cfg.CONTROL_DT),
                integrated_distance_m=integrated_distance,
                maximum_route_progress_m=maximum_route_progress,
                minimum_capture_speed_error_mps=(
                    minimum_capture_speed_error
                    if math.isfinite(minimum_capture_speed_error)
                    else None
                ),
                collision_count=carla_if.get_episode_collision_count(),
                camera_fixture=capture_identity,
            )
        finally:
            writer.close()
            if safety_adapter is not None:
                safety_adapter.close()
            carla_if.cleanup()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a short CARLA-only spawn and drivetrain diagnostic.

The four default cases isolate two variables that can make a stationary ego
look like a controller failure:

* CARLA's authored spawn transform versus its projected driving-lane center.
* Automatic transmission versus an explicitly forced manual first gear.

No model, cameras, pedestrians, or traffic-manager actors are involved.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import carla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument(
        "--spawn-mode",
        choices=("authored", "centered", "both"),
        default="both",
    )
    parser.add_argument(
        "--gear-mode",
        choices=("automatic", "manual", "both"),
        default="both",
    )
    parser.add_argument("--settle-ticks", type=int, default=5)
    parser.add_argument("--drive-ticks", type=int, default=30)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.1)
    parser.add_argument("--throttle", type=float, default=0.6)
    parser.add_argument("--movement-threshold-m", type=float, default=0.1)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()

    if not 0 <= args.port <= 65535:
        parser.error("--port must be within [0, 65535]")
    if args.spawn_index < 0:
        parser.error("--spawn-index must be nonnegative")
    if args.settle_ticks < 0 or args.drive_ticks < 1:
        parser.error("--settle-ticks must be nonnegative and --drive-ticks must be positive")
    if args.fixed_delta_seconds <= 0:
        parser.error("--fixed-delta-seconds must be positive")
    if not 0.0 <= args.throttle <= 1.0:
        parser.error("--throttle must be within [0, 1]")
    if args.movement_threshold_m < 0:
        parser.error("--movement-threshold-m must be nonnegative")
    return args


class EventWriter:
    def __init__(self, path: Path | None):
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True)
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


def copy_transform(transform: carla.Transform) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def select_spawn_transform(
    world: carla.World,
    spawn_index: int,
    spawn_mode: str,
) -> tuple[carla.Transform, dict[str, Any]]:
    spawn_points = list(world.get_map().get_spawn_points())
    if spawn_index >= len(spawn_points):
        raise ValueError(f"spawn index {spawn_index} is outside [0, {len(spawn_points) - 1}]")

    authored = copy_transform(spawn_points[spawn_index])
    selected = copy_transform(authored)
    waypoint_fields: dict[str, Any] = {}
    if spawn_mode == "centered":
        waypoint = world.get_map().get_waypoint(
            authored.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError("authored spawn has no nearby driving waypoint")
        selected = copy_transform(waypoint.transform)
        # Match CarlaMayo's empty-road implementation exactly.
        selected.location.z = authored.location.z
        waypoint_fields = {
            "road_id": int(waypoint.road_id),
            "section_id": int(waypoint.section_id),
            "lane_id": int(waypoint.lane_id),
            "lane_width_m": float(waypoint.lane_width),
            "is_junction": bool(waypoint.is_junction),
        }

    lateral_shift = math.hypot(
        float(selected.location.x - authored.location.x),
        float(selected.location.y - authored.location.y),
    )
    return selected, {
        "authored_transform": transform_fields(authored),
        "selected_transform": transform_fields(selected),
        "authored_to_selected_xy_m": lateral_shift,
        **waypoint_fields,
    }


def transform_fields(transform: carla.Transform) -> dict[str, float]:
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "pitch": float(transform.rotation.pitch),
        "yaw": float(transform.rotation.yaw),
        "roll": float(transform.rotation.roll),
    }


def vector_fields(vector: carla.Vector3D) -> dict[str, float]:
    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z),
    }


def control_fields(control: carla.VehicleControl) -> dict[str, Any]:
    return {
        "throttle": float(control.throttle),
        "steer": float(control.steer),
        "brake": float(control.brake),
        "hand_brake": bool(control.hand_brake),
        "reverse": bool(control.reverse),
        "manual_gear_shift": bool(control.manual_gear_shift),
        "gear": int(control.gear),
    }


def observed_physics_state(actor: carla.Actor) -> bool | None:
    value = getattr(actor, "is_simulating_physics", None)
    if callable(value):
        try:
            return bool(value())
        except (RuntimeError, TypeError):
            return None
    if value is None:
        return None
    return bool(value)


def case_values(value: str, first: str, second: str) -> tuple[str, ...]:
    return (first, second) if value == "both" else (value,)


def apply_drive_control(
    vehicle: carla.Vehicle,
    *,
    gear_mode: str,
    throttle: float,
) -> None:
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=float(throttle),
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=gear_mode == "manual",
            gear=1 if gear_mode == "manual" else 0,
        )
    )


def run_case(
    client: carla.Client,
    writer: EventWriter,
    args: argparse.Namespace,
    *,
    spawn_mode: str,
    gear_mode: str,
) -> dict[str, Any]:
    case_id = f"{spawn_mode}-{gear_mode}"
    world = None
    vehicle = None
    original_settings = None
    try:
        world = client.load_world(args.map)
        world.wait_for_tick(20.0)
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta_seconds)
        world.apply_settings(settings)

        transform, spawn_metadata = select_spawn_transform(
            world,
            args.spawn_index,
            spawn_mode,
        )
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "hero")
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            raise RuntimeError(f"CARLA refused {spawn_mode} spawn transform")

        vehicle.set_simulate_physics(True)
        vehicle.set_target_velocity(carla.Vector3D())
        vehicle.set_target_angular_velocity(carla.Vector3D())
        writer.emit(
            {
                "event": "case_start",
                "case": case_id,
                "spawn_mode": spawn_mode,
                "gear_mode": gear_mode,
                "vehicle_id": int(vehicle.id),
                "simulate_physics_requested": True,
                "simulate_physics_observed": observed_physics_state(vehicle),
                **spawn_metadata,
            }
        )

        settle_control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=gear_mode == "manual",
            gear=1 if gear_mode == "manual" else 0,
        )
        for _ in range(args.settle_ticks):
            vehicle.apply_control(settle_control)
            world.tick()

        initial_transform = vehicle.get_transform()
        initial_location = initial_transform.location
        max_speed_kmh = 0.0
        max_displacement_m = 0.0
        observed_gears: set[int] = set()

        for tick_index in range(args.drive_ticks):
            apply_drive_control(
                vehicle,
                gear_mode=gear_mode,
                throttle=args.throttle,
            )
            frame_id = int(world.tick())
            snapshot = world.get_snapshot()
            actor_snapshot = snapshot.find(vehicle.id)
            if actor_snapshot is None:
                raise RuntimeError(f"ego vehicle missing from snapshot frame {snapshot.frame}")

            actor_transform = actor_snapshot.get_transform()
            velocity = actor_snapshot.get_velocity()
            speed_kmh = (
                math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
                * 3.6
            )
            displacement_m = actor_transform.location.distance(initial_location)
            actual_control = vehicle.get_control()
            max_speed_kmh = max(max_speed_kmh, speed_kmh)
            max_displacement_m = max(max_displacement_m, displacement_m)
            observed_gears.add(int(actual_control.gear))

            writer.emit(
                {
                    "event": "tick",
                    "case": case_id,
                    "tick_index": tick_index,
                    "frame_id": frame_id,
                    "snapshot_frame_id": int(snapshot.frame),
                    "simulation_time_s": float(snapshot.timestamp.elapsed_seconds),
                    "transform": transform_fields(actor_transform),
                    "velocity_mps": vector_fields(velocity),
                    "speed_kmh": speed_kmh,
                    "displacement_m": displacement_m,
                    "actual_control": control_fields(actual_control),
                    "simulate_physics_observed": observed_physics_state(vehicle),
                }
            )

        final_transform = vehicle.get_transform()
        final_displacement_m = final_transform.location.distance(initial_location)
        summary = {
            "event": "case_summary",
            "case": case_id,
            "spawn_mode": spawn_mode,
            "gear_mode": gear_mode,
            "ticks": int(args.drive_ticks),
            "throttle": float(args.throttle),
            "initial_transform": transform_fields(initial_transform),
            "final_transform": transform_fields(final_transform),
            "final_displacement_m": float(final_displacement_m),
            "max_displacement_m": float(max_displacement_m),
            "max_speed_kmh": float(max_speed_kmh),
            "observed_gears": sorted(observed_gears),
            "moved": bool(max_displacement_m >= args.movement_threshold_m),
        }
        writer.emit(summary)
        return summary
    finally:
        if vehicle is not None:
            try:
                vehicle.destroy()
            except RuntimeError as exc:
                writer.emit(
                    {
                        "event": "cleanup_warning",
                        "case": case_id,
                        "message": f"vehicle destroy failed: {exc}",
                    }
                )
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError as exc:
                writer.emit(
                    {
                        "event": "cleanup_warning",
                        "case": case_id,
                        "message": f"world settings restore failed: {exc}",
                    }
                )


def main() -> int:
    args = parse_args()
    writer = EventWriter(args.output_jsonl)
    failures = 0
    summaries = []
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        client.get_world()
        writer.emit(
            {
                "event": "run_start",
                "host": args.host,
                "port": int(args.port),
                "map": args.map,
                "spawn_index": int(args.spawn_index),
                "settle_ticks": int(args.settle_ticks),
                "drive_ticks": int(args.drive_ticks),
                "fixed_delta_seconds": float(args.fixed_delta_seconds),
                "throttle": float(args.throttle),
            }
        )

        spawn_modes = case_values(args.spawn_mode, "authored", "centered")
        gear_modes = case_values(args.gear_mode, "automatic", "manual")
        for spawn_mode, gear_mode in itertools.product(spawn_modes, gear_modes):
            try:
                summaries.append(
                    run_case(
                        client,
                        writer,
                        args,
                        spawn_mode=spawn_mode,
                        gear_mode=gear_mode,
                    )
                )
            except Exception as exc:
                failures += 1
                writer.emit(
                    {
                        "event": "case_error",
                        "case": f"{spawn_mode}-{gear_mode}",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                time.sleep(0.5)

        writer.emit(
            {
                "event": "run_summary",
                "case_count": len(spawn_modes) * len(gear_modes),
                "completed_case_count": len(summaries),
                "failed_case_count": failures,
                "cases": summaries,
            }
        )
        return 1 if failures else 0
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())

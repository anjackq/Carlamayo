"""Deterministic route-derived oracle trajectories for controller experiments.

The oracle is deliberately model-independent.  It provides a known-good
spatiotemporal reference so Alpamayo prediction quality can be separated from
PID, route-authorization, gearbox, and CARLA vehicle-physics behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config as cfg
from .geometry import world_points_to_model_ego
from .route_navigation import RoutePlan


@dataclass(frozen=True)
class OracleTrajectory:
    world_points: np.ndarray
    model_points: np.ndarray
    waypoint_times_s: np.ndarray
    route_progress_m: np.ndarray
    speed_profile_mps: np.ndarray
    target_speed_mps: float
    stop_at_destination: bool

    def __post_init__(self) -> None:
        count = int(cfg.TRAJECTORY_NUM_POINTS)
        expected = {
            "world_points": (count, 3),
            "model_points": (count, 3),
            "waypoint_times_s": (count,),
            "route_progress_m": (count,),
            "speed_profile_mps": (count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            frozen = np.array(value, dtype=np.float64, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if not math.isfinite(float(self.target_speed_mps)) or self.target_speed_mps < 0:
            raise ValueError("target_speed_mps must be finite and nonnegative")


def _unique_route_geometry(route: RoutePlan) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.asarray(route.cumulative_distance_m, dtype=np.float64)
    points = np.asarray([point.xyz for point in route.points], dtype=np.float64)
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-9])
    cumulative = cumulative[keep]
    points = points[keep]
    if len(cumulative) < 2 or cumulative[-1] <= cumulative[0]:
        raise ValueError("route must contain nonzero geometric extent")
    return cumulative, points


def interpolate_route_world_points(
    route: RoutePlan,
    progress_m: Any,
) -> np.ndarray:
    """Interpolate XYZ along route arc length without changing its geometry."""

    progress = np.asarray(progress_m, dtype=np.float64)
    if progress.ndim != 1 or not np.isfinite(progress).all():
        raise ValueError("progress_m must be a finite one-dimensional array")
    cumulative, points = _unique_route_geometry(route)
    clipped = np.clip(progress, cumulative[0], cumulative[-1])
    return np.stack(
        [
            np.interp(clipped, cumulative, points[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )


def project_route_progress(
    route: RoutePlan,
    point_xyz: Any,
    *,
    start_index: int = 0,
) -> tuple[float, float]:
    """Project one world point to route arc length using a monotonic suffix."""

    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape not in {(2,), (3,)} or not np.isfinite(point).all():
        raise ValueError("point_xyz must contain two or three finite values")
    cumulative = np.asarray(route.cumulative_distance_m, dtype=np.float64)
    points = np.asarray([item.xyz for item in route.points], dtype=np.float64)
    start = min(max(0, int(start_index) - 1), len(points) - 2)
    best_distance = math.inf
    best_progress = float(cumulative[start])
    for index in range(start, len(points) - 1):
        segment = points[index + 1, :2] - points[index, :2]
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            fraction = 0.0
            projected = points[index, :2]
        else:
            fraction = float(
                np.clip(
                    np.dot(point[:2] - points[index, :2], segment) / length_sq,
                    0.0,
                    1.0,
                )
            )
            projected = points[index, :2] + fraction * segment
        distance = float(np.linalg.norm(point[:2] - projected))
        if distance < best_distance:
            best_distance = distance
            best_progress = float(
                cumulative[index]
                + fraction * (cumulative[index + 1] - cumulative[index])
            )
    return best_progress, best_distance


def build_oracle_route_trajectory(
    route: RoutePlan,
    *,
    start_progress_m: float,
    source_simulation_time_s: float,
    capture_pose_world: Any,
    current_speed_mps: float,
    target_speed_mps: float,
    maximum_acceleration_mps2: float = 2.0,
    comfortable_deceleration_mps2: float = 2.5,
    stop_at_destination: bool = False,
    stop_progress_m: float | None = None,
) -> OracleTrajectory:
    """Create a 64-point route trajectory with bounded longitudinal motion."""

    numeric = (
        start_progress_m,
        source_simulation_time_s,
        current_speed_mps,
        target_speed_mps,
        maximum_acceleration_mps2,
        comfortable_deceleration_mps2,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("oracle trajectory inputs must be finite")
    if (
        start_progress_m < 0.0
        or source_simulation_time_s < 0.0
        or current_speed_mps < 0.0
        or target_speed_mps < 0.0
        or maximum_acceleration_mps2 <= 0.0
        or comfortable_deceleration_mps2 <= 0.0
    ):
        raise ValueError("oracle speed, time, progress, and acceleration inputs are invalid")
    if stop_progress_m is not None:
        try:
            stop_progress_m = float(stop_progress_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("stop_progress_m must be finite or None") from exc
        if (
            not math.isfinite(stop_progress_m)
            or stop_progress_m < start_progress_m
            or stop_progress_m > route.length_m
        ):
            raise ValueError("stop_progress_m must lie on the remaining route")

    dt = float(cfg.TRAJECTORY_WAYPOINT_DT)
    count = int(cfg.TRAJECTORY_NUM_POINTS)
    route_end = float(
        stop_progress_m
        if stop_progress_m is not None
        else route.length_m
    )
    stop_at_end = bool(stop_at_destination or stop_progress_m is not None)
    progress = float(np.clip(start_progress_m, 0.0, route_end))
    speed = float(current_speed_mps)
    progress_profile = np.empty(count, dtype=np.float64)
    speed_profile = np.empty(count, dtype=np.float64)

    for index in range(count):
        remaining = max(0.0, route_end - progress)
        desired_speed = float(target_speed_mps)
        if stop_at_end:
            braking_speed = math.sqrt(
                max(0.0, 2.0 * float(comfortable_deceleration_mps2) * remaining)
            )
            desired_speed = min(desired_speed, braking_speed)
        if desired_speed >= speed:
            next_speed = min(
                desired_speed,
                speed + float(maximum_acceleration_mps2) * dt,
            )
        else:
            next_speed = max(
                desired_speed,
                speed - float(comfortable_deceleration_mps2) * dt,
            )
        next_progress = min(
            route_end,
            progress + 0.5 * (speed + next_speed) * dt,
        )
        if route_end - next_progress <= 1e-6:
            next_progress = route_end
            if stop_at_end:
                next_speed = 0.0
        progress = next_progress
        speed = max(0.0, next_speed)
        progress_profile[index] = progress
        speed_profile[index] = speed

    world_points = interpolate_route_world_points(route, progress_profile)
    model_points = world_points_to_model_ego(capture_pose_world, world_points)
    waypoint_times = float(source_simulation_time_s) + dt * np.arange(
        1,
        count + 1,
        dtype=np.float64,
    )
    return OracleTrajectory(
        world_points=world_points,
        model_points=model_points,
        waypoint_times_s=waypoint_times,
        route_progress_m=progress_profile,
        speed_profile_mps=speed_profile,
        target_speed_mps=float(target_speed_mps),
        stop_at_destination=stop_at_end,
    )


__all__ = [
    "OracleTrajectory",
    "build_oracle_route_trajectory",
    "interpolate_route_world_points",
    "project_route_progress",
]

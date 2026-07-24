"""PID controller helpers and official CARLA follower."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import carla
import numpy as np

from . import config as cfg
from .trajectory_runtime import (
    detect_terminal_stop_index,
    target_speed_from_timestamps,
)


def _resolve_vehicle_pid_controller():
    """Import VehiclePIDController, auto-adding env/relative CARLA agent paths."""
    try:
        from agents.navigation.controller import VehiclePIDController as _VehiclePIDController
        return _VehiclePIDController
    except ImportError:
        pass

    candidate_roots = []
    for env_key in ("CARLA_ROOT", "CARLA_HOME"):
        v = os.environ.get(env_key)
        if v:
            candidate_roots.append(os.path.expanduser(v))

    candidate_roots.append(os.path.expanduser(cfg.CARLA_AGENT_ROOT))

    for root in candidate_roots:
        agents_parent = os.path.abspath(os.path.join(root, "PythonAPI", "carla"))
        if os.path.isdir(agents_parent) and agents_parent not in sys.path:
            sys.path.append(agents_parent)

    try:
        from agents.navigation.controller import VehiclePIDController as _VehiclePIDController
        return _VehiclePIDController
    except ImportError as e:
        raise ImportError(
            "VehiclePIDController not found. Set CARLA_ROOT or add "
            "'<CARLA_ROOT>/PythonAPI/carla' to PYTHONPATH."
        ) from e


def alpamayo_to_carla_local(wp_ego):
    """Convert Alpamayo local frame (y=left) to CARLA local frame (y=right)."""
    wp_local = np.asarray(wp_ego, dtype=np.float64).copy()
    wp_local[:, 1] *= -1.0
    return wp_local


def local_to_world(vehicle_tf, wp_local):
    """Convert local waypoints to world using CARLA transform."""
    wp_world = []
    for p in wp_local:
        loc_w = vehicle_tf.transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        wp_world.append([loc_w.x, loc_w.y, loc_w.z])
    return np.asarray(wp_world, dtype=np.float64)


@dataclass(frozen=True)
class RawTargetWaypoint:
    """Minimal waypoint-like target for CARLA VehiclePIDController."""

    transform: carla.Transform


class OfficialPIDFollower:
    """CARLA official PID follower for raw Alpamayo targets."""

    def __init__(self, world, vehicle):
        VehiclePIDController = _resolve_vehicle_pid_controller()
        self.world = world
        self.vehicle = vehicle
        args_lateral = {
            "K_P": cfg.PID_LAT_KP,
            "K_I": cfg.PID_LAT_KI,
            "K_D": cfg.PID_LAT_KD,
            "dt": cfg.CONTROL_DT,
        }
        args_longitudinal = {
            "K_P": cfg.PID_LON_KP,
            "K_I": cfg.PID_LON_KI,
            "K_D": cfg.PID_LON_KD,
            "dt": cfg.CONTROL_DT,
        }
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral=args_lateral,
            args_longitudinal=args_longitudinal,
            max_throttle=cfg.THROTTLE_MAX,
            max_brake=cfg.BRAKE_MAX,
            max_steering=0.8,
        )
        self._active_plan_id = None
        self._progress_index = 0
        self._progress_s_m = 0.0

    def reset_plan_progress(self, plan_id=None):
        """Reset monotonic fixed-world path progress for a new plan."""

        self._active_plan_id = plan_id
        self._progress_index = 0
        self._progress_s_m = 0.0

    @staticmethod
    def _target_speed_from_timestamps(
        wp_world,
        waypoint_times_s,
        start_idx,
        *,
        capture_origin_world=None,
        terminal_stop_index=None,
    ):
        """Derive speed from temporal waypoint spacing, including terminal stops."""

        return target_speed_from_timestamps(
            wp_world,
            waypoint_times_s,
            start_idx,
            capture_origin_world=capture_origin_world,
            terminal_stop_index=terminal_stop_index,
        )

    @staticmethod
    def _launch_floor_speed(
        wp_world,
        waypoint_times_s,
        start_idx,
        current_speed,
        *,
        terminal_stop_index=None,
    ):
        """Minimum launch speed so a fresh moving plan pulls a stopped ego off the line.

        Returns 0.0 (no floor) once the ego is already rolling, or whenever the
        plan does not intend meaningful forward motion over the near horizon --
        so stop, creep, and terminal-stop plans are unaffected.  The floor keys
        off the ego's actual speed rather than plan age or progress, which is what
        makes it survive the ~1 s proposal-replacement cadence: a fresh plan can
        no longer reset an in-progress launch back to its stationary prefix.  It
        never exceeds the plan's own intended near-horizon peak speed.
        """

        if current_speed >= float(cfg.PID_LAUNCH_ENGAGE_SPEED_MPS):
            return 0.0

        points = np.asarray(wp_world, dtype=np.float64)
        times = np.asarray(waypoint_times_s, dtype=np.float64)
        if len(points) < 2 or len(points) != len(times):
            return 0.0

        segment_distance = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        segment_dt = np.diff(times)
        speeds = np.divide(
            segment_distance,
            segment_dt,
            out=np.zeros_like(segment_distance),
            where=segment_dt > 1e-6,
        )
        window_end = len(speeds)
        if terminal_stop_index is not None:
            window_end = min(window_end, max(0, int(terminal_stop_index)))
        window_start = min(max(0, int(start_idx)), window_end)
        horizon_segments = max(
            1,
            int(round(float(cfg.PID_LAUNCH_HORIZON_S) / float(cfg.TRAJECTORY_WAYPOINT_DT))),
        )
        window = speeds[window_start : min(window_end, window_start + horizon_segments)]
        intended_peak = float(np.max(window)) if len(window) else 0.0
        if intended_peak < float(cfg.PID_LAUNCH_MIN_INTENT_MPS):
            return 0.0
        return float(min(intended_peak, float(cfg.PID_LAUNCH_SPEED_MPS)))

    @staticmethod
    def _full_brake(mode, *, controller_state="FALLBACK", **debug):
        return 0.0, 0.0, 1.0, {
            "mode": mode,
            "controller_state": controller_state,
            "target_speed_mps": 0.0,
            "bypass_smoothing": True,
            **debug,
        }

    @staticmethod
    def _cumulative_distance(points):
        if len(points) == 0:
            return np.empty((0,), dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segment_lengths)])

    def _project_monotonic_progress(self, points, cumulative):
        """Project ego onto the polyline without allowing progress to decrease."""

        if len(points) <= 1:
            self._progress_index = 0
            return self._progress_s_m

        ego_loc = self.vehicle.get_transform().location
        ego_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float64)
        start_segment = max(0, min(self._progress_index - 1, len(points) - 2))
        best_distance_sq = float("inf")
        best_progress = self._progress_s_m

        for segment_index in range(start_segment, len(points) - 1):
            start = points[segment_index, :2]
            delta = points[segment_index + 1, :2] - start
            length_sq = float(np.dot(delta, delta))
            if length_sq <= 1e-12:
                fraction = 0.0
                projected = start
            else:
                fraction = float(np.clip(np.dot(ego_xy - start, delta) / length_sq, 0.0, 1.0))
                projected = start + fraction * delta
            distance_sq = float(np.dot(ego_xy - projected, ego_xy - projected))
            candidate_progress = float(
                cumulative[segment_index]
                + fraction * (cumulative[segment_index + 1] - cumulative[segment_index])
            )
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_progress = candidate_progress

        self._progress_s_m = max(self._progress_s_m, best_progress)
        self._progress_index = max(
            self._progress_index,
            int(np.searchsorted(cumulative, self._progress_s_m, side="right") - 1),
        )
        self._progress_index = min(self._progress_index, len(points) - 1)
        return self._progress_s_m

    def _pick_fixed_world_target(
        self,
        wp_world,
        speed_mps,
        *,
        terminal_stop_index=None,
        maximum_authorized_waypoint_index=None,
    ):
        """Choose lateral target from measured geometric path progress.

        Waypoint timestamps describe the plan's longitudinal speed profile; they
        do not prove that the physical ego has reached the matching waypoint.
        Steering therefore advances only from the ego's monotonic projection
        onto the fixed-world path.  This prevents longitudinal lag from being
        added to the configured lateral lookahead on curves.
        """

        points = np.asarray(wp_world, dtype=np.float64)
        lookahead_m = float(
            np.clip(
                cfg.PID_LOOKAHEAD_MIN_M + cfg.PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
                cfg.PID_LOOKAHEAD_MIN_M,
                cfg.PID_LOOKAHEAD_MAX_M,
            )
        )
        if len(points) == 0:
            return None, 0, lookahead_m, float("inf"), 0.0, np.empty((0,))

        ego_loc = self.vehicle.get_transform().location
        ego_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float64)
        cumulative = self._cumulative_distance(points)
        progress_s = self._project_monotonic_progress(points, cumulative)
        maximum_target_index = len(points) - 1
        if terminal_stop_index is not None:
            maximum_target_index = min(
                maximum_target_index,
                max(0, int(terminal_stop_index)),
            )
        if maximum_authorized_waypoint_index is not None:
            maximum_target_index = min(
                maximum_target_index,
                max(0, int(maximum_authorized_waypoint_index)),
            )
        maximum_target_s = float(cumulative[maximum_target_index])
        if progress_s > maximum_target_s + 1e-6:
            return (
                None,
                maximum_target_index,
                lookahead_m,
                float("inf"),
                progress_s,
                cumulative,
            )
        target_s = min(progress_s + lookahead_m, maximum_target_s)
        target_idx = int(np.searchsorted(cumulative, target_s, side="left"))
        target_idx = min(
            max(self._progress_index, target_idx),
            maximum_target_index,
        )

        target = points[target_idx]
        target_distance = float(np.linalg.norm(target[:2] - ego_xy))
        target_wp = RawTargetWaypoint(
            carla.Transform(
                carla.Location(x=float(target[0]), y=float(target[1]), z=float(target[2])),
                carla.Rotation(),
            )
        )
        return target_wp, target_idx, lookahead_m, target_distance, progress_s, cumulative

    def _pick_target(self, wp_world, speed_mps):
        lookahead_m = float(
            np.clip(
                cfg.PID_LOOKAHEAD_MIN_M + cfg.PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
                cfg.PID_LOOKAHEAD_MIN_M,
                cfg.PID_LOOKAHEAD_MAX_M,
            )
        )
        if len(wp_world) == 0:
            return None, 0, lookahead_m
        if len(wp_world) == 1:
            target_idx = 0
        else:
            seg = np.linalg.norm(np.diff(wp_world[:, :2], axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            target_idx = int(min(np.searchsorted(cum, lookahead_m), len(wp_world) - 1))
        loc = carla.Location(
            x=float(wp_world[target_idx, 0]),
            y=float(wp_world[target_idx, 1]),
            z=float(wp_world[target_idx, 2]),
        )
        target_wp = RawTargetWaypoint(carla.Transform(loc, carla.Rotation()))
        return target_wp, target_idx, lookahead_m

    def compute_control(self, vehicle_tf, wp_ego, speed_mps):
        wp_local = alpamayo_to_carla_local(wp_ego)
        wp_world = local_to_world(vehicle_tf, wp_local)
        traj_extent = float(np.max(np.linalg.norm(wp_local[:, :2], axis=1)))
        target_speed_kmh = float(
            np.clip(
                cfg.PID_TARGET_SPEED_MIN_KMH + cfg.PID_TARGET_SPEED_EXTENT_GAIN * traj_extent,
                cfg.PID_TARGET_SPEED_MIN_KMH,
                cfg.PID_TARGET_SPEED_MAX_KMH,
            )
        )
        target_wp, target_idx, lookahead_m = self._pick_target(wp_world, speed_mps)
        if target_wp is None:
            return 0.0, 0.0, 0.0, {
                "mode": "official_pid_no_target",
                "traj_extent": traj_extent,
            }
        control = self.pid.run_step(target_speed_kmh, target_wp)
        debug = {
            "mode": "official_pid",
            "target_speed_kmh": target_speed_kmh,
            "lookahead_m": lookahead_m,
            "target_idx": int(target_idx),
            "target_wp_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_raw_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_projected_to_road": False,
            "traj_extent": traj_extent,
        }
        return float(control.steer), float(control.throttle), float(control.brake), debug

    def compute_world_control(
        self,
        *,
        plan_id,
        wp_world,
        waypoint_times_s,
        current_simulation_time_s,
        speed_mps,
        stop_requested=False,
        terminal_stop_index=None,
        capture_origin_world=None,
        target_speed_cap_mps=None,
        maximum_authorized_waypoint_index=None,
    ):
        """Track a timestamped fixed-world path without re-anchoring it to ego.

        Invalid or exhausted input fails closed.  The legacy ``compute_control``
        method above remains available for callers that still provide ego-frame
        waypoints.
        """

        try:
            points = np.asarray(wp_world, dtype=np.float64)
            times = np.asarray(waypoint_times_s, dtype=np.float64)
            current_time = float(current_simulation_time_s)
            current_speed = float(speed_mps)
        except (TypeError, ValueError):
            return self._full_brake(
                "invalid_trajectory",
                rejection_reason="non_numeric_trajectory_input",
            )
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or times.ndim != 1
            or len(points) != len(times)
            or len(points) == 0
            or not np.isfinite(points).all()
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0.0)
            or not np.isfinite(current_time)
            or not np.isfinite(current_speed)
            or current_speed < 0.0
        ):
            return self._full_brake(
                "invalid_trajectory",
                rejection_reason="invalid_fixed_world_trajectory",
            )

        if plan_id != self._active_plan_id:
            self.reset_plan_progress(plan_id)

        try:
            if target_speed_cap_mps is not None:
                target_speed_cap_mps = float(target_speed_cap_mps)
                if not math.isfinite(target_speed_cap_mps) or target_speed_cap_mps < 0.0:
                    raise ValueError
            if maximum_authorized_waypoint_index is not None:
                maximum_authorized_waypoint_index = int(
                    maximum_authorized_waypoint_index
                )
                if not 0 <= maximum_authorized_waypoint_index < len(points):
                    raise ValueError
        except (TypeError, ValueError):
            return self._full_brake(
                "invalid_safety_constraint",
                rejection_reason="invalid_road_execution_constraint",
            )

        first_future_idx = int(
            np.searchsorted(times, current_time, side="right")
        )
        if first_future_idx >= len(points):
            return self._full_brake("trajectory_exhausted")

        inferred_stop_index = detect_terminal_stop_index(
            points,
            waypoint_dt_s=float(np.median(np.diff(times))),
        )
        if terminal_stop_index is None:
            terminal_stop_index = inferred_stop_index
        else:
            try:
                terminal_stop_index = int(terminal_stop_index)
            except (TypeError, ValueError):
                return self._full_brake(
                    "invalid_trajectory",
                    rejection_reason="invalid_terminal_stop_index",
                )
            if not 0 <= terminal_stop_index < len(points):
                return self._full_brake(
                    "invalid_trajectory",
                    rejection_reason="invalid_terminal_stop_index",
                )
        if stop_requested and terminal_stop_index is None:
            # An explicit caller stop request is fail-safe even if its geometry
            # does not contain a classifiable stationary tail.
            return self._full_brake(
                "explicit_stop",
                controller_state="STOPPED",
                target_idx=first_future_idx,
            )
        if terminal_stop_index is not None and first_future_idx > terminal_stop_index:
            return self._full_brake(
                "terminal_stop_time_reached",
                controller_state=(
                    "STOPPED"
                    if current_speed <= float(cfg.PID_STOP_SPEED_THRESHOLD_MPS)
                    else "DECELERATING"
                ),
                target_idx=int(terminal_stop_index),
            )
        if (
            maximum_authorized_waypoint_index is not None
            and first_future_idx > maximum_authorized_waypoint_index
        ):
            return self._full_brake(
                "road_safe_prefix_exhausted",
                controller_state="ROAD_CONSTRAINED_DECELERATING",
                target_idx=int(maximum_authorized_waypoint_index),
            )

        (
            target_wp,
            target_idx,
            lookahead_m,
            target_distance,
            progress_s,
            cumulative,
        ) = self._pick_fixed_world_target(
            points,
            current_speed,
            terminal_stop_index=terminal_stop_index,
            maximum_authorized_waypoint_index=maximum_authorized_waypoint_index,
        )
        if target_wp is None:
            if (
                terminal_stop_index is not None
                and progress_s
                > float(cumulative[int(terminal_stop_index)]) + 1e-6
            ):
                return self._full_brake(
                    "terminal_stop",
                    controller_state=(
                        "STOPPED"
                        if current_speed
                        <= float(cfg.PID_STOP_SPEED_THRESHOLD_MPS)
                        else "DECELERATING"
                    ),
                    target_idx=int(terminal_stop_index),
                    progress_s_m=progress_s,
                )
            if (
                maximum_authorized_waypoint_index is not None
                and progress_s
                > float(
                    cumulative[int(maximum_authorized_waypoint_index)]
                )
                + 1e-6
            ):
                return self._full_brake(
                    "road_safe_prefix_exhausted",
                    controller_state="ROAD_CONSTRAINED_DECELERATING",
                    target_idx=int(maximum_authorized_waypoint_index),
                    progress_s_m=progress_s,
                )
            return self._full_brake("invalid_trajectory", rejection_reason="no_target")

        profile_index = max(first_future_idx, self._progress_index)
        temporal_progress_s = float(cumulative[first_future_idx])
        time_geometry_gap_m = temporal_progress_s - progress_s
        steering_target_path_distance_m = max(
            0.0,
            float(cumulative[target_idx]) - progress_s,
        )
        control_limit_index = terminal_stop_index
        if maximum_authorized_waypoint_index is not None:
            control_limit_index = (
                maximum_authorized_waypoint_index
                if control_limit_index is None
                else min(control_limit_index, maximum_authorized_waypoint_index)
            )

        target_speed_mps = self._target_speed_from_timestamps(
            points,
            times,
            profile_index,
            capture_origin_world=capture_origin_world,
            terminal_stop_index=control_limit_index,
        )

        # Break the launch deadlock: pull a stopped ego off the line when the plan
        # intends forward motion, before the terminal-stop braking clamp so an
        # approaching stop can still override the floor downward.
        launch_floor_mps = self._launch_floor_speed(
            points,
            times,
            first_future_idx,
            current_speed,
            terminal_stop_index=control_limit_index,
        )
        if launch_floor_mps > 0.0:
            target_speed_mps = max(target_speed_mps, launch_floor_mps)

        distance_to_stop = None
        if terminal_stop_index is not None:
            stop_s = float(cumulative[int(terminal_stop_index)])
            distance_to_stop = max(0.0, stop_s - progress_s)
            braking_speed = math.sqrt(
                max(0.0, 2.0 * float(cfg.PID_COMFORTABLE_DECEL_MPS2) * distance_to_stop)
            )
            target_speed_mps = min(target_speed_mps, braking_speed)
            if distance_to_stop <= float(cfg.PID_STOP_POSITION_TOLERANCE_M):
                return self._full_brake(
                    "terminal_stop",
                    controller_state=(
                        "STOPPED"
                        if current_speed <= float(cfg.PID_STOP_SPEED_THRESHOLD_MPS)
                        else "DECELERATING"
                    ),
                    target_idx=int(terminal_stop_index),
                    distance_to_stop_m=distance_to_stop,
                    progress_s_m=progress_s,
                )

        unconstrained_target_speed_mps = target_speed_mps
        road_speed_limited = (
            target_speed_cap_mps is not None
            and target_speed_cap_mps < target_speed_mps
        )
        if target_speed_cap_mps is not None:
            target_speed_mps = min(target_speed_mps, target_speed_cap_mps)

        control = self.pid.run_step(target_speed_mps * 3.6, target_wp)
        if road_speed_limited:
            controller_state = "ROAD_CONSTRAINED_DECELERATING"
        else:
            controller_state = (
                "DECELERATING"
                if target_speed_mps + float(cfg.PID_DECELERATION_STATE_DELTA_MPS) < current_speed
                else "TRACKING"
            )
        throttle = float(control.throttle)
        brake = float(control.brake)
        if target_speed_mps <= float(cfg.PID_STOP_SPEED_THRESHOLD_MPS):
            throttle = 0.0
        return float(control.steer), throttle, brake, {
            "mode": "fixed_world_pid",
            "controller_state": controller_state,
            "target_speed_mps": target_speed_mps,
            "lookahead_m": lookahead_m,
            "target_idx": int(target_idx),
            "steering_target_index": int(target_idx),
            "target_distance_m": target_distance,
            "steering_target_path_distance_m": (
                steering_target_path_distance_m
            ),
            "target_wp_xyz": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
                float(target_wp.transform.location.z),
            ],
            "progress_index": int(self._progress_index),
            "progress_s_m": progress_s,
            "steering_reference": "geometric_progress",
            "steering_reference_index": int(self._progress_index),
            "steering_reference_s_m": progress_s,
            "first_future_index": first_future_idx,
            "speed_profile_index": int(profile_index),
            "temporal_progress_s_m": temporal_progress_s,
            "time_geometry_gap_m": time_geometry_gap_m,
            "terminal_stop_index": terminal_stop_index,
            "distance_to_stop_m": distance_to_stop,
            "launch_floor_mps": launch_floor_mps,
            "unconstrained_target_speed_mps": unconstrained_target_speed_mps,
            "road_speed_cap_mps": target_speed_cap_mps,
            "road_speed_limited": road_speed_limited,
            "maximum_authorized_waypoint_index": maximum_authorized_waypoint_index,
            "bypass_smoothing": False,
        }

"""Diagnostic source-speed reachability facts for frozen trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from . import config as cfg


class PhysicalReachabilityStatus(str, Enum):
    REACHABLE = "REACHABLE"
    TOO_LONG = "TOO_LONG"
    TOO_SHORT_TO_STOP = "TOO_SHORT_TO_STOP"


class SourceSpeedPriorStatus(str, Enum):
    CONSISTENT = "CONSISTENT"
    ACCELERATION_PRIOR = "ACCELERATION_PRIOR"
    DECELERATION_PRIOR = "DECELERATION_PRIOR"
    STOP_PRIOR = "STOP_PRIOR"


@dataclass(frozen=True)
class TrajectoryReachabilityProfile:
    source_speed_mps: float
    history_mean_speed_mps: float
    history_terminal_speed_mps: float
    history_acceleration_mps2: float
    horizon_s: float
    path_length_m: float
    terminal_displacement_m: float
    minimum_reachable_distance_m: float
    maximum_reachable_distance_m: float
    required_constant_acceleration_mps2: float
    physical_status: PhysicalReachabilityStatus
    source_speed_prior_status: SourceSpeedPriorStatus
    near_horizon_s: float
    near_path_length_m: float
    near_minimum_reachable_distance_m: float
    near_maximum_reachable_distance_m: float
    near_required_constant_acceleration_mps2: float
    near_physical_status: PhysicalReachabilityStatus
    near_source_speed_prior_status: SourceSpeedPriorStatus

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_speed_mps": float(self.source_speed_mps),
            "history_mean_speed_mps": float(self.history_mean_speed_mps),
            "history_terminal_speed_mps": float(self.history_terminal_speed_mps),
            "history_acceleration_mps2": float(self.history_acceleration_mps2),
            "horizon_s": float(self.horizon_s),
            "path_length_m": float(self.path_length_m),
            "terminal_displacement_m": float(self.terminal_displacement_m),
            "minimum_reachable_distance_m": float(
                self.minimum_reachable_distance_m
            ),
            "maximum_reachable_distance_m": float(
                self.maximum_reachable_distance_m
            ),
            "required_constant_acceleration_mps2": float(
                self.required_constant_acceleration_mps2
            ),
            "physical_status": self.physical_status.value,
            "source_speed_prior_status": self.source_speed_prior_status.value,
            "near_horizon_s": float(self.near_horizon_s),
            "near_path_length_m": float(self.near_path_length_m),
            "near_minimum_reachable_distance_m": float(
                self.near_minimum_reachable_distance_m
            ),
            "near_maximum_reachable_distance_m": float(
                self.near_maximum_reachable_distance_m
            ),
            "near_required_constant_acceleration_mps2": float(
                self.near_required_constant_acceleration_mps2
            ),
            "near_physical_status": self.near_physical_status.value,
            "near_source_speed_prior_status": (
                self.near_source_speed_prior_status.value
            ),
        }


def _distance_bounds(
    speed_mps: float,
    horizon_s: float,
    *,
    maximum_acceleration_mps2: float,
    comfortable_deceleration_mps2: float,
) -> tuple[float, float]:
    stop_time = speed_mps / comfortable_deceleration_mps2
    if stop_time <= horizon_s:
        minimum = speed_mps * speed_mps / (2.0 * comfortable_deceleration_mps2)
    else:
        minimum = (
            speed_mps * horizon_s
            - 0.5 * comfortable_deceleration_mps2 * horizon_s * horizon_s
        )
    maximum = (
        speed_mps * horizon_s
        + 0.5 * maximum_acceleration_mps2 * horizon_s * horizon_s
    )
    return max(0.0, minimum), max(0.0, maximum)


def _physical_status(
    distance_m: float,
    minimum_m: float,
    maximum_m: float,
    *,
    tolerance_m: float,
) -> PhysicalReachabilityStatus:
    if distance_m > maximum_m + tolerance_m:
        return PhysicalReachabilityStatus.TOO_LONG
    if distance_m + tolerance_m < minimum_m:
        return PhysicalReachabilityStatus.TOO_SHORT_TO_STOP
    return PhysicalReachabilityStatus.REACHABLE


def _prior_status(
    distance_m: float,
    horizon_s: float,
    source_speed_mps: float,
    *,
    acceleration_tolerance_mps2: float,
) -> tuple[float, SourceSpeedPriorStatus]:
    required_acceleration = (
        2.0 * (distance_m - source_speed_mps * horizon_s)
        / (horizon_s * horizon_s)
    )
    if distance_m <= float(cfg.PID_STOP_POSITION_TOLERANCE_M):
        status = SourceSpeedPriorStatus.STOP_PRIOR
    elif required_acceleration > acceleration_tolerance_mps2:
        status = SourceSpeedPriorStatus.ACCELERATION_PRIOR
    elif required_acceleration < -acceleration_tolerance_mps2:
        status = SourceSpeedPriorStatus.DECELERATION_PRIOR
    else:
        status = SourceSpeedPriorStatus.CONSISTENT
    return required_acceleration, status


def _history_motion(history_xyz: np.ndarray, history_dt_s: float):
    segment_speeds = np.linalg.norm(
        np.diff(history_xyz[:, :2], axis=0),
        axis=1,
    ) / history_dt_s
    mean_speed = float(np.mean(segment_speeds))
    terminal_speed = float(np.median(segment_speeds[-3:]))
    if len(segment_speeds) >= 8:
        early = float(np.median(segment_speeds[:4]))
        late = float(np.median(segment_speeds[-4:]))
        center_duration = max(
            history_dt_s,
            (len(segment_speeds) - 4) * history_dt_s,
        )
        acceleration = (late - early) / center_duration
    else:
        acceleration = 0.0
    return mean_speed, terminal_speed, acceleration


def compute_trajectory_reachability_profile(
    model_points,
    ego_history_xyz,
    *,
    source_speed_mps: float | None = None,
    waypoint_dt_s: float = cfg.TRAJECTORY_WAYPOINT_DT,
    history_dt_s: float = cfg.CONTROL_DT,
    near_horizon_s: float = cfg.SAFETY_EXECUTION_HORIZON_S,
    maximum_acceleration_mps2: float = 2.0,
    comfortable_deceleration_mps2: float = 2.5,
    physical_distance_tolerance_m: float = 0.5,
    source_prior_acceleration_tolerance_mps2: float = 0.75,
) -> TrajectoryReachabilityProfile:
    """Compare a prediction with physical and source-speed distance envelopes."""

    points = np.asarray(model_points, dtype=np.float64)
    history = np.asarray(ego_history_xyz, dtype=np.float64)
    scalars = (
        waypoint_dt_s,
        history_dt_s,
        near_horizon_s,
        maximum_acceleration_mps2,
        comfortable_deceleration_mps2,
        physical_distance_tolerance_m,
        source_prior_acceleration_tolerance_mps2,
    )
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or len(points) < 1
        or history.ndim != 2
        or history.shape[1] < 2
        or len(history) < 2
        or not np.isfinite(points).all()
        or not np.isfinite(history).all()
        or not all(math.isfinite(float(value)) and float(value) > 0.0 for value in scalars)
    ):
        raise ValueError("invalid_reachability_input")

    history_mean, history_terminal, history_acceleration = _history_motion(
        history,
        float(history_dt_s),
    )
    speed = history_terminal if source_speed_mps is None else float(source_speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("invalid_source_speed")

    path = np.vstack([np.zeros((1, 2), dtype=np.float64), points[:, :2]])
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    terminal_displacement = float(np.linalg.norm(points[-1, :2]))
    horizon = float(len(points) * waypoint_dt_s)
    near_horizon = min(float(near_horizon_s), horizon)
    near_segment_count = min(
        len(segment_lengths),
        max(1, int(math.ceil(near_horizon / float(waypoint_dt_s)))),
    )
    near_path_length = float(np.sum(segment_lengths[:near_segment_count]))

    minimum, maximum = _distance_bounds(
        speed,
        horizon,
        maximum_acceleration_mps2=float(maximum_acceleration_mps2),
        comfortable_deceleration_mps2=float(comfortable_deceleration_mps2),
    )
    near_minimum, near_maximum = _distance_bounds(
        speed,
        near_horizon,
        maximum_acceleration_mps2=float(maximum_acceleration_mps2),
        comfortable_deceleration_mps2=float(comfortable_deceleration_mps2),
    )
    required_acceleration, prior_status = _prior_status(
        path_length,
        horizon,
        speed,
        acceleration_tolerance_mps2=float(
            source_prior_acceleration_tolerance_mps2
        ),
    )
    near_required_acceleration, near_prior_status = _prior_status(
        near_path_length,
        near_horizon,
        speed,
        acceleration_tolerance_mps2=float(
            source_prior_acceleration_tolerance_mps2
        ),
    )
    return TrajectoryReachabilityProfile(
        source_speed_mps=speed,
        history_mean_speed_mps=history_mean,
        history_terminal_speed_mps=history_terminal,
        history_acceleration_mps2=history_acceleration,
        horizon_s=horizon,
        path_length_m=path_length,
        terminal_displacement_m=terminal_displacement,
        minimum_reachable_distance_m=minimum,
        maximum_reachable_distance_m=maximum,
        required_constant_acceleration_mps2=required_acceleration,
        physical_status=_physical_status(
            path_length,
            minimum,
            maximum,
            tolerance_m=float(physical_distance_tolerance_m),
        ),
        source_speed_prior_status=prior_status,
        near_horizon_s=near_horizon,
        near_path_length_m=near_path_length,
        near_minimum_reachable_distance_m=near_minimum,
        near_maximum_reachable_distance_m=near_maximum,
        near_required_constant_acceleration_mps2=near_required_acceleration,
        near_physical_status=_physical_status(
            near_path_length,
            near_minimum,
            near_maximum,
            tolerance_m=float(physical_distance_tolerance_m),
        ),
        near_source_speed_prior_status=near_prior_status,
    )


__all__ = [
    "PhysicalReachabilityStatus",
    "SourceSpeedPriorStatus",
    "TrajectoryReachabilityProfile",
    "compute_trajectory_reachability_profile",
]

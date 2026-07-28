"""Validation and fixed-world lifecycle for Alpamayo trajectories.

This module is deliberately independent of CARLA.  A model-space proposal is
validated and anchored to the source observation pose exactly once; subsequent
control ticks only select future points from that immutable world-space path.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from . import config as cfg
from .geometry import meaningful_path_tangent_xy, model_ego_points_to_world
from .safety_shield import densify_path


class TrajectoryValidationError(ValueError):
    """A model trajectory is unsafe or incompatible with the runtime contract."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TrajectoryMotionClass(str, Enum):
    """Intent class derived only from timestamped trajectory geometry."""

    MOVING = "MOVING"
    DELAYED_START = "DELAYED_START"
    CREEP_OR_STALL = "CREEP_OR_STALL"
    EXPLICIT_STOP = "EXPLICIT_STOP"


@dataclass(frozen=True)
class TrajectoryMotionProfile:
    """Elapsed-prefix-aware motion facts shared by selection and telemetry."""

    first_future_index: int
    remaining_horizon_s: float
    initial_target_speed_mps: float
    near_term_path_length_m: float
    near_term_mean_speed_mps: float
    near_term_peak_speed_mps: float
    remaining_path_length_m: float
    remaining_peak_speed_mps: float
    peak_smoothed_deceleration_mps2: float
    time_to_effective_stop_s: float | None
    motion_class: TrajectoryMotionClass

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "first_future_index": int(self.first_future_index),
            "remaining_horizon_s": float(self.remaining_horizon_s),
            "initial_target_speed_mps": float(self.initial_target_speed_mps),
            "near_term_path_length_m": float(self.near_term_path_length_m),
            "near_term_mean_speed_mps": float(self.near_term_mean_speed_mps),
            "near_term_peak_speed_mps": float(self.near_term_peak_speed_mps),
            "remaining_path_length_m": float(self.remaining_path_length_m),
            "remaining_peak_speed_mps": float(self.remaining_peak_speed_mps),
            "peak_smoothed_deceleration_mps2": float(
                self.peak_smoothed_deceleration_mps2
            ),
            "time_to_effective_stop_s": (
                None
                if self.time_to_effective_stop_s is None
                else float(self.time_to_effective_stop_s)
            ),
            "motion_class": self.motion_class.value,
        }


@dataclass(frozen=True)
class FixedWorldTrajectory:
    """One model proposal anchored to its exact source pose and timestamps.

    ``waypoint_times_s`` contains absolute CARLA simulation timestamps.  The
    arrays produced by :func:`build_fixed_world_trajectory` are owned copies and
    marked read-only so moving the ego later cannot move the plan.
    """

    plan_id: str
    source_frame_id: int
    source_simulation_time_s: float
    capture_pose_world: np.ndarray
    model_points: np.ndarray
    world_points: np.ndarray
    waypoint_times_s: np.ndarray
    coc_text: str
    stop_requested: bool
    prompt_revision: int
    respawn_revision: int
    selected_candidate_index: int = 0
    terminal_stop_index: int | None = None
    navigation_context: Any | None = None

    @property
    def horizon_end_s(self) -> float:
        return float(self.waypoint_times_s[-1])

    @property
    def waypoint_offsets_s(self) -> np.ndarray:
        return self.waypoint_times_s - self.source_simulation_time_s


@dataclass(frozen=True)
class PlanExecutionValidity:
    """Validity of one fixed-world plan at a control timestamp."""

    valid: bool
    rejection_reason: str | None
    source_age_s: float
    remaining_horizon_s: float
    first_future_index: int | None


@dataclass(frozen=True)
class PlanAlignmentValidity:
    """Whether the current ego pose still agrees with the plan corridor."""

    valid: bool
    rejection_reason: str | None
    tracking_error_m: float
    heading_error_deg: float


def target_speed_from_timestamps(
    points: Any,
    waypoint_times_s: Any,
    start_idx: int,
    *,
    capture_origin_world: Any | None = None,
    terminal_stop_index: int | None = None,
    window_segments: int = 6,
) -> float:
    """Return the controller's median speed over the next timestamped segments."""

    try:
        path = np.asarray(points, dtype=np.float64)
        times = np.asarray(waypoint_times_s, dtype=np.float64)
        index = int(start_idx)
        window_count = int(window_segments)
    except (TypeError, ValueError):
        return 0.0
    if (
        path.ndim != 2
        or path.shape[1] != 3
        or times.ndim != 1
        or len(path) != len(times)
        or len(path) == 0
        or not np.isfinite(path).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
        or index < 0
        or window_count < 1
    ):
        return 0.0

    speed_index_offset = 0
    if capture_origin_world is not None:
        try:
            origin = np.asarray(capture_origin_world, dtype=np.float64)
        except (TypeError, ValueError):
            origin = np.empty((0,), dtype=np.float64)
        if origin.shape == (3,) and np.isfinite(origin).all():
            first_dt = (
                float(times[1] - times[0])
                if len(times) > 1
                else float(cfg.TRAJECTORY_WAYPOINT_DT)
            )
            path = np.vstack([origin, path])
            times = np.concatenate([[times[0] - first_dt], times])
            speed_index_offset = 1

    if len(path) < 2:
        return 0.0
    segment_distance = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    segment_dt = np.diff(times)
    speeds = np.divide(
        segment_distance,
        segment_dt,
        out=np.zeros_like(segment_distance),
        where=segment_dt > 1e-6,
    )
    segment_idx = min(max(0, index + speed_index_offset), len(speeds) - 1)
    window = speeds[segment_idx : min(len(speeds), segment_idx + window_count)]
    target_speed = float(np.median(window)) if len(window) else 0.0
    if terminal_stop_index is not None and index >= int(terminal_stop_index):
        target_speed = 0.0
    return float(np.clip(target_speed, 0.0, cfg.TRAJECTORY_MAX_SPEED_MPS))


def _clip_timed_path_at(
    points: np.ndarray,
    times: np.ndarray,
    current_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discard elapsed geometry while retaining a partial crossing segment."""

    if current_time_s <= float(times[0]):
        return points.copy(), times.copy()
    if current_time_s >= float(times[-1]):
        return points[-1:].copy(), times[-1:].copy()

    upper = int(np.searchsorted(times, current_time_s, side="right"))
    lower = upper - 1
    dt = float(times[upper] - times[lower])
    fraction = 0.0 if dt <= 0.0 else (current_time_s - float(times[lower])) / dt
    interpolated = points[lower] + float(fraction) * (points[upper] - points[lower])
    clipped_points = np.vstack([interpolated, points[upper:]])
    clipped_times = np.concatenate([[current_time_s], times[upper:]])
    return clipped_points, clipped_times


def compute_trajectory_motion_profile(
    plan: FixedWorldTrajectory,
    current_simulation_time_s: float,
    *,
    execution_horizon_s: float | None = None,
) -> TrajectoryMotionProfile:
    """Compute motion intent after slicing the plan at the execution timestamp."""

    try:
        current_time = float(current_simulation_time_s)
        source_time = float(plan.source_simulation_time_s)
        points = np.asarray(plan.world_points, dtype=np.float64)
        waypoint_times = np.asarray(plan.waypoint_times_s, dtype=np.float64)
        capture_pose = np.asarray(plan.capture_pose_world, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrajectoryValidationError("invalid_plan_geometry") from exc
    if (
        not math.isfinite(current_time)
        or not math.isfinite(source_time)
        or current_time < source_time - float(cfg.TRAJECTORY_TIME_EPSILON_S)
        or points.ndim != 2
        or points.shape[1] != 3
        or waypoint_times.shape != (len(points),)
        or capture_pose.shape != (4, 4)
        or not np.isfinite(points).all()
        or not np.isfinite(waypoint_times).all()
        or not np.isfinite(capture_pose).all()
        or np.any(np.diff(waypoint_times) <= 0.0)
    ):
        raise TrajectoryValidationError("invalid_plan_geometry")

    first_future = int(np.searchsorted(waypoint_times, current_time, side="right"))
    origin = capture_pose[:3, 3]
    full_points = np.vstack([origin, points])
    full_times = np.concatenate([[source_time], waypoint_times])
    clipped_points, clipped_times = _clip_timed_path_at(
        full_points,
        full_times,
        current_time,
    )
    remaining_horizon = max(0.0, float(full_times[-1]) - current_time)

    if len(clipped_points) < 2:
        segment_distance = np.empty((0,), dtype=np.float64)
        segment_dt = np.empty((0,), dtype=np.float64)
        speeds = np.empty((0,), dtype=np.float64)
    else:
        segment_distance = np.linalg.norm(
            np.diff(clipped_points[:, :2], axis=0),
            axis=1,
        )
        segment_dt = np.diff(clipped_times)
        speeds = np.divide(
            segment_distance,
            segment_dt,
            out=np.zeros_like(segment_distance),
            where=segment_dt > 1e-6,
        )

    horizon = float(
        cfg.SAFETY_EXECUTION_HORIZON_S
        if execution_horizon_s is None
        else execution_horizon_s
    )
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise TrajectoryValidationError("invalid_motion_execution_horizon")
    execution_duration = min(horizon, remaining_horizon)
    execution_end = current_time + execution_duration
    near_distance = 0.0
    near_speeds = []
    for index, distance in enumerate(segment_distance):
        start = float(clipped_times[index])
        end = float(clipped_times[index + 1])
        overlap = max(0.0, min(end, execution_end) - max(start, current_time))
        if overlap <= 0.0 or end <= start:
            continue
        near_distance += float(distance) * overlap / (end - start)
        near_speeds.append(float(speeds[index]))
        if end >= execution_end:
            break

    near_mean = (
        near_distance / execution_duration if execution_duration > 1e-9 else 0.0
    )
    near_peak = max(near_speeds, default=0.0)
    remaining_distance = float(np.sum(segment_distance))
    remaining_peak = float(np.max(speeds, initial=0.0))

    smoothed = np.asarray(
        [
            float(np.median(speeds[index : index + 5]))
            for index in range(len(speeds))
        ],
        dtype=np.float64,
    )
    peak_deceleration = 0.0
    if len(smoothed) >= 2:
        smoothed_dt = np.maximum(segment_dt[1:], 1e-6)
        decelerations = -(np.diff(smoothed) / smoothed_dt)
        peak_deceleration = float(np.max(decelerations, initial=0.0))

    time_to_effective_stop = None
    minimum_stop_segments = int(cfg.TRAJECTORY_STOP_TAIL_MIN_POINTS)
    if len(speeds) >= minimum_stop_segments:
        stop_mask = speeds <= float(cfg.PID_STOP_SPEED_THRESHOLD_MPS)
        for start in range(0, len(stop_mask) - minimum_stop_segments + 1):
            if bool(np.all(stop_mask[start : start + minimum_stop_segments])):
                time_to_effective_stop = max(
                    0.0,
                    float(clipped_times[start]) - current_time,
                )
                break

    initial_target = target_speed_from_timestamps(
        points,
        waypoint_times,
        min(first_future, max(0, len(points) - 1)),
        capture_origin_world=origin,
        terminal_stop_index=plan.terminal_stop_index,
    )
    moving_speed = float(cfg.PID_LAUNCH_MIN_INTENT_MPS)
    moving_distance = moving_speed * horizon
    if plan.terminal_stop_index is not None:
        motion_class = TrajectoryMotionClass.EXPLICIT_STOP
    elif near_mean >= moving_speed and near_distance >= moving_distance:
        motion_class = TrajectoryMotionClass.MOVING
    elif remaining_peak >= moving_speed:
        motion_class = TrajectoryMotionClass.DELAYED_START
    else:
        motion_class = TrajectoryMotionClass.CREEP_OR_STALL

    return TrajectoryMotionProfile(
        first_future_index=first_future,
        remaining_horizon_s=remaining_horizon,
        initial_target_speed_mps=initial_target,
        near_term_path_length_m=near_distance,
        near_term_mean_speed_mps=near_mean,
        near_term_peak_speed_mps=near_peak,
        remaining_path_length_m=remaining_distance,
        remaining_peak_speed_mps=remaining_peak,
        peak_smoothed_deceleration_mps2=peak_deceleration,
        time_to_effective_stop_s=time_to_effective_stop,
        motion_class=motion_class,
    )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TrajectoryValidationError(f"invalid_{name}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TrajectoryValidationError(f"invalid_{name}") from exc
    if result < 0:
        raise TrajectoryValidationError(f"invalid_{name}")
    return int(result)


def _readonly_copy(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and array.shape != shape:
        raise TrajectoryValidationError(f"invalid_shape:{array.shape}")
    if not np.isfinite(array).all():
        raise TrajectoryValidationError("non_finite_values")
    array.setflags(write=False)
    return array


def _trajectory_array(points: Any) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError("invalid_trajectory_values") from exc
    expected_shape = (int(cfg.TRAJECTORY_NUM_POINTS), 3)
    if array.shape != expected_shape:
        raise TrajectoryValidationError(
            f"invalid_shape:{array.shape};expected:{expected_shape}"
        )
    if not np.isfinite(array).all():
        raise TrajectoryValidationError("non_finite_points")
    return array.copy()


def detect_terminal_stop_index(
    points: Any,
    *,
    waypoint_dt_s: float | None = None,
) -> int | None:
    """Return the first index of a stable terminal stop tail, if present.

    Stop intent is geometric.  A proposal that remains close to its starting
    point is a stationary plan.  A moving proposal is considered stopping only
    when a sufficiently long terminal tail has near-zero step speed and remains
    clustered around its final point.  CoC text is intentionally not consulted.
    """

    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        return None
    if not np.isfinite(array).all():
        return None

    planar = array[:, :2]
    displacement_from_start = np.linalg.norm(planar - planar[0], axis=1)
    if float(np.max(displacement_from_start, initial=0.0)) <= float(
        cfg.TRAJECTORY_STOP_MAX_DISPLACEMENT_M
    ):
        return 0

    minimum_points = int(cfg.TRAJECTORY_STOP_TAIL_MIN_POINTS)
    if minimum_points < 2 or len(planar) < minimum_points:
        return None

    dt = float(
        cfg.TRAJECTORY_WAYPOINT_DT if waypoint_dt_s is None else waypoint_dt_s
    )
    if not math.isfinite(dt) or dt <= 0.0:
        return None
    maximum_step = min(
        float(cfg.TRAJECTORY_STOP_MAX_STEP_M),
        float(cfg.PID_STOP_SPEED_THRESHOLD_MPS) * dt,
    )
    cluster_radius = float(cfg.TRAJECTORY_STOP_CLUSTER_RADIUS_M)

    for start in range(0, len(planar) - minimum_points + 1):
        tail = planar[start:]
        tail_steps = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        distance_from_endpoint = np.linalg.norm(tail - tail[-1], axis=1)
        if (
            float(np.max(tail_steps, initial=0.0)) <= maximum_step
            and float(np.max(distance_from_endpoint, initial=0.0)) <= cluster_radius
        ):
            return start
    return None


def validate_model_trajectory(points: Any) -> tuple[np.ndarray, int | None]:
    """Validate one strict ``(64, 3)`` model proposal and find stop intent."""

    array = _trajectory_array(points)
    # The capture origin is the implicit t=0 point.  Validate that first jump as
    # well as all model-to-model steps; otherwise a constant path hundreds of
    # metres away could be misclassified as a stationary stop.
    steps = np.concatenate(
        [
            [float(np.linalg.norm(array[0, :2]))],
            np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1),
        ]
    )
    if float(np.max(steps, initial=0.0)) > float(cfg.TRAJECTORY_MAX_STEP_M):
        raise TrajectoryValidationError("excessive_waypoint_step")
    if float(np.max(np.abs(array[:, 1]), initial=0.0)) > float(
        cfg.TRAJECTORY_MAX_LATERAL_M
    ):
        raise TrajectoryValidationError("excessive_lateral_displacement")

    terminal_stop_index = detect_terminal_stop_index(array)
    if terminal_stop_index is None:
        if float(np.max(array[:, 0], initial=0.0)) < float(
            cfg.TRAJECTORY_MIN_FORWARD_PROGRESS_M
        ):
            raise TrajectoryValidationError("insufficient_forward_progress")
        if float(np.min(array[:, 0], initial=0.0)) < -float(
            cfg.TRAJECTORY_MAX_BACKWARD_M
        ):
            raise TrajectoryValidationError("excessive_backward_motion")

    return array, terminal_stop_index


def classify_and_validate_model_trajectory(points: Any) -> tuple[np.ndarray, bool]:
    """Backward-compatible validation API returning a stop-intent boolean."""

    array, terminal_stop_index = validate_model_trajectory(points)
    return array, terminal_stop_index is not None


def build_fixed_world_trajectory(
    *,
    plan_id: str,
    source_frame_id: int,
    source_simulation_time_s: float,
    capture_pose_world: Any,
    model_points: Any,
    coc_text: str,
    prompt_revision: int,
    respawn_revision: int,
    selected_candidate_index: int = 0,
    navigation_context: Any | None = None,
) -> FixedWorldTrajectory:
    """Validate and anchor a model trajectory to its capture pose once."""

    normalized_plan_id = str(plan_id).strip()
    if not normalized_plan_id:
        raise TrajectoryValidationError("invalid_plan_id")
    frame_id = _nonnegative_int(source_frame_id, "source_frame_id")
    prompt_rev = _nonnegative_int(prompt_revision, "prompt_revision")
    respawn_rev = _nonnegative_int(respawn_revision, "respawn_revision")
    candidate_index = _nonnegative_int(selected_candidate_index, "selected_candidate_index")

    points, terminal_stop_index = validate_model_trajectory(model_points)
    try:
        pose = np.array(capture_pose_world, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError("invalid_capture_pose") from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise TrajectoryValidationError("invalid_capture_pose")

    try:
        source_time = float(source_simulation_time_s)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError("invalid_source_simulation_time") from exc
    if not math.isfinite(source_time) or source_time < 0.0:
        raise TrajectoryValidationError("invalid_source_simulation_time")

    waypoint_times = source_time + float(cfg.TRAJECTORY_WAYPOINT_DT) * np.arange(
        1,
        len(points) + 1,
        dtype=np.float64,
    )
    try:
        world_points = model_ego_points_to_world(pose, points)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise TrajectoryValidationError("invalid_capture_pose") from exc

    pose = _readonly_copy(pose, shape=(4, 4))
    points = _readonly_copy(points, shape=(int(cfg.TRAJECTORY_NUM_POINTS), 3))
    world_points = _readonly_copy(
        world_points,
        shape=(int(cfg.TRAJECTORY_NUM_POINTS), 3),
    )
    waypoint_times = _readonly_copy(
        waypoint_times,
        shape=(int(cfg.TRAJECTORY_NUM_POINTS),),
    )

    return FixedWorldTrajectory(
        plan_id=normalized_plan_id,
        source_frame_id=frame_id,
        source_simulation_time_s=source_time,
        capture_pose_world=pose,
        model_points=points,
        world_points=world_points,
        waypoint_times_s=waypoint_times,
        coc_text=str(coc_text or ""),
        stop_requested=terminal_stop_index is not None,
        prompt_revision=prompt_rev,
        respawn_revision=respawn_rev,
        selected_candidate_index=candidate_index,
        terminal_stop_index=terminal_stop_index,
        navigation_context=navigation_context,
    )


def validate_plan_for_execution(
    plan: FixedWorldTrajectory,
    current_simulation_time_s: float,
    *,
    current_prompt_revision: int | None = None,
    current_respawn_revision: int | None = None,
    maximum_plan_age_s: float | None = None,
    minimum_remaining_horizon_s: float | None = None,
) -> PlanExecutionValidity:
    """Reject revision-stale, time-stale, malformed, or exhausted plans."""

    try:
        current_time = float(current_simulation_time_s)
    except (TypeError, ValueError):
        current_time = math.nan
    if not math.isfinite(current_time) or current_time < 0.0:
        return PlanExecutionValidity(
            valid=False,
            rejection_reason="invalid_current_simulation_time",
            source_age_s=0.0,
            remaining_horizon_s=0.0,
            first_future_index=None,
        )

    try:
        source_time = float(plan.source_simulation_time_s)
        times = np.asarray(plan.waypoint_times_s, dtype=np.float64)
        points = np.asarray(plan.world_points, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return PlanExecutionValidity(
            valid=False,
            rejection_reason="invalid_plan_geometry",
            source_age_s=0.0,
            remaining_horizon_s=0.0,
            first_future_index=None,
        )
    expected_count = int(cfg.TRAJECTORY_NUM_POINTS)
    if (
        not math.isfinite(source_time)
        or source_time < 0.0
        or times.shape != (expected_count,)
        or points.shape != (expected_count, 3)
        or not np.isfinite(times).all()
        or not np.isfinite(points).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        return PlanExecutionValidity(
            valid=False,
            rejection_reason="invalid_plan_geometry",
            source_age_s=(
                max(0.0, current_time - source_time)
                if math.isfinite(source_time)
                else 0.0
            ),
            remaining_horizon_s=0.0,
            first_future_index=None,
        )

    raw_source_age = current_time - source_time
    source_age = max(0.0, raw_source_age)
    remaining = max(0.0, float(times[-1]) - current_time)
    first_future = int(np.searchsorted(times, current_time, side="right"))
    epsilon = float(cfg.TRAJECTORY_TIME_EPSILON_S)
    max_age = float(
        cfg.TRAJECTORY_MAX_PLAN_AGE_S
        if maximum_plan_age_s is None
        else maximum_plan_age_s
    )
    min_horizon = float(
        cfg.TRAJECTORY_MIN_REMAINING_HORIZON_S
        if minimum_remaining_horizon_s is None
        else minimum_remaining_horizon_s
    )

    reason = None
    if (
        current_prompt_revision is not None
        and int(current_prompt_revision) != int(plan.prompt_revision)
    ):
        reason = "prompt_revision_mismatch"
    elif (
        current_respawn_revision is not None
        and int(current_respawn_revision) != int(plan.respawn_revision)
    ):
        reason = "respawn_revision_mismatch"
    elif raw_source_age < -epsilon:
        reason = "source_time_in_future"
    elif source_age > max_age + epsilon:
        reason = "plan_source_age_exceeded"
    elif remaining < min_horizon - epsilon:
        reason = "insufficient_remaining_horizon"
    elif first_future >= len(points):
        reason = "trajectory_exhausted"

    return PlanExecutionValidity(
        valid=reason is None,
        rejection_reason=reason,
        source_age_s=source_age,
        remaining_horizon_s=remaining,
        first_future_index=None if reason is not None else first_future,
    )


def validate_plan_alignment(
    plan: FixedWorldTrajectory,
    current_simulation_time_s: float,
    current_pose_world: Any,
    *,
    maximum_tracking_error_m: float | None = None,
    maximum_heading_error_deg: float | None = None,
) -> PlanAlignmentValidity:
    """Measure cross-track and heading error against the fixed-world path.

    The waypoint timestamps determine whether any executable horizon remains,
    but they do not prescribe the ego's exact longitudinal position.  In
    particular, an asynchronously produced plan can arrive while the ego is
    still near its capture pose.  Projecting onto the complete fixed path from
    that capture pose to the remaining horizon treats that expected along-track
    lag as lag, rather than incorrectly reporting it as lateral error.
    """

    try:
        current_time = float(current_simulation_time_s)
        pose = np.asarray(current_pose_world, dtype=np.float64)
    except (TypeError, ValueError):
        current_time = math.nan
        pose = np.empty((0, 0), dtype=np.float64)
    if (
        not math.isfinite(current_time)
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
    ):
        return PlanAlignmentValidity(
            False,
            "invalid_current_pose",
            0.0,
            0.0,
        )

    try:
        capture_pose = np.asarray(plan.capture_pose_world, dtype=np.float64)
        points = np.asarray(plan.world_points, dtype=np.float64)
        times = np.asarray(plan.waypoint_times_s, dtype=np.float64)
        source_time = float(plan.source_simulation_time_s)
    except (AttributeError, TypeError, ValueError):
        return PlanAlignmentValidity(False, "invalid_plan_geometry", 0.0, 0.0)

    expected_count = int(cfg.TRAJECTORY_NUM_POINTS)
    if (
        capture_pose.shape != (4, 4)
        or points.shape != (expected_count, 3)
        or times.shape != (expected_count,)
        or not np.isfinite(capture_pose).all()
        or not np.isfinite(points).all()
        or not np.isfinite(times).all()
        or not math.isfinite(source_time)
        or source_time < 0.0
        or np.any(np.diff(times) <= 0.0)
        or float(times[0]) <= source_time
    ):
        return PlanAlignmentValidity(False, "invalid_plan_geometry", 0.0, 0.0)

    epsilon = float(cfg.TRAJECTORY_TIME_EPSILON_S)
    if current_time < source_time - epsilon:
        return PlanAlignmentValidity(False, "source_time_in_future", 0.0, 0.0)
    first_future = int(np.searchsorted(times, current_time, side="right"))
    if first_future >= len(points):
        return PlanAlignmentValidity(False, "trajectory_exhausted", 0.0, 0.0)

    # The elapsed prefix remains part of the alignment corridor so an ego that
    # waited for inference can project to its actual along-track location.  It
    # is never returned to the controller as a target; ``first_future`` above
    # only establishes that the fixed path still has an executable remainder.
    path_xy = np.vstack([capture_pose[:2, 3], points[:, :2]])
    ego_xy = pose[:2, 3]
    segment_delta = np.diff(path_xy, axis=0)
    segment_length_sq = np.einsum("ij,ij->i", segment_delta, segment_delta)
    moving_segments = segment_length_sq > 1e-12

    tangent = None
    if np.any(moving_segments):
        starts = path_xy[:-1][moving_segments]
        deltas = segment_delta[moving_segments]
        lengths_sq = segment_length_sq[moving_segments]
        fractions = np.clip(
            np.einsum("ij,ij->i", ego_xy - starts, deltas) / lengths_sq,
            0.0,
            1.0,
        )
        projections = starts + fractions[:, None] * deltas
        errors = np.linalg.norm(projections - ego_xy, axis=1)
        closest = int(np.argmin(errors))
        tracking_error = float(errors[closest])
        segment_indices = np.flatnonzero(moving_segments)
        closest_segment_index = int(segment_indices[closest])
        closest_projection = projections[closest]
        heading_path = np.insert(
            path_xy,
            closest_segment_index + 1,
            closest_projection,
            axis=0,
        )
        tangent = meaningful_path_tangent_xy(
            heading_path,
            closest_segment_index + 1,
            lookahead_m=float(cfg.TRAJECTORY_HEADING_LOOKAHEAD_M),
            minimum_displacement_m=float(
                cfg.TRAJECTORY_HEADING_MIN_DISPLACEMENT_M
            ),
        )
    else:
        # A geometrically stationary stop has no meaningful path heading.
        tracking_error = float(np.linalg.norm(ego_xy - path_xy[0]))

    if tangent is None:
        heading_error = 0.0
    else:
        ego_forward = pose[:2, 0]
        ego_forward_norm = float(np.linalg.norm(ego_forward))
        if ego_forward_norm <= 1e-6:
            return PlanAlignmentValidity(
                False,
                "invalid_current_heading",
                tracking_error,
                0.0,
            )
        ego_forward = ego_forward / ego_forward_norm
        heading_error = math.degrees(
            math.acos(float(np.clip(np.dot(ego_forward, tangent), -1.0, 1.0)))
        )

    max_tracking = float(
        cfg.TRAJECTORY_MAX_TRACKING_ERROR_M
        if maximum_tracking_error_m is None
        else maximum_tracking_error_m
    )
    max_heading = float(
        cfg.TRAJECTORY_MAX_HEADING_ERROR_DEG
        if maximum_heading_error_deg is None
        else maximum_heading_error_deg
    )
    reason = None
    if tracking_error > max_tracking:
        reason = "plan_tracking_error_exceeded"
    elif heading_error > max_heading:
        reason = "plan_heading_error_exceeded"
    return PlanAlignmentValidity(
        reason is None,
        reason,
        tracking_error,
        heading_error,
    )


def validate_world_trajectory_drivable(
    plan: FixedWorldTrajectory,
    is_drivable: Callable[[np.ndarray], bool],
    *,
    stride: int | None = None,
) -> None:
    """Reject a plan when a <=0.5 m interpolated sample is not drivable.

    ``stride`` remains accepted for source compatibility but no longer weakens
    the spatial safety guarantee.
    """

    del stride
    dense_points = densify_path(
        plan.world_points,
        max_spacing_m=float(cfg.SAFETY_PATH_SAMPLE_SPACING_M),
    )
    for index, point in enumerate(dense_points):
        if not bool(is_drivable(np.asarray(point, dtype=np.float64))):
            raise TrajectoryValidationError(f"off_driving_lane:sample={index}")


__all__ = [
    "FixedWorldTrajectory",
    "PlanExecutionValidity",
    "PlanAlignmentValidity",
    "TrajectoryMotionClass",
    "TrajectoryMotionProfile",
    "TrajectoryValidationError",
    "build_fixed_world_trajectory",
    "classify_and_validate_model_trajectory",
    "compute_trajectory_motion_profile",
    "detect_terminal_stop_index",
    "target_speed_from_timestamps",
    "validate_model_trajectory",
    "validate_plan_for_execution",
    "validate_plan_alignment",
    "validate_world_trajectory_drivable",
]

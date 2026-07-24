"""Immutable metadata contract for one retained standby trajectory.

The standby slot deliberately stores no road assessment.  Road envelopes are
tick-relative authority and must be recomputed before a retained trajectory can
be activated.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .trajectory_runtime import FixedWorldTrajectory


_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


class StandbyLifecycleStatus(str, Enum):
    """Telemetry states for the single retained-standby slot."""

    STORED = "STORED"
    REPLACED = "REPLACED"
    ACTIVATED = "ACTIVATED"
    DISCARDED = "DISCARDED"
    CLEARED = "CLEARED"


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(result)


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _status_text(value: Any, name: str) -> str:
    result = str(getattr(value, "value", value)).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _immutable_float_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    require_finite: bool = True,
) -> np.ndarray:
    try:
        source = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if shape is not None and source.shape != shape:
        raise ValueError(f"{name} has shape {source.shape}, expected {shape}")
    if require_finite and not np.isfinite(source).all():
        raise ValueError(f"{name} must contain only finite values")

    # A bytes-backed ndarray cannot be made writable again with setflags().
    # This is stronger than merely marking an owned ndarray read-only.
    contiguous = np.ascontiguousarray(source, dtype=np.float64)
    result = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(contiguous.shape)
    result.setflags(write=False)
    return result


def _snapshot_plan(plan: FixedWorldTrajectory) -> FixedWorldTrajectory:
    if not isinstance(plan, FixedWorldTrajectory):
        raise TypeError("plan must be a FixedWorldTrajectory")

    plan_id = str(plan.plan_id).strip()
    if not plan_id:
        raise ValueError("plan.plan_id must be nonempty")
    source_frame_id = _nonnegative_int(
        plan.source_frame_id,
        "plan.source_frame_id",
    )
    source_time = _nonnegative_float(
        plan.source_simulation_time_s,
        "plan.source_simulation_time_s",
    )
    prompt_revision = _nonnegative_int(
        plan.prompt_revision,
        "plan.prompt_revision",
    )
    respawn_revision = _nonnegative_int(
        plan.respawn_revision,
        "plan.respawn_revision",
    )
    selected_index = _nonnegative_int(
        plan.selected_candidate_index,
        "plan.selected_candidate_index",
    )
    if not isinstance(plan.stop_requested, (bool, np.bool_)):
        raise ValueError("plan.stop_requested must be boolean")

    capture_pose = _immutable_float_array(
        plan.capture_pose_world,
        "plan.capture_pose_world",
        shape=(4, 4),
    )
    model_points = _immutable_float_array(
        plan.model_points,
        "plan.model_points",
    )
    world_points = _immutable_float_array(
        plan.world_points,
        "plan.world_points",
    )
    waypoint_times = _immutable_float_array(
        plan.waypoint_times_s,
        "plan.waypoint_times_s",
    )
    if (
        model_points.ndim != 2
        or model_points.shape[1:] != (3,)
        or len(model_points) == 0
        or world_points.shape != model_points.shape
        or waypoint_times.shape != (len(model_points),)
    ):
        raise ValueError("plan trajectory arrays have incompatible shapes")
    if np.any(np.diff(waypoint_times) <= 0.0):
        raise ValueError("plan.waypoint_times_s must be strictly increasing")
    if float(waypoint_times[0]) <= source_time:
        raise ValueError("plan waypoint times must follow the source time")

    terminal_stop_index = plan.terminal_stop_index
    if terminal_stop_index is not None:
        terminal_stop_index = _nonnegative_int(
            terminal_stop_index,
            "plan.terminal_stop_index",
        )
        if terminal_stop_index >= len(model_points):
            raise ValueError("plan.terminal_stop_index is out of range")

    return FixedWorldTrajectory(
        plan_id=plan_id,
        source_frame_id=source_frame_id,
        source_simulation_time_s=source_time,
        capture_pose_world=capture_pose,
        model_points=model_points,
        world_points=world_points,
        waypoint_times_s=waypoint_times,
        coc_text=str(plan.coc_text or ""),
        stop_requested=bool(plan.stop_requested),
        prompt_revision=prompt_revision,
        respawn_revision=respawn_revision,
        selected_candidate_index=selected_index,
        terminal_stop_index=terminal_stop_index,
    )


@dataclass(frozen=True)
class RetainedStandbyPlan:
    """One accepted-but-retained candidate, frozen at inference arrival.

    The object owns immutable copies of all trajectory arrays.  It intentionally
    has no ``RoadExecutionEnvelope`` field: callers must reassess the plan using
    current ego/map facts immediately before activation.
    """

    plan: FixedWorldTrajectory
    trajectory_samples: np.ndarray
    selected_candidate_index: int
    coc_sha256: str
    original_admission_status: str
    original_handoff_status: str
    inference_time_s: float
    trajectory_timestamp_s: float
    source_loop_tick_id: int
    source_elapsed_proxy_s: float
    retained_loop_tick_id: int
    retained_simulation_time_s: float
    verified_empty_road: bool

    def __post_init__(self) -> None:
        plan = _snapshot_plan(self.plan)
        samples = _immutable_float_array(
            self.trajectory_samples,
            "trajectory_samples",
            require_finite=False,
        )
        if (
            samples.ndim != 3
            or samples.shape[0] == 0
            or samples.shape[1:] != plan.model_points.shape
        ):
            raise ValueError(
                "trajectory_samples must have shape "
                f"(K, {len(plan.model_points)}, 3)"
            )

        selected_index = _nonnegative_int(
            self.selected_candidate_index,
            "selected_candidate_index",
        )
        if selected_index >= len(samples):
            raise ValueError("selected_candidate_index is out of range")
        if selected_index != plan.selected_candidate_index:
            raise ValueError(
                "selected_candidate_index must match "
                "plan.selected_candidate_index"
            )
        if (
            not np.isfinite(samples[selected_index]).all()
            or not np.array_equal(samples[selected_index], plan.model_points)
        ):
            raise ValueError(
                "selected trajectory sample must match plan.model_points"
            )

        coc_sha256 = str(self.coc_sha256).strip()
        if _SHA256_RE.fullmatch(coc_sha256) is None:
            raise ValueError("coc_sha256 must be a 64-character hexadecimal digest")

        source_loop_tick = _nonnegative_int(
            self.source_loop_tick_id,
            "source_loop_tick_id",
        )
        retained_loop_tick = _nonnegative_int(
            self.retained_loop_tick_id,
            "retained_loop_tick_id",
        )
        if retained_loop_tick < source_loop_tick:
            raise ValueError(
                "retained_loop_tick_id cannot precede source_loop_tick_id"
            )

        retained_simulation_time = _nonnegative_float(
            self.retained_simulation_time_s,
            "retained_simulation_time_s",
        )
        if retained_simulation_time < plan.source_simulation_time_s:
            raise ValueError(
                "retained_simulation_time_s cannot precede plan source time"
            )
        if not isinstance(self.verified_empty_road, (bool, np.bool_)):
            raise ValueError("verified_empty_road must be boolean")

        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "trajectory_samples", samples)
        object.__setattr__(self, "selected_candidate_index", selected_index)
        object.__setattr__(self, "coc_sha256", coc_sha256.lower())
        object.__setattr__(
            self,
            "original_admission_status",
            _status_text(
                self.original_admission_status,
                "original_admission_status",
            ),
        )
        object.__setattr__(
            self,
            "original_handoff_status",
            _status_text(
                self.original_handoff_status,
                "original_handoff_status",
            ),
        )
        object.__setattr__(
            self,
            "inference_time_s",
            _nonnegative_float(self.inference_time_s, "inference_time_s"),
        )
        object.__setattr__(
            self,
            "trajectory_timestamp_s",
            _nonnegative_float(
                self.trajectory_timestamp_s,
                "trajectory_timestamp_s",
            ),
        )
        object.__setattr__(self, "source_loop_tick_id", source_loop_tick)
        object.__setattr__(
            self,
            "source_elapsed_proxy_s",
            _nonnegative_float(
                self.source_elapsed_proxy_s,
                "source_elapsed_proxy_s",
            ),
        )
        object.__setattr__(self, "retained_loop_tick_id", retained_loop_tick)
        object.__setattr__(
            self,
            "retained_simulation_time_s",
            retained_simulation_time,
        )
        object.__setattr__(
            self,
            "verified_empty_road",
            bool(self.verified_empty_road),
        )

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    @property
    def source_order_key(self) -> tuple[int, float, int, float]:
        """Stable ordering for asynchronous results from one runtime epoch."""

        return (
            self.source_loop_tick_id,
            self.source_elapsed_proxy_s,
            self.plan.source_frame_id,
            self.plan.source_simulation_time_s,
        )

    def is_newer_than(self, other: RetainedStandbyPlan | None) -> bool:
        """Return whether this standby came from a strictly newer observation."""

        if other is None:
            return True
        if not isinstance(other, RetainedStandbyPlan):
            raise TypeError("other must be a RetainedStandbyPlan or None")
        return self.source_order_key > other.source_order_key

    def source_age_s(self, current_sim_time: float) -> float:
        current = _nonnegative_float(current_sim_time, "current_sim_time")
        return max(0.0, current - self.plan.source_simulation_time_s)

    def remaining_horizon_s(self, current_sim_time: float) -> float:
        current = _nonnegative_float(current_sim_time, "current_sim_time")
        return max(0.0, self.plan.horizon_end_s - current)

    def to_json_dict(
        self,
        current_simulation_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Return small additive telemetry metadata without trajectory payloads."""

        source_age = None
        remaining_horizon = None
        if current_simulation_time_s is not None:
            source_age = self.source_age_s(current_simulation_time_s)
            remaining_horizon = self.remaining_horizon_s(
                current_simulation_time_s
            )
        return {
            "plan_id": self.plan_id,
            "source_frame_id": int(self.plan.source_frame_id),
            "source_simulation_time_s": float(
                self.plan.source_simulation_time_s
            ),
            "source_loop_tick_id": int(self.source_loop_tick_id),
            "source_elapsed_proxy_s": float(self.source_elapsed_proxy_s),
            "source_order_key": list(self.source_order_key),
            "retained_loop_tick_id": int(self.retained_loop_tick_id),
            "retained_simulation_time_s": float(
                self.retained_simulation_time_s
            ),
            "selected_candidate_index": int(self.selected_candidate_index),
            "candidate_count": int(len(self.trajectory_samples)),
            "coc_sha256": self.coc_sha256,
            "original_admission_status": self.original_admission_status,
            "original_handoff_status": self.original_handoff_status,
            "inference_time_s": float(self.inference_time_s),
            "trajectory_timestamp_s": float(self.trajectory_timestamp_s),
            "verified_empty_road": bool(self.verified_empty_road),
            "prompt_revision": int(self.plan.prompt_revision),
            "respawn_revision": int(self.plan.respawn_revision),
            "stop_requested": bool(self.plan.stop_requested),
            "horizon_end_s": float(self.plan.horizon_end_s),
            "source_age_s": source_age,
            "remaining_horizon_s": remaining_horizon,
        }


__all__ = [
    "RetainedStandbyPlan",
    "StandbyLifecycleStatus",
]

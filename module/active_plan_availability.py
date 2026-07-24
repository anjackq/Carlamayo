"""Bounded availability policy for an already-admitted active trajectory.

Candidate validation remains deliberately stricter.  This module only decides
whether a fixed-world active plan may bridge the small gap between the normal
freshness window and the exact road-safety execution horizon.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .safety_shield import AssessmentStatus


_BRIDGEABLE_REJECTIONS = frozenset(
    {
        "plan_source_age_exceeded",
        "insufficient_remaining_horizon",
    }
)


class ActivePlanAvailabilityStatus(str, Enum):
    """Execution availability of the current fixed-world plan."""

    STANDARD_EXECUTION = "STANDARD_EXECUTION"
    BRIDGED_FULL_SAFE = "BRIDGED_FULL_SAFE"
    BRIDGE_DENIED = "BRIDGE_DENIED"
    NO_ACTIVE_PLAN = "NO_ACTIVE_PLAN"


@dataclass(frozen=True)
class ActivePlanAvailabilityDecision:
    """Serializable, side-effect-free active-plan availability outcome."""

    status: ActivePlanAvailabilityStatus
    execution_allowed: bool
    bridge_attempted: bool
    original_rejection_reason: str | None
    denial_reason: str | None
    source_age_s: float | None
    remaining_horizon_s: float | None
    bridge_deadline_age_s: float
    bridge_min_remaining_horizon_s: float

    @property
    def bridge_active(self) -> bool:
        return self.status is ActivePlanAvailabilityStatus.BRIDGED_FULL_SAFE

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "execution_allowed": bool(self.execution_allowed),
            "bridge_attempted": bool(self.bridge_attempted),
            "bridge_active": bool(self.bridge_active),
            "original_rejection_reason": self.original_rejection_reason,
            "denial_reason": self.denial_reason,
            "source_age_s": self.source_age_s,
            "remaining_horizon_s": self.remaining_horizon_s,
            "bridge_deadline_age_s": float(self.bridge_deadline_age_s),
            "bridge_min_remaining_horizon_s": float(
                self.bridge_min_remaining_horizon_s
            ),
        }


def _status(value: Any) -> AssessmentStatus | None:
    try:
        return AssessmentStatus(getattr(value, "status"))
    except (AttributeError, TypeError, ValueError):
        return None


def _reserve_status(envelope: Any) -> str | None:
    try:
        value = envelope.stopping_reserve_profile.status
    except AttributeError:
        return None
    return str(getattr(value, "value", value))


def _finite_nonnegative(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _nonnegative_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(operator.index(value))
    except TypeError:
        return None
    return result if result >= 0 else None


def decide_active_plan_availability(
    *,
    active_plan_present: bool,
    standard_validity: Any | None,
    bridge_validity: Any | None,
    alignment_validity: Any | None,
    road_envelope: Any | None,
    bridge_deadline_age_s: float,
    bridge_min_remaining_horizon_s: float,
) -> ActivePlanAvailabilityDecision:
    """Allow a bounded active-only bridge only for an exact fully-safe path."""

    raw_deadline = _finite_nonnegative(bridge_deadline_age_s)
    raw_minimum_horizon = _finite_nonnegative(bridge_min_remaining_horizon_s)
    policy_bounds_valid = (
        raw_deadline is not None and raw_minimum_horizon is not None
    )
    deadline = raw_deadline if raw_deadline is not None else 0.0
    minimum_horizon = (
        raw_minimum_horizon if raw_minimum_horizon is not None else 0.0
    )
    if not active_plan_present:
        return ActivePlanAvailabilityDecision(
            status=ActivePlanAvailabilityStatus.NO_ACTIVE_PLAN,
            execution_allowed=False,
            bridge_attempted=False,
            original_rejection_reason=None,
            denial_reason="no_active_plan",
            source_age_s=None,
            remaining_horizon_s=None,
            bridge_deadline_age_s=deadline,
            bridge_min_remaining_horizon_s=minimum_horizon,
        )

    original_reason = getattr(standard_validity, "rejection_reason", None)
    source_age = _finite_nonnegative(
        getattr(standard_validity, "source_age_s", None)
    )
    remaining = _finite_nonnegative(
        getattr(standard_validity, "remaining_horizon_s", None)
    )
    alignment_valid = bool(getattr(alignment_validity, "valid", False))
    if bool(getattr(standard_validity, "valid", False)) and alignment_valid:
        return ActivePlanAvailabilityDecision(
            status=ActivePlanAvailabilityStatus.STANDARD_EXECUTION,
            execution_allowed=True,
            bridge_attempted=False,
            original_rejection_reason=None,
            denial_reason=None,
            source_age_s=source_age,
            remaining_horizon_s=remaining,
            bridge_deadline_age_s=deadline,
            bridge_min_remaining_horizon_s=minimum_horizon,
        )

    common = {
        "status": ActivePlanAvailabilityStatus.BRIDGE_DENIED,
        "execution_allowed": False,
        "original_rejection_reason": original_reason,
        "source_age_s": source_age,
        "remaining_horizon_s": remaining,
        "bridge_deadline_age_s": deadline,
        "bridge_min_remaining_horizon_s": minimum_horizon,
    }
    if bool(getattr(standard_validity, "valid", False)):
        return ActivePlanAvailabilityDecision(
            bridge_attempted=False,
            denial_reason=(
                getattr(alignment_validity, "rejection_reason", None)
                or "active_plan_alignment_invalid"
            ),
            **common,
        )
    if original_reason not in _BRIDGEABLE_REJECTIONS:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=False,
            denial_reason=original_reason or "active_plan_invalid",
            **common,
        )
    if not policy_bounds_valid:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="invalid_bridge_policy_bounds",
            **common,
        )
    if not bool(getattr(bridge_validity, "valid", False)):
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason=(
                getattr(bridge_validity, "rejection_reason", None)
                or "bridge_validity_failed"
            ),
            **common,
        )
    bridge_age = _finite_nonnegative(
        getattr(bridge_validity, "source_age_s", None)
    )
    bridge_remaining = _finite_nonnegative(
        getattr(bridge_validity, "remaining_horizon_s", None)
    )
    first_future_index = _nonnegative_index(
        getattr(bridge_validity, "first_future_index", None)
    )
    if bridge_age is None or bridge_remaining is None or first_future_index is None:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="invalid_bridge_timing",
            **common,
        )
    epsilon_s = 1e-6
    if bridge_age > deadline + epsilon_s:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_deadline_exceeded",
            **common,
        )
    if bridge_remaining < minimum_horizon - epsilon_s:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_horizon_below_minimum",
            **common,
        )
    if not alignment_valid:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason=(
                getattr(alignment_validity, "rejection_reason", None)
                or "active_plan_alignment_invalid"
            ),
            **common,
        )
    if road_envelope is None:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_road_assessment_unavailable",
            **common,
        )

    clearance = getattr(
        road_envelope,
        "current_ego_clearance_road",
        getattr(road_envelope, "current_ego_road", None),
    )
    road_requirements = (
        ("bridge_current_ego_not_safe", getattr(road_envelope, "current_ego_road", None)),
        ("bridge_current_clearance_not_safe", clearance),
        ("bridge_near_term_not_safe", getattr(road_envelope, "near_term_path_road", None)),
        ("bridge_full_path_not_safe", getattr(road_envelope, "full_path_road", None)),
    )
    for denial_reason, assessment in road_requirements:
        if _status(assessment) is not AssessmentStatus.SAFE:
            return ActivePlanAvailabilityDecision(
                bridge_attempted=True,
                denial_reason=denial_reason,
                **common,
            )
    if bool(getattr(road_envelope, "recovery_required", False)):
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_recovery_not_authorized",
            **common,
        )
    if bool(getattr(road_envelope, "emergency_required", True)):
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_emergency_required",
            **common,
        )
    last_safe_index = _nonnegative_index(
        getattr(road_envelope, "last_safe_waypoint_index", None)
    )
    if last_safe_index is None:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_no_authorized_waypoint",
            **common,
        )
    if last_safe_index < first_future_index:
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_authorized_prefix_exhausted",
            **common,
        )
    if _reserve_status(road_envelope) != "UNBOUNDED":
        return ActivePlanAvailabilityDecision(
            bridge_attempted=True,
            denial_reason="bridge_reserve_not_unbounded",
            **common,
        )
    return ActivePlanAvailabilityDecision(
        status=ActivePlanAvailabilityStatus.BRIDGED_FULL_SAFE,
        execution_allowed=True,
        bridge_attempted=True,
        original_rejection_reason=original_reason,
        denial_reason=None,
        source_age_s=source_age,
        remaining_horizon_s=remaining,
        bridge_deadline_age_s=deadline,
        bridge_min_remaining_horizon_s=minimum_horizon,
    )


__all__ = [
    "ActivePlanAvailabilityDecision",
    "ActivePlanAvailabilityStatus",
    "decide_active_plan_availability",
]

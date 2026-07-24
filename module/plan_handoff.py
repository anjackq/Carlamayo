"""Pure bounded handoff policy between an active and a candidate trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


ACCEPTED_ADMISSION_STATUSES = frozenset(
    {
        "ACCEPT_FULLY_SAFE",
        "ACCEPT_SAFE_PREFIX",
        "ACCEPT_RECOVERY_PREFIX",
    }
)
_RESERVE_TIER = {
    "UNBOUNDED": 0,
    "ROBUST": 0,
    "RECOVERY": 1,
    "FRAGILE": 2,
    "UNAVAILABLE": 3,
}
_ADMISSION_TIER = {
    "ACCEPT_FULLY_SAFE": 0,
    "ACCEPT_SAFE_PREFIX": 1,
    "ACCEPT_RECOVERY_PREFIX": 2,
}
_MOTION_TIER = {
    "MOVING": 0,
    "DELAYED_START": 1,
    "CREEP_OR_STALL": 2,
    "EXPLICIT_STOP": 3,
}


class PlanHandoffStatus(str, Enum):
    ACTIVATE_NO_ACTIVE = "ACTIVATE_NO_ACTIVE"
    ACTIVATE_SAFETY_IMPROVEMENT = "ACTIVATE_SAFETY_IMPROVEMENT"
    ACTIVATE_FRESH = "ACTIVATE_FRESH"
    ACTIVATE_RETENTION_DEADLINE = "ACTIVATE_RETENTION_DEADLINE"
    RETAIN_ACTIVE_STOPPING_RESERVE = "RETAIN_ACTIVE_STOPPING_RESERVE"
    RETAIN_ACTIVE_MOTION_QUALITY = "RETAIN_ACTIVE_MOTION_QUALITY"
    NO_EXECUTABLE_PLAN = "NO_EXECUTABLE_PLAN"


@dataclass(frozen=True)
class PlanHandoffDecision:
    """Serializable outcome without mutating either trajectory."""

    status: PlanHandoffStatus
    activate_candidate: bool
    candidate_plan_id: str | None
    active_plan_id: str | None
    candidate_admission_status: str | None
    active_admission_status: str | None
    candidate_motion_class: str | None
    active_motion_class: str | None
    candidate_reserve_status: str | None
    active_reserve_status: str | None
    active_remaining_horizon_s: float | None
    reason: str

    @property
    def retain_active(self) -> bool:
        return not self.activate_candidate and self.active_plan_id is not None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "activate_candidate": bool(self.activate_candidate),
            "retain_active": bool(self.retain_active),
            "candidate_plan_id": self.candidate_plan_id,
            "active_plan_id": self.active_plan_id,
            "candidate_admission_status": self.candidate_admission_status,
            "active_admission_status": self.active_admission_status,
            "candidate_motion_class": self.candidate_motion_class,
            "active_motion_class": self.active_motion_class,
            "candidate_reserve_status": self.candidate_reserve_status,
            "active_reserve_status": self.active_reserve_status,
            "active_remaining_horizon_s": self.active_remaining_horizon_s,
            "reason": self.reason,
        }


def _value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    result = str(raw).strip()
    return result or None


def _finite_horizon(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _safety_tier(admission: str | None, reserve: str | None) -> tuple[int, int]:
    return (
        _RESERVE_TIER.get(reserve or "UNAVAILABLE", 3),
        _ADMISSION_TIER.get(admission or "", 99),
    )


def decide_plan_handoff(
    *,
    candidate_plan_id: str | None,
    candidate_admission_status: Any,
    candidate_motion_class: Any,
    candidate_reserve_status: Any,
    candidate_explicit_stop: bool,
    active_plan_id: str | None,
    active_admission_status: Any,
    active_motion_class: Any,
    active_reserve_status: Any,
    active_remaining_horizon_s: float | None,
    active_executable: bool,
    verified_empty_road: bool,
    retention_deadline_s: float = 3.0,
) -> PlanHandoffDecision:
    """Decide replacement after both plans have independently passed admission."""

    candidate_admission = _value(candidate_admission_status)
    active_admission = _value(active_admission_status)
    candidate_motion = _value(candidate_motion_class)
    active_motion = _value(active_motion_class)
    candidate_reserve = _value(candidate_reserve_status)
    active_reserve = _value(active_reserve_status)
    active_horizon = _finite_horizon(active_remaining_horizon_s)

    common = {
        "candidate_plan_id": (
            str(candidate_plan_id) if candidate_plan_id is not None else None
        ),
        "active_plan_id": str(active_plan_id) if active_plan_id is not None else None,
        "candidate_admission_status": candidate_admission,
        "active_admission_status": active_admission,
        "candidate_motion_class": candidate_motion,
        "active_motion_class": active_motion,
        "candidate_reserve_status": candidate_reserve,
        "active_reserve_status": active_reserve,
        "active_remaining_horizon_s": active_horizon,
    }
    candidate_admitted = candidate_admission in ACCEPTED_ADMISSION_STATUSES
    if not candidate_admitted:
        return PlanHandoffDecision(
            status=PlanHandoffStatus.NO_EXECUTABLE_PLAN,
            activate_candidate=False,
            reason="candidate_not_admitted",
            **common,
        )
    if not active_executable or active_plan_id is None:
        return PlanHandoffDecision(
            status=PlanHandoffStatus.ACTIVATE_NO_ACTIVE,
            activate_candidate=True,
            reason="no_executable_active_plan",
            **common,
        )

    candidate_safety = _safety_tier(candidate_admission, candidate_reserve)
    active_safety = _safety_tier(active_admission, active_reserve)
    if candidate_safety < active_safety:
        return PlanHandoffDecision(
            status=PlanHandoffStatus.ACTIVATE_SAFETY_IMPROVEMENT,
            activate_candidate=True,
            reason="candidate_safety_tier_improved",
            **common,
        )
    if candidate_explicit_stop or candidate_motion == "EXPLICIT_STOP":
        return PlanHandoffDecision(
            status=PlanHandoffStatus.ACTIVATE_FRESH,
            activate_candidate=True,
            reason="explicit_stop_bypasses_motion_retention",
            **common,
        )
    if active_horizon is None or active_horizon <= float(retention_deadline_s):
        return PlanHandoffDecision(
            status=PlanHandoffStatus.ACTIVATE_RETENTION_DEADLINE,
            activate_candidate=True,
            reason="active_plan_retention_deadline_reached",
            **common,
        )
    if (
        candidate_reserve == "FRAGILE"
        and active_reserve in {"ROBUST", "UNBOUNDED"}
    ):
        return PlanHandoffDecision(
            status=PlanHandoffStatus.RETAIN_ACTIVE_STOPPING_RESERVE,
            activate_candidate=False,
            reason="active_plan_has_guarded_stopping_reserve",
            **common,
        )
    if (
        verified_empty_road
        and candidate_motion in _MOTION_TIER
        and active_motion in _MOTION_TIER
        and _MOTION_TIER[candidate_motion] > _MOTION_TIER[active_motion]
    ):
        return PlanHandoffDecision(
            status=PlanHandoffStatus.RETAIN_ACTIVE_MOTION_QUALITY,
            activate_candidate=False,
            reason="candidate_motion_quality_worse_than_active",
            **common,
        )
    return PlanHandoffDecision(
        status=PlanHandoffStatus.ACTIVATE_FRESH,
        activate_candidate=True,
        reason="fresh_admitted_candidate",
        **common,
    )


__all__ = [
    "ACCEPTED_ADMISSION_STATUSES",
    "PlanHandoffDecision",
    "PlanHandoffStatus",
    "decide_plan_handoff",
]

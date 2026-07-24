"""Deterministic ranking for already-evaluated Alpamayo trajectory candidates.

This module does not query CARLA and does not decide whether a path is safe.
The closed-loop runtime first performs generic validation, fixed-world
alignment, and road admission for every candidate.  This module only ranks the
resulting facts so continuity cannot select an unevaluated trajectory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


ACCEPTED_ADMISSION_STATUSES = frozenset(
    {
        "ACCEPT_FULLY_SAFE",
        "ACCEPT_SAFE_PREFIX",
        "ACCEPT_RECOVERY_PREFIX",
    }
)
_ADMISSION_QUALITY = {
    "ACCEPT_FULLY_SAFE": 0,
    "ACCEPT_SAFE_PREFIX": 1,
    "ACCEPT_RECOVERY_PREFIX": 2,
}
_RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)
_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)
ROUTE_LATERAL_DEADBAND_M = 0.75


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite or None") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or None")
    return result


@dataclass(frozen=True)
class CandidateEvaluation:
    """The validation and road facts used to rank one model sample."""

    candidate_index: int
    plan_id: str
    admission_status: str | None
    rejection_reason: str | None
    stop_requested: bool
    forward_progress_m: float
    representative_lateral_m: float
    full_path_margin_m: float | None
    continuity_m: float | None

    def __post_init__(self) -> None:
        if isinstance(self.candidate_index, bool) or int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be a non-negative integer")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        plan_id = str(self.plan_id).strip()
        if not plan_id:
            raise ValueError("plan_id must be non-empty")
        object.__setattr__(self, "plan_id", plan_id)
        admission = (
            None
            if self.admission_status is None
            else str(self.admission_status).strip() or None
        )
        object.__setattr__(self, "admission_status", admission)
        object.__setattr__(
            self,
            "rejection_reason",
            None
            if self.rejection_reason is None
            else str(self.rejection_reason).strip() or None,
        )
        object.__setattr__(self, "stop_requested", bool(self.stop_requested))
        progress = _finite_or_none(self.forward_progress_m, "forward_progress_m")
        lateral = _finite_or_none(
            self.representative_lateral_m,
            "representative_lateral_m",
        )
        if progress is None or lateral is None:
            raise ValueError("candidate progress and lateral displacement are required")
        object.__setattr__(self, "forward_progress_m", progress)
        object.__setattr__(self, "representative_lateral_m", lateral)
        object.__setattr__(
            self,
            "full_path_margin_m",
            _finite_or_none(self.full_path_margin_m, "full_path_margin_m"),
        )
        object.__setattr__(
            self,
            "continuity_m",
            _finite_or_none(self.continuity_m, "continuity_m"),
        )

    @property
    def admitted(self) -> bool:
        return self.admission_status in ACCEPTED_ADMISSION_STATUSES


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate plus the transparent lexicographic ranking terms."""

    evaluation: CandidateEvaluation
    category_rank: int
    stop_penalty: int
    admission_quality_rank: int
    route_mismatch_rank: float
    negative_full_path_margin: float
    continuity_rank_m: float
    negative_forward_progress_m: float

    @property
    def ranking_key(self) -> tuple[float, ...]:
        return (
            float(self.category_rank),
            float(self.stop_penalty),
            float(self.admission_quality_rank),
            float(self.route_mismatch_rank),
            float(self.negative_full_path_margin),
            float(self.continuity_rank_m),
            float(self.negative_forward_progress_m),
            float(self.evaluation.candidate_index),
        )

    def to_json_dict(self) -> dict[str, Any]:
        evaluation = self.evaluation
        return {
            "candidate_index": evaluation.candidate_index,
            "plan_id": evaluation.plan_id,
            "admission_status": evaluation.admission_status,
            "admitted": evaluation.admitted,
            "rejection_reason": evaluation.rejection_reason,
            "stop_requested": evaluation.stop_requested,
            "forward_progress_m": evaluation.forward_progress_m,
            "representative_lateral_m": evaluation.representative_lateral_m,
            "full_path_margin_m": evaluation.full_path_margin_m,
            "continuity_m": evaluation.continuity_m,
            "category_rank": self.category_rank,
            "stop_penalty": self.stop_penalty,
            "admission_quality_rank": self.admission_quality_rank,
            "route_mismatch_rank": self.route_mismatch_rank,
            "ranking_key": [
                value if math.isfinite(value) else None
                for value in self.ranking_key
            ],
        }


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    selected_plan_id: str
    selected_admitted: bool
    desired_turn: str | None
    selection_reason: str
    ranked_candidates: tuple[RankedCandidate, ...]

    @property
    def selected(self) -> RankedCandidate:
        return next(
            candidate
            for candidate in self.ranked_candidates
            if candidate.evaluation.candidate_index == self.selected_index
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_index": self.selected_index,
            "selected_plan_id": self.selected_plan_id,
            "selected_admitted": self.selected_admitted,
            "desired_turn": self.desired_turn,
            "selection_reason": self.selection_reason,
            "candidate_evaluations": [
                candidate.to_json_dict() for candidate in self.ranked_candidates
            ],
        }


def navigation_turn_direction(navigation_text: str | None) -> str | None:
    """Return one unambiguous turn direction from navigation text."""

    text = str(navigation_text or "")
    has_right = bool(_RIGHT_RE.search(text))
    has_left = bool(_LEFT_RE.search(text))
    if has_right == has_left:
        return None
    return "right" if has_right else "left"


def _route_mismatch_rank(
    lateral_m: float,
    desired_turn: str | None,
) -> float:
    if desired_turn is None:
        return 0.0
    if abs(lateral_m) < ROUTE_LATERAL_DEADBAND_M:
        return 0.25
    # Alpamayo uses y-left, so positive is left and negative is right.
    matches = lateral_m > 0.0 if desired_turn == "left" else lateral_m < 0.0
    return 0.0 if matches else 1.0


def _category_rank(admission_status: str | None) -> int:
    if admission_status in ACCEPTED_ADMISSION_STATUSES:
        return 0
    if admission_status == "REJECT_RETAIN_ACTIVE":
        return 1
    if admission_status == "REJECT_FALLBACK_STOP":
        return 2
    return 3


def rank_candidate_evaluations(
    evaluations: Iterable[CandidateEvaluation],
    *,
    navigation_text: str | None,
    prefer_moving: bool,
) -> CandidateSelection:
    """Rank all evaluated candidates with safety eligibility first.

    ``prefer_moving`` is intended only for deterministic empty-road
    diagnostics.  It lets an admitted moving safe prefix outrank an admitted
    stationary full-safe proposal.  Normal traffic leaves stop intent neutral.
    """

    candidates = tuple(evaluations)
    if not candidates:
        raise ValueError("at least one candidate evaluation is required")
    indices = [candidate.candidate_index for candidate in candidates]
    if len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be unique")

    desired_turn = navigation_turn_direction(navigation_text)
    ranked = []
    for candidate in candidates:
        category = _category_rank(candidate.admission_status)
        admission_quality = _ADMISSION_QUALITY.get(candidate.admission_status, 99)
        stop_penalty = int(
            category == 0 and bool(prefer_moving) and candidate.stop_requested
        )
        margin_rank = (
            -candidate.full_path_margin_m
            if candidate.full_path_margin_m is not None
            else float("inf")
        )
        continuity_rank = (
            candidate.continuity_m
            if candidate.continuity_m is not None
            else float("inf")
        )
        ranked.append(
            RankedCandidate(
                evaluation=candidate,
                category_rank=category,
                stop_penalty=stop_penalty,
                admission_quality_rank=admission_quality,
                route_mismatch_rank=_route_mismatch_rank(
                    candidate.representative_lateral_m,
                    desired_turn,
                ),
                negative_full_path_margin=margin_rank,
                continuity_rank_m=continuity_rank,
                negative_forward_progress_m=-candidate.forward_progress_m,
            )
        )

    ranked_candidates = tuple(sorted(ranked, key=lambda item: item.ranking_key))
    selected = ranked_candidates[0]
    if selected.evaluation.admitted:
        reason = "best_admitted_candidate"
    elif selected.evaluation.admission_status == "REJECT_RETAIN_ACTIVE":
        reason = "no_admissible_candidate_retain_active"
    elif selected.evaluation.admission_status == "REJECT_FALLBACK_STOP":
        reason = "no_admissible_candidate_fallback_stop"
    else:
        reason = "all_candidates_failed_validation"
    return CandidateSelection(
        selected_index=selected.evaluation.candidate_index,
        selected_plan_id=selected.evaluation.plan_id,
        selected_admitted=selected.evaluation.admitted,
        desired_turn=desired_turn,
        selection_reason=reason,
        ranked_candidates=ranked_candidates,
    )


__all__ = [
    "ACCEPTED_ADMISSION_STATUSES",
    "CandidateEvaluation",
    "CandidateSelection",
    "RankedCandidate",
    "navigation_turn_direction",
    "rank_candidate_evaluations",
]

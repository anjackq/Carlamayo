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
from enum import Enum
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
_RESERVE_QUALITY = {
    "UNBOUNDED": 0,
    "ROBUST": 0,
    "RECOVERY": 1,
    "FRAGILE": 2,
    "UNAVAILABLE": 3,
}
_MOTION_QUALITY = {
    "MOVING": 0,
    "DELAYED_START": 1,
    "CREEP_OR_STALL": 2,
    "EXPLICIT_STOP": 3,
}
_ROUTE_QUALITY = {
    "MATCH": 0,
    "DEVIATE": 1,
    "UNKNOWN": 2,
    None: 3,
}
_PHYSICAL_REACHABILITY_QUALITY = {
    "REACHABLE": 0,
    "TOO_LONG": 1,
    "TOO_SHORT_TO_STOP": 1,
    None: 2,
}
_SOURCE_SPEED_PRIOR_QUALITY = {
    "CONSISTENT": 0,
    "STOP_PRIOR": 1,
    "ACCELERATION_PRIOR": 1,
    "DECELERATION_PRIOR": 1,
    None: 2,
}
_RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)
_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)
ROUTE_LATERAL_DEADBAND_M = 0.75


class CandidateRankingPolicy(str, Enum):
    CURRENT = "current"
    REACHABILITY_FIRST = "reachability-first"


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
    motion_class: str | None = None
    initial_target_speed_mps: float | None = None
    stopping_reserve_status: str | None = None
    stopping_reserve_m: float | None = None
    time_to_first_bad_s: float | None = None
    near_term_route_status: str | None = None
    full_path_route_status: str | None = None
    route_branch_match: bool | None = None
    route_cross_track_error_m: float | None = None
    near_physical_status: str | None = None
    full_physical_status: str | None = None
    near_source_speed_prior_status: str | None = None
    near_required_constant_acceleration_mps2: float | None = None

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
        motion_class = (
            None if self.motion_class is None else str(self.motion_class).strip() or None
        )
        if motion_class is not None and motion_class not in _MOTION_QUALITY:
            raise ValueError(f"unknown motion_class: {motion_class}")
        object.__setattr__(self, "motion_class", motion_class)
        reserve_status = (
            None
            if self.stopping_reserve_status is None
            else str(self.stopping_reserve_status).strip() or None
        )
        if reserve_status is not None and reserve_status not in _RESERVE_QUALITY:
            raise ValueError(f"unknown stopping_reserve_status: {reserve_status}")
        object.__setattr__(self, "stopping_reserve_status", reserve_status)
        object.__setattr__(
            self,
            "initial_target_speed_mps",
            _finite_or_none(
                self.initial_target_speed_mps,
                "initial_target_speed_mps",
            ),
        )
        object.__setattr__(
            self,
            "stopping_reserve_m",
            _finite_or_none(self.stopping_reserve_m, "stopping_reserve_m"),
        )
        object.__setattr__(
            self,
            "time_to_first_bad_s",
            _finite_or_none(self.time_to_first_bad_s, "time_to_first_bad_s"),
        )
        for attribute in ("near_term_route_status", "full_path_route_status"):
            raw_status = getattr(self, attribute)
            route_status = (
                None
                if raw_status is None
                else str(raw_status).strip().upper() or None
            )
            if route_status not in _ROUTE_QUALITY:
                raise ValueError(f"unknown {attribute}: {route_status}")
            object.__setattr__(self, attribute, route_status)
        object.__setattr__(
            self,
            "route_branch_match",
            (
                None
                if self.route_branch_match is None
                else bool(self.route_branch_match)
            ),
        )
        object.__setattr__(
            self,
            "route_cross_track_error_m",
            _finite_or_none(
                self.route_cross_track_error_m,
                "route_cross_track_error_m",
            ),
        )
        for attribute in ("near_physical_status", "full_physical_status"):
            raw_status = getattr(self, attribute)
            physical_status = (
                None
                if raw_status is None
                else str(raw_status).strip().upper() or None
            )
            if physical_status not in _PHYSICAL_REACHABILITY_QUALITY:
                raise ValueError(f"unknown {attribute}: {physical_status}")
            object.__setattr__(self, attribute, physical_status)
        prior_status = (
            None
            if self.near_source_speed_prior_status is None
            else str(self.near_source_speed_prior_status).strip().upper() or None
        )
        if prior_status not in _SOURCE_SPEED_PRIOR_QUALITY:
            raise ValueError(
                "unknown near_source_speed_prior_status: "
                f"{prior_status}"
            )
        object.__setattr__(
            self,
            "near_source_speed_prior_status",
            prior_status,
        )
        object.__setattr__(
            self,
            "near_required_constant_acceleration_mps2",
            _finite_or_none(
                self.near_required_constant_acceleration_mps2,
                "near_required_constant_acceleration_mps2",
            ),
        )

    @property
    def admitted(self) -> bool:
        return self.admission_status in ACCEPTED_ADMISSION_STATUSES

    @property
    def reachability_complete(self) -> bool:
        return bool(
            self.near_physical_status is not None
            and self.full_physical_status is not None
            and self.near_source_speed_prior_status is not None
            and self.near_required_constant_acceleration_mps2 is not None
        )


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate plus the transparent lexicographic ranking terms."""

    evaluation: CandidateEvaluation
    ranking_policy: CandidateRankingPolicy
    category_rank: int
    reserve_fragility_rank: int
    near_route_rank: int
    motion_quality_rank: int
    stop_penalty: int
    near_physical_reachability_rank: int
    full_physical_reachability_rank: int
    source_speed_prior_rank: int
    absolute_required_acceleration_rank: float
    admission_quality_rank: int
    negative_stopping_reserve_m: float
    negative_time_to_first_bad_s: float
    route_mismatch_rank: float
    negative_full_path_margin: float
    speed_continuity_rank_mps: float
    continuity_rank_m: float
    negative_forward_progress_m: float

    @property
    def current_ranking_key(self) -> tuple[float, ...]:
        return (
            float(self.category_rank),
            float(self.reserve_fragility_rank),
            float(self.motion_quality_rank),
            float(self.admission_quality_rank),
            float(self.negative_stopping_reserve_m),
            float(self.negative_time_to_first_bad_s),
            float(self.route_mismatch_rank),
            float(self.negative_full_path_margin),
            float(self.speed_continuity_rank_mps),
            float(self.continuity_rank_m),
            float(self.negative_forward_progress_m),
            float(self.evaluation.candidate_index),
        )

    @property
    def reachability_ranking_key(self) -> tuple[float, ...]:
        return (
            float(self.category_rank),
            float(self.reserve_fragility_rank),
            float(self.near_route_rank),
            float(self.stop_penalty),
            float(self.near_physical_reachability_rank),
            float(self.admission_quality_rank),
            float(self.route_mismatch_rank),
            float(self.full_physical_reachability_rank),
            float(self.motion_quality_rank),
            float(self.source_speed_prior_rank),
            float(self.absolute_required_acceleration_rank),
            float(self.negative_stopping_reserve_m),
            float(self.negative_time_to_first_bad_s),
            float(self.negative_full_path_margin),
            float(self.speed_continuity_rank_mps),
            float(self.continuity_rank_m),
            float(self.negative_forward_progress_m),
            float(self.evaluation.candidate_index),
        )

    @property
    def ranking_key(self) -> tuple[float, ...]:
        if self.ranking_policy is CandidateRankingPolicy.REACHABILITY_FIRST:
            return self.reachability_ranking_key
        return self.current_ranking_key

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
            "motion_class": evaluation.motion_class,
            "initial_target_speed_mps": evaluation.initial_target_speed_mps,
            "stopping_reserve_status": evaluation.stopping_reserve_status,
            "stopping_reserve_m": evaluation.stopping_reserve_m,
            "time_to_first_bad_s": evaluation.time_to_first_bad_s,
            "near_term_route_status": evaluation.near_term_route_status,
            "full_path_route_status": evaluation.full_path_route_status,
            "route_branch_match": evaluation.route_branch_match,
            "route_cross_track_error_m": evaluation.route_cross_track_error_m,
            "near_physical_status": evaluation.near_physical_status,
            "full_physical_status": evaluation.full_physical_status,
            "near_source_speed_prior_status": (
                evaluation.near_source_speed_prior_status
            ),
            "near_required_constant_acceleration_mps2": (
                evaluation.near_required_constant_acceleration_mps2
            ),
            "reachability_complete": evaluation.reachability_complete,
            "ranking_policy": self.ranking_policy.value,
            "category_rank": self.category_rank,
            "reserve_fragility_rank": self.reserve_fragility_rank,
            "near_route_rank": self.near_route_rank,
            "motion_quality_rank": self.motion_quality_rank,
            "stop_penalty": self.stop_penalty,
            "near_physical_reachability_rank": (
                self.near_physical_reachability_rank
            ),
            "full_physical_reachability_rank": (
                self.full_physical_reachability_rank
            ),
            "source_speed_prior_rank": self.source_speed_prior_rank,
            "absolute_required_acceleration_rank": (
                self.absolute_required_acceleration_rank
            ),
            "admission_quality_rank": self.admission_quality_rank,
            "speed_continuity_rank_mps": self.speed_continuity_rank_mps,
            "route_mismatch_rank": self.route_mismatch_rank,
            "ranking_key": [
                value if math.isfinite(value) else None
                for value in self.ranking_key
            ],
            "current_ranking_key": [
                value if math.isfinite(value) else None
                for value in self.current_ranking_key
            ],
            "reachability_ranking_key": [
                value if math.isfinite(value) else None
                for value in self.reachability_ranking_key
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
    requested_ranking_policy: CandidateRankingPolicy = (
        CandidateRankingPolicy.CURRENT
    )
    effective_ranking_policy: CandidateRankingPolicy = (
        CandidateRankingPolicy.CURRENT
    )
    ranking_fallback_reason: str | None = None

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
            "requested_ranking_policy": (
                self.requested_ranking_policy.value
            ),
            "effective_ranking_policy": self.effective_ranking_policy.value,
            "ranking_fallback_reason": self.ranking_fallback_reason,
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
    current_speed_mps: float | None = None,
    ranking_policy: CandidateRankingPolicy | str = (
        CandidateRankingPolicy.CURRENT
    ),
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
    try:
        requested_policy = CandidateRankingPolicy(ranking_policy)
    except ValueError as exc:
        raise ValueError(f"unknown candidate ranking policy: {ranking_policy}") from exc
    ranking_fallback_reason = None
    effective_policy = requested_policy
    if (
        requested_policy is CandidateRankingPolicy.REACHABILITY_FIRST
        and not all(candidate.reachability_complete for candidate in candidates)
    ):
        effective_policy = CandidateRankingPolicy.CURRENT
        ranking_fallback_reason = "incomplete_reachability_batch"

    desired_turn = navigation_turn_direction(navigation_text)
    current_speed = _finite_or_none(current_speed_mps, "current_speed_mps")
    ranked = []
    for candidate in candidates:
        category = _category_rank(candidate.admission_status)
        admission_quality = _ADMISSION_QUALITY.get(candidate.admission_status, 99)
        reserve_status = candidate.stopping_reserve_status or "UNAVAILABLE"
        reserve_fragility = _RESERVE_QUALITY[reserve_status]
        effective_motion_class = candidate.motion_class or (
            "EXPLICIT_STOP" if candidate.stop_requested else "MOVING"
        )
        motion_quality = (
            _MOTION_QUALITY[effective_motion_class] if prefer_moving else 0
        )
        stop_penalty = int(prefer_moving and effective_motion_class == "EXPLICIT_STOP")
        if reserve_status == "UNBOUNDED":
            reserve_rank = float("-inf")
            time_rank = float("-inf")
        else:
            reserve_rank = (
                -candidate.stopping_reserve_m
                if candidate.stopping_reserve_m is not None
                else float("inf")
            )
            time_rank = (
                -candidate.time_to_first_bad_s
                if candidate.time_to_first_bad_s is not None
                else float("inf")
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
        speed_continuity_rank = 0.0
        if prefer_moving:
            if (
                current_speed is None
                or candidate.initial_target_speed_mps is None
            ):
                speed_continuity_rank = float("inf")
            else:
                speed_continuity_rank = abs(
                    candidate.initial_target_speed_mps - current_speed
                )
        if candidate.full_path_route_status is not None:
            route_mismatch_rank = {
                "MATCH": 0.0,
                "DEVIATE": 1.0,
                "UNKNOWN": 2.0,
            }[candidate.full_path_route_status]
            if candidate.route_branch_match is False:
                route_mismatch_rank += 1.0
        else:
            route_mismatch_rank = _route_mismatch_rank(
                candidate.representative_lateral_m,
                desired_turn,
            )
        near_route_rank = _ROUTE_QUALITY[candidate.near_term_route_status]
        near_physical_rank = _PHYSICAL_REACHABILITY_QUALITY[
            candidate.near_physical_status
        ]
        full_physical_rank = _PHYSICAL_REACHABILITY_QUALITY[
            candidate.full_physical_status
        ]
        source_prior_rank = _SOURCE_SPEED_PRIOR_QUALITY[
            candidate.near_source_speed_prior_status
        ]
        required_acceleration_rank = (
            abs(candidate.near_required_constant_acceleration_mps2)
            if candidate.near_required_constant_acceleration_mps2 is not None
            else float("inf")
        )
        ranked.append(
            RankedCandidate(
                evaluation=candidate,
                ranking_policy=effective_policy,
                category_rank=category,
                reserve_fragility_rank=reserve_fragility,
                near_route_rank=near_route_rank,
                motion_quality_rank=motion_quality,
                stop_penalty=stop_penalty,
                near_physical_reachability_rank=near_physical_rank,
                full_physical_reachability_rank=full_physical_rank,
                source_speed_prior_rank=source_prior_rank,
                absolute_required_acceleration_rank=(
                    required_acceleration_rank
                ),
                admission_quality_rank=admission_quality,
                negative_stopping_reserve_m=reserve_rank,
                negative_time_to_first_bad_s=time_rank,
                route_mismatch_rank=route_mismatch_rank,
                negative_full_path_margin=margin_rank,
                speed_continuity_rank_mps=speed_continuity_rank,
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
        requested_ranking_policy=requested_policy,
        effective_ranking_policy=effective_policy,
        ranking_fallback_reason=ranking_fallback_reason,
    )


__all__ = [
    "ACCEPTED_ADMISSION_STATUSES",
    "CandidateEvaluation",
    "CandidateRankingPolicy",
    "CandidateSelection",
    "RankedCandidate",
    "navigation_turn_direction",
    "rank_candidate_evaluations",
]

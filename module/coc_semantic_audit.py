"""Diagnostic-only, source-synchronized Alpamayo CoC semantic audit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from .route_navigation import NavigationAction, NavigationContext


class ClaimPolarity(str, Enum):
    AFFIRM = "AFFIRM"
    NEGATE = "NEGATE"
    UNCERTAIN = "UNCERTAIN"


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    POLICY_CONFLICT = "POLICY_CONFLICT"
    TRAJECTORY_MISMATCH = "TRAJECTORY_MISMATCH"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True)
class SceneActorTruth:
    actor_id: int
    actor_class: str
    distance_m: float


@dataclass(frozen=True)
class SceneTrafficControlTruth:
    control_type: str
    state: str | None
    distance_m: float
    route_relevant: bool


@dataclass(frozen=True)
class SceneTruthSnapshot:
    source_frame_id: int
    source_simulation_time_s: float
    ego_road_id: int | None
    ego_section_id: int | None
    ego_lane_id: int | None
    ego_is_junction: bool | None
    navigation_context: NavigationContext | None
    dynamic_actors: tuple[SceneActorTruth, ...]
    traffic_controls: tuple[SceneTrafficControlTruth, ...]
    strict_lane_policy_enabled: bool
    scenario_annotations: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_frame_id": int(self.source_frame_id),
            "source_simulation_time_s": float(self.source_simulation_time_s),
            "ego_road_id": self.ego_road_id,
            "ego_section_id": self.ego_section_id,
            "ego_lane_id": self.ego_lane_id,
            "ego_is_junction": self.ego_is_junction,
            "navigation_context": (
                None
                if self.navigation_context is None
                else self.navigation_context.to_json_dict()
            ),
            "dynamic_actors": [
                {
                    "actor_id": int(actor.actor_id),
                    "actor_class": actor.actor_class,
                    "distance_m": float(actor.distance_m),
                }
                for actor in self.dynamic_actors
            ],
            "traffic_controls": [
                {
                    "control_type": control.control_type,
                    "state": control.state,
                    "distance_m": float(control.distance_m),
                    "route_relevant": bool(control.route_relevant),
                }
                for control in self.traffic_controls
            ],
            "strict_lane_policy_enabled": bool(self.strict_lane_policy_enabled),
            "scenario_annotations": list(self.scenario_annotations),
        }


@dataclass(frozen=True)
class CoCClaim:
    category: str
    value: str
    polarity: ClaimPolarity
    evidence_text: str

    def to_json_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "value": self.value,
            "polarity": self.polarity.value,
            "evidence_text": self.evidence_text,
        }


@dataclass(frozen=True)
class AuditedClaim:
    claim: CoCClaim
    verdict: ClaimVerdict
    reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self.claim.to_json_dict(),
            "verdict": self.verdict.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CoCSemanticAudit:
    claims: tuple[AuditedClaim, ...]
    audit_error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        verdict_counts = {verdict.value: 0 for verdict in ClaimVerdict}
        for audited in self.claims:
            verdict_counts[audited.verdict.value] += 1
        return {
            "claims": [claim.to_json_dict() for claim in self.claims],
            "verdict_counts": verdict_counts,
            "positive_hallucination_count": sum(
                1
                for claim in self.claims
                if claim.claim.polarity is ClaimPolarity.AFFIRM
                and claim.verdict is ClaimVerdict.CONTRADICTED
                and claim.claim.category in {"actor", "traffic_light", "traffic_sign"}
            ),
            "audit_error": self.audit_error,
        }


_CLAIM_PATTERNS = (
    ("lane", "change_left", re.compile(r"\b(?:change|merge)(?:\s+\w+){0,2}\s+left\b")),
    ("lane", "change_right", re.compile(r"\b(?:change|merge)(?:\s+\w+){0,2}\s+right\b")),
    ("lane", "keep", re.compile(r"\b(?:keep|stay|remain)(?:\s+\w+){0,2}\s+lane\b")),
    ("turn", "left", re.compile(r"\b(?:turn|curve|bend)(?:s|ing)?\s+left\b")),
    ("turn", "right", re.compile(r"\b(?:turn|curve|bend)(?:s|ing)?\s+right\b")),
    ("motion", "stop", re.compile(r"\b(?:stop|stopping|halt)\b")),
    ("motion", "slow", re.compile(r"\b(?:slow|slowing|decelerate|decelerating)\b")),
    ("motion", "accelerate", re.compile(r"\b(?:accelerate|accelerating|speed up)\b")),
    ("actor", "vehicle", re.compile(r"\b(?:vehicle|car|truck|bus)\b")),
    ("actor", "pedestrian", re.compile(r"\b(?:pedestrian|person|walker)\b")),
    ("actor", "cyclist", re.compile(r"\b(?:cyclist|bicycle|bike)\b")),
    (
        "traffic_light",
        "red",
        re.compile(r"\b(?:red (?:traffic )?light|(?:traffic )?light is red)\b"),
    ),
    (
        "traffic_light",
        "yellow",
        re.compile(r"\b(?:yellow (?:traffic )?light|(?:traffic )?light is yellow)\b"),
    ),
    (
        "traffic_light",
        "green",
        re.compile(r"\b(?:green (?:traffic )?light|(?:traffic )?light is green)\b"),
    ),
    ("traffic_sign", "stop", re.compile(r"\bstop sign\b")),
    ("traffic_sign", "yield", re.compile(r"\byield sign\b")),
    ("road_context", "junction", re.compile(r"\b(?:junction|intersection)\b")),
    ("road_context", "roundabout", re.compile(r"\broundabout\b")),
)
_NEGATION_RE = re.compile(r"\b(?:no|not|without|none|isn't|is not)\b")
_UNCERTAINTY_RE = re.compile(
    r"\b(?:potential|possibly|possible|may|might|could|uncertain|appears)\b"
)


def _sentence_fragments(text: str) -> Iterable[str]:
    for fragment in re.split(r"(?<=[.!?;])\s+|\n+", text):
        fragment = fragment.strip()
        if fragment:
            yield fragment


def parse_coc_claims(coc_text: str) -> tuple[CoCClaim, ...]:
    """Extract conservative, polarity-aware semantic claims."""

    claims: list[CoCClaim] = []
    for fragment in _sentence_fragments(str(coc_text or "").lower()):
        for category, value, pattern in _CLAIM_PATTERNS:
            match = pattern.search(fragment)
            if match is None:
                continue
            window = fragment[max(0, match.start() - 40) : match.end() + 15]
            if _UNCERTAINTY_RE.search(window):
                polarity = ClaimPolarity.UNCERTAIN
            elif _NEGATION_RE.search(window):
                polarity = ClaimPolarity.NEGATE
            else:
                polarity = ClaimPolarity.AFFIRM
            claims.append(
                CoCClaim(
                    category=category,
                    value=value,
                    polarity=polarity,
                    evidence_text=fragment,
                )
            )
    return tuple(claims)


def _presence_verdict(
    claim: CoCClaim,
    present: bool,
    *,
    reason_prefix: str,
) -> AuditedClaim:
    if claim.polarity is ClaimPolarity.UNCERTAIN:
        return AuditedClaim(claim, ClaimVerdict.NOT_EVALUABLE, "uncertain_claim")
    supported = present if claim.polarity is ClaimPolarity.AFFIRM else not present
    return AuditedClaim(
        claim,
        ClaimVerdict.SUPPORTED if supported else ClaimVerdict.CONTRADICTED,
        f"{reason_prefix}_{'present' if present else 'absent'}",
    )


def _motion_value(motion_profile: Any) -> str:
    motion_class = getattr(motion_profile, "motion_class", "")
    return str(getattr(motion_class, "value", motion_class)).upper()


def audit_coc_semantics(
    coc_text: str,
    snapshot: SceneTruthSnapshot,
    *,
    route_assessment: Any | None = None,
    motion_profile: Any | None = None,
) -> CoCSemanticAudit:
    """Audit CoC against source-time truth without producing control authority."""

    try:
        claims = parse_coc_claims(coc_text)
        results: list[AuditedClaim] = []
        actor_classes = {
            actor.actor_class.lower()
            for actor in snapshot.dynamic_actors
            if float(actor.distance_m) <= 60.0
        }
        controls = tuple(
            control for control in snapshot.traffic_controls if control.route_relevant
        )
        nav_action = (
            snapshot.navigation_context.action
            if snapshot.navigation_context is not None
            else None
        )
        for claim in claims:
            if claim.category == "actor":
                aliases = {
                    "vehicle": {"vehicle", "car", "truck", "bus"},
                    "pedestrian": {"pedestrian", "person", "walker"},
                    "cyclist": {"cyclist", "bicycle", "bike"},
                }[claim.value]
                results.append(
                    _presence_verdict(
                        claim,
                        bool(actor_classes & aliases),
                        reason_prefix=claim.value,
                    )
                )
            elif claim.category == "traffic_light":
                present = any(
                    control.control_type.lower() == "traffic_light"
                    and str(control.state or "").lower() == claim.value
                    for control in controls
                )
                results.append(
                    _presence_verdict(
                        claim,
                        present,
                        reason_prefix=f"{claim.value}_traffic_light",
                    )
                )
            elif claim.category == "traffic_sign":
                present = any(
                    control.control_type.lower() == f"{claim.value}_sign"
                    for control in controls
                )
                results.append(
                    _presence_verdict(
                        claim,
                        present,
                        reason_prefix=f"{claim.value}_sign",
                    )
                )
            elif claim.category == "lane":
                if (
                    claim.value.startswith("change")
                    and claim.polarity is ClaimPolarity.AFFIRM
                    and snapshot.strict_lane_policy_enabled
                    and (
                        snapshot.navigation_context is None
                        or not snapshot.navigation_context.lane_change_authorized
                    )
                ):
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.POLICY_CONFLICT,
                            "strict_lane_policy_forbids_lane_change",
                        )
                    )
                else:
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.NOT_EVALUABLE,
                            "lane_claim_requires_trajectory_relation",
                        )
                    )
            elif claim.category == "turn":
                expected = {
                    "left": NavigationAction.LEFT,
                    "right": NavigationAction.RIGHT,
                }[claim.value]
                if route_assessment is not None and str(
                    getattr(route_assessment, "branch_match", True)
                ).lower() == "false":
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.TRAJECTORY_MISMATCH,
                            "candidate_uses_unauthorized_route_branch",
                        )
                    )
                elif nav_action is None:
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.NOT_EVALUABLE,
                            "navigation_context_unavailable",
                        )
                    )
                else:
                    supported = nav_action is expected
                    if claim.polarity is ClaimPolarity.NEGATE:
                        supported = not supported
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.SUPPORTED if supported else ClaimVerdict.CONTRADICTED,
                            f"route_action_{nav_action.value.lower()}",
                        )
                    )
            elif claim.category == "motion":
                motion_value = _motion_value(motion_profile)
                supported = {
                    "stop": motion_value == "EXPLICIT_STOP",
                    "slow": (
                        float(
                            getattr(
                                motion_profile,
                                "peak_smoothed_deceleration_mps2",
                                0.0,
                            )
                        )
                        > 0.25
                    ),
                    "accelerate": (
                        motion_value in {"MOVING", "DELAYED_START"}
                        and float(
                            getattr(motion_profile, "near_term_peak_speed_mps", 0.0)
                        )
                        > float(
                            getattr(motion_profile, "initial_target_speed_mps", 0.0)
                        )
                    ),
                }[claim.value]
                if claim.polarity is ClaimPolarity.NEGATE:
                    supported = not supported
                if claim.polarity is ClaimPolarity.UNCERTAIN or motion_profile is None:
                    verdict = ClaimVerdict.NOT_EVALUABLE
                else:
                    verdict = (
                        ClaimVerdict.SUPPORTED
                        if supported
                        else ClaimVerdict.TRAJECTORY_MISMATCH
                    )
                results.append(
                    AuditedClaim(claim, verdict, f"motion_class_{motion_value or 'unknown'}")
                )
            elif claim.value == "junction":
                present = bool(snapshot.ego_is_junction) or nav_action in {
                    NavigationAction.STRAIGHT,
                    NavigationAction.LEFT,
                    NavigationAction.RIGHT,
                }
                results.append(
                    _presence_verdict(claim, present, reason_prefix="junction")
                )
            elif claim.value == "roundabout":
                annotated = "roundabout" in {
                    item.lower() for item in snapshot.scenario_annotations
                }
                if not annotated:
                    results.append(
                        AuditedClaim(
                            claim,
                            ClaimVerdict.NOT_EVALUABLE,
                            "roundabout_not_scenario_annotated",
                        )
                    )
                else:
                    results.append(
                        _presence_verdict(
                            claim,
                            True,
                            reason_prefix="annotated_roundabout",
                        )
                    )
        return CoCSemanticAudit(tuple(results))
    except Exception as exc:
        return CoCSemanticAudit((), audit_error=f"{type(exc).__name__}:{exc}")


__all__ = [
    "AuditedClaim",
    "ClaimPolarity",
    "ClaimVerdict",
    "CoCClaim",
    "CoCSemanticAudit",
    "SceneActorTruth",
    "SceneTrafficControlTruth",
    "SceneTruthSnapshot",
    "audit_coc_semantics",
    "parse_coc_claims",
]

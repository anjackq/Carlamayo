#!/usr/bin/env python3
"""Summarize route-policy and CoC live-gate facts from runtime JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.carla_route_adapter import FIXED_TOWN03_DESTINATION_XYZ
from module.route_navigation import (
    NavigationAction,
    NavigationManeuverPhase,
    format_navigation_prompt,
)


STATIONARY_SPEED_MPS = 0.2
ABSORBING_STOP_TICKS = 50
ROUTE_COMPLETION_GATE = 0.95
HARD_BRAKE_THRESHOLD = 0.8


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(event)
    if not events:
        raise ValueError(f"{path}: telemetry is empty")
    return events


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nearest_rank_percentile(
    values: Iterable[float],
    percentile: float,
) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(
        len(ordered) - 1,
        max(0, int(math.ceil(float(percentile) * len(ordered))) - 1),
    )
    return float(ordered[index])


def _position_xy(event: dict[str, Any]) -> tuple[float, float] | None:
    position = event.get("ego_position_world")
    if not isinstance(position, dict):
        return None
    x = _finite_float(position.get("x"))
    y = _finite_float(position.get("y"))
    return None if x is None or y is None else (x, y)


def _integrated_distance(ticks: Iterable[dict[str, Any]]) -> float:
    positions = [
        position
        for event in ticks
        if (position := _position_xy(event)) is not None
    ]
    return float(
        sum(
            math.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(positions, positions[1:])
        )
    )


def _navigation_contexts(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") not in {
            "episode_start",
            "inference_submitted",
        }:
            continue
        context = event.get("navigation_context")
        if isinstance(context, dict):
            contexts.append(context)
    return contexts


def _prompt_mismatch(context: dict[str, Any]) -> bool:
    try:
        action = NavigationAction(str(context["action"]))
        distance = float(context["distance_to_maneuver_m"])
        phase = NavigationManeuverPhase(
            str(context.get("maneuver_phase", "APPROACH"))
        )
    except (KeyError, TypeError, ValueError):
        return True
    expected = (
        ""
        if context.get("tracker_status") == "ROUTE_UNAVAILABLE"
        else format_navigation_prompt(
            action,
            distance,
            maneuver_phase=phase,
        )
    )
    return str(context.get("text", "")) != expected


def _road_status(event: dict[str, Any]) -> str | None:
    envelope = event.get("road_execution_envelope")
    if not isinstance(envelope, dict):
        return None
    assessment = envelope.get("current_ego_road")
    return (
        str(assessment.get("status"))
        if isinstance(assessment, dict) and assessment.get("status") is not None
        else None
    )


def _selected_candidate_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event_type") == "candidate_evaluation"
        and event.get("selected") is True
    ]


def _longest_stationary_streak(
    ticks: Iterable[dict[str, Any]],
) -> int:
    longest = 0
    current = 0
    for tick in ticks:
        speed = _finite_float(tick.get("speed_mps"))
        if speed is not None and speed <= STATIONARY_SPEED_MPS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _stop_go_restart_count(
    ticks: Iterable[dict[str, Any]],
) -> int:
    stopped = False
    restarts = 0
    for tick in ticks:
        speed = _finite_float(tick.get("speed_mps"))
        if speed is None:
            continue
        if speed <= STATIONARY_SPEED_MPS:
            stopped = True
        elif stopped and speed >= 0.5:
            restarts += 1
            stopped = False
    return restarts


def _applied_brake(tick: dict[str, Any]) -> float | None:
    applied = tick.get("applied_control")
    if isinstance(applied, dict):
        return _finite_float(applied.get("brake"))
    return _finite_float(tick.get("echoed_brake"))


def _navigation_phase(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return "UNKNOWN"
    action = str(context.get("action") or "UNKNOWN")
    maneuver_phase = str(
        context.get("maneuver_phase") or "APPROACH"
    )
    distance = _finite_float(context.get("distance_to_maneuver_m"))
    if maneuver_phase == "ACTIVE":
        phase = "ACTIVE"
    elif distance is not None and distance <= 15.0:
        phase = "APPROACH_NEAR"
    else:
        phase = "APPROACH_FAR"
    return f"{action}/{phase}"


def _selection_coverage(
    events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    proposal_contexts = {
        str(event["proposal_id"]): event.get("navigation_context")
        for event in events
        if event.get("event_type") == "alpamayo_proposal"
        and event.get("proposal_id") is not None
    }
    selections = [
        event
        for event in events
        if event.get("event_type") == "candidate_selection"
    ]
    selection_compute_values = [
        value
        for selection in selections
        if (
            value := _finite_float(
                selection.get("selection_compute_ms")
            )
        )
        is not None
    ]
    phase_rows: dict[str, dict[str, int]] = {}
    near_available = 0
    full_available = 0
    current_near_misses = 0
    reachability_near_misses = 0
    effective_near_misses = 0
    current_full_misses = 0
    reachability_full_misses = 0
    effective_full_misses = 0
    batch_fallbacks = 0
    selection_changes = 0
    for selection in selections:
        proposal_id = str(selection.get("proposal_id") or "")
        phase = _navigation_phase(proposal_contexts.get(proposal_id))
        phase_row = phase_rows.setdefault(
            phase,
            {
                "requests": 0,
                "near_executable_available": 0,
                "full_turn_executable_available": 0,
                "current_near_misses": 0,
                "reachability_near_misses": 0,
                "effective_near_misses": 0,
                "current_full_misses": 0,
                "reachability_full_misses": 0,
                "effective_full_misses": 0,
            },
        )
        phase_row["requests"] += 1
        near_indices = {
            int(value)
            for value in (
                selection.get("near_executable_candidate_indices")
                or ()
            )
        }
        full_indices = {
            int(value)
            for value in (
                selection.get(
                    "full_turn_executable_candidate_indices"
                )
                or ()
            )
        }
        current_index = selection.get(
            "current_shadow_selected_index"
        )
        reachability_index = selection.get(
            "reachability_shadow_selected_index"
        )
        effective_index = selection.get(
            "selected_candidate_index"
        )
        if near_indices:
            near_available += 1
            phase_row["near_executable_available"] += 1
            if current_index is None or int(current_index) not in near_indices:
                current_near_misses += 1
                phase_row["current_near_misses"] += 1
            if (
                reachability_index is None
                or int(reachability_index) not in near_indices
            ):
                reachability_near_misses += 1
                phase_row["reachability_near_misses"] += 1
            if (
                effective_index is None
                or int(effective_index) not in near_indices
            ):
                effective_near_misses += 1
                phase_row["effective_near_misses"] += 1
        if full_indices:
            full_available += 1
            phase_row["full_turn_executable_available"] += 1
            if current_index is None or int(current_index) not in full_indices:
                current_full_misses += 1
                phase_row["current_full_misses"] += 1
            if (
                reachability_index is None
                or int(reachability_index) not in full_indices
            ):
                reachability_full_misses += 1
                phase_row["reachability_full_misses"] += 1
            if (
                effective_index is None
                or int(effective_index) not in full_indices
            ):
                effective_full_misses += 1
                phase_row["effective_full_misses"] += 1
        batch_fallbacks += int(
            selection.get("candidate_ranking_fallback_reason")
            is not None
        )
        selection_changes += int(
            selection.get("shadow_selection_changed") is True
        )
    return {
        "request_count": len(selections),
        "near_executable_available_requests": near_available,
        "full_turn_executable_available_requests": full_available,
        "near_executable_coverage_at_k": (
            near_available / len(selections) if selections else None
        ),
        "full_turn_executable_coverage_at_k": (
            full_available / len(selections) if selections else None
        ),
        "current_near_conditional_misses": current_near_misses,
        "reachability_near_conditional_misses": (
            reachability_near_misses
        ),
        "effective_near_conditional_misses": effective_near_misses,
        "current_full_conditional_misses": current_full_misses,
        "reachability_full_conditional_misses": (
            reachability_full_misses
        ),
        "effective_full_conditional_misses": effective_full_misses,
        "shadow_selection_change_count": selection_changes,
        "reachability_batch_fallback_count": batch_fallbacks,
        "selection_compute_p95_ms": _nearest_rank_percentile(
            selection_compute_values,
            0.95,
        ),
        "phase": dict(sorted(phase_rows.items())),
    }


def _accepted_unauthorized(candidate: dict[str, Any]) -> bool:
    if candidate.get("admitted") is not True:
        return False
    route = candidate.get("candidate_route_assessment")
    if not isinstance(route, dict):
        return True
    return (
        route.get("current_route_status") != "MATCH"
        or route.get("near_term_route_status") != "MATCH"
    )


def _route_only_override(event: dict[str, Any]) -> bool:
    if event.get("direct_safety_trigger") is not True:
        return False
    values = [
        event.get("safety_override_reason"),
        event.get("safety_override_type"),
        *(event.get("safety_reason_codes") or []),
    ]
    return any("route" in str(value).lower() for value in values if value is not None)


def summarize_route_rollout(
    events: list[dict[str, Any]],
    *,
    destination_xyz: tuple[float, float, float] = FIXED_TOWN03_DESTINATION_XYZ,
) -> dict[str, Any]:
    start = next(
        (event for event in events if event.get("event_type") == "episode_start"),
        {},
    )
    episode_summary = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "episode_summary"
        ),
        {},
    )
    ticks = [event for event in events if event.get("event_type") == "tick"]
    contexts = _navigation_contexts(events)
    candidates = _selected_candidate_events(events)
    all_candidate_events = [
        event
        for event in events
        if event.get("event_type") == "candidate_evaluation"
    ]
    selection_coverage = _selection_coverage(events)

    route_length = _finite_float(
        (start.get("route_startup_facts") or {}).get("route_length_m")
    )
    progress_values = [
        progress
        for context in contexts
        if (progress := _finite_float(context.get("route_progress_m"))) is not None
    ]
    max_progress = max(progress_values, default=0.0)
    completion = (
        min(1.0, max_progress / route_length)
        if route_length is not None and route_length > 0.0
        else None
    )

    final_tick = ticks[-1] if ticks else {}
    final_position = _position_xy(final_tick)
    final_error = (
        math.hypot(
            final_position[0] - destination_xyz[0],
            final_position[1] - destination_xyz[1],
        )
        if final_position is not None
        else None
    )
    final_speed = _finite_float(final_tick.get("speed_mps"))

    stationary_suffix = 0
    for tick in reversed(ticks):
        speed = _finite_float(tick.get("speed_mps"))
        if speed is None or speed > STATIONARY_SPEED_MPS:
            break
        stationary_suffix += 1
    longest_stationary_streak = _longest_stationary_streak(ticks)
    stop_go_restarts = _stop_go_restart_count(ticks)
    hard_brake_ticks = sum(
        (brake := _applied_brake(tick)) is not None
        and brake >= HARD_BRAKE_THRESHOLD
        for tick in ticks
    )

    audit_verdicts: Counter[str] = Counter()
    positive_hallucinations = 0
    audit_errors = 0
    for candidate in candidates:
        audit = candidate.get("coc_semantic_audit")
        if not isinstance(audit, dict):
            continue
        if audit.get("audit_error"):
            audit_errors += 1
        positive_hallucinations += int(audit.get("positive_hallucination_count") or 0)
        counts = audit.get("verdict_counts")
        if isinstance(counts, dict):
            for verdict, count in counts.items():
                audit_verdicts[str(verdict)] += int(count)

    fallback_ticks = sum(
        tick.get("applied_control_source") == "FALLBACK" for tick in ticks
    )
    direct_overrides = sum(
        tick.get("direct_safety_trigger") is True for tick in ticks
    )
    latch_only_ticks = sum(tick.get("latch_only") is True for tick in ticks)
    collisions = max(
        (int(tick.get("collision_count") or 0) for tick in ticks),
        default=int(episode_summary.get("episode_collision_count") or 0),
    )
    prompt_mismatches = sum(_prompt_mismatch(context) for context in contexts)
    route_unavailable_contexts = sum(
        context.get("tracker_status") == "ROUTE_UNAVAILABLE"
        for context in contexts
    )
    route_unavailable_events = sum(
        event.get("event_type") == "route_unavailable"
        for event in events
    )
    route_unavailable_samples = max(
        route_unavailable_contexts,
        route_unavailable_events,
    )
    unauthorized_acceptances = sum(
        _accepted_unauthorized(candidate) for candidate in candidates
    )
    route_only_overrides = sum(_route_only_override(tick) for tick in ticks)
    ego_unsafe_ticks = sum(_road_status(tick) in {"UNSAFE", "UNKNOWN"} for tick in ticks)
    route_constraint_ticks = sum(
        tick.get("controller_state") == "ROUTE_POLICY_CONSTRAINT"
        for tick in ticks
    )
    absorbing_stop = (
        stationary_suffix >= ABSORBING_STOP_TICKS
        and (completion is None or completion < ROUTE_COMPLETION_GATE)
    )
    selected_motion_classes = Counter(
        str(candidate.get("motion_class") or "UNKNOWN")
        for candidate in candidates
    )
    selected_near_physical_statuses = Counter(
        str(candidate.get("near_physical_status") or "UNKNOWN")
        for candidate in candidates
    )
    candidate_near_physical_statuses = Counter()
    candidate_source_prior_statuses = Counter()
    reachability_profile_errors = 0
    reachability_compute_ms = []
    for candidate in all_candidate_events:
        profile = candidate.get("trajectory_reachability_profile")
        if isinstance(profile, dict):
            candidate_near_physical_statuses[
                str(profile.get("near_physical_status") or "UNKNOWN")
            ] += 1
            candidate_source_prior_statuses[
                str(
                    profile.get(
                        "near_source_speed_prior_status"
                    )
                    or "UNKNOWN"
                )
            ] += 1
        elif candidate.get("trajectory_reachability_error"):
            reachability_profile_errors += 1
        latency = _finite_float(
            candidate.get("trajectory_reachability_compute_ms")
        )
        if latency is not None:
            reachability_compute_ms.append(latency)
    governor_modes = Counter()
    for tick in ticks:
        debug = tick.get("controller_debug")
        governor = (
            debug.get("low_speed_longitudinal_governor")
            if isinstance(debug, dict)
            else None
        )
        if isinstance(governor, dict):
            governor_modes[str(governor.get("mode") or "UNKNOWN")] += 1
    inference_simulation_durations = []
    for event in events:
        if event.get("event_type") != "inference_result":
            continue
        source_time = _finite_float(
            event.get("source_simulation_time_s")
        )
        arrival_time = _finite_float(
            event.get("arrival_simulation_time_s")
        )
        if source_time is not None and arrival_time is not None:
            inference_simulation_durations.append(
                max(0.0, arrival_time - source_time)
            )
    max_inference_simulation_duration = max(
        inference_simulation_durations,
        default=None,
    )
    reachability_latency_p95 = (
        _nearest_rank_percentile(
            reachability_compute_ms,
            0.95,
        )
    )
    reachability_audit_enabled = bool(
        start.get("trajectory_reachability_audit", False)
    )
    ranking_policy = str(
        start.get("candidate_ranking_policy") or "current"
    )
    selected_plan_ids = {
        str(candidate.get("candidate_plan_id"))
        for candidate in candidates
        if candidate.get("candidate_plan_id") is not None
    }
    handoff_events = [
        event
        for event in events
        if event.get("event_type") == "plan_handoff"
        and event.get("candidate_plan_id") is not None
    ]
    selected_handoffs = [
        event
        for event in handoff_events
        if str(event.get("candidate_plan_id")) in selected_plan_ids
    ]
    handoff_statuses = Counter(
        str(event.get("status") or "UNKNOWN")
        for event in selected_handoffs
    )

    return {
        "run_id": start.get("run_id"),
        "scenario_seed": start.get("scenario_seed"),
        "stop_reason": episode_summary.get("stop_reason"),
        "tick_count": len(ticks),
        "route": {
            "length_m": route_length,
            "max_progress_m": max_progress,
            "completion_ratio": completion,
            "final_destination_error_m": final_error,
            "prompt_context_count": len(contexts),
            "prompt_truth_mismatch_count": prompt_mismatches,
            "route_unavailable_context_count": route_unavailable_samples,
            "accepted_unauthorized_prefix_count": unauthorized_acceptances,
            "route_only_emergency_override_ticks": route_only_overrides,
            "route_constraint_ticks": route_constraint_ticks,
        },
        "motion": {
            "integrated_distance_m": _integrated_distance(ticks),
            "final_speed_mps": final_speed,
            "stationary_suffix_ticks": stationary_suffix,
            "longest_stationary_streak_ticks": (
                longest_stationary_streak
            ),
            "stop_go_restart_count": stop_go_restarts,
            "hard_brake_ticks": hard_brake_ticks,
            "absorbing_stop": absorbing_stop,
        },
        "candidate_policy": {
            "ranking_policy": ranking_policy,
            "reachability_audit_enabled": (
                reachability_audit_enabled
            ),
            **selection_coverage,
            "selected_motion_class_counts": dict(
                sorted(selected_motion_classes.items())
            ),
            "selected_near_physical_status_counts": dict(
                sorted(selected_near_physical_statuses.items())
            ),
            "candidate_near_physical_status_counts": dict(
                sorted(candidate_near_physical_statuses.items())
            ),
            "candidate_near_source_prior_counts": dict(
                sorted(candidate_source_prior_statuses.items())
            ),
            "reachability_profile_error_count": (
                reachability_profile_errors
            ),
            "reachability_compute_p95_ms": (
                reachability_latency_p95
            ),
            "selected_handoff_count": len(selected_handoffs),
            "selected_activated_count": sum(
                event.get("activate_candidate") is True
                for event in selected_handoffs
            ),
            "selected_retained_active_count": sum(
                event.get("retain_active") is True
                for event in selected_handoffs
            ),
            "selected_handoff_status_counts": dict(
                sorted(handoff_statuses.items())
            ),
        },
        "controller": {
            "low_speed_governor_mode_counts": dict(
                sorted(governor_modes.items())
            ),
        },
        "synchronous_contract": {
            "execution": start.get("execution"),
            "inference_interval_count": len(
                inference_simulation_durations
            ),
            "maximum_inference_simulation_duration_s": (
                max_inference_simulation_duration
            ),
        },
        "safety": {
            "collisions": collisions,
            "current_ego_road_unsafe_or_unknown_ticks": ego_unsafe_ticks,
            "fallback_ticks": fallback_ticks,
            "direct_override_ticks": direct_overrides,
            "latch_only_ticks": latch_only_ticks,
        },
        "coc_audit": {
            "selected_candidate_audit_count": sum(
                isinstance(candidate.get("coc_semantic_audit"), dict)
                for candidate in candidates
            ),
            "verdict_counts": dict(sorted(audit_verdicts.items())),
            "positive_hallucination_count": positive_hallucinations,
            "audit_error_count": audit_errors,
        },
        "gates": {
            "collision_free": collisions == 0,
            "ego_road_safe": ego_unsafe_ticks == 0,
            "prompt_grounded": prompt_mismatches == 0,
            "route_authorization_strict": unauthorized_acceptances == 0,
            "no_route_emergency_override": route_only_overrides == 0,
            "no_absorbing_stop": not absorbing_stop,
            "route_completion_95_percent": (
                completion is not None and completion >= ROUTE_COMPLETION_GATE
            ),
            "destination_stop": (
                final_error is not None
                and final_error <= 1.5
                and final_speed is not None
                and final_speed <= STATIONARY_SPEED_MPS
            ),
            "synchronous_inference_paused": (
                start.get("execution") != "sync"
                or max_inference_simulation_duration in {None, 0.0}
            ),
            "reachability_profiles_complete": (
                not reachability_audit_enabled
                or reachability_profile_errors == 0
            ),
            "effective_near_selector_recall": (
                not reachability_audit_enabled
                or selection_coverage[
                    "effective_near_conditional_misses"
                ]
                == 0
            ),
            "effective_full_selector_recall": (
                not reachability_audit_enabled
                or selection_coverage[
                    "effective_full_conditional_misses"
                ]
                == 0
            ),
        },
    }


def _parse_destination(raw: str) -> tuple[float, float, float]:
    pieces = raw.split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("destination must use X,Y,Z")
    try:
        destination = tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("destination must use X,Y,Z") from exc
    if not all(math.isfinite(value) for value in destination):
        raise argparse.ArgumentTypeError("destination must be finite")
    return destination  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_jsonl", type=Path)
    parser.add_argument(
        "--destination",
        type=_parse_destination,
        default=FIXED_TOWN03_DESTINATION_XYZ,
        help="destination X,Y,Z (defaults to the fixed Town03 live gate)",
    )
    parser.add_argument(
        "--require-all-gates",
        action="store_true",
        help="return nonzero when any reported gate fails",
    )
    args = parser.parse_args()
    try:
        events = _read_events(args.telemetry_jsonl)
        summary = summarize_route_rollout(
            events,
            destination_xyz=args.destination,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_all_gates and not all(summary["gates"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from module.route_navigation import NavigationAction, format_navigation_prompt


STATIONARY_SPEED_MPS = 0.2
ABSORBING_STOP_TICKS = 50
ROUTE_COMPLETION_GATE = 0.95


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
    except (KeyError, TypeError, ValueError):
        return True
    expected = (
        ""
        if context.get("tracker_status") == "ROUTE_UNAVAILABLE"
        else format_navigation_prompt(action, distance)
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
    route_unavailable_samples = sum(
        context.get("tracker_status") == "ROUTE_UNAVAILABLE"
        for context in contexts
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
            "absorbing_stop": absorbing_stop,
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

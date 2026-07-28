#!/usr/bin/env python3
"""Summarize synchronous oracle-route experiment telemetry."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(payload)
    if not events:
        raise ValueError(f"{path}: telemetry is empty")
    return events


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested_status(event: dict[str, Any], *keys: str) -> str | None:
    value: Any = event
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return None if value is None else str(value)


def _stop_go_cycles(speeds: Sequence[float], *, target_speed_mps: float) -> int:
    moving_threshold = min(0.5, max(0.2, target_speed_mps * 0.4))
    state = "stopped"
    cycles = 0
    has_moved = False
    for speed in speeds:
        if state == "stopped" and speed >= moving_threshold:
            if has_moved:
                cycles += 1
            has_moved = True
            state = "moving"
        elif state == "moving" and speed <= 0.1:
            state = "stopped"
    return cycles


def summarize(events: Sequence[dict[str, Any]], *, source: str | None = None) -> dict[str, Any]:
    start = next(
        (event for event in events if event.get("event_type") == "episode_start"),
        {},
    )
    summary = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "episode_summary"
        ),
        {},
    )
    ticks = [event for event in events if event.get("event_type") == "tick"]
    target_speed = _finite(start.get("target_speed_mps")) or 0.0
    speeds = [
        speed
        for event in ticks
        if (speed := _finite(event.get("speed_mps"))) is not None
    ]
    settled_speeds = [
        speed
        for event in ticks
        if (_finite(event.get("simulation_time_s")) or 0.0)
        - (_finite(ticks[0].get("simulation_time_s")) or 0.0)
        >= 3.0
        and (speed := _finite(event.get("speed_mps"))) is not None
    ] if ticks else []
    progress = [
        value
        for event in ticks
        if (value := _finite(event.get("route_progress_m"))) is not None
    ]
    progress_regressions = sum(
        current + 1e-6 < previous
        for previous, current in zip(progress, progress[1:])
    )
    controller_states = Counter(
        str(event.get("controller_state") or "UNKNOWN") for event in ticks
    )
    route_near_statuses = Counter(
        status
        for event in ticks
        if (
            status := _nested_status(
                event,
                "route_candidate_assessment",
                "near_term_route_status",
            )
        )
        is not None
    )
    road_current_statuses = Counter(
        status
        for event in ticks
        if (
            status := _nested_status(
                event,
                "road_envelope",
                "current_ego_road",
                "status",
            )
        )
        is not None
    )
    override_ticks = sum(
        _nested_status(event, "safety_decision", "safety_override_applied")
        == "True"
        for event in ticks
    )
    collisions = max(
        (int(event.get("collision_count") or 0) for event in ticks),
        default=int(summary.get("collision_count") or 0),
    )
    fixture = summary.get("camera_fixture")
    return {
        "source": source,
        "execution": start.get("execution"),
        "trajectory_source": start.get("trajectory_source"),
        "scenario_seed": start.get("scenario_seed"),
        "target_speed_mps": target_speed,
        "stop_reason": summary.get("stop_reason"),
        "tick_count": len(ticks),
        "simulation_duration_s": _finite(summary.get("simulation_duration_s")),
        "integrated_distance_m": _finite(summary.get("integrated_distance_m")),
        "maximum_route_progress_m": (
            _finite(summary.get("maximum_route_progress_m"))
            if summary.get("maximum_route_progress_m") is not None
            else max(progress, default=0.0)
        ),
        "progress_regression_count": progress_regressions,
        "mean_speed_mps": fmean(speeds) if speeds else None,
        "peak_speed_mps": max(speeds) if speeds else None,
        "settled_speed_mae_mps": (
            fmean(abs(speed - target_speed) for speed in settled_speeds)
            if settled_speeds
            else None
        ),
        "stop_go_cycle_count": _stop_go_cycles(
            speeds,
            target_speed_mps=target_speed,
        ),
        "controller_state_counts": dict(sorted(controller_states.items())),
        "current_ego_road_status_counts": dict(sorted(road_current_statuses.items())),
        "near_term_route_status_counts": dict(sorted(route_near_statuses.items())),
        "safety_override_ticks": override_ticks,
        "collision_count": collisions,
        "fixture_captured": isinstance(fixture, dict),
        "fixture_id": fixture.get("fixture_id") if isinstance(fixture, dict) else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = [
        summarize(_read_events(path), source=str(path))
        for path in args.runtime_jsonl
    ]
    payload: Any = results[0] if len(results) == 1 else results
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

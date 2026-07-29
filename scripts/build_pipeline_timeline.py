#!/usr/bin/env python3
"""Build a dual-clock CarlaMayo pipeline timeline from runtime JSONL.

The CSV is the machine-readable source of truth.  The optional PNG keeps wall
time and CARLA simulation time on separate panels so blocking SYNC inference is
visible without being mistaken for simulated plan age.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TimelineRow:
    event_type: str
    lane: str
    wall_time_s: float | None
    simulation_time_s: float | None
    wall_duration_s: float | None = None
    simulation_duration_s: float | None = None
    loop_tick_id: int | None = None
    source_loop_tick_id: int | None = None
    request_id: int | None = None
    proposal_id: str | None = None
    active_plan_id: str | None = None
    controller_state: str | None = None
    speed_mps: float | None = None
    target_speed_mps: float | None = None
    throttle: float | None = None
    brake: float | None = None
    steering: float | None = None
    gear: int | None = None
    route_status: str | None = None
    route_progress_m: float | None = None
    detail: str | None = None


_LANES = {
    "tick": "CARLA / CONTROL",
    "inference_submitted": "ALPAMAYO",
    "inference_result": "ALPAMAYO",
    "candidate_selection": "ROAD / ROUTE SELECTOR",
    "plan_handoff": "PLAN HANDOFF",
    "prompt_revision_changed": "NAVIGATION",
    "route_unavailable": "NAVIGATION",
    "route_replan": "NAVIGATION",
    "safety_decision": "SAFETY",
    "control": "CARLA / CONTROL",
}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first(event: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if event.get(key) is not None:
            return event[key]
    return None


def _simulation_time(event: dict[str, Any]) -> float | None:
    return _finite_float(
        _first(
            event,
            "simulation_time_s",
            "decision_simulation_time_s",
            "arrival_simulation_time_s",
            "source_simulation_time_s",
        )
    )


def _route_fields(event: dict[str, Any]) -> tuple[str | None, float | None]:
    context = event.get("navigation_context")
    if not isinstance(context, dict):
        context = {}
    status = _first(
        event,
        "route_status",
        "near_term_route_status",
    )
    if status is None:
        status = context.get("tracker_status")
    progress = _finite_float(
        _first(event, "route_progress_m", "maximum_route_progress_m")
    )
    if progress is None:
        progress = _finite_float(context.get("route_progress_m"))
    return (None if status is None else str(status), progress)


def _detail(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "")
    if event_type == "candidate_selection":
        return (
            f"selected={event.get('selected_candidate_index')};"
            f"compute_ms={event.get('selection_compute_ms')};"
            f"road_ms={event.get('road_batch_wall_ms')}"
        )
    if event_type == "plan_handoff":
        return str(_first(event, "status", "handoff_status", "reason") or "")
    if event_type == "prompt_revision_changed":
        context = event.get("navigation_context")
        if isinstance(context, dict):
            return str(context.get("text") or event.get("reason") or "")
    if event_type == "route_replan":
        return str(_first(event, "status", "reason") or "")
    if event_type == "inference_result":
        return str(_first(event, "status", "rejection_reason") or "")
    return None


def build_timeline_rows(events: Sequence[dict[str, Any]]) -> list[TimelineRow]:
    """Normalize runtime events and add one interval per inference request."""

    rows: list[TimelineRow] = []
    submissions: dict[int, dict[str, Any]] = {}
    for event in events:
        event_type = str(event.get("event_type") or "")
        if not event_type:
            continue
        request_id = _integer(event.get("request_id"))
        if event_type == "inference_submitted" and request_id is not None:
            submissions[request_id] = event

        controller_debug = event.get("controller_debug")
        if not isinstance(controller_debug, dict):
            controller_debug = {}
        applied = event.get("applied_control")
        if not isinstance(applied, dict):
            applied = {}
        echoed = event.get("echoed_control")
        if not isinstance(echoed, dict):
            echoed = {}
        route_status, route_progress = _route_fields(event)
        row = TimelineRow(
            event_type=event_type,
            lane=_LANES.get(event_type, "OTHER"),
            wall_time_s=_finite_float(event.get("wall_elapsed_s")),
            simulation_time_s=_simulation_time(event),
            loop_tick_id=_integer(
                _first(event, "loop_tick_id", "arrival_loop_tick_id")
            ),
            source_loop_tick_id=_integer(event.get("source_loop_tick_id")),
            request_id=request_id,
            proposal_id=(
                None
                if event.get("proposal_id") is None
                else str(event.get("proposal_id"))
            ),
            active_plan_id=(
                None
                if event.get("active_plan_id") is None
                else str(event.get("active_plan_id"))
            ),
            controller_state=(
                None
                if event.get("controller_state") is None
                else str(event.get("controller_state"))
            ),
            speed_mps=_finite_float(event.get("speed_mps")),
            target_speed_mps=_finite_float(
                _first(event, "target_speed_mps")
                if event.get("target_speed_mps") is not None
                else controller_debug.get("target_speed_mps")
            ),
            throttle=_finite_float(applied.get("throttle")),
            brake=_finite_float(applied.get("brake")),
            steering=_finite_float(applied.get("steering")),
            gear=_integer(_first(echoed, "gear")),
            route_status=route_status,
            route_progress_m=route_progress,
            detail=_detail(event),
        )
        rows.append(row)

        if event_type == "inference_result" and request_id in submissions:
            submitted = submissions[request_id]
            wall_start = _finite_float(submitted.get("wall_elapsed_s"))
            wall_end = _finite_float(event.get("wall_elapsed_s"))
            sim_start = _finite_float(submitted.get("source_simulation_time_s"))
            sim_end = _finite_float(
                _first(event, "arrival_simulation_time_s", "source_simulation_time_s")
            )
            rows.append(
                TimelineRow(
                    event_type="inference_interval",
                    lane="ALPAMAYO",
                    wall_time_s=wall_start,
                    simulation_time_s=sim_start,
                    wall_duration_s=(
                        max(0.0, wall_end - wall_start)
                        if wall_start is not None and wall_end is not None
                        else None
                    ),
                    simulation_duration_s=(
                        max(0.0, sim_end - sim_start)
                        if sim_start is not None and sim_end is not None
                        else None
                    ),
                    source_loop_tick_id=_integer(
                        submitted.get("source_loop_tick_id")
                    ),
                    request_id=request_id,
                    detail=(
                        f"model_s={event.get('model_inference_latency_s')};"
                        f"processing_ms={event.get('result_processing_ms')}"
                    ),
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            math.inf if row.wall_time_s is None else row.wall_time_s,
            row.event_type != "inference_interval",
        ),
    )


def read_runtime_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(payload)
    return events


def write_timeline_csv(rows: Iterable[TimelineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(TimelineRow)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in names})


def render_timeline_png(rows: Sequence[TimelineRow], path: Path) -> None:
    """Render compact wall/simulation panels; matplotlib is optional."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PNG output requires matplotlib") from exc

    lane_order = (
        "NAVIGATION",
        "ALPAMAYO",
        "ROAD / ROUTE SELECTOR",
        "PLAN HANDOFF",
        "CARLA / CONTROL",
        "SAFETY",
    )
    lane_y = {lane: index for index, lane in enumerate(reversed(lane_order))}
    colors = {
        "NAVIGATION": "#377eb8",
        "ALPAMAYO": "#984ea3",
        "ROAD / ROUTE SELECTOR": "#ff7f00",
        "PLAN HANDOFF": "#a65628",
        "CARLA / CONTROL": "#4daf4a",
        "SAFETY": "#e41a1c",
    }
    figure, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    for axis, clock, duration, title in (
        (axes[0], "wall_time_s", "wall_duration_s", "Wall clock"),
        (
            axes[1],
            "simulation_time_s",
            "simulation_duration_s",
            "CARLA simulation clock",
        ),
    ):
        for row in rows:
            if row.lane not in lane_y:
                continue
            x = getattr(row, clock)
            if x is None:
                continue
            width = getattr(row, duration)
            y = lane_y[row.lane]
            color = colors[row.lane]
            if row.event_type == "inference_interval" and width is not None:
                axis.broken_barh([(x, max(width, 1e-6))], (y - 0.28, 0.56), facecolors=color)
            else:
                axis.scatter([x], [y], s=8, color=color, alpha=0.65)
        axis.set_yticks(
            [lane_y[lane] for lane in lane_order],
            labels=list(lane_order),
        )
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel("seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_jsonl", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    events = read_runtime_events(args.runtime_jsonl)
    rows = build_timeline_rows(events)
    write_timeline_csv(rows, args.csv)
    if args.png is not None:
        render_timeline_png(rows, args.png)
    print(
        json.dumps(
            {
                "event_count": len(events),
                "timeline_row_count": len(rows),
                "csv": str(args.csv),
                "png": None if args.png is None else str(args.png),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

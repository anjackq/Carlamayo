import csv
from pathlib import Path

import pytest

from scripts.build_pipeline_timeline import (
    build_timeline_rows,
    write_timeline_csv,
)


def _events():
    return [
        {
            "event_type": "inference_submitted",
            "request_id": 7,
            "source_loop_tick_id": 20,
            "source_simulation_time_s": 12.0,
            "wall_elapsed_s": 4.0,
        },
        {
            "event_type": "candidate_selection",
            "request_id": 7,
            "proposal_id": "run:7",
            "source_simulation_time_s": 12.0,
            "wall_elapsed_s": 5.3,
            "selected_candidate_index": 2,
            "selection_compute_ms": 200.0,
            "road_batch_wall_ms": 150.0,
        },
        {
            "event_type": "inference_result",
            "request_id": 7,
            "source_loop_tick_id": 20,
            "arrival_loop_tick_id": 20,
            "source_simulation_time_s": 12.0,
            "arrival_simulation_time_s": 12.0,
            "wall_elapsed_s": 5.5,
            "model_inference_latency_s": 1.3,
            "result_processing_ms": 200.0,
            "status": "accepted_plan",
        },
        {
            "event_type": "tick",
            "loop_tick_id": 20,
            "simulation_time_s": 12.0,
            "wall_elapsed_s": 5.6,
            "active_plan_id": "run:7/candidate-2",
            "controller_state": "TRACKING",
            "speed_mps": 1.5,
            "controller_debug": {"target_speed_mps": 2.0},
            "applied_control": {
                "throttle": 0.4,
                "brake": 0.0,
                "steering": 0.1,
            },
            "echoed_control": {"gear": 1},
            "navigation_context": {
                "tracker_status": "AVAILABLE",
                "route_progress_m": 8.0,
            },
        },
    ]


def test_timeline_separates_wall_and_simulation_inference_duration():
    rows = build_timeline_rows(_events())
    interval = next(row for row in rows if row.event_type == "inference_interval")

    assert interval.wall_time_s == pytest.approx(4.0)
    assert interval.wall_duration_s == pytest.approx(1.5)
    assert interval.simulation_time_s == pytest.approx(12.0)
    assert interval.simulation_duration_s == pytest.approx(0.0)


def test_timeline_extracts_control_and_route_fields():
    tick = next(row for row in build_timeline_rows(_events()) if row.event_type == "tick")

    assert tick.lane == "CARLA / CONTROL"
    assert tick.active_plan_id == "run:7/candidate-2"
    assert tick.target_speed_mps == pytest.approx(2.0)
    assert tick.throttle == pytest.approx(0.4)
    assert tick.gear == 1
    assert tick.route_status == "AVAILABLE"
    assert tick.route_progress_m == pytest.approx(8.0)


def test_timeline_csv_is_machine_readable(tmp_path: Path):
    output = tmp_path / "timeline.csv"
    rows = build_timeline_rows(_events())

    write_timeline_csv(rows, output)

    with output.open(newline="", encoding="utf-8") as stream:
        parsed = list(csv.DictReader(stream))
    assert len(parsed) == len(rows)
    assert {row["event_type"] for row in parsed} >= {
        "inference_interval",
        "tick",
    }

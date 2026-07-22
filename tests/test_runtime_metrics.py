import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from module.runtime_metrics import JsonlWriter, RuntimeMetrics, to_json_safe


class ControllerState(Enum):
    TRACKING = "tracking"


class ArrayLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class OpaqueImage:
    shape = (4, 1080, 1920, 3)

    def tolist(self):
        raise AssertionError("opaque camera pixels must not be serialized")


@dataclass(frozen=True)
class Decision:
    frame_id: int
    state: ControllerState
    target: ArrayLike
    score: ScalarLike


@dataclass(frozen=True)
class ObservationRecord:
    frame_id: int
    camera_images: OpaqueImage

    def to_json_dict(self):
        return {
            "frame_id": self.frame_id,
            "camera_shape": self.camera_images.shape,
        }


def test_to_json_safe_recurses_without_numpy_dependency(tmp_path):
    decision = Decision(
        frame_id=12,
        state=ControllerState.TRACKING,
        target=ArrayLike([[1.0, 2.0], [3.0, float("nan")]]),
        score=ScalarLike(0.75),
    )

    converted = to_json_safe(
        {
            "decision": decision,
            "output": Path(tmp_path / "events.jsonl"),
            "tags": {"control", "tracking"},
        }
    )

    assert converted["decision"] == {
        "frame_id": 12,
        "state": "tracking",
        "target": [[1.0, 2.0], [3.0, None]],
        "score": 0.75,
    }
    assert converted["output"] == str(tmp_path / "events.jsonl")
    assert converted["tags"] == ["control", "tracking"]
    json.dumps(converted, allow_nan=False)


def test_to_json_safe_prefers_record_summary_over_dataclass_expansion():
    record = ObservationRecord(frame_id=9, camera_images=OpaqueImage())

    assert to_json_safe(record) == {
        "frame_id": 9,
        "camera_shape": [4, 1080, 1920, 3],
    }


def test_jsonl_writer_is_append_only_and_context_managed(tmp_path):
    output = tmp_path / "nested" / "events.jsonl"
    output.parent.mkdir()
    output.write_text('{"event_type":"existing"}\n', encoding="utf-8")

    writer = JsonlWriter(output)
    with writer as active_writer:
        active_writer.write_event(
            "control",
            Decision(4, ControllerState.TRACKING, ArrayLike([1, 2]), ScalarLike(0.5)),
            source_age_s=ScalarLike(0.2),
        )
        assert not writer.closed

    assert writer.closed
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0] == {"event_type": "existing"}
    assert records[1]["event_type"] == "control"
    assert records[1]["state"] == "tracking"
    assert records[1]["source_age_s"] == 0.2


def test_runtime_metrics_aggregates_percentiles_and_event_outcomes():
    clock_values = iter([100.0, 106.5])
    metrics = RuntimeMetrics(max_samples=100, clock=lambda: next(clock_values))

    for index in range(1, 101):
        metrics.record_event(
            "inference",
            inference_latency_s=float(index),
            source_age_s=float(index) / 10.0,
            source_age_proxy_s=float(index) / 5.0,
            simulation_time_s=10.0 + index / 10.0,
        )
    metrics.record_event("validation", valid=False, rejection_reason="expired")
    metrics.record_event(
        "control",
        safety_override_applied=True,
        safety_override_reason="obstacle",
        fallback_state="emergency_brake",
        fallback_reason="no_valid_plan",
    )

    summary = metrics.final_summary(episode_id="town03-001")

    assert summary["episode_id"] == "town03-001"
    assert summary["wall_duration_s"] == pytest.approx(6.5)
    assert summary["simulation_duration_s"] == pytest.approx(9.9)
    assert summary["event_counts"] == {"control": 1, "inference": 100, "validation": 1}
    assert summary["inference_latency_s"]["count"] == 100
    assert summary["inference_latency_s"]["p50"] == pytest.approx(50.5)
    assert summary["inference_latency_s"]["p95"] == pytest.approx(95.05)
    assert summary["inference_latency_s"]["p99"] == pytest.approx(99.01)
    assert summary["source_age_s"]["p50"] == pytest.approx(5.05)
    assert summary["source_age_proxy_s"]["p50"] == pytest.approx(10.1)
    assert summary["rejections"] == {"total": 1, "reasons": {"expired": 1}}
    assert summary["safety_overrides"] == {"total": 1, "reasons": {"obstacle": 1}}
    assert summary["fallbacks"] == {"total": 1, "reasons": {"no_valid_plan": 1}}


def test_collision_total_survives_per_ego_counter_reset():
    metrics = RuntimeMetrics()

    for count in (0, 1, 2, 0, 1, 3):
        metrics.record_event("tick", collision_count=count)

    assert metrics.final_summary()["collisions"]["total"] == 5


@pytest.mark.parametrize("value", [True, 1.5, "2", -1])
def test_collision_counters_require_exact_nonnegative_integers(value):
    metrics = RuntimeMetrics()

    with pytest.raises((TypeError, ValueError)):
        metrics.record_collision_count(value)


def test_simulation_duration_uses_minimum_and_maximum_event_times():
    clock_values = iter([0.0, 1.0])
    metrics = RuntimeMetrics(clock=lambda: next(clock_values))
    for simulation_time_s in (5.0, 3.0, 9.0, -1.0):
        metrics.record_event("tick", simulation_time_s=simulation_time_s)

    assert metrics.final_summary()["simulation_duration_s"] == pytest.approx(6.0)


def test_summary_counts_event_statuses_and_controller_states():
    metrics = RuntimeMetrics()
    metrics.record_event("inference_result", status="accepted_plan")
    metrics.record_event("inference_result", status="error")
    metrics.record_event("tick", controller_state="TRACKING")
    metrics.record_event("tick", controller_state="TRACKING")

    summary = metrics.final_summary()

    assert summary["event_status_counts"] == {
        "inference_result.accepted_plan": 1,
        "inference_result.error": 1,
    }
    assert summary["controller_states"] == {"TRACKING": 2}


def test_aggregation_memory_and_category_cardinality_are_bounded():
    metrics = RuntimeMetrics(max_samples=7, max_categories=3, reservoir_seed=3)

    for index in range(1_000):
        metrics.record_event(
            f"event-{index}",
            latency_s=index,
            age_s=index / 10,
            rejected=True,
            rejection_reason=f"reason-{index}",
        )

    summary = metrics.final_summary()

    assert summary["inference_latency_s"]["count"] == 1_000
    assert summary["inference_latency_s"]["sample_count"] == 7
    assert summary["source_age_s"]["sample_count"] == 7
    assert len(summary["event_counts"]) <= 4  # three retained keys plus overflow
    assert len(summary["rejections"]["reasons"]) <= 4
    assert summary["event_counts"]["__other__"] == 997
    assert summary["rejections"]["reasons"]["__other__"] == 997


def test_final_summary_can_be_written_as_last_jsonl_event(tmp_path):
    output = tmp_path / "episode.jsonl"
    metrics = RuntimeMetrics()
    metrics.record_event("collision")

    with JsonlWriter(output) as writer:
        writer.write_event("collision", frame_id=42)
        written = metrics.write_final_summary(writer, ended_reason="time_limit")

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[-1] == written
    assert records[-1]["event_type"] == "episode_summary"
    assert records[-1]["ended_reason"] == "time_limit"
    assert records[-1]["collisions"]["total"] == 1

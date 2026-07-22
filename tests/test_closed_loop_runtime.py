import importlib
import json
import queue as stdlib_queue
import sys
import threading
import types
from collections import Counter

import pytest


sys.modules.setdefault(
    "carla",
    types.SimpleNamespace(
        command=types.SimpleNamespace(DestroyActor=lambda actor_id: ("destroy", actor_id)),
        VehicleControl=lambda: types.SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0),
        Vector3D=lambda: types.SimpleNamespace(),
    ),
)
closed_loop = importlib.import_module("carlamayo_closed_loop")


class _FakeWorld:
    def __init__(self, fixed_delta_seconds):
        self._settings = types.SimpleNamespace(fixed_delta_seconds=float(fixed_delta_seconds))

    def get_settings(self):
        return self._settings


class _FakeCarlaInterface:
    def __init__(self, *, fixed_delta_seconds=0.1, on_tick=None, fail_on_tick=None):
        self.world = _FakeWorld(fixed_delta_seconds)
        self.ego_vehicle = object()
        self.tick_count = 0
        self.cleanup_count = 0
        self.applied_controls = []
        self._on_tick = on_tick
        self._fail_on_tick = fail_on_tick

    def connect(self):
        pass

    def load_map(self, _map_name):
        pass

    def spawn_ego_vehicle(self):
        pass

    def enable_synchronous_mode(self):
        pass

    def spawn_npcs(self, **_kwargs):
        pass

    def setup_cameras(self):
        pass

    def setup_collision_sensor(self):
        pass

    def tick(self):
        next_tick = self.tick_count + 1
        if self._fail_on_tick == next_tick:
            raise RuntimeError(f"synthetic tick failure {next_tick}")
        self.tick_count = next_tick
        if self._on_tick is not None:
            self._on_tick(self.tick_count)

    def get_ego_state(self):
        return {"speed": 0.0}

    def update_history(self, _state):
        pass

    def get_collision_count(self):
        return 0

    def get_episode_collision_count(self):
        return 0

    def get_last_collision_event(self):
        return None

    def get_camera_images(self):
        return [closed_loop.np.zeros((1, 1, 3), dtype=closed_loop.np.uint8)]

    def get_history_in_local_frame(self):
        return object(), object()

    def apply_control(self, steering, throttle, brake):
        self.applied_controls.append((steering, throttle, brake))

    def cleanup(self):
        self.cleanup_count += 1


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_stream_summary_invariants(records):
    assert records
    assert records[-1]["event_type"] == "episode_summary"
    assert sum(record["event_type"] == "episode_summary" for record in records) == 1

    run_ids = {record["run_id"] for record in records}
    assert len(run_ids) == 1
    for record in records:
        assert record["schema_version"] == 1
        assert isinstance(record["event_type"], str)
        json.dumps(record, allow_nan=False)

    events = records[:-1]
    summary = records[-1]
    expected_counts = Counter(record["event_type"] for record in events)
    assert summary["total_events"] == len(events)
    assert summary["event_counts"] == dict(sorted(expected_counts.items()))


def _install_common_fakes(monkeypatch, args, carla_if, *, num_frames):
    monkeypatch.setattr(closed_loop, "parse_args", lambda: args)
    monkeypatch.setattr(closed_loop, "configure_cuda_linalg_library", lambda _name: None)
    monkeypatch.setattr(
        closed_loop,
        "load_model",
        lambda _quantization, device_map: (object(), object()),
    )
    monkeypatch.setattr(closed_loop, "CARLAInterface", lambda: carla_if)
    monkeypatch.setattr(closed_loop, "OfficialPIDFollower", lambda *_args: object())
    monkeypatch.setattr(closed_loop, "prepare_model_input", lambda *_args: object())
    monkeypatch.setattr(closed_loop, "run_vqa", lambda *_args, **_kwargs: {"answer": "ok"})
    monkeypatch.setattr(
        closed_loop,
        "extract_answer_text",
        lambda extra: extra.get("answer", "") if isinstance(extra, dict) else "",
    )
    monkeypatch.setattr(closed_loop.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(closed_loop.time, "sleep", lambda _seconds: None)

    monkeypatch.setattr(closed_loop.cfg, "SAVE_VIDEO", False)
    monkeypatch.setattr(closed_loop.cfg, "CONTROL_DT", 0.1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_FRAMES", num_frames)
    monkeypatch.setattr(closed_loop.cfg, "NUM_CAMERAS", 1)
    monkeypatch.setattr(closed_loop.cfg, "IMG_HEIGHT", 1)
    monkeypatch.setattr(closed_loop.cfg, "IMG_WIDTH", 1)
    monkeypatch.setattr(closed_loop.cfg, "IMG_CHANNELS", 3)
    monkeypatch.setattr(closed_loop.cfg, "NPC_VEHICLE_COUNT", 0)
    monkeypatch.setattr(closed_loop.cfg, "NPC_WALKER_COUNT", 0)


@pytest.mark.parametrize(
    ("episode_limit_s", "expected_ticks", "expected_duration_s"),
    [
        (0.05, 1, 0.1),
        (0.2, 2, 0.2),
        (0.25, 3, 0.3),
    ],
)
def test_episode_limit_stops_on_first_successful_tick_boundary(
    monkeypatch,
    tmp_path,
    episode_limit_s,
    expected_ticks,
    expected_duration_s,
):
    telemetry_path = tmp_path / "episode.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            str(episode_limit_s),
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=100)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    summary = records[-1]
    assert carla_if.tick_count == expected_ticks
    assert carla_if.cleanup_count == 1
    assert summary["stop_reason"] == "max_episode_seconds"
    assert summary["loop_tick_count"] == expected_ticks
    assert summary["simulation_duration_proxy_s"] == pytest.approx(expected_duration_s)
    assert summary["event_counts"]["tick"] == expected_ticks
    _assert_stream_summary_invariants(records)


def test_async_request_has_one_correlated_terminal_result_and_valid_jsonl(
    monkeypatch,
    tmp_path,
):
    result_enqueued = threading.Event()
    original_queue_class = stdlib_queue.Queue
    queue_number = 0

    class _ResultQueue(original_queue_class):
        def put_nowait(self, item):
            result = super().put_nowait(item)
            if isinstance(item, dict) and item.get("request_id") is not None:
                result_enqueued.set()
            return result

    def queue_factory(*args, **kwargs):
        nonlocal queue_number
        queue_number += 1
        queue_class = _ResultQueue if queue_number == 2 else original_queue_class
        return queue_class(*args, **kwargs)

    def wait_for_result_on_second_tick(tick_count):
        if tick_count == 2:
            assert result_enqueued.wait(timeout=2.0), "async worker did not enqueue a result"

    telemetry_path = tmp_path / "async.jsonl"
    args = closed_loop.parse_args(
        [
            "--async",
            "--mode",
            "vqa",
            "--vqa-question",
            "What is ahead?",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.2",
        ]
    )
    carla_if = _FakeCarlaInterface(
        fixed_delta_seconds=0.1,
        on_tick=wait_for_result_on_second_tick,
    )
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    monkeypatch.setattr(closed_loop.queue, "Queue", queue_factory)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    submitted = [record for record in records if record["event_type"] == "inference_submitted"]
    terminal = [record for record in records if record["event_type"] == "inference_result"]
    assert Counter(record["request_id"] for record in submitted) == Counter(
        record["request_id"] for record in terminal
    )
    assert len(submitted) == len(terminal) == 1
    assert terminal[0]["status"] == "completed_vqa"
    assert terminal[0]["rejected"] is False
    assert terminal[0]["source_loop_tick_id"] == submitted[0]["source_loop_tick_id"]
    assert terminal[0]["arrival_loop_tick_id"] >= terminal[0]["source_loop_tick_id"]
    assert terminal[0]["inference_wall_latency_s"] >= 0.0

    ticks = [record for record in records if record["event_type"] == "tick"]
    assert len(ticks) == 2
    for tick in ticks:
        assert {
            "loop_tick_id",
            "controller_state",
            "requested_control",
            "applied_control",
            "collision_count",
        } <= tick.keys()

    summary = records[-1]
    assert summary["event_status_counts"] == {"inference_result.completed_vqa": 1}
    assert summary["inference_latency_s"]["count"] == 1
    _assert_stream_summary_invariants(records)


def test_episode_end_terminalizes_a_queued_async_request_once(monkeypatch, tmp_path):
    class _DormantThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    telemetry_path = tmp_path / "cancelled.jsonl"
    args = closed_loop.parse_args(
        [
            "--async",
            "--mode",
            "vqa",
            "--vqa-question",
            "What is ahead?",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    monkeypatch.setattr(closed_loop.threading, "Thread", _DormantThread)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    submitted = [record for record in records if record["event_type"] == "inference_submitted"]
    terminal = [record for record in records if record["event_type"] == "inference_result"]
    assert len(submitted) == len(terminal) == 1
    assert terminal[0]["request_id"] == submitted[0]["request_id"]
    assert terminal[0]["status"] == "cancelled"
    assert terminal[0]["rejection_reason"] == "request_queue_cleared:episode_end"
    assert terminal[0]["inference_wall_latency_s"] is None
    assert terminal[0]["request_lifetime_s"] >= 0.0

    summary = records[-1]
    assert summary["event_status_counts"] == {"inference_result.cancelled": 1}
    assert summary["rejections"] == {
        "total": 1,
        "reasons": {"request_queue_cleared:episode_end": 1},
    }
    assert summary["inference_latency_s"]["count"] == 0
    _assert_stream_summary_invariants(records)


def test_runtime_error_is_recorded_before_the_final_summary(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "error.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.2",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1, fail_on_tick=1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=100)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    assert [record["event_type"] for record in records] == [
        "episode_start",
        "runtime_error",
        "episode_summary",
    ]
    summary = records[-1]
    assert summary["stop_reason"] == "error"
    assert summary["error"] == "synthetic tick failure 1"
    assert summary["loop_tick_count"] == 0
    assert carla_if.cleanup_count == 1
    _assert_stream_summary_invariants(records)

import importlib
import json
import queue as stdlib_queue
import sys
import threading
import types
from collections import Counter

import pytest

from module.safety_shield import ObstacleAssessment, RoadContainmentAssessment


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


class _FakeEgoVehicle:
    def __init__(self):
        self.transform = object()

    def get_transform(self):
        return self.transform


class _FakeCarlaInterface:
    def __init__(self, *, fixed_delta_seconds=0.1, on_tick=None, fail_on_tick=None):
        self.world = _FakeWorld(fixed_delta_seconds)
        self.ego_vehicle = _FakeEgoVehicle()
        self.tick_count = 0
        self.cleanup_count = 0
        self.applied_controls = []
        self.loaded_maps = []
        self.scenario_seeds = []
        self.ego_spawn_requests = []
        self.npc_spawn_requests = []
        self.setup_cameras_count = 0
        self.setup_collision_sensor_count = 0
        self._on_tick = on_tick
        self._fail_on_tick = fail_on_tick

    def connect(self):
        pass

    def load_map(self, map_name, *, force_reload=False):
        self.loaded_maps.append((map_name, force_reload))

    def set_scenario_seed(self, seed):
        self.scenario_seeds.append(seed)

    def spawn_ego_vehicle(
        self,
        *,
        spawn_index=None,
        center_on_driving_lane=False,
    ):
        self.ego_spawn_requests.append(
            {
                "spawn_index": spawn_index,
                "center_on_driving_lane": center_on_driving_lane,
            }
        )

    def enable_synchronous_mode(self):
        pass

    def spawn_npcs(self, **kwargs):
        self.npc_spawn_requests.append(dict(kwargs))

    def get_non_ego_dynamic_actor_census(self):
        return {
            "non_ego_vehicle_count": 0,
            "walker_count": 0,
            "walker_controller_count": 0,
        }

    def setup_cameras(self):
        self.setup_cameras_count += 1

    def setup_collision_sensor(self):
        self.setup_collision_sensor_count += 1

    def tick(self):
        next_tick = self.tick_count + 1
        if self._fail_on_tick == next_tick:
            raise RuntimeError(f"synthetic tick failure {next_tick}")
        self.tick_count = next_tick
        if self._on_tick is not None:
            self._on_tick(self.tick_count)
        return types.SimpleNamespace()

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
    monkeypatch.setattr(
        closed_loop,
        "prepare_model_input",
        lambda *_args, **_kwargs: object(),
    )
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


def test_empty_road_setup_forces_fresh_map_fixed_spawn_and_zero_npcs(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "empty-road.jsonl"
    args = closed_loop.parse_args(
        [
            "--empty-road",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=100)
    rng_events = []
    monkeypatch.setattr(
        closed_loop,
        "seed_runtime_randomness",
        lambda seed: rng_events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        closed_loop,
        "load_model",
        lambda _quantization, device_map: (
            rng_events.append(("load_model", device_map)) or object(),
            object(),
        ),
    )
    preflight = RoadContainmentAssessment.safe(
        sample_count=5,
        min_margin_m=0.42,
        quality="carla_ground_truth_spawn_preflight",
    )
    preflight_transforms = []
    safety_adapter = types.SimpleNamespace(
        assess_ego_transform=lambda transform: (
            preflight_transforms.append(transform) or preflight
        )
    )
    monkeypatch.setattr(
        closed_loop,
        "CarlaGroundTruthSafetyAdapter",
        lambda world, ego_vehicle, policy: safety_adapter,
    )

    closed_loop.main()

    assert rng_events == [
        ("seed", closed_loop.cfg.EMPTY_ROAD_SCENARIO_SEED),
        ("load_model", "auto"),
        ("seed", closed_loop.cfg.EMPTY_ROAD_SCENARIO_SEED),
    ]
    assert carla_if.loaded_maps == [(closed_loop.cfg.CARLA_MAP, True)]
    assert carla_if.scenario_seeds == [closed_loop.cfg.EMPTY_ROAD_SCENARIO_SEED]
    assert carla_if.ego_spawn_requests == [
        {
            "spawn_index": closed_loop.cfg.EMPTY_ROAD_EGO_SPAWN_INDEX,
            "center_on_driving_lane": True,
        }
    ]
    assert carla_if.npc_spawn_requests == [
        {"num_vehicles": 0, "num_walkers": 0}
    ]
    assert preflight_transforms == [carla_if.ego_vehicle.transform]

    records = _read_jsonl(telemetry_path)
    episode_start = next(
        record for record in records if record["event_type"] == "episode_start"
    )
    assert episode_start["scenario"] == "empty_road"
    assert episode_start["scenario_seed"] == 0
    assert episode_start["ego_spawn_index"] == 0
    assert episode_start["requested_npc_vehicle_count"] == 0
    assert episode_start["requested_npc_walker_count"] == 0
    assert episode_start["empty_road_footprint_preflight"] == preflight.to_json_dict()
    _assert_stream_summary_invariants(records)


def test_unsafe_empty_road_footprint_aborts_before_oom_free_model_load(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "unsafe-empty-road.jsonl"
    args = closed_loop.parse_args(
        [
            "--empty-road",
            "--oom-free",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=100)
    preflight = RoadContainmentAssessment.unsafe(
        (
            "road_not_contained",
            "footprint_corner_off_driving_lane",
        ),
        sample_count=5,
        min_margin_m=-0.52,
        first_bad_sample_index=2,
        quality="carla_ground_truth_spawn_preflight",
    )
    monkeypatch.setattr(
        closed_loop,
        "CarlaGroundTruthSafetyAdapter",
        lambda world, ego_vehicle, policy: types.SimpleNamespace(
            assess_ego_transform=lambda transform: preflight
        ),
    )
    model_loads = []
    monkeypatch.setitem(
        sys.modules,
        "module.oom_offload",
        types.SimpleNamespace(
            load_offloaded_model=lambda **_kwargs: (
                model_loads.append("loaded") or object(),
                object(),
            )
        ),
    )

    closed_loop.main()

    assert model_loads == []
    assert carla_if.setup_cameras_count == 0
    assert carla_if.setup_collision_sensor_count == 0
    assert carla_if.tick_count == 0
    assert carla_if.cleanup_count == 1
    assert carla_if.applied_controls[-1] == (0.0, 0.0, 1.0)

    records = _read_jsonl(telemetry_path)
    runtime_error = next(
        record for record in records if record["event_type"] == "runtime_error"
    )
    assert "empty-road ego footprint is not safely contained" in runtime_error["error"]
    assert not any(record["event_type"] == "episode_start" for record in records)
    summary = records[-1]
    assert summary["stop_reason"] == "error"
    assert "empty-road ego footprint is not safely contained" in summary["error"]
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
    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
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
    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
    terminal = [
        record for record in records if record["event_type"] == "inference_result"
    ]
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


def test_runtime_error_after_nominal_throttle_applies_final_brake_and_returns_one(
    monkeypatch,
    tmp_path,
):
    class _TrackingFollower:
        def compute_world_control(self, **_kwargs):
            return 0.0, 0.6, 0.0, {
                "controller_state": "TRACKING",
                "target_speed_mps": 5.0,
                "bypass_smoothing": False,
            }

        def reset_plan_progress(self, *_args):
            pass

    class _SafeAdapter:
        def __init__(self, *_args):
            pass

        def assess(self, **_kwargs):
            return types.SimpleNamespace(
                road=RoadContainmentAssessment.safe(sample_count=5),
                obstacles=ObstacleAssessment.safe(evaluated_actor_count=0),
            )

        def reset(self, *_args):
            pass

    telemetry_path = tmp_path / "post-throttle-error.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.3",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1, fail_on_tick=2)
    carla_if.get_camera_images = lambda: closed_loop.np.zeros(
        (4, 1, 1, 3),
        dtype=closed_loop.np.uint8,
    )
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_CAMERAS", 4)
    monkeypatch.setattr(
        closed_loop,
        "OfficialPIDFollower",
        lambda *_args: _TrackingFollower(),
    )
    monkeypatch.setattr(closed_loop, "CarlaGroundTruthSafetyAdapter", _SafeAdapter)
    monkeypatch.setattr(
        closed_loop,
        "run_inference",
        lambda *_args, **_kwargs: (object(), {"cot": "Follow the lane."}),
    )
    moving_points = closed_loop.np.zeros((1, 64, 3), dtype=closed_loop.np.float64)
    moving_points[0, :, 0] = closed_loop.np.arange(1, 65) * 0.2
    monkeypatch.setattr(
        closed_loop,
        "extract_trajectory_samples",
        lambda _prediction: moving_points.copy(),
    )
    monkeypatch.setattr(
        closed_loop,
        "create_visualization_frame",
        lambda cam_img, *_args, **_kwargs: cam_img,
    )

    exit_code = closed_loop.main()

    assert exit_code == 1
    assert len(carla_if.applied_controls) == 2
    assert carla_if.applied_controls[0][1] > 0.0
    assert carla_if.applied_controls[0][2] == 0.0
    assert carla_if.applied_controls[-1] == (0.0, 0.0, 1.0)

    records = _read_jsonl(telemetry_path)
    runtime_error = next(
        record for record in records if record["event_type"] == "runtime_error"
    )
    assert runtime_error["error"] == "synthetic tick failure 2"
    assert runtime_error["fail_closed_brake_applied"] is True
    assert records[-1]["stop_reason"] == "error"
    assert records[-1]["error"] == "synthetic tick failure 2"
    _assert_stream_summary_invariants(records)


def test_sync_keyboard_interrupt_terminalizes_request_and_returns_130(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "sync-keyboard-interrupt.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.2",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)

    def interrupt_inference(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(closed_loop, "run_inference", interrupt_inference)

    exit_code = closed_loop.main()

    assert exit_code == 130
    assert carla_if.applied_controls == [(0.0, 0.0, 1.0)]
    records = _read_jsonl(telemetry_path)
    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
    terminal = [
        record for record in records if record["event_type"] == "inference_result"
    ]
    assert len(submitted) == len(terminal) == 1
    assert terminal[0]["request_id"] == submitted[0]["request_id"]
    assert terminal[0]["status"] == "interrupted"
    assert terminal[0]["rejected"] is True
    assert (
        terminal[0]["rejection_reason"]
        == "keyboard_interrupt_during_sync_inference"
    )
    stop_event = next(
        record
        for record in records
        if record["event_type"] == "episode_stop_requested"
    )
    assert stop_event["emergency_brake_applied"] is True
    assert not [record for record in records if record["event_type"] == "runtime_error"]
    assert records[-1]["stop_reason"] == "keyboard_interrupt"
    _assert_stream_summary_invariants(records)


def test_sync_vqa_failure_is_not_retried_on_each_tick(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "sync-vqa-error.jsonl"
    args = closed_loop.parse_args(
        [
            "--mode",
            "vqa",
            "--vqa-question",
            "What is ahead?",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.5",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    vqa_calls = 0

    def fail_vqa(*_args, **_kwargs):
        nonlocal vqa_calls
        vqa_calls += 1
        raise RuntimeError("synthetic VQA failure")

    monkeypatch.setattr(closed_loop, "run_vqa", fail_vqa)

    exit_code = closed_loop.main()

    assert exit_code == 0
    assert vqa_calls == 1
    assert carla_if.tick_count == 5
    assert carla_if.applied_controls == [(0.0, 0.0, 1.0)] * 5

    records = _read_jsonl(telemetry_path)
    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
    terminal = [
        record for record in records if record["event_type"] == "inference_result"
    ]
    assert len(submitted) == len(terminal) == 1
    assert terminal[0]["request_id"] == submitted[0]["request_id"]
    assert terminal[0]["status"] == "error"
    assert terminal[0]["rejection_reason"].startswith("model_inference_error:")
    assert not [record for record in records if record["event_type"] == "runtime_error"]
    assert records[-1]["stop_reason"] == "max_episode_seconds"
    _assert_stream_summary_invariants(records)


def test_startup_model_load_error_emits_runtime_error_and_summary(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "startup-model-load-error.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    carla_if.ego_vehicle = None
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)

    def fail_model_load(*_args, **_kwargs):
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(closed_loop, "load_model", fail_model_load)

    exit_code = closed_loop.main()

    assert exit_code == 1
    assert carla_if.tick_count == 0
    assert carla_if.cleanup_count == 1
    assert carla_if.applied_controls == []
    records = _read_jsonl(telemetry_path)
    assert [record["event_type"] for record in records] == [
        "runtime_error",
        "episode_summary",
    ]
    runtime_error, summary = records
    assert runtime_error["error"] == "synthetic model load failure"
    assert runtime_error["emergency_brake_attempted"] is False
    assert runtime_error["fail_closed_brake_applied"] is False
    assert summary["stop_reason"] == "error"
    assert summary["error"] == "synthetic model load failure"
    assert summary["loop_tick_count"] == 0
    _assert_stream_summary_invariants(records)


def test_summary_exact_timing_flags_are_false_without_exact_observations(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "zero-exact-observations.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1, fail_on_tick=1)

    def unexpected_synchronized_observation(*_args, **_kwargs):
        raise AssertionError("tick failure should prevent observation capture")

    carla_if.get_synchronized_observation = unexpected_synchronized_observation
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)

    exit_code = closed_loop.main()

    assert exit_code == 1
    records = _read_jsonl(telemetry_path)
    episode_start = next(
        record for record in records if record["event_type"] == "episode_start"
    )
    summary = records[-1]
    assert episode_start["exact_sensor_frame_ids_available"] is True
    assert summary["loop_tick_count"] == 0
    assert summary["exact_source_frame_ids_available"] is False
    assert summary["exact_plan_age_available"] is False
    assert summary["source_age_s"]["count"] == 0
    _assert_stream_summary_invariants(records)


def test_sync_normal_inference_failure_keeps_episode_running_and_brakes(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "sync-inference-error.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.3",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)

    def fail_inference(*_args, **_kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(closed_loop, "run_inference", fail_inference)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    assert not [record for record in records if record["event_type"] == "runtime_error"]
    assert records[-1]["stop_reason"] == "max_episode_seconds"
    assert records[-1]["loop_tick_count"] == 3
    assert carla_if.applied_controls == [(0.0, 0.0, 1.0)] * 3

    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
    terminal = [
        record for record in records if record["event_type"] == "inference_result"
    ]
    assert len(submitted) == len(terminal) == 1
    assert terminal[0]["status"] == "error"
    assert terminal[0]["rejection_reason"].startswith("model_inference_error:")

    ticks = [record for record in records if record["event_type"] == "tick"]
    full_brake = {"steering": 0.0, "throttle": 0.0, "brake": 1.0}
    assert len(ticks) == 3
    assert all(tick["controller_state"] == "WAITING_FOR_PLAN" for tick in ticks)
    assert all(tick["applied_control_source"] == "FALLBACK" for tick in ticks)
    assert all(tick["applied_control"] == full_brake for tick in ticks)


def test_invalid_pid_output_fails_closed_without_runtime_error(monkeypatch, tmp_path):
    class _NaNFollower:
        def compute_world_control(self, **_kwargs):
            return float("nan"), 0.6, 0.0, {
                "controller_state": "TRACKING",
                "target_speed_mps": 5.0,
                "bypass_smoothing": False,
            }

        def reset_plan_progress(self, *_args):
            pass

    class _SafeAdapter:
        def __init__(self, *_args):
            pass

        def assess(self, **_kwargs):
            return types.SimpleNamespace(
                road=RoadContainmentAssessment.safe(sample_count=5),
                obstacles=ObstacleAssessment.safe(evaluated_actor_count=0),
            )

        def reset(self, *_args):
            pass

    telemetry_path = tmp_path / "invalid-pid.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    carla_if.get_camera_images = lambda: closed_loop.np.zeros(
        (4, 1, 1, 3),
        dtype=closed_loop.np.uint8,
    )
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_CAMERAS", 4)
    monkeypatch.setattr(
        closed_loop,
        "OfficialPIDFollower",
        lambda *_args: _NaNFollower(),
    )
    monkeypatch.setattr(closed_loop, "CarlaGroundTruthSafetyAdapter", _SafeAdapter)
    monkeypatch.setattr(
        closed_loop,
        "run_inference",
        lambda *_args, **_kwargs: (object(), {"cot": "Follow the lane."}),
    )
    moving_points = closed_loop.np.zeros((1, 64, 3), dtype=closed_loop.np.float64)
    moving_points[0, :, 0] = closed_loop.np.arange(1, 65) * 0.2
    monkeypatch.setattr(
        closed_loop,
        "extract_trajectory_samples",
        lambda _prediction: moving_points.copy(),
    )
    monkeypatch.setattr(
        closed_loop,
        "create_visualization_frame",
        lambda cam_img, *_args, **_kwargs: cam_img,
    )

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    assert not [record for record in records if record["event_type"] == "runtime_error"]
    assert records[-1]["stop_reason"] == "max_episode_seconds"
    assert carla_if.applied_controls == [(0.0, 0.0, 1.0)]

    tick = next(record for record in records if record["event_type"] == "tick")
    assert tick["controller_state"] == "INVALID_CONTROLLER_OUTPUT"
    assert tick["fallback_state"] == "INVALID_CONTROLLER_OUTPUT"
    assert tick["control_origin"] == "fail_closed_controller_validation"
    assert tick["requested_control"] is None
    assert tick["nominal_control"] is None
    assert tick["applied_control"] == {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
    assert tick["applied_control_source"] == "SAFETY_OVERRIDE"
    assert tick["safety_override_applied"] is True
    assert tick["safety_override_reason"] == "nominal_control_unavailable"
    assert tick["safety_assessment"]["arbitration_errors"]


def test_sync_normal_inference_uses_one_second_simulation_time_cadence(
    monkeypatch,
    tmp_path,
):
    telemetry_path = tmp_path / "sync-cadence.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "2.05",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    inference_calls = 0

    def fail_inference(*_args, **_kwargs):
        nonlocal inference_calls
        inference_calls += 1
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(closed_loop, "run_inference", fail_inference)

    closed_loop.main()

    records = _read_jsonl(telemetry_path)
    submitted = [
        record for record in records if record["event_type"] == "inference_submitted"
    ]
    source_times = [record["source_simulation_time_s"] for record in submitted]

    assert inference_calls == len(submitted) == 3
    assert source_times == pytest.approx([0.1, 1.1, 2.1])
    assert all(
        later - earlier >= 1.0 - 1e-9
        for earlier, later in zip(source_times, source_times[1:])
    )
    assert not [record for record in records if record["event_type"] == "runtime_error"]
    assert records[-1]["stop_reason"] == "max_episode_seconds"


def test_obstacle_override_is_the_only_control_applied_for_a_nominal_throttle(
    monkeypatch,
    tmp_path,
):
    class _TrackingFollower:
        def compute_world_control(self, **_kwargs):
            return 0.2, 0.6, 0.0, {
                "controller_state": "TRACKING",
                "target_speed_mps": 5.0,
                "bypass_smoothing": False,
            }

        def reset_plan_progress(self, *_args):
            pass

    class _UnsafeAdapter:
        def __init__(self, *_args):
            pass

        def assess(self, **_kwargs):
            return types.SimpleNamespace(
                road=RoadContainmentAssessment.safe(sample_count=5),
                obstacles=ObstacleAssessment.unknown(("synthetic_obstacle",)),
            )

        def reset(self, *_args):
            pass

    telemetry_path = tmp_path / "safety.jsonl"
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            "0.1",
        ]
    )
    carla_if = _FakeCarlaInterface(fixed_delta_seconds=0.1)
    carla_if.get_camera_images = lambda: closed_loop.np.zeros(
        (4, 1, 1, 3),
        dtype=closed_loop.np.uint8,
    )
    _install_common_fakes(monkeypatch, args, carla_if, num_frames=1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_CAMERAS", 4)
    monkeypatch.setattr(closed_loop, "OfficialPIDFollower", lambda *_args: _TrackingFollower())
    monkeypatch.setattr(closed_loop, "CarlaGroundTruthSafetyAdapter", _UnsafeAdapter)
    monkeypatch.setattr(
        closed_loop,
        "run_inference",
        lambda *_args, **_kwargs: (object(), {"cot": "Brake for the obstacle."}),
    )
    moving_points = closed_loop.np.zeros((1, 64, 3), dtype=closed_loop.np.float64)
    moving_points[0, :, 0] = closed_loop.np.arange(1, 65) * 0.2
    monkeypatch.setattr(
        closed_loop,
        "extract_trajectory_samples",
        lambda _prediction: moving_points.copy(),
    )
    monkeypatch.setattr(
        closed_loop,
        "create_visualization_frame",
        lambda cam_img, *_args, **_kwargs: cam_img,
    )

    closed_loop.main()

    assert carla_if.applied_controls == [(0.0, 0.0, 1.0)]
    records = _read_jsonl(telemetry_path)
    tick = next(record for record in records if record["event_type"] == "tick")
    assert tick["requested_control"]["throttle"] == pytest.approx(0.6)
    assert tick["nominal_control"]["throttle"] > 0.0
    assert tick["applied_control"] == {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
    assert tick["applied_control_source"] == "SAFETY_OVERRIDE"
    assert tick["safety_override_applied"] is True
    assert tick["safety_override_reason"] == "obstacle_assessment_unknown"
    proposals = [
        record for record in records if record["event_type"] == "alpamayo_proposal"
    ]
    assert len(proposals) == 1
    assert proposals[0]["coc_text_full"] == "Brake for the obstacle."

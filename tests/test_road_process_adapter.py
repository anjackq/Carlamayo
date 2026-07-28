"""Adapter-level process shadow, fallback, cache, and lifecycle tests."""

from __future__ import annotations

import hashlib
import os
import types
from collections import deque

import pytest

from module import carla_safety_adapter
from module.carla_safety_adapter import CarlaGroundTruthSafetyAdapter
from module.road_assessment_backend import (
    FootprintQueryResult,
    ProcessRoadBatchStats,
    RoadProcessBackendCrashed,
    RoadProcessBackendTimeout,
    WaypointPrimitive,
)
from module.safety_shield import AssessmentStatus
from tests.test_carla_safety_adapter import (
    FakeActor,
    FakeBoundingBox,
    FakeLocation,
    FakeWorld,
    RecordingMap,
    _adapter,
    _long_plan,
    _tick_context,
)


class _OpenDriveRecordingMap(RecordingMap):
    name = "TinyRoad"

    def to_opendrive(self):
        return "<OpenDRIVE/>"


class _SnapshotRecordingMap(RecordingMap):
    name = "SharedMapName"

    def __init__(self, *snapshots):
        super().__init__()
        self.snapshots = tuple(snapshots)
        self.to_opendrive_count = 0

    def to_opendrive(self):
        index = min(
            self.to_opendrive_count,
            len(self.snapshots) - 1,
        )
        self.to_opendrive_count += 1
        return self.snapshots[index]


def _opendrive_map_digest(opendrive: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"carla_opendrive")
    digest.update(b"\0")
    digest.update(opendrive.encode("utf-8"))
    return digest.hexdigest()


class _FakeProcessBackend:
    def __init__(self, script):
        self.script = script
        self.query_calls = []
        self.close_count = 0

    @staticmethod
    def _matching_result(query, *, mismatch=False):
        waypoints = []
        for point_index, point in enumerate(query.points_xyz):
            waypoints.append(
                WaypointPrimitive(
                    found=True,
                    road_id=1,
                    lane_id=(-2 if mismatch and point_index == 0 else -1),
                    is_junction=False,
                    lane_center_x=float(point[0]),
                    lane_center_y=0.0,
                    lane_yaw_deg=0.0,
                    lane_width=4.0,
                )
            )
        return FootprintQueryResult(
            query_id=query.query_id,
            waypoints=tuple(waypoints),
            error_type=None,
            map_query_count=5,
            worker_query_ms=0.1,
            worker_pid=9100,
        )

    def query(self, queries):
        queries = tuple(queries)
        self.query_calls.append(queries)
        commissioning_batch = len(self.query_calls) == 1
        action = (
            self.script.popleft()
            if self.script
            else "match"
        )
        if isinstance(action, BaseException):
            raise action
        results = tuple(
            self._matching_result(
                query,
                mismatch=action == "mismatch" and index == 0,
            )
            for index, query in enumerate(queries)
        )
        return results, ProcessRoadBatchStats(
            map_query_wall_ms=1.0,
            worker_query_sum_ms=0.1 * len(results),
            map_query_count=5 * len(results),
            worker_count=2,
            chunk_count=max(1, (len(results) + 15) // 16),
            commissioning_batch=commissioning_batch,
            query_deadline_ms=(
                500.0 if commissioning_batch else 250.0
            ),
        )

    def close(self):
        self.close_count += 1


class _ProcessBackendFactory:
    def __init__(self, *instance_scripts):
        self.instance_scripts = deque(
            deque(script) for script in instance_scripts
        )
        self.calls = []
        self.instances = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        script = (
            self.instance_scripts.popleft()
            if self.instance_scripts
            else deque()
        )
        instance = _FakeProcessBackend(script)
        self.instances.append(instance)
        return instance


@pytest.fixture(autouse=True)
def _fake_carla_types(monkeypatch):
    driving = object()
    monkeypatch.setattr(
        carla_safety_adapter,
        "carla",
        types.SimpleNamespace(
            Location=FakeLocation,
            LaneType=types.SimpleNamespace(Driving=driving),
        ),
    )


def _process_adapter(carla_map, factory):
    serial_adapter, world, ego = _adapter(carla_map)
    adapter = CarlaGroundTruthSafetyAdapter(
        world,
        ego,
        road_assessment_backend="process",
        road_assessment_workers=2,
        process_backend_factory=factory,
    )
    return adapter, world, ego


def _semantic_payload(envelope):
    payload = envelope.to_json_dict()
    payload.pop("stopping_reserve_compute_ms", None)
    return payload


def test_first_batch_shadow_pass_then_process_only_preserves_exact_semantics():
    carla_map = _OpenDriveRecordingMap()
    factory = _ProcessBackendFactory(("match", "match"))
    adapter, _world, ego = _process_adapter(carla_map, factory)
    serial, _serial_world, serial_ego = _adapter(
        _OpenDriveRecordingMap()
    )
    first_plan = _long_plan(plan_id="shadow-first", start_time_s=5.0)

    first = adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=(first_plan,),
    )
    expected = serial.assess_plan_road(
        tick_context=_tick_context(
            serial_ego,
            frame=50,
            simulation_time_s=5.0,
        ),
        plan=first_plan,
    )

    assert _semantic_payload(first[0]) == _semantic_payload(expected)
    assert adapter.last_road_batch_stats.backend_status == "process_shadow_pass"
    assert adapter.last_road_batch_stats.shadow_parity_status == "pass"
    assert adapter.last_road_batch_stats.commissioning_batch is True
    assert adapter.last_road_batch_stats.query_deadline_ms == pytest.approx(
        500.0
    )
    assert adapter.last_road_batch_stats.process_attempt_ms >= 0.0
    assert adapter.last_road_batch_stats.shadow_serial_ms > 0.0
    assert adapter.last_road_batch_stats.shadow_serial_query_count > 0
    assert (
        adapter.last_road_batch_stats.shadow_parity_ms
        >= adapter.last_road_batch_stats.shadow_serial_ms
    )
    assert (
        adapter.last_road_batch_stats.map_query_ms
        < adapter.last_road_batch_stats.query_deadline_ms
    )
    assert (
        adapter.last_road_batch_stats.map_query_ms
        != adapter.last_road_batch_stats.query_deadline_ms
    )
    assert len(factory.instances[0].query_calls) == 1

    carla_map.calls.clear()
    second_plan = _long_plan(
        plan_id="process-only",
        start_time_s=5.1,
        step_m=0.11,
    )
    second = adapter.assess_plans_road(
        tick_context=_tick_context(
            ego,
            frame=51,
            simulation_time_s=5.1,
        ),
        plans=(second_plan,),
    )

    assert second[0].current_ego_road.status is AssessmentStatus.SAFE
    assert adapter.last_road_batch_stats.backend_status == "process"
    assert adapter.last_road_batch_stats.shadow_parity_status == "not_run"
    assert adapter.last_road_batch_stats.commissioning_batch is False
    assert adapter.last_road_batch_stats.query_deadline_ms == pytest.approx(
        250.0
    )
    assert adapter.last_road_batch_stats.shadow_serial_ms == 0.0
    assert adapter.last_road_batch_stats.shadow_serial_query_count == 0
    assert len(factory.instances[0].query_calls) == 2
    # The second profile is process-backed; only current ego uses parent map.
    assert len(carla_map.calls) == 5


def test_shadow_mismatch_returns_serial_and_sticky_disables_process():
    carla_map = _OpenDriveRecordingMap()
    factory = _ProcessBackendFactory(("mismatch",))
    adapter, _world, ego = _process_adapter(carla_map, factory)

    envelope = adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=(_long_plan(plan_id="mismatch", start_time_s=5.0),),
    )[0]

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.SAFE
    assert (
        adapter.last_road_batch_stats.backend_status
        == "process_disabled_shadow_mismatch"
    )
    assert adapter.last_road_batch_stats.shadow_parity_status == "fail"
    assert factory.instances[0].close_count == 1
    assert len(factory.instances[0].query_calls) == 1

    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=51, simulation_time_s=5.1),
        plans=(
            _long_plan(
                plan_id="after-mismatch",
                start_time_s=5.1,
                step_m=0.11,
            ),
        ),
    )
    assert len(factory.instances[0].query_calls) == 1
    assert (
        adapter.last_road_batch_stats.backend_status
        == "process_disabled_shadow_mismatch"
    )
    assert adapter.last_road_batch_stats.shadow_parity_status == "fail"

    adapter.reset(ego)
    assert len(factory.instances) == 1


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_reason"),
    [
        (
            RoadProcessBackendTimeout("synthetic timeout"),
            "process_serial_fallback_timeout",
            "timeout",
        ),
        (
            RoadProcessBackendCrashed("synthetic crash"),
            "process_serial_fallback_crash",
            "crash",
        ),
    ],
)
def test_process_failure_uses_exact_serial_fallback_until_reset(
    failure,
    expected_status,
    expected_reason,
):
    carla_map = _OpenDriveRecordingMap()
    factory = _ProcessBackendFactory(
        (failure,),
        ("match",),
    )
    adapter, _world, ego = _process_adapter(carla_map, factory)
    serial, _serial_world, serial_ego = _adapter(
        _OpenDriveRecordingMap()
    )
    plan = _long_plan(plan_id="fallback", start_time_s=5.0)

    actual = adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=(plan,),
    )[0]
    expected = serial.assess_plan_road(
        tick_context=_tick_context(
            serial_ego,
            frame=50,
            simulation_time_s=5.0,
        ),
        plan=plan,
    )

    assert _semantic_payload(actual) == _semantic_payload(expected)
    stats = adapter.last_road_batch_stats
    assert stats.backend_status == expected_status
    assert stats.fallback_reason == expected_reason
    assert stats.serial_fallback_ms >= 0.0
    assert factory.instances[0].close_count == 1
    assert stats.commissioning_batch is True
    assert stats.query_deadline_ms == pytest.approx(500.0)
    assert stats.process_attempt_ms >= 0.0

    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=51, simulation_time_s=5.1),
        plans=(
            _long_plan(
                plan_id="degraded",
                start_time_s=5.1,
                step_m=0.11,
            ),
        ),
    )
    assert len(factory.instances[0].query_calls) == 1
    assert adapter.last_road_batch_stats.backend_status == "process_degraded"

    adapter.reset(ego)
    assert len(factory.instances) == 2
    recovered = adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=52, simulation_time_s=5.2),
        plans=(
            _long_plan(
                plan_id="recovered",
                start_time_s=5.2,
                step_m=0.12,
            ),
        ),
    )
    assert recovered[0].current_ego_road.status is AssessmentStatus.SAFE
    assert adapter.last_road_batch_stats.backend_status == "process_shadow_pass"


def test_steady_timeout_reports_steady_deadline_after_successful_commissioning():
    factory = _ProcessBackendFactory(
        ("match", RoadProcessBackendTimeout("steady timeout")),
    )
    adapter, _world, ego = _process_adapter(
        _OpenDriveRecordingMap(),
        factory,
    )
    first_plan = _long_plan(
        plan_id="commissioning-success",
        start_time_s=5.0,
    )
    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=(first_plan,),
    )
    assert adapter.last_road_batch_stats.backend_status == "process_shadow_pass"

    second_plan = _long_plan(
        plan_id="steady-timeout",
        start_time_s=5.1,
        step_m=0.11,
    )
    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=51, simulation_time_s=5.1),
        plans=(second_plan,),
    )

    stats = adapter.last_road_batch_stats
    assert stats.backend_status == "process_serial_fallback_timeout"
    assert stats.fallback_reason == "timeout"
    assert stats.commissioning_batch is False
    assert stats.query_deadline_ms == pytest.approx(250.0)
    assert stats.serial_fallback_ms >= 0.0
    assert factory.instances[0].close_count == 1


def test_process_failure_falls_back_to_exact_serial_for_the_whole_batch():
    carla_map = _OpenDriveRecordingMap()
    factory = _ProcessBackendFactory(
        (RoadProcessBackendCrashed("dispatch failed"),),
    )
    adapter, _world, ego = _process_adapter(carla_map, factory)
    serial, _serial_world, serial_ego = _adapter(
        _OpenDriveRecordingMap()
    )
    plans = tuple(
        _long_plan(
            plan_id=f"batch-fallback-{index}",
            start_time_s=5.0,
            step_m=0.10 + index * 0.01,
        )
        for index in range(3)
    )

    actual = adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=plans,
    )
    expected = serial.assess_plans_road(
        tick_context=_tick_context(
            serial_ego,
            frame=50,
            simulation_time_s=5.0,
        ),
        plans=plans,
    )

    assert tuple(map(_semantic_payload, actual)) == tuple(
        map(_semantic_payload, expected)
    )
    assert len(factory.instances[0].query_calls) == 1
    assert len(factory.instances[0].query_calls[0]) > len(plans)
    assert (
        adapter.last_road_batch_stats.backend_status
        == "process_serial_fallback_crash"
    )
    assert adapter.last_road_batch_stats.fallback_reason == "crash"
    assert adapter.last_road_batch_stats.profile_cache_misses == 3


@pytest.mark.parametrize(
    (
        "failure",
        "expected_status",
        "expected_reason",
        "expected_shadow_status",
    ),
    [
        (
            RoadProcessBackendTimeout("synthetic timeout"),
            "process_degraded_cache_only",
            "timeout",
            "not_run",
        ),
        (
            "mismatch",
            "process_disabled_shadow_mismatch",
            "shadow_parity_mismatch",
            "fail",
        ),
    ],
)
def test_cache_only_batch_preserves_process_failure_telemetry(
    failure,
    expected_status,
    expected_reason,
    expected_shadow_status,
):
    factory = _ProcessBackendFactory((failure,))
    adapter, _world, ego = _process_adapter(
        _OpenDriveRecordingMap(),
        factory,
    )
    plan = _long_plan(plan_id="cached-after-failure", start_time_s=5.0)

    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=50, simulation_time_s=5.0),
        plans=(plan,),
    )
    assert len(factory.instances[0].query_calls) == 1

    adapter.assess_plans_road(
        tick_context=_tick_context(ego, frame=51, simulation_time_s=5.1),
        plans=(plan,),
    )

    stats = adapter.last_road_batch_stats
    assert stats.profile_cache_hits == 1
    assert stats.profile_cache_misses == 0
    assert stats.backend_status == expected_status
    assert stats.fallback_reason == expected_reason
    assert stats.shadow_parity_status == expected_shadow_status
    assert len(factory.instances[0].query_calls) == 1


def test_process_adapter_init_and_reset_each_use_one_exact_opendrive_snapshot():
    first_snapshot = "<OpenDRIVE name='first'/>"
    changed_snapshot = "<OpenDRIVE name='changed'/>"
    unexpected_snapshot = "<OpenDRIVE name='unexpected'/>"
    carla_map = _SnapshotRecordingMap(
        first_snapshot,
        changed_snapshot,
        unexpected_snapshot,
    )
    factory = _ProcessBackendFactory(("match",), ("match",))

    ego = FakeActor(
        1,
        "vehicle.ego",
        bounding_box=FakeBoundingBox(),
    )
    world = FakeWorld(carla_map, (ego,))
    adapter = CarlaGroundTruthSafetyAdapter(
        world,
        ego,
        road_assessment_backend="process",
        road_assessment_workers=2,
        process_backend_factory=factory,
    )

    assert carla_map.to_opendrive_count == 1
    assert len(factory.calls) == 1
    assert factory.calls[0]["opendrive"] == first_snapshot
    assert (
        factory.calls[0]["map_digest"]
        == _opendrive_map_digest(first_snapshot)
    )

    adapter.reset(ego)

    assert carla_map.to_opendrive_count == 2
    assert len(factory.calls) == 2
    assert factory.calls[1]["opendrive"] == changed_snapshot
    assert (
        factory.calls[1]["map_digest"]
        == _opendrive_map_digest(changed_snapshot)
    )
    adapter.close()


def test_serial_map_identity_uses_opendrive_not_only_shared_map_name():
    first_snapshot = "<OpenDRIVE name='first'/>"
    second_snapshot = "<OpenDRIVE name='second'/>"
    first_adapter, _world, _ego = _adapter(
        _SnapshotRecordingMap(first_snapshot)
    )
    second_adapter, _world, _ego = _adapter(
        _SnapshotRecordingMap(second_snapshot)
    )

    assert first_adapter._map.name == second_adapter._map.name
    assert first_adapter._map_digest == _opendrive_map_digest(first_snapshot)
    assert second_adapter._map_digest == _opendrive_map_digest(
        second_snapshot
    )
    assert first_adapter._map_digest != second_adapter._map_digest


def test_same_plan_id_changed_geometry_fails_before_process_dispatch():
    factory = _ProcessBackendFactory(("match",))
    adapter, _world, ego = _process_adapter(
        _OpenDriveRecordingMap(),
        factory,
    )
    context = _tick_context(ego, frame=50, simulation_time_s=5.0)
    adapter.assess_plans_road(
        tick_context=context,
        plans=(_long_plan(plan_id="same-id", step_m=0.10),),
    )
    process_query_count = len(factory.instances[0].query_calls)

    changed = adapter.assess_plans_road(
        tick_context=context,
        plans=(_long_plan(plan_id="same-id", step_m=0.12),),
    )[0]

    assert len(factory.instances[0].query_calls) == process_query_count
    assert changed.near_term_path_road.status is AssessmentStatus.UNKNOWN
    assert changed.emergency_required is True
    assert "plan_id_geometry_mismatch" in changed.full_path_road.reason_codes


def test_clear_reset_and_close_manage_exactly_one_worker_generation():
    factory = _ProcessBackendFactory(("match",), ("match",))
    adapter, _world, ego = _process_adapter(
        _OpenDriveRecordingMap(),
        factory,
    )
    first = factory.instances[0]

    adapter.clear_plan_caches()
    assert len(factory.instances) == 1
    assert first.close_count == 0

    adapter.reset(ego)
    assert first.close_count == 1
    assert len(factory.instances) == 2
    second = factory.instances[1]

    adapter.close()
    adapter.close()
    assert second.close_count == 1


def test_default_process_worker_count_respects_cpu_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))

    assert CarlaGroundTruthSafetyAdapter._resolve_process_worker_count(None) == 6
    assert CarlaGroundTruthSafetyAdapter._resolve_process_worker_count(3) == 3
    with pytest.raises(ValueError):
        CarlaGroundTruthSafetyAdapter._resolve_process_worker_count(0)

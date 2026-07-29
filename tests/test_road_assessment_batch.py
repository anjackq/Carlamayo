"""Exact serial road-assessment batch contract tests.

These tests use the same deterministic fake CARLA map as the single-plan
adapter suite, but exercise only the additive batch/cache contracts.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest

from module import carla_safety_adapter
from module.carla_safety_adapter import CarlaGroundTruthSafetyAdapter
from module.safety_shield import AssessmentStatus, SafetyPolicy
from tests.test_carla_safety_adapter import (
    FakeBoundingBox,
    FakeLocation,
    FakeWaypoint,
    RecordingMap,
    _adapter,
    _long_plan,
    _tick_context,
)


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


def _plan_with_step(*, plan_id: str, step_m: float):
    return _long_plan(
        plan_id=plan_id,
        start_time_s=5.0,
        step_m=step_m,
    )


def _semantic_payload(envelope):
    payload = copy.deepcopy(envelope.to_json_dict())
    # Wall-clock profiling is deliberately not part of envelope semantics.
    payload.pop("stopping_reserve_compute_ms", None)
    return payload


def test_serial_batch_matches_ordered_individual_envelopes():
    plans = tuple(
        _plan_with_step(plan_id=f"batch-{index}", step_m=step_m)
        for index, step_m in enumerate((0.10, 0.11, 0.12))
    )
    batch_adapter, _world, batch_ego = _adapter(RecordingMap())
    context = _tick_context(batch_ego, simulation_time_s=5.0)

    batch = batch_adapter.assess_plans_road(
        tick_context=context,
        plans=plans,
    )

    expected = []
    for plan in plans:
        individual_adapter, _world, individual_ego = _adapter(RecordingMap())
        expected.append(
            individual_adapter.assess_plan_road(
                tick_context=_tick_context(
                    individual_ego,
                    simulation_time_s=5.0,
                ),
                plan=plan,
            )
        )
    assert len(batch) == len(plans)
    assert [_semantic_payload(item) for item in batch] == [
        _semantic_payload(item) for item in expected
    ]
    stats = batch_adapter.last_road_batch_stats
    assert stats.backend_status == "serial"
    assert stats.plan_count == 3
    assert stats.worker_count == 1
    assert stats.chunk_count == 1
    assert stats.error_count == 0


def test_current_ego_footprint_is_queried_once_per_batch_frame():
    carla_map = RecordingMap()
    adapter, _world, ego = _adapter(carla_map)
    current_footprint_calls = 0
    original_query = adapter._query_footprint

    def record_query(*, center_xyz, **kwargs):
        nonlocal current_footprint_calls
        if np.allclose(np.asarray(center_xyz)[:2], (0.0, 0.0)):
            current_footprint_calls += 1
        return original_query(center_xyz=center_xyz, **kwargs)

    adapter._query_footprint = record_query
    plans = tuple(
        _plan_with_step(plan_id=f"query-{index}", step_m=step_m)
        for index, step_m in enumerate((0.10, 0.11, 0.12))
    )

    context = _tick_context(ego, simulation_time_s=5.0)
    adapter.assess_plans_road(
        tick_context=context,
        plans=plans,
    )
    adapter.assess_plans_road(
        tick_context=context,
        plans=plans,
    )

    assert current_footprint_calls == 1
    adapter.assess_plans_road(
        tick_context=_tick_context(
            ego,
            frame=51,
            simulation_time_s=5.1,
        ),
        plans=plans,
    )
    assert current_footprint_calls == 2
    stats = adapter.last_road_batch_stats
    assert stats.map_query_count == 5
    assert stats.profile_cache_hits == 3


def test_profile_cache_reuses_exact_geometry_across_different_plan_ids():
    carla_map = RecordingMap()
    adapter, _world, ego = _adapter(carla_map)
    context = _tick_context(ego, simulation_time_s=5.0)
    first = _plan_with_step(plan_id="cache-source", step_m=0.10)
    same_geometry = _plan_with_step(plan_id="cache-alias", step_m=0.10)

    adapter.assess_plans_road(tick_context=context, plans=(first,))
    carla_map.calls.clear()
    second = adapter.assess_plans_road(
        tick_context=context,
        plans=(same_geometry,),
    )

    assert second[0].current_ego_road.status is AssessmentStatus.SAFE
    stats = adapter.last_road_batch_stats
    assert stats.profile_cache_hits == 1
    assert stats.profile_cache_misses == 0
    assert stats.pose_count == 0
    # The exact current footprint and trajectory profile are both frame-local
    # cache hits for this second logical plan.
    assert stats.map_query_count == len(carla_map.calls) == 0


def test_transient_map_error_does_not_poison_shared_geometry_cache():
    query_count = 0

    def fail_once_during_first_profile(location):
        nonlocal query_count
        query_count += 1
        # The first five queries assess the current ego footprint.  Fail the
        # first trajectory query, then let the exact map recover.
        if query_count == 6:
            raise RuntimeError("transient map failure")
        return FakeWaypoint(location)

    carla_map = RecordingMap(fail_once_during_first_profile)
    adapter, _world, ego = _adapter(carla_map)
    first = _plan_with_step(plan_id="transient-source", step_m=0.10)
    same_geometry = _plan_with_step(
        plan_id="transient-recovery",
        step_m=0.10,
    )

    contaminated = adapter.assess_plans_road(
        tick_context=_tick_context(
            ego,
            frame=50,
            simulation_time_s=5.0,
        ),
        plans=(first,),
    )

    assert contaminated[0].current_ego_road.status is AssessmentStatus.SAFE
    assert contaminated[0].full_path_road.status is AssessmentStatus.UNKNOWN
    assert adapter.last_road_batch_stats.backend_status == "serial_degraded"
    assert adapter.last_road_batch_stats.error_count == 1
    assert len(adapter._road_profile_cache) == 0

    recovered = adapter.assess_plans_road(
        tick_context=_tick_context(
            ego,
            frame=51,
            simulation_time_s=5.1,
        ),
        plans=(same_geometry,),
    )

    assert recovered[0].current_ego_road.status is AssessmentStatus.SAFE
    assert recovered[0].full_path_road.status is AssessmentStatus.SAFE
    assert adapter.last_road_batch_stats.backend_status == "serial"
    assert adapter.last_road_batch_stats.error_count == 0
    assert adapter.last_road_batch_stats.profile_cache_hits == 0
    assert adapter.last_road_batch_stats.profile_cache_misses == 1
    assert len(adapter._road_profile_cache) == 1


def test_same_plan_id_with_changed_geometry_fails_closed_without_cache_alias():
    adapter, _world, ego = _adapter(RecordingMap())
    context = _tick_context(ego, simulation_time_s=5.0)
    adapter.assess_plans_road(
        tick_context=context,
        plans=(_plan_with_step(plan_id="reused-id", step_m=0.10),),
    )

    changed = adapter.assess_plans_road(
        tick_context=context,
        plans=(_plan_with_step(plan_id="reused-id", step_m=0.12),),
    )

    assert len(changed) == 1
    envelope = changed[0]
    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.UNKNOWN
    assert envelope.full_path_road.status is AssessmentStatus.UNKNOWN
    assert envelope.emergency_required is True
    assert any(
        "plan_id_geometry_mismatch" in reason
        for reason in envelope.full_path_road.reason_codes
    )
    stats = adapter.last_road_batch_stats
    assert stats.error_count == 1
    assert stats.profile_cache_hits == 0


def test_exact_profile_key_changes_with_map_policy_bbox_and_fallback_yaw():
    plan = _plan_with_step(plan_id="cache-key", step_m=0.10)
    geometry_digest = carla_safety_adapter._plan_geometry_digest(plan)
    base_map = types.SimpleNamespace(name="Town03")
    other_map = types.SimpleNamespace(name="Town04")
    base_policy = SafetyPolicy()
    other_policy = SafetyPolicy(
        lateral_clearance_m=base_policy.lateral_clearance_m + 0.01,
    )
    base_bbox = FakeBoundingBox(half_length=0.4, half_width=0.3)
    other_bbox = FakeBoundingBox(half_length=0.4, half_width=0.31)

    def key(*, map_value=base_map, policy=base_policy, bbox=base_bbox, yaw=0.0):
        return carla_safety_adapter._road_profile_cache_key(
            geometry_digest=geometry_digest,
            map_digest=carla_safety_adapter._map_content_digest(map_value),
            policy_digest=carla_safety_adapter._policy_content_digest(policy),
            bounding_box_digest=(
                carla_safety_adapter._bounding_box_content_digest(bbox)
            ),
            fallback_yaw_rad=yaw,
        )

    base = key()
    variants = {
        key(map_value=other_map),
        key(policy=other_policy),
        key(bbox=other_bbox),
        key(yaw=0.125),
    }
    assert base not in variants
    assert len(variants) == 4


def test_non_degenerate_geometry_reuses_profile_when_ego_yaw_changes():
    carla_map = RecordingMap()
    adapter, _world, ego = _adapter(carla_map)
    first = _plan_with_step(plan_id="yaw-source", step_m=0.10)
    alias = _plan_with_step(plan_id="yaw-alias", step_m=0.10)

    adapter.assess_plans_road(
        tick_context=_tick_context(
            ego,
            frame=50,
            simulation_time_s=5.0,
        ),
        plans=(first,),
    )
    changed_yaw_context = _tick_context(
        ego,
        frame=51,
        simulation_time_s=5.1,
    )
    changed_yaw_context.ego_transform.rotation.yaw = 12.5
    carla_map.calls.clear()

    adapter.assess_plans_road(
        tick_context=changed_yaw_context,
        plans=(alias,),
    )

    assert adapter.last_road_batch_stats.profile_cache_hits == 1
    assert adapter.last_road_batch_stats.profile_cache_misses == 0
    assert adapter.last_road_batch_stats.map_query_count == 5


def test_stationary_geometry_keeps_exact_fallback_yaw_in_cache_key():
    plan = _plan_with_step(plan_id="stationary", step_m=0.0)

    assert carla_safety_adapter._effective_profile_fallback_yaw(
        plan,
        0.25,
    ) == pytest.approx(0.25)
    assert carla_safety_adapter._effective_profile_fallback_yaw(
        _plan_with_step(plan_id="moving", step_m=0.10),
        0.25,
    ) == 0.0

"""Simulator-free integration tests for exact K-candidate road batches."""

from __future__ import annotations

import types

import pytest

from module.carla_safety_adapter import (
    RoadExecutionEnvelope,
    StoppingReserveProfile,
    StoppingReserveStatus,
)
from module.safety_shield import (
    ObstacleAssessment,
    RoadContainmentAssessment,
)
from tests.test_closed_loop_runtime import (
    _FakeCarlaInterface,
    _install_common_fakes,
    _read_jsonl,
    closed_loop,
)


def _safe_envelope(*, margin_m: float = 0.5) -> RoadExecutionEnvelope:
    safe = RoadContainmentAssessment.safe(
        sample_count=5,
        min_margin_m=margin_m,
    )
    return RoadExecutionEnvelope(
        current_ego_road=safe,
        current_ego_clearance_road=safe,
        near_term_path_road=safe,
        full_path_road=safe,
        last_safe_waypoint_index=63,
        time_to_first_bad_s=None,
        distance_to_first_bad_m=None,
        target_speed_cap_mps=None,
        emergency_required=False,
        stopping_reserve_profile=StoppingReserveProfile(
            raw_physical_stopping_cap_mps=None,
            target_speed_cap_mps=None,
            guard_speed_mps=2.5,
            required_stopping_distance_m=None,
            stopping_reserve_m=None,
            status=StoppingReserveStatus.UNBOUNDED,
        ),
        stopping_reserve_compute_ms=0.05,
    )


class _TrackingFollower:
    def compute_world_control(self, **_kwargs):
        return 0.0, 0.4, 0.0, {
            "controller_state": "TRACKING",
            "target_speed_mps": 2.5,
            "bypass_smoothing": False,
        }

    def reset_plan_progress(self, *_args):
        pass


class _BatchStats:
    def to_json_dict(self):
        return {
            "densify_heading_ms": 1.25,
            "map_query_ms": 2.5,
            "aggregate_ms": 0.75,
            "pose_count": 192,
            "map_query_count": 965,
            "worker_count": 1,
            "chunk_count": 1,
            "profile_cache_hits": 0,
            "backend_status": "serial",
        }


class _RecordingBatchAdapter:
    batch_failure_mode = None
    instances = []

    def __init__(self, *_args):
        self.batch_calls = []
        self.legacy_candidate_calls = []
        self.last_road_batch_stats = _BatchStats()
        type(self).instances.append(self)

    def assess_plans_road(self, *, tick_context, plans):
        del tick_context
        indices = tuple(int(plan.selected_candidate_index) for plan in plans)
        self.batch_calls.append(indices)
        if len(self.batch_calls) >= 2 and self.batch_failure_mode == "raise":
            raise RuntimeError("synthetic batch failure")
        if len(self.batch_calls) >= 2 and self.batch_failure_mode == "short":
            return tuple(_safe_envelope() for _ in plans[:-1])
        if self.batch_failure_mode == "raise_first":
            raise RuntimeError("synthetic first-batch failure")
        return tuple(
            _safe_envelope(margin_m=0.5 - index * 0.05)
            for index in indices
        )

    def assess_plan_road(self, *, plan, **_kwargs):
        # This legacy method remains necessary for active-plan tick checks.
        # Candidate selection must use assess_plans_road instead.
        self.legacy_candidate_calls.append(plan.plan_id)
        return _safe_envelope()

    def assess(self, **_kwargs):
        envelope = _safe_envelope()
        return types.SimpleNamespace(
            road=envelope.current_ego_road,
            obstacles=ObstacleAssessment.safe(evaluated_actor_count=0),
            current_ego_road=envelope.current_ego_road,
            proposed_path_road=envelope.full_path_road,
            road_envelope=envelope,
        )

    def reset(self, *_args):
        pass


def _run_k3(
    monkeypatch,
    tmp_path,
    *,
    max_episode_seconds: float,
    failure_mode: str | None = None,
):
    telemetry_path = tmp_path / f"k3-road-batch-{failure_mode or 'ok'}.jsonl"
    args = closed_loop.parse_args(
        [
            "--num-traj-samples",
            "3",
            "--telemetry-jsonl",
            str(telemetry_path),
            "--max-episode-seconds",
            str(max_episode_seconds),
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
        lambda *_args: _TrackingFollower(),
    )
    _RecordingBatchAdapter.instances.clear()
    monkeypatch.setattr(
        _RecordingBatchAdapter,
        "batch_failure_mode",
        failure_mode,
    )
    monkeypatch.setattr(
        closed_loop,
        "CarlaGroundTruthSafetyAdapter",
        _RecordingBatchAdapter,
    )
    monkeypatch.setattr(
        closed_loop,
        "run_inference",
        lambda *_args, **_kwargs: (
            object(),
            {
                "cot": closed_loop.np.array(
                    [["candidate zero", "candidate one", "candidate two"]],
                    dtype=object,
                )
            },
        ),
    )
    candidates = closed_loop.np.zeros(
        (3, 64, 3),
        dtype=closed_loop.np.float64,
    )
    candidates[:, :, 0] = closed_loop.np.arange(1, 65) * 0.2
    monkeypatch.setattr(
        closed_loop,
        "extract_trajectory_samples",
        lambda _prediction: candidates.copy(),
    )
    monkeypatch.setattr(
        closed_loop,
        "create_visualization_frame",
        lambda cam_img, *_args, **_kwargs: cam_img,
    )

    closed_loop.main()

    return (
        _RecordingBatchAdapter.instances[-1],
        carla_if,
        _read_jsonl(telemetry_path),
    )


def test_k3_candidate_selection_uses_one_ordered_batch_and_emits_additive_stats(
    monkeypatch,
    tmp_path,
):
    adapter, _carla_if, records = _run_k3(
        monkeypatch,
        tmp_path,
        max_episode_seconds=0.1,
    )

    assert adapter.batch_calls == [(0, 1, 2)]
    evaluations = [
        record
        for record in records
        if record["event_type"] == "candidate_evaluation"
    ]
    assert [record["candidate_index"] for record in evaluations] == [0, 1, 2]
    assert [record["candidate_plan_id"].rsplit("-", 1)[-1]
            for record in evaluations] == ["0", "1", "2"]

    selection = next(
        record
        for record in records
        if record["event_type"] == "candidate_selection"
    )
    assert selection["selected_candidate_index"] == 0
    assert selection["backend_status"] == "serial"
    assert selection["densify_heading_ms"] == pytest.approx(1.25)
    assert selection["map_query_ms"] == pytest.approx(2.5)
    assert selection["aggregate_ms"] == pytest.approx(0.75)
    assert selection["pose_count"] == 192
    assert selection["map_query_count"] == 965
    assert selection["worker_count"] == 1
    assert selection["chunk_count"] == 1
    assert selection["profile_cache_hits"] == 0
    for field in (
        "selection_compute_ms",
        "selection_latency_ms",
        "validation_ms",
        "road_batch_wall_ms",
        "ranking_ms",
        "telemetry_emit_ms",
    ):
        assert field in selection
    terminal = next(
        record
        for record in records
        if record["event_type"] == "inference_result"
    )
    assert "result_processing_ms" in terminal
    assert terminal["result_processing_ms"] is not None


@pytest.mark.parametrize("failure_mode", ["raise", "short"])
def test_batch_failure_retains_executable_active_plan(
    monkeypatch,
    tmp_path,
    failure_mode,
):
    adapter, _carla_if, records = _run_k3(
        monkeypatch,
        tmp_path,
        max_episode_seconds=1.2,
        failure_mode=failure_mode,
    )

    assert adapter.batch_calls[:2] == [(0, 1, 2), (0, 1, 2)]
    admissions = [
        record
        for record in records
        if record["event_type"] == "plan_admission"
    ]
    assert admissions[0]["admission_status"] == "ACCEPT_FULLY_SAFE"
    assert admissions[1]["admission_status"] == "REJECT_RETAIN_ACTIVE"
    active_plan_id = admissions[0]["proposal_id"]
    post_failure_ticks = [
        record
        for record in records
        if record["event_type"] == "tick" and record["loop_tick_id"] >= 11
    ]
    assert post_failure_ticks
    assert all(record["active_plan_id"] == active_plan_id
               for record in post_failure_ticks)
    assert all(record["fallback_state"] == "NONE"
               for record in post_failure_ticks)


def test_first_batch_failure_without_active_plan_falls_back_closed(
    monkeypatch,
    tmp_path,
):
    adapter, carla_if, records = _run_k3(
        monkeypatch,
        tmp_path,
        max_episode_seconds=0.1,
        failure_mode="raise_first",
    )

    assert adapter.batch_calls == [(0, 1, 2)]
    selection = next(
        record
        for record in records
        if record["event_type"] == "candidate_selection"
    )
    assert selection["selected_admitted"] is False
    admission = next(
        record for record in records if record["event_type"] == "plan_admission"
    )
    assert admission["admission_status"] == "REJECT_FALLBACK_STOP"
    tick = next(record for record in records if record["event_type"] == "tick")
    assert tick["active_plan_id"] is None
    assert tick["fallback_state"] == "WAITING_FOR_PLAN"
    assert carla_if.applied_controls[-1] == (0.0, 0.0, 1.0)

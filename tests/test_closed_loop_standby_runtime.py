import importlib
import json
import sys
import types

import numpy as np
import pytest


sys.modules.setdefault(
    "carla",
    types.SimpleNamespace(
        command=types.SimpleNamespace(
            DestroyActor=lambda actor_id: ("destroy", actor_id)
        ),
        VehicleControl=lambda: types.SimpleNamespace(
            steer=0.0,
            throttle=0.0,
            brake=0.0,
        ),
        Vector3D=lambda: types.SimpleNamespace(),
    ),
)

from module.carla_safety_adapter import (  # noqa: E402
    RoadExecutionEnvelope,
    StoppingReserveProfile,
    StoppingReserveStatus,
)
from module.safety_shield import (  # noqa: E402
    ObstacleAssessment,
    RoadContainmentAssessment,
)


closed_loop = importlib.import_module("carlamayo_closed_loop")


class _FakeWorld:
    def __init__(self):
        self._settings = types.SimpleNamespace(fixed_delta_seconds=0.1)

    def get_settings(self):
        return self._settings


class _FakeEgoVehicle:
    def get_transform(self):
        return object()


class _FakeCarlaInterface:
    def __init__(self):
        self.world = _FakeWorld()
        self.ego_vehicle = _FakeEgoVehicle()
        self.tick_count = 0
        self.applied_controls = []

    def connect(self):
        pass

    def load_map(self, *_args, **_kwargs):
        pass

    def set_scenario_seed(self, _seed):
        pass

    def spawn_ego_vehicle(self, **_kwargs):
        pass

    def enable_synchronous_mode(self):
        pass

    def spawn_npcs(self, **_kwargs):
        pass

    def get_non_ego_dynamic_actor_census(self):
        return {
            "non_ego_vehicle_count": 0,
            "walker_count": 0,
            "walker_controller_count": 0,
        }

    def setup_cameras(self):
        pass

    def setup_collision_sensor(self):
        pass

    def tick(self):
        self.tick_count += 1
        return types.SimpleNamespace()

    def get_ego_state(self):
        return {"speed": 0.0}

    def update_history(self, _state):
        pass

    def get_camera_images(self):
        return np.zeros((4, 1, 1, 3), dtype=np.uint8)

    def get_history_in_local_frame(self):
        return object(), object()

    def get_collision_count(self):
        return 0

    def get_episode_collision_count(self):
        return 0

    def get_last_collision_event(self):
        return None

    def apply_control(self, steering, throttle, brake):
        self.applied_controls.append((steering, throttle, brake))

    def get_applied_control(self):
        if not self.applied_controls:
            return None
        steering, throttle, brake = self.applied_controls[-1]
        return {
            "echoed_steer": float(steering),
            "echoed_throttle": float(throttle),
            "echoed_brake": float(brake),
            "gear": 1 if throttle > 0.0 else 0,
        }

    def cleanup(self):
        pass


class _TrackingFollower:
    def compute_world_control(self, **_kwargs):
        return 0.0, 0.4, 0.0, {
            "controller_state": "TRACKING",
            "target_speed_mps": 2.0,
            "bypass_smoothing": False,
        }

    def reset_plan_progress(self, *_args):
        pass


class _StandbyRoadAdapter:
    """Give proposal 2 a safe prefix, optionally only on its first assessment."""

    def __init__(
        self,
        *,
        unsafe_on_reassessment=False,
        carla_if=None,
        stale_active_control_envelope=False,
        bridge_denial_tick=None,
        active_degrades_after_first_assessment=False,
        candidate_fully_safe=False,
    ):
        self.unsafe_on_reassessment = bool(unsafe_on_reassessment)
        self.carla_if = carla_if
        self.stale_active_control_envelope = bool(
            stale_active_control_envelope
        )
        self.bridge_denial_tick = bridge_denial_tick
        self.active_degrades_after_first_assessment = bool(
            active_degrades_after_first_assessment
        )
        self.candidate_fully_safe = bool(candidate_fully_safe)
        self.standby_assessment_count = 0
        self.active_assessment_count = 0
        self.active_assessment_statuses = []
        self.active_control_statuses = []
        self.safe_road = RoadContainmentAssessment.safe(
            sample_count=5,
            min_margin_m=0.5,
        )
        self.full_path_unsafe = RoadContainmentAssessment.unsafe(
            ("road_not_contained",),
            sample_count=5,
            min_margin_m=-0.4,
            first_bad_sample_index=4,
        )
        self.near_term_unsafe = RoadContainmentAssessment.unsafe(
            ("road_not_contained",),
            sample_count=5,
            min_margin_m=-0.2,
            first_bad_sample_index=0,
        )
        self.standby_assessment_statuses = []

    def _reserve(self, status):
        if status is StoppingReserveStatus.UNBOUNDED:
            return StoppingReserveProfile(
                raw_physical_stopping_cap_mps=None,
                target_speed_cap_mps=None,
                guard_speed_mps=2.0,
                required_stopping_distance_m=None,
                stopping_reserve_m=None,
                status=status,
            )
        return StoppingReserveProfile(
            raw_physical_stopping_cap_mps=6.0,
            target_speed_cap_mps=4.0,
            guard_speed_mps=2.0,
            required_stopping_distance_m=3.0,
            stopping_reserve_m=6.0,
            status=status,
        )

    def _full_safe_envelope(self):
        return RoadExecutionEnvelope(
            current_ego_road=self.safe_road,
            current_ego_clearance_road=self.safe_road,
            near_term_path_road=self.safe_road,
            full_path_road=self.safe_road,
            last_safe_waypoint_index=63,
            time_to_first_bad_s=None,
            distance_to_first_bad_m=None,
            target_speed_cap_mps=None,
            emergency_required=False,
            stopping_reserve_profile=self._reserve(
                StoppingReserveStatus.UNBOUNDED
            ),
        )

    def _safe_prefix_envelope(self, *, near_term_unsafe=False):
        return RoadExecutionEnvelope(
            current_ego_road=self.safe_road,
            current_ego_clearance_road=self.safe_road,
            near_term_path_road=(
                self.near_term_unsafe if near_term_unsafe else self.safe_road
            ),
            full_path_road=self.full_path_unsafe,
            last_safe_waypoint_index=5 if near_term_unsafe else 48,
            time_to_first_bad_s=0.5 if near_term_unsafe else 4.8,
            distance_to_first_bad_m=1.0 if near_term_unsafe else 9.0,
            target_speed_cap_mps=0.0 if near_term_unsafe else 4.0,
            emergency_required=False,
            stopping_reserve_profile=self._reserve(
                StoppingReserveStatus.ROBUST
            ),
        )

    def _unexecutable_envelope(self):
        return RoadExecutionEnvelope(
            current_ego_road=self.safe_road,
            current_ego_clearance_road=self.safe_road,
            near_term_path_road=self.near_term_unsafe,
            full_path_road=self.full_path_unsafe,
            last_safe_waypoint_index=None,
            time_to_first_bad_s=0.0,
            distance_to_first_bad_m=0.0,
            target_speed_cap_mps=0.0,
            emergency_required=False,
            stopping_reserve_profile=self._reserve(
                StoppingReserveStatus.FRAGILE
            ),
        )

    @staticmethod
    def _request_number(plan):
        return str(plan.plan_id).rsplit(":", 1)[-1].split("/", 1)[0]

    @classmethod
    def _is_standby_plan(cls, plan):
        return cls._request_number(plan) == "2"

    def assess_plan_road(self, *, plan, **_kwargs):
        if self._is_standby_plan(plan):
            self.standby_assessment_count += 1
            unsafe = (
                self.unsafe_on_reassessment
                and self.standby_assessment_count >= 2
            )
            envelope = (
                self._full_safe_envelope()
                if self.candidate_fully_safe
                else self._safe_prefix_envelope(near_term_unsafe=unsafe)
            )
            self.standby_assessment_statuses.append(
                envelope.near_term_path_road.status.value
            )
            return envelope

        self.active_assessment_count += 1
        envelope = (
            self._safe_prefix_envelope()
            if self.active_degrades_after_first_assessment
            and self.active_assessment_count >= 2
            else self._full_safe_envelope()
        )
        self.active_assessment_statuses.append(
            envelope.full_path_road.status.value
        )
        return envelope

    def assess(self, *, plan, **_kwargs):
        if self._is_standby_plan(plan):
            envelope = (
                self._full_safe_envelope()
                if self.candidate_fully_safe
                else self._safe_prefix_envelope()
            )
        else:
            deny_bridge = (
                self.bridge_denial_tick is not None
                and self.carla_if is not None
                and self.carla_if.tick_count >= int(self.bridge_denial_tick)
            )
            stale_control = (
                self.stale_active_control_envelope
                and self.standby_assessment_count > 0
            )
            envelope = (
                self._unexecutable_envelope()
                if deny_bridge or stale_control
                else self._full_safe_envelope()
            )
            self.active_control_statuses.append(
                envelope.near_term_path_road.status.value
            )
        return types.SimpleNamespace(
            road=self.safe_road,
            obstacles=ObstacleAssessment.safe(evaluated_actor_count=0),
            current_ego_road=self.safe_road,
            proposed_path_road=envelope.full_path_road,
            road_envelope=envelope,
        )

    def reset(self, *_args):
        pass


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _run_standby_scenario(
    monkeypatch,
    tmp_path,
    *,
    unsafe_on_reassessment=False,
    episode_seconds=3.7,
    num_traj_samples=1,
    trajectory_supplier=None,
    adapter_options=None,
    waypoint_dt_s=None,
):
    telemetry_path = tmp_path / (
        "standby-unsafe.jsonl"
        if unsafe_on_reassessment
        else "standby-activation.jsonl"
    )
    argv = [
        "--telemetry-jsonl",
        str(telemetry_path),
        "--max-episode-seconds",
        str(episode_seconds),
    ]
    if num_traj_samples != 1:
        argv.extend(["--num-traj-samples", str(num_traj_samples)])
    args = closed_loop.parse_args(argv)
    carla_if = _FakeCarlaInterface()
    options = dict(adapter_options or {})
    adapter = _StandbyRoadAdapter(
        unsafe_on_reassessment=unsafe_on_reassessment,
        carla_if=carla_if,
        **options,
    )

    monkeypatch.setattr(closed_loop, "parse_args", lambda: args)
    monkeypatch.setattr(
        closed_loop,
        "configure_cuda_linalg_library",
        lambda _name: None,
    )
    monkeypatch.setattr(
        closed_loop,
        "load_model",
        lambda _quantization, device_map: (object(), object()),
    )
    monkeypatch.setattr(closed_loop, "CARLAInterface", lambda: carla_if)
    monkeypatch.setattr(
        closed_loop,
        "OfficialPIDFollower",
        lambda *_args: _TrackingFollower(),
    )
    monkeypatch.setattr(
        closed_loop,
        "CarlaGroundTruthSafetyAdapter",
        lambda *_args: adapter,
    )
    monkeypatch.setattr(
        closed_loop,
        "prepare_model_input",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        closed_loop,
        "run_inference",
        lambda *_args, **_kwargs: (
            object(),
            {"cot": "Continue in the current lane."},
        ),
    )
    monkeypatch.setattr(
        closed_loop,
        "extract_answer_text",
        lambda extra: extra.get("answer", "") if isinstance(extra, dict) else "",
    )
    monkeypatch.setattr(
        closed_loop,
        "create_visualization_frame",
        lambda cam_img, *_args, **_kwargs: cam_img,
    )
    monkeypatch.setattr(closed_loop.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(closed_loop.time, "sleep", lambda _seconds: None)

    monkeypatch.setattr(closed_loop.cfg, "SAVE_VIDEO", False)
    monkeypatch.setattr(closed_loop.cfg, "CONTROL_DT", 0.1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_FRAMES", 1)
    monkeypatch.setattr(closed_loop.cfg, "NUM_CAMERAS", 4)
    monkeypatch.setattr(closed_loop.cfg, "IMG_HEIGHT", 1)
    monkeypatch.setattr(closed_loop.cfg, "IMG_WIDTH", 1)
    monkeypatch.setattr(closed_loop.cfg, "IMG_CHANNELS", 3)
    monkeypatch.setattr(closed_loop.cfg, "NPC_VEHICLE_COUNT", 0)
    monkeypatch.setattr(closed_loop.cfg, "NPC_WALKER_COUNT", 0)
    if waypoint_dt_s is not None:
        monkeypatch.setattr(
            closed_loop.cfg,
            "TRAJECTORY_WAYPOINT_DT",
            float(waypoint_dt_s),
        )

    valid = np.zeros((1, 64, 3), dtype=np.float64)
    valid[0, :, 0] = np.arange(1, 65, dtype=np.float64) * 0.2
    extraction_count = 0

    def extract_then_reject_later(_prediction):
        nonlocal extraction_count
        extraction_count += 1
        if trajectory_supplier is not None:
            return trajectory_supplier(extraction_count)
        result = valid.copy()
        if extraction_count >= 3:
            result[:, :, 1] = 100.0
        return result

    monkeypatch.setattr(
        closed_loop,
        "extract_trajectory_samples",
        extract_then_reject_later,
    )

    closed_loop.main()
    return _read_jsonl(telemetry_path), adapter


def test_retained_candidate_survives_invalid_result_and_activates_at_deadline(
    monkeypatch,
    tmp_path,
):
    records, adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        unsafe_on_reassessment=False,
    )
    standby_events = [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]

    assert [event["lifecycle_status"] for event in standby_events] == [
        "STORED",
        "ACTIVATED",
    ]
    stored, activated = standby_events
    assert stored["standby"]["original_admission_status"] == "ACCEPT_SAFE_PREFIX"
    assert stored["standby"]["original_handoff_status"] == (
        "RETAIN_ACTIVE_SAFETY_TIER"
    )
    assert activated["trigger"] == "RETENTION_DEADLINE"
    assert activated["reassessment"]["validity"]["valid"] is True
    assert activated["reassessment"]["alignment"]["valid"] is True
    assert activated["reassessment"]["admission_status"] == "ACCEPT_SAFE_PREFIX"
    assert activated["reassessment"]["handoff"]["status"] == (
        "ACTIVATE_RETENTION_DEADLINE"
    )
    assert adapter.standby_assessment_statuses == ["SAFE", "SAFE"]

    rejected_later = next(
        record
        for record in records
        if record["event_type"] == "inference_result"
        and record["request_id"] == 3
    )
    assert rejected_later["status"] == "rejected_plan"
    same_tick = next(
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] == rejected_later["arrival_loop_tick_id"]
    )
    assert same_tick["standby_plan_id"] == stored["standby"]["plan_id"]

    activation_handoff = next(
        record
        for record in records
        if record["event_type"] == "plan_handoff"
        and record["handoff_source"] == "standby_reassessment"
    )
    assert activation_handoff["status"] == "ACTIVATE_RETENTION_DEADLINE"
    ticks_after_activation = [
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] >= activated["loop_tick_id"]
    ]
    assert ticks_after_activation
    assert all(
        tick["active_plan_id"] == stored["standby"]["plan_id"]
        for tick in ticks_after_activation
    )
    assert all(tick["standby_plan_id"] is None for tick in ticks_after_activation)
    assert not [
        tick
        for tick in ticks_after_activation
        if tick["fallback_state"] == "WAITING_FOR_PLAN"
    ]


def test_unsafe_fresh_reassessment_discards_standby_without_reusing_envelope(
    monkeypatch,
    tmp_path,
):
    records, adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        unsafe_on_reassessment=True,
    )
    standby_events = [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]

    assert [event["lifecycle_status"] for event in standby_events] == [
        "STORED",
        "DISCARDED",
    ]
    stored, discarded = standby_events
    assert adapter.standby_assessment_count == 2
    assert adapter.standby_assessment_statuses == ["SAFE", "UNSAFE"]
    assert discarded["trigger"] == "RETENTION_DEADLINE"
    assert discarded["reason"] == "standby_admission:REJECT_RETAIN_ACTIVE"
    assert discarded["reassessment"]["admission_status"] == (
        "REJECT_RETAIN_ACTIVE"
    )
    assert (
        discarded["reassessment"]["road_envelope"]["near_term_path_road"][
            "status"
        ]
        == "UNSAFE"
    )
    assert discarded["reassessment"]["road_envelope"] != (
        stored["standby"].get("road_envelope")
    )
    assert not [
        record
        for record in records
        if record["event_type"] == "standby_plan"
        and record["lifecycle_status"] == "ACTIVATED"
    ]
    ticks_after_discard = [
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] >= discarded["loop_tick_id"]
    ]
    assert ticks_after_discard
    assert all(tick["standby_plan_id"] is None for tick in ticks_after_discard)
    assert all(
        str(tick["active_plan_id"]).endswith(":1")
        for tick in ticks_after_discard
    )


def test_selected_finite_candidate_with_nonfinite_siblings_is_stored(
    monkeypatch,
    tmp_path,
):
    def mixed_candidates(_request_number):
        samples = np.zeros((3, 64, 3), dtype=np.float64)
        samples[0] = np.nan
        samples[1, :, 0] = np.arange(1, 65, dtype=np.float64) * 0.2
        samples[2] = np.inf
        return samples

    records, _adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        episode_seconds=1.2,
        num_traj_samples=3,
        trajectory_supplier=mixed_candidates,
    )
    standby_events = [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]

    assert [event["lifecycle_status"] for event in standby_events] == [
        "STORED"
    ]
    stored = standby_events[0]
    assert stored["standby"]["selected_candidate_index"] == 1
    assert stored["standby"]["candidate_count"] == 3
    assert not [
        record
        for record in records
        if record["event_type"] == "standby_plan"
        and str(record["reason"]).startswith("standby_snapshot_error:")
    ]
    request_two = next(
        record
        for record in records
        if record["event_type"] == "inference_result"
        and record["request_id"] == 2
    )
    assert request_two["status"] == "retained_active_plan"
    evaluations = [
        record
        for record in records
        if record["event_type"] == "candidate_evaluation"
        and str(record["proposal_id"]).endswith(":2")
    ]
    assert [record["selected"] for record in evaluations] == [
        False,
        True,
        False,
    ]
    assert evaluations[0]["rejection_reason"] == "non_finite_points"
    assert evaluations[2]["rejection_reason"] == "non_finite_points"


def test_stale_unsafe_control_envelope_does_not_consume_safe_standby(
    monkeypatch,
    tmp_path,
):
    records, adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        episode_seconds=1.5,
        adapter_options={"stale_active_control_envelope": True},
    )
    standby_events = [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]

    assert [event["lifecycle_status"] for event in standby_events] == [
        "STORED"
    ]
    assert "UNSAFE" in adapter.active_control_statuses
    assert adapter.active_assessment_statuses[-1] == "SAFE"
    assert adapter.standby_assessment_count == 1
    ticks_after_retention = [
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] >= 11
    ]
    assert ticks_after_retention
    assert all(
        tick["standby_plan_id"] == standby_events[0]["standby"]["plan_id"]
        for tick in ticks_after_retention
    )
    assert not [
        event
        for event in standby_events
        if event["lifecycle_status"] in {"ACTIVATED", "DISCARDED"}
    ]


def test_bridge_road_denial_activates_standby_in_same_tick(
    monkeypatch,
    tmp_path,
):
    records, _adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        episode_seconds=4.7,
        waypoint_dt_s=0.2,
        adapter_options={"bridge_denial_tick": 46},
    )
    bridge_denial = next(
        record
        for record in records
        if record["event_type"] == "active_plan_availability"
        and record["status"] == "BRIDGE_DENIED"
    )
    activation = next(
        record
        for record in records
        if record["event_type"] == "standby_plan"
        and record["lifecycle_status"] == "ACTIVATED"
    )

    assert bridge_denial["denial_reason"] == "bridge_near_term_not_safe"
    bridge_index = records.index(bridge_denial)
    activation_index = records.index(activation)
    assert bridge_index < activation_index
    assert not [
        record
        for record in records[bridge_index + 1 : activation_index]
        if record["event_type"] == "tick"
    ]
    assert bridge_denial["source_age_s"] == pytest.approx(4.5)
    assert activation["loop_tick_id"] == 46
    assert activation["simulation_time_s"] == pytest.approx(4.6)
    assert activation["trigger"] == "ACTIVE_UNAVAILABLE"
    assert activation["reassessment"]["handoff"]["status"] == (
        "ACTIVATE_NO_ACTIVE"
    )
    activation_tick = next(
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] == activation["loop_tick_id"]
    )
    assert activation_tick["active_plan_id"] == activation["standby"]["plan_id"]
    assert activation_tick["fallback_state"] == "NONE"
    assert activation_tick["active_plan_availability_status"] == (
        "STANDARD_EXECUTION"
    )
    assert activation_tick["active_plan_availability"]["source_age_s"] == (
        pytest.approx(3.5)
    )
    assert activation_tick["active_plan_availability"][
        "remaining_horizon_s"
    ] == pytest.approx(9.3)
    assert not [
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] == activation["loop_tick_id"]
        and record["fallback_state"] == "WAITING_FOR_PLAN"
    ]


def test_handoff_uses_fresh_degraded_active_tier_not_historical_alias(
    monkeypatch,
    tmp_path,
):
    records, adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        episode_seconds=1.2,
        adapter_options={
            "active_degrades_after_first_assessment": True,
            "candidate_fully_safe": True,
        },
    )
    handoff = next(
        record
        for record in records
        if record["event_type"] == "plan_handoff"
        and str(record["candidate_plan_id"]).endswith(":2")
    )

    assert adapter.active_assessment_statuses[:2] == ["SAFE", "UNSAFE"]
    assert handoff["active_plan_admission_status_compatibility_alias"] == (
        "ACCEPT_FULLY_SAFE"
    )
    assert handoff["active_admission_status"] == "ACCEPT_SAFE_PREFIX"
    assert handoff["candidate_admission_status"] == "ACCEPT_FULLY_SAFE"
    assert handoff["status"] == "ACTIVATE_SAFETY_IMPROVEMENT"
    assert handoff["activate_candidate"] is True
    assert not [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]


def test_explicit_stop_activates_immediately_and_never_enters_standby(
    monkeypatch,
    tmp_path,
):
    moving = np.zeros((1, 64, 3), dtype=np.float64)
    moving[0, :, 0] = np.arange(1, 65, dtype=np.float64) * 0.2
    explicit_stop = np.zeros((1, 64, 3), dtype=np.float64)
    explicit_stop[0, :20, 0] = (
        np.arange(1, 21, dtype=np.float64) * 0.1
    )
    explicit_stop[0, 20:, 0] = 2.0

    def moving_then_stop(request_number):
        return moving.copy() if request_number == 1 else explicit_stop.copy()

    records, _adapter = _run_standby_scenario(
        monkeypatch,
        tmp_path,
        episode_seconds=1.2,
        trajectory_supplier=moving_then_stop,
    )
    handoff = next(
        record
        for record in records
        if record["event_type"] == "plan_handoff"
        and str(record["candidate_plan_id"]).endswith(":2")
    )

    assert handoff["candidate_admission_status"] == "ACCEPT_SAFE_PREFIX"
    assert handoff["candidate_motion_class"] == "EXPLICIT_STOP"
    assert handoff["status"] == "ACTIVATE_FRESH"
    assert handoff["reason"] == "explicit_stop_bypasses_motion_retention"
    assert handoff["activate_candidate"] is True
    assert not [
        record
        for record in records
        if record["event_type"] == "standby_plan"
    ]
    stop_validation = next(
        record
        for record in records
        if record["event_type"] == "plan_validation"
        and str(record["proposal_id"]).endswith(":2")
    )
    assert stop_validation["stop_requested"] is True
    tick = next(
        record
        for record in records
        if record["event_type"] == "tick"
        and record["loop_tick_id"] == 11
    )
    assert str(tick["active_plan_id"]).endswith(":2")

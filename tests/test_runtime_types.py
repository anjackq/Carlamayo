import json
from dataclasses import FrozenInstanceError

import pytest

from module.runtime_types import (
    ControlDecision,
    InferenceRequest,
    PlanValidation,
    SynchronizedObservation,
    TrajectoryPlan,
    VisualizationSnapshot,
    summarize_payload,
    to_json_dict,
    to_json_line,
)


class FakeArray:
    def __init__(self, shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype

    def __len__(self):
        return self.shape[0]


def test_synchronized_observation_is_frozen_and_summarizes_payloads():
    observation = SynchronizedObservation(
        frame_id=12,
        simulation_time_s=1.2,
        ego_pose_world=FakeArray((4, 4)),
        ego_velocity_world=FakeArray((3,)),
        camera_images=FakeArray((4, 1080, 1920, 3), dtype="uint8"),
        camera_ids=[0, 1, 2, 6],
        camera_intrinsics={"front": FakeArray((3, 3))},
        camera_extrinsics={"front": FakeArray((4, 4))},
        ego_history=FakeArray((4, 4, 4)),
        ego_history_frame_ids=[9, 10, 11, 12],
        ego_history_simulation_times_s=[0.9, 1.0, 1.1, 1.2],
    )

    serialized = observation.to_json_dict()

    assert observation.camera_ids == (0, 1, 2, 6)
    assert serialized["record_type"] == "SynchronizedObservation"
    assert serialized["camera_images"] == {
        "type": "FakeArray",
        "shape": [4, 1080, 1920, 3],
        "dtype": "uint8",
        "length": 4,
    }
    assert serialized["payloads_summarized"] is True
    with pytest.raises(FrozenInstanceError):
        observation.frame_id = 13


def test_synchronized_observation_validates_bundle_and_history_identity():
    base = {
        "frame_id": 12,
        "simulation_time_s": 1.2,
        "ego_pose_world": object(),
        "ego_velocity_world": object(),
        "camera_images": FakeArray((1, 2, 2, 3)),
        "camera_ids": (0,),
        "camera_intrinsics": {0: object()},
        "camera_extrinsics": {0: object()},
        "ego_history": FakeArray((2, 4, 4)),
        "ego_history_frame_ids": (11, 12),
        "ego_history_simulation_times_s": (1.1, 1.2),
    }

    with pytest.raises(ValueError, match="camera_images length"):
        SynchronizedObservation(**{**base, "camera_ids": (0, 1)})
    with pytest.raises(ValueError, match="ego_history length"):
        SynchronizedObservation(**{**base, "ego_history_frame_ids": (10, 11, 12)})
    with pytest.raises(ValueError, match="camera_intrinsics"):
        SynchronizedObservation(**{**base, "camera_intrinsics": None})


def test_inference_request_keeps_source_identity_and_revision_metadata():
    request = InferenceRequest(
        source_frame_id=25,
        source_simulation_time_s=2.5,
        capture_pose_world=FakeArray((4, 4)),
        image_frames=FakeArray((4, 4, 3, 1080, 1920), dtype="uint8"),
        ego_history_xyz=FakeArray((1, 1, 16, 3)),
        ego_history_rot=FakeArray((1, 1, 16, 3, 3)),
        camera_ids=(0, 1, 2, 6),
        submission_wall_time_s=100.25,
        navigation_prompt="Turn left",
        navigation_weight=1.2,
        prompt_revision=3,
        respawn_revision=4,
    )

    serialized = to_json_dict(request)

    assert serialized["source_frame_id"] == 25
    assert serialized["camera_ids"] == [0, 1, 2, 6]
    assert serialized["prompt_revision"] == 3
    assert serialized["respawn_revision"] == 4
    assert serialized["image_frames"]["shape"] == [4, 4, 3, 1080, 1920]


def test_trajectory_plan_validates_timestamps_and_point_counts():
    waypoint_times = [0.1 * index for index in range(1, 65)]
    plan = TrajectoryPlan(
        plan_id="plan-25-0",
        source_frame_id=25,
        source_simulation_time_s=2.5,
        capture_pose_world=FakeArray((4, 4)),
        selected_ego_frame_points=FakeArray((64, 3)),
        world_frame_points=FakeArray((64, 3)),
        waypoint_times_s=waypoint_times,
        coc_text="Keep the lane clear.",
        candidate_metadata={"scores": [0.1, 0.2]},
        inference_wall_latency_s=2.4,
        prompt_revision=3,
        respawn_revision=4,
        selected_candidate_index=1,
    )

    serialized = plan.to_json_dict()

    assert plan.waypoint_times_s == tuple(waypoint_times)
    assert serialized["candidate_metadata"] == {"scores": [0.1, 0.2]}
    assert serialized["world_frame_points"]["shape"] == [64, 3]

    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryPlan(
            plan_id="bad-times",
            source_frame_id=1,
            source_simulation_time_s=0.1,
            capture_pose_world=object(),
            selected_ego_frame_points=[(0, 0, 0), (1, 0, 0)],
            world_frame_points=[(0, 0, 0), (1, 0, 0)],
            waypoint_times_s=[0.2, 0.1],
            coc_text="",
            candidate_metadata=None,
            inference_wall_latency_s=1.0,
            prompt_revision=0,
            respawn_revision=0,
        )

    with pytest.raises(ValueError, match="does not match"):
        TrajectoryPlan(
            plan_id="bad-count",
            source_frame_id=1,
            source_simulation_time_s=0.1,
            capture_pose_world=object(),
            selected_ego_frame_points=[(0, 0, 0)] * 63,
            world_frame_points=[(0, 0, 0)] * 64,
            waypoint_times_s=waypoint_times,
            coc_text="",
            candidate_metadata=None,
            inference_wall_latency_s=1.0,
            prompt_revision=0,
            respawn_revision=0,
        )


def test_plan_validation_enforces_valid_and_rejected_state_consistency():
    accepted = PlanValidation(
        valid=True,
        rejection_reason=None,
        source_age_s=1.2,
        remaining_horizon_s=5.2,
        lateral_drift_m=0.1,
        heading_drift_deg=1.5,
        first_usable_waypoint_index=12,
    )
    rejected = PlanValidation(
        valid=False,
        rejection_reason="expired",
        source_age_s=6.5,
        remaining_horizon_s=0.0,
        lateral_drift_m=0.0,
        heading_drift_deg=0.0,
        first_usable_waypoint_index=None,
    )

    assert accepted.to_json_dict()["first_usable_waypoint_index"] == 12
    assert rejected.to_json_dict()["rejection_reason"] == "expired"

    with pytest.raises(ValueError, match="invalid plan"):
        PlanValidation(False, None, 1.0, 1.0, 0.0, 0.0, None)
    with pytest.raises(ValueError, match="valid plan"):
        PlanValidation(True, "bad", 1.0, 1.0, 0.0, 0.0, 1)


def test_control_decision_preserves_requested_and_applied_controls():
    decision = ControlDecision(
        frame_id=31,
        simulation_time_s=3.1,
        controller_state="EMERGENCY_BRAKE",
        target_point_world=(10.0, 2.0, 0.0),
        target_speed_mps=4.0,
        requested_steering=0.2,
        requested_throttle=0.5,
        requested_brake=0.0,
        applied_steering=0.0,
        applied_throttle=0.0,
        applied_brake=1.0,
        fallback_state="FALLBACK",
        fallback_reason="obstacle",
        safety_override_applied=True,
        safety_override_type="emergency_brake",
        safety_override_reason="minimum TTC",
        source_plan_id="plan-25-0",
        source_plan_frame_id=25,
    )

    serialized = decision.to_json_dict()

    assert serialized["requested_control"] == {
        "steering": 0.2,
        "throttle": 0.5,
        "brake": 0.0,
    }
    assert serialized["applied_control"] == {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 1.0,
    }
    assert serialized["target_point_world"] == [10.0, 2.0, 0.0]
    assert serialized["safety_override_applied"] is True


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("frame_id", -1),
        ("simulation_time_s", float("nan")),
        ("requested_steering", 1.01),
        ("requested_throttle", -0.01),
        ("applied_brake", 1.01),
    ],
)
def test_control_decision_rejects_invalid_identity_time_and_control_fields(
    field_name,
    field_value,
):
    kwargs = {
        "frame_id": 1,
        "simulation_time_s": 0.1,
        "controller_state": "TRACKING",
        "target_point_world": (1.0, 0.0, 0.0),
        "target_speed_mps": 2.0,
        "requested_steering": 0.0,
        "requested_throttle": 0.2,
        "requested_brake": 0.0,
        "applied_steering": 0.0,
        "applied_throttle": 0.2,
        "applied_brake": 0.0,
        "source_plan_id": "plan-1",
        "source_plan_frame_id": 1,
    }
    kwargs[field_name] = field_value

    with pytest.raises((TypeError, ValueError)):
        ControlDecision(**kwargs)


def test_safety_override_metadata_is_consistent():
    base = {
        "frame_id": 1,
        "simulation_time_s": 0.1,
        "controller_state": "WAITING",
        "target_point_world": None,
        "target_speed_mps": 0.0,
        "requested_steering": 0.0,
        "requested_throttle": 0.0,
        "requested_brake": 1.0,
        "applied_steering": 0.0,
        "applied_throttle": 0.0,
        "applied_brake": 1.0,
    }

    with pytest.raises(ValueError, match="requires type and reason"):
        ControlDecision(**base, safety_override_applied=True)
    with pytest.raises(ValueError, match="inactive safety override"):
        ControlDecision(**base, safety_override_reason="unexpected")


def test_tracking_control_requires_target_and_source_plan_identity():
    base = {
        "frame_id": 1,
        "simulation_time_s": 0.1,
        "controller_state": "TRACKING",
        "target_point_world": (1.0, 0.0, 0.0),
        "target_speed_mps": 1.0,
        "requested_steering": 0.0,
        "requested_throttle": 0.1,
        "requested_brake": 0.0,
        "applied_steering": 0.0,
        "applied_throttle": 0.1,
        "applied_brake": 0.0,
    }

    with pytest.raises(ValueError, match="source plan identity"):
        ControlDecision(**base)


def test_visualization_snapshot_tracks_source_and_current_time_and_payload_layers():
    snapshot = VisualizationSnapshot(
        current_frame_id=40,
        current_simulation_time_s=4.0,
        source_observation_frame_id=25,
        source_observation_simulation_time_s=2.5,
        source_camera_bundle=FakeArray((4, 1080, 1920, 3), dtype="uint8"),
        current_display_camera=FakeArray((1080, 1920, 3), dtype="uint8"),
        candidate_trajectories=FakeArray((4, 64, 3)),
        selected_candidate_index=2,
        selected_proposal=FakeArray((64, 3)),
        controller_reference_trajectory=FakeArray((38, 3)),
        controller_target_point=(10.0, 1.0, 0.0),
        requested_control=(0.2, 0.4, 0.0),
        ego_trail=FakeArray((20, 3)),
        applied_control=(0.0, 0.0, 1.0),
        safety_zones={"stop": [(0, 0), (1, 1)]},
        conflict_zones={},
        safety_override_applied=True,
        safety_override_type="brake",
        safety_override_reason="conflict zone",
        coc_text="Brake for the pedestrian.",
        navigation_prompt="Continue straight",
        inference_backend="REMOTE",
        inference_state="ready",
        inference_latency_s=2.3,
        plan_source_age_s=1.5,
        remaining_horizon_s=4.9,
        fallback_state="NONE",
    )

    serialized = snapshot.to_json_dict()

    assert snapshot.inference_backend == "remote"
    assert serialized["source_observation_frame_id"] == 25
    assert serialized["candidate_trajectories"]["shape"] == [4, 64, 3]
    assert serialized["requested_control"] == [0.2, 0.4, 0.0]
    assert serialized["applied_control"] == [0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="frame and time"):
        VisualizationSnapshot(
            current_frame_id=40,
            current_simulation_time_s=4.0,
            source_observation_frame_id=25,
        )


def test_json_helpers_produce_compact_parseable_json_without_raw_payloads():
    payload = FakeArray((2, 3), dtype="uint8")
    summary = summarize_payload(payload)
    line = to_json_line({"payload": payload, "finite": 1.5})

    assert summary == {
        "type": "FakeArray",
        "shape": [2, 3],
        "dtype": "uint8",
        "length": 2,
    }
    assert json.loads(line) == {"finite": 1.5, "payload": summary}
    assert "FakeArray object at" not in line


def test_json_helper_rejects_nonfinite_value_from_custom_serializer():
    class BadRecord:
        def to_json_dict(self):
            return {"nested": {"value": float("nan")}}

    with pytest.raises(ValueError, match="non-finite"):
        to_json_dict(BadRecord())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SynchronizedObservation(
            -1,
            0.0,
            object(),
            object(),
            object(),
            (0,),
            object(),
            object(),
            object(),
            (0,),
            (0.0,),
        ),
        lambda: InferenceRequest(
            1,
            float("inf"),
            object(),
            object(),
            object(),
            object(),
            (0,),
            1.0,
        ),
        lambda: VisualizationSnapshot(1, -0.1),
    ],
)
def test_contracts_reject_invalid_essential_frame_and_time_fields(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()

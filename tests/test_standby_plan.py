from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pytest

from module.standby_plan import RetainedStandbyPlan, StandbyLifecycleStatus
from module.trajectory_runtime import build_fixed_world_trajectory


def _samples(*, selected_index=1):
    samples = np.zeros((3, 64, 3), dtype=np.float64)
    samples[:, :, 0] = np.arange(1, 65, dtype=np.float64)[None, :] * 0.1
    samples[1, :, 1] = 0.05
    samples[2, :, 1] = -0.05
    return samples, selected_index


def _standby(
    *,
    plan_id="plan-7",
    source_loop_tick_id=100,
    source_elapsed_proxy_s=10.0,
    source_frame_id=500,
    source_simulation_time_s=20.0,
    retained_loop_tick_id=108,
    retained_simulation_time_s=20.8,
    **overrides,
):
    samples, selected_index = _samples()
    plan = build_fixed_world_trajectory(
        plan_id=plan_id,
        source_frame_id=source_frame_id,
        source_simulation_time_s=source_simulation_time_s,
        capture_pose_world=np.eye(4),
        model_points=samples[selected_index],
        coc_text="Continue in the current lane.",
        prompt_revision=2,
        respawn_revision=3,
        selected_candidate_index=selected_index,
    )
    values = {
        "plan": plan,
        "trajectory_samples": samples,
        "selected_candidate_index": selected_index,
        "coc_sha256": hashlib.sha256(plan.coc_text.encode()).hexdigest(),
        "original_admission_status": "ACCEPT_SAFE_PREFIX",
        "original_handoff_status": "RETAIN_ACTIVE_SAFETY_TIER",
        "inference_time_s": 4.7,
        "trajectory_timestamp_s": 12345.6,
        "source_loop_tick_id": source_loop_tick_id,
        "source_elapsed_proxy_s": source_elapsed_proxy_s,
        "retained_loop_tick_id": retained_loop_tick_id,
        "retained_simulation_time_s": retained_simulation_time_s,
        "verified_empty_road": True,
    }
    values.update(overrides)
    return RetainedStandbyPlan(**values), samples, plan


def test_lifecycle_status_values_are_stable():
    assert [status.value for status in StandbyLifecycleStatus] == [
        "STORED",
        "REPLACED",
        "ACTIVATED",
        "DISCARDED",
        "CLEARED",
    ]


def test_standby_owns_deeply_read_only_trajectory_snapshots():
    standby, source_samples, source_plan = _standby()
    selected_before = standby.trajectory_samples[1, 0].copy()
    model_before = standby.plan.model_points[0].copy()

    source_samples[1, 0] = 999.0
    # Exercise a manually replaced plan array too; standby must not alias it.
    mutable_model_points = source_plan.model_points.copy()
    mutable_plan = replace(source_plan, model_points=mutable_model_points)
    copied, _, _ = _standby(plan=mutable_plan)
    mutable_model_points[0] = 888.0

    np.testing.assert_array_equal(
        standby.trajectory_samples[1, 0],
        selected_before,
    )
    np.testing.assert_array_equal(standby.plan.model_points[0], model_before)
    np.testing.assert_array_equal(copied.plan.model_points[0], model_before)
    for array in (
        standby.trajectory_samples,
        standby.plan.capture_pose_world,
        standby.plan.model_points,
        standby.plan.world_points,
        standby.plan.waypoint_times_s,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError):
        standby.trajectory_samples[0, 0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        standby.source_loop_tick_id = 200


def test_source_ordering_is_strict_and_none_is_replaceable():
    older, _, _ = _standby()
    same_source, _, _ = _standby(
        plan_id="same-source-different-plan",
        retained_loop_tick_id=120,
        retained_simulation_time_s=22.0,
    )
    newer, _, _ = _standby(
        plan_id="newer",
        source_loop_tick_id=101,
        source_elapsed_proxy_s=10.1,
        source_frame_id=501,
        source_simulation_time_s=20.1,
    )

    assert older.source_order_key == (100, 10.0, 500, 20.0)
    assert older.is_newer_than(None) is True
    assert same_source.is_newer_than(older) is False
    assert newer.is_newer_than(older) is True
    assert older.is_newer_than(newer) is False
    with pytest.raises(TypeError):
        older.is_newer_than(object())


def test_timing_and_json_metadata_are_bounded_and_payload_free():
    standby, _, _ = _standby()

    assert standby.plan_id == "plan-7"
    assert standby.source_age_s(22.0) == pytest.approx(2.0)
    assert standby.remaining_horizon_s(22.0) == pytest.approx(4.4)
    assert standby.source_age_s(19.0) == pytest.approx(0.0)
    assert standby.remaining_horizon_s(30.0) == pytest.approx(0.0)

    metadata = standby.to_json_dict(current_simulation_time_s=22.0)
    assert metadata["source_age_s"] == pytest.approx(2.0)
    assert metadata["remaining_horizon_s"] == pytest.approx(4.4)
    assert metadata["candidate_count"] == 3
    assert metadata["source_order_key"] == [100, 10.0, 500, 20.0]
    assert metadata["original_admission_status"] == "ACCEPT_SAFE_PREFIX"
    assert "trajectory_samples" not in metadata
    assert "road_envelope" not in metadata
    assert standby.to_json_dict()["source_age_s"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inference_time_s", np.nan),
        ("inference_time_s", -0.1),
        ("trajectory_timestamp_s", np.inf),
        ("source_elapsed_proxy_s", -0.1),
        ("retained_simulation_time_s", np.nan),
        ("source_loop_tick_id", -1),
        ("retained_loop_tick_id", -1),
    ],
)
def test_rejects_nonfinite_or_negative_timing_and_indices(field, value):
    with pytest.raises(ValueError):
        _standby(**{field: value})


def test_rejects_inconsistent_retention_and_candidate_metadata():
    with pytest.raises(ValueError, match="cannot precede source_loop_tick_id"):
        _standby(retained_loop_tick_id=99)
    with pytest.raises(ValueError, match="cannot precede plan source time"):
        _standby(retained_simulation_time_s=19.9)
    with pytest.raises(ValueError, match="64-character"):
        _standby(coc_sha256="not-a-hash")
    with pytest.raises(ValueError, match="out of range"):
        _standby(selected_candidate_index=3)
    with pytest.raises(ValueError, match="must match"):
        _standby(selected_candidate_index=0)


def test_rejects_mutated_or_nonfinite_selected_sample():
    standby, samples, _ = _standby()
    samples[standby.selected_candidate_index, 4, 0] += 0.01
    with pytest.raises(ValueError, match="must match plan.model_points"):
        _standby(trajectory_samples=samples)

    invalid_samples = standby.trajectory_samples.copy()
    invalid_samples[standby.selected_candidate_index, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must match plan.model_points"):
        _standby(trajectory_samples=invalid_samples)


def test_nonfinite_unselected_sample_is_preserved_without_affecting_standby():
    standby, samples, _ = _standby()
    samples[0, 0, 0] = np.nan
    samples[2, 1, 1] = np.inf

    retained, _, _ = _standby(trajectory_samples=samples)

    assert np.isnan(retained.trajectory_samples[0, 0, 0])
    assert np.isinf(retained.trajectory_samples[2, 1, 1])
    np.testing.assert_array_equal(
        retained.trajectory_samples[retained.selected_candidate_index],
        standby.plan.model_points,
    )
    assert retained.trajectory_samples.flags.writeable is False

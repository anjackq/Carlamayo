import math
import types

import numpy as np
import pytest

from module.geometry import (
    camera_intrinsic_matrix,
    carla_ego_points_to_model,
    carla_relative_rotation_to_model,
    carla_rotation_matrix,
    model_ego_points_to_carla,
    model_ego_points_to_world,
    pose_matrix_from_components,
    pose_matrix_from_state,
    pose_matrix_from_transform,
    transform_points,
    world_points_to_model_ego,
)


def test_model_world_point_round_trip_is_below_one_micrometre():
    capture_pose = pose_matrix_from_components(
        123.4,
        -51.2,
        3.7,
        roll_deg=3.0,
        pitch_deg=-7.0,
        yaw_deg=137.0,
    )
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 2.25, -0.1],
            [12.0, -4.5, 0.8],
            [-0.25, 0.75, 1.2],
        ],
        dtype=np.float64,
    )

    world_points = model_ego_points_to_world(capture_pose, model_points)
    recovered = world_points_to_model_ego(capture_pose, world_points)

    assert float(np.max(np.abs(recovered - model_points))) < 1e-6


def test_pose_and_inverse_transform_round_trip_is_below_one_micrometre():
    pose = pose_matrix_from_components(
        -20.0,
        8.0,
        1.5,
        roll_deg=-4.0,
        pitch_deg=11.0,
        yaw_deg=-73.0,
    )
    local_points = np.array(
        [[0.0, 0.0, 0.0], [3.0, -1.0, 0.5], [-2.0, 4.0, -0.2]],
        dtype=np.float64,
    )

    world_points = transform_points(pose, local_points)
    recovered = transform_points(np.linalg.inv(pose), world_points)

    assert float(np.max(np.abs(recovered - local_points))) < 1e-6
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0, abs=1e-12)


def test_carla_y_right_and_model_y_left_have_opposite_signs():
    carla_points = np.array(
        [[5.0, 2.0, 1.0], [5.0, -3.0, 1.0]],
        dtype=np.float64,
    )

    model_points = carla_ego_points_to_model(carla_points)
    recovered = model_ego_points_to_carla(model_points)

    np.testing.assert_allclose(model_points[:, 0], carla_points[:, 0])
    np.testing.assert_allclose(model_points[:, 1], -carla_points[:, 1])
    np.testing.assert_allclose(model_points[:, 2], carla_points[:, 2])
    np.testing.assert_allclose(recovered, carla_points, atol=1e-12)


def test_world_path_anchored_at_capture_does_not_follow_later_ego_pose():
    source_pose = pose_matrix_from_components(10.0, 20.0, 0.0, yaw_deg=90.0)
    later_pose = pose_matrix_from_components(-30.0, 50.0, 0.0, yaw_deg=-45.0)
    model_path = np.array(
        [[1.0, 0.0, 0.0], [5.0, 1.0, 0.0], [9.0, 2.0, 0.0]],
        dtype=np.float64,
    )

    fixed_world_path = model_ego_points_to_world(source_pose, model_path)
    original_values = fixed_world_path.copy()
    path_if_incorrectly_reanchored = model_ego_points_to_world(later_pose, model_path)

    np.testing.assert_array_equal(fixed_world_path, original_values)
    assert not np.allclose(fixed_world_path, path_if_incorrectly_reanchored)
    # At source yaw +90 degrees, model forward maps to positive world y.
    np.testing.assert_allclose(fixed_world_path[0], [10.0, 21.0, 0.0], atol=1e-12)


def test_relative_rotation_basis_flip_reverses_yaw_and_round_trips():
    carla_yaw = carla_rotation_matrix(0.0, 0.0, 30.0)
    model_rotation = carla_relative_rotation_to_model(carla_yaw)
    expected_model_yaw = np.array(
        [
            [math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0],
            [-math.sin(math.radians(30.0)), math.cos(math.radians(30.0)), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(model_rotation, expected_model_yaw, atol=1e-12)
    np.testing.assert_allclose(
        carla_relative_rotation_to_model(model_rotation),
        carla_yaw,
        atol=1e-12,
    )
    np.testing.assert_allclose(model_rotation.T @ model_rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(model_rotation) == pytest.approx(1.0, abs=1e-12)


def test_pose_factories_copy_transform_and_match_state_components():
    matrix = pose_matrix_from_components(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    transform = types.SimpleNamespace(get_matrix=lambda: matrix)
    copied = pose_matrix_from_transform(transform)
    state_pose = pose_matrix_from_state(
        {"x": 1.0, "y": 2.0, "z": 3.0, "roll": 4.0, "pitch": 5.0, "yaw": 6.0}
    )

    matrix[0, 3] = 999.0

    assert copied[0, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(copied, state_pose, atol=1e-12)


def test_camera_intrinsics_have_expected_focal_length_and_principal_point():
    width, height, fov = 1920, 1080, 120.0
    intrinsic = camera_intrinsic_matrix(width, height, fov)
    expected_focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    assert intrinsic.dtype == np.float64
    assert intrinsic[0, 0] == pytest.approx(expected_focal)
    assert intrinsic[1, 1] == pytest.approx(expected_focal)
    assert intrinsic[0, 2] == pytest.approx(width / 2.0)
    assert intrinsic[1, 2] == pytest.approx(height / 2.0)
    np.testing.assert_allclose(intrinsic[2], [0.0, 0.0, 1.0])

    principal_axis_pixel = intrinsic @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(principal_axis_pixel[:2], [width / 2.0, height / 2.0])


@pytest.mark.parametrize(
    ("width", "height", "fov"),
    [(0, 1080, 90.0), (1920, -1, 90.0), (1920, 1080, 0.0), (1920, 1080, 180.0)],
)
def test_camera_intrinsics_reject_invalid_calibration(width, height, fov):
    with pytest.raises(ValueError):
        camera_intrinsic_matrix(width, height, fov)

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from module.camera_geometry import FThetaProjection, PinholeProjection
from module.visualization import (
    VideoRecorder,
    create_open_loop_visualization_frame,
    create_visualization_frame,
    project_trajectory_to_image,
    project_world_polyline_to_image,
    project_world_points_to_camera,
    project_world_trajectory_to_image,
    save_open_loop_video,
)


def test_project_trajectory_to_image_draws_selected_path_in_red():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.1, 0.0], [8.0, 0.2, 0.0]]])

    rendered = project_trajectory_to_image(image, trajectory, selected_idx=0)

    assert rendered.shape == image.shape
    assert rendered[..., 0].max() == 255
    assert rendered.sum() > 0


def test_project_trajectory_to_image_accepts_torch_tensor():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    trajectory = torch.tensor([[[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]], dtype=torch.float32)

    rendered = project_trajectory_to_image(image, trajectory)

    assert rendered.shape == image.shape
    assert rendered.sum() > 0


def test_project_trajectory_to_image_rejects_invalid_rank():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    trajectory = np.zeros((2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Expected trajectory with ndim 2 or 3"):
        project_trajectory_to_image(image, trajectory)


def test_exact_world_projection_uses_carla_camera_axes():
    intrinsic = np.array(
        [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    pixels, valid = project_world_points_to_camera(points, np.eye(4), intrinsic)

    np.testing.assert_allclose(pixels[:3], [[80.0, 60.0], [90.0, 60.0], [80.0, 50.0]])
    assert valid.tolist() == [True, True, True, False]


def test_world_projection_uses_model_facing_projection_object():
    projection = PinholeProjection(
        width=160,
        height=120,
        fx=100.0,
        fy=100.0,
        cx=80.0,
        cy=60.0,
    )
    points = np.array(
        [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]]
    )

    pixels, valid = project_world_points_to_camera(
        points,
        np.eye(4),
        projection,
    )

    np.testing.assert_allclose(
        pixels,
        [[80.0, 60.0], [90.0, 60.0], [80.0, 50.0]],
        atol=1e-9,
    )
    assert valid.all()


def test_ftheta_world_projection_matches_known_angles():
    projection = FThetaProjection(
        width=100,
        height=80,
        cx=50.0,
        cy=40.0,
        angle_to_radius_coefficients=(0.0, 50.0),
        radius_to_angle_coefficients=(0.0, 1.0 / 50.0),
    )
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 10.0 * math.tan(0.4), 0.0],
            [10.0, 0.0, 10.0 * math.tan(0.2)],
        ]
    )

    pixels, valid = project_world_points_to_camera(
        points,
        np.eye(4),
        projection,
    )

    np.testing.assert_allclose(
        pixels,
        [[50.0, 40.0], [70.0, 40.0], [50.0, 30.0]],
        atol=1e-6,
    )
    assert valid.all()


def test_calibrated_world_trajectory_draws_on_the_current_image():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsic = np.array(
        [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array([[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]])

    rendered = project_world_trajectory_to_image(image, points, np.eye(4), intrinsic)

    assert rendered.shape == image.shape
    assert rendered[..., 1].max() == 255


def test_calibrated_world_trajectory_distinguishes_safe_future_and_first_bad():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsic = np.array(
        [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [6.0, 3.0, 0.0],
            [8.0, 6.0, 0.0],
        ]
    )

    rendered = project_world_trajectory_to_image(
        image,
        points,
        np.eye(4),
        intrinsic,
        last_safe_waypoint_index=1,
    )

    assert np.any(np.all(rendered == (0, 255, 80), axis=2))
    assert np.any(np.all(rendered == (255, 210, 0), axis=2))
    assert np.any(np.all(rendered == (255, 0, 0), axis=2))


def test_diagnostic_world_polyline_uses_distinct_color_and_dash_pattern():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    intrinsic = np.array(
        [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array(
        [[2.0, 0.0, 0.0], [4.0, 0.1, 0.0], [6.0, 0.2, 0.0], [8.0, 0.3, 0.0]]
    )

    rendered = project_world_polyline_to_image(
        image,
        points,
        np.eye(4),
        intrinsic,
        color=(255, 80, 255),
        dashed=True,
    )

    assert np.any(np.all(rendered == (255, 80, 255), axis=2))


def test_create_visualization_frame_preserves_rgb_shape_and_adds_overlay():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])

    frame = create_visualization_frame(
        image,
        trajectory,
        selected_idx=0,
        frame_count=3,
        inference_time=0.25,
        cot_text="Stop because the light is red.",
        speed_kmh=12.0,
        steering=0.1,
        navigation_text="Stop at the light",
        navigation_weight=1.0,
        paused=True,
    )

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert frame.sum() > 0


def test_create_open_loop_visualization_frame_adds_header_and_cot_overlay():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])

    frame = create_open_loop_visualization_frame(
        image,
        trajectory,
        frame_count=1,
        total_frames=2,
        inference_time=0.12,
        cot_text="Follow the lane.",
    )

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert frame.sum() > 0


def test_video_recorder_no_frames_does_not_create_output(tmp_path):
    output_path = tmp_path / "empty.mp4"

    VideoRecorder(output_path).save()

    assert not output_path.exists()


def test_video_recorder_streams_frames_and_publishes_preview(tmp_path):
    output_path = tmp_path / "streamed.mp4"
    preview_path = tmp_path / "latest.jpg"
    recorder = VideoRecorder(output_path, fps=5, preview_path=preview_path)

    recorder.add_frame(np.zeros((80, 120, 3), dtype=np.uint8))

    assert recorder.frame_count == 1
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0

    recorder.save()

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_open_loop_video_writes_nonempty_video(tmp_path):
    output_path = tmp_path / "open_loop.mp4"
    predictions = [np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])]
    camera_images = [np.zeros((80, 120, 3), dtype=np.uint8)]

    save_open_loop_video(
        predictions,
        camera_images,
        ["Follow the lane."],
        [0.1],
        output_path,
        fps=5,
    )

    assert Path(output_path).exists()
    assert output_path.stat().st_size > 0

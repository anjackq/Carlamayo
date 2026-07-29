import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from module.camera_geometry import (
    CAMERA_PROFILE_RIG_FRAME,
    CameraCalibrationError,
    FThetaProjection,
    PinholeProjection,
    build_ftheta_remap,
    camera_pose_in_carla_actor,
    load_camera_rig_profile,
    remap_rgb_image,
    validate_vehicle_dimensions,
)
from module.geometry import carla_rotation_matrix


def _camera_payload(name, camera_id, *, coefficients=(0.0, 800.0)):
    return {
        "name": name,
        "alpamayo_id": camera_id,
        "resolution": [1920, 1080],
        "sensor_to_rig_matrix": [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 1.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "projection": {
            "type": "ftheta",
            "principal_point": [960.0, 540.0],
            "angle_to_radius_coefficients": list(coefficients),
            "radius_to_angle_coefficients": [0.0, 1.0 / 800.0],
        },
    }


def _profile_payload():
    return {
        "schema_version": 1,
        "profile_id": "unit-test-profile",
        "dataset_revision": "unit-test-revision",
        "source_clip_id": "synthetic",
        "platform_class": "hyperion_8",
        "rig_frame": CAMERA_PROFILE_RIG_FRAME,
        "vehicle": {
            "length": 4.8,
            "width": 2.1,
            "height": 1.5,
            "rear_axle_to_bbox_center": 1.35,
        },
        "cameras": [
            _camera_payload("cam_front_left", 0),
            _camera_payload("cam_front_wide", 1),
            _camera_payload("cam_front_right", 2),
            _camera_payload("cam_front_tele", 6),
        ],
    }


def _write_profile(tmp_path, mutate=None):
    payload = _profile_payload()
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bbox():
    return SimpleNamespace(
        location=SimpleNamespace(x=0.1, y=-0.2, z=0.8),
        extent=SimpleNamespace(x=2.4, y=1.05, z=0.75),
    )


def test_pinhole_pixel_ray_round_trip_and_invalid_masks():
    projection = PinholeProjection.from_horizontal_fov(1920, 1080, 120.0)
    pixels = np.array([[0.0, 0.0], [960.0, 540.0], [1919.0, 1079.0]])

    rays, valid_pixels = projection.pixel_to_ray(pixels)
    reconstructed, valid_rays = projection.ray_to_pixel(rays)

    assert valid_pixels.tolist() == [True, True, True]
    assert valid_rays.tolist() == [True, True, True]
    np.testing.assert_allclose(reconstructed, pixels, atol=1e-9)
    assert not projection.ray_to_pixel([[0.0, 0.0, -1.0]])[1][0]


def test_ftheta_pixel_ray_round_trip_is_subpixel_and_center_is_finite():
    projection = FThetaProjection(
        width=1920,
        height=1080,
        cx=958.25,
        cy=541.5,
        angle_to_radius_coefficients=(0.0, 800.0),
        radius_to_angle_coefficients=(0.0, 1.0 / 800.0),
    )
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1919.0, 47),
        np.linspace(0.0, 1079.0, 29),
    )
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    pixels = np.concatenate([pixels, [[projection.cx, projection.cy]]], axis=0)

    rays, valid_pixels = projection.pixel_to_ray(pixels)
    reconstructed, valid_rays = projection.ray_to_pixel(rays)
    errors = np.linalg.norm(reconstructed - pixels, axis=1)

    assert valid_pixels.all()
    assert valid_rays.all()
    assert np.isfinite(rays).all()
    assert np.median(errors) < 0.05
    assert np.quantile(errors, 0.99) < 0.25
    np.testing.assert_allclose(rays[-1], [0.0, 0.0, 1.0], atol=1e-12)


def test_ftheta_rejects_non_monotonic_polynomial():
    with pytest.raises(CameraCalibrationError, match="strictly monotonic"):
        FThetaProjection(
            width=1920,
            height=1080,
            cx=960.0,
            cy=540.0,
            angle_to_radius_coefficients=(0.0, -800.0),
            radius_to_angle_coefficients=(0.0, -1.0 / 800.0),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (
            lambda payload: payload["cameras"].reverse(),
            r"left/wide/right/tele",
        ),
        (
            lambda payload: payload["cameras"][0].update(alpamayo_id=3),
            r"\[0, 1, 2, 6\]",
        ),
        (
            lambda payload: payload["cameras"][0].update(resolution=[1280, 720]),
            "1920x1080",
        ),
        (
            lambda payload: payload["cameras"][0]["projection"].update(
                principal_point=[math.nan, 540.0]
            ),
            "finite",
        ),
        (
            lambda payload: payload["cameras"][0]["projection"].update(
                angle_to_radius_coefficients=[0.0, -800.0]
            ),
            "strictly monotonic",
        ),
    ],
)
def test_profile_rejects_invalid_contract(tmp_path, mutate, message):
    path = _write_profile(tmp_path, mutate)

    with pytest.raises(CameraCalibrationError, match=message):
        load_camera_rig_profile(path)


def test_profile_hash_pose_conversion_and_vehicle_validation(tmp_path):
    profile = load_camera_rig_profile(_write_profile(tmp_path))
    bbox = _bbox()

    errors = validate_vehicle_dimensions(profile, bbox)
    pose = camera_pose_in_carla_actor(profile, profile.cameras[0], bbox)
    rotation = carla_rotation_matrix(pose["roll"], pose["pitch"], pose["yaw"])

    assert len(profile.sha256) == 64
    assert max(errors.values()) < 0.10
    assert pose["x"] == pytest.approx(0.1 + 2.0 - 1.35, abs=1e-12)
    assert pose["y"] == pytest.approx(-0.2 - 0.5, abs=1e-12)
    assert pose["z"] == pytest.approx(0.8 - 0.75 + 1.4, abs=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)


def test_vehicle_dimension_mismatch_fails_closed(tmp_path):
    profile = load_camera_rig_profile(_write_profile(tmp_path))
    bbox = _bbox()
    bbox.extent.x = 3.0

    with pytest.raises(CameraCalibrationError, match="exceeds 10%"):
        validate_vehicle_dimensions(profile, bbox)


def test_overscan_remap_covers_output_and_preserves_uint8_contract():
    projection = FThetaProjection(
        width=1920,
        height=1080,
        cx=960.0,
        cy=540.0,
        angle_to_radius_coefficients=(0.0, 800.0),
        radius_to_angle_coefficients=(0.0, 1.0 / 800.0),
    )

    remap = build_ftheta_remap(projection)
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    source[..., 0] = np.arange(1920, dtype=np.uint16)[None, :] % 256
    warped = remap_rgb_image(source, remap)

    assert remap.source_fov_deg < 160.0
    assert remap.valid_ratio >= 0.995
    assert warped.shape == (1080, 1920, 3)
    assert warped.dtype == np.uint8

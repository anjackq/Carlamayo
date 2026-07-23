import os

import numpy as np
import pytest

from module.camera_fixture import (
    CameraFixtureError,
    load_camera_fixture,
    save_camera_fixture,
)


def _fixture_payload():
    return {
        "images_array": np.zeros((4, 4, 1080, 1920, 3), dtype=np.uint8),
        "history_xyz": np.zeros((16, 3), dtype=np.float32),
        "history_rot": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
        "camera_ids": (0, 1, 2, 6),
        "frame_ids": (20, 21, 22, 23),
        "simulation_times_s": (2.0, 2.1, 2.2, 2.3),
        "capture_pose_world": np.eye(4),
        "metadata": {
            "camera_alignment_mode": "baseline",
            "camera_profile_sha256": None,
            "navigation_text": "Continue straight for 50m.",
            "map": "Town03",
            "spawn_index": 0,
            "scenario_seed": 0,
            "synthetic_scene": {"empty_road": True},
        },
    }


def test_camera_fixture_round_trip_is_private_and_pickle_free(tmp_path):
    path = tmp_path / "fixture.npz"

    identity = save_camera_fixture(path, **_fixture_payload())
    loaded = load_camera_fixture(path)

    assert len(identity["fixture_id"]) == 16
    assert len(identity["fixture_sha256"]) == 64
    assert loaded["image_frames"].shape == (4, 4, 3, 1080, 1920)
    assert loaded["camera_ids"].tolist() == [0, 1, 2, 6]
    assert loaded["metadata"]["fixture_id"] == identity["fixture_id"]
    assert loaded["fixture_sha256"] == identity["fixture_sha256"]
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_camera_fixture_rejects_incomplete_history_contract(tmp_path):
    payload = _fixture_payload()
    payload["history_xyz"] = np.zeros((15, 3), dtype=np.float32)

    with pytest.raises(CameraFixtureError, match=r"shape \(16,3\)"):
        save_camera_fixture(tmp_path / "bad.npz", **payload)


def test_camera_fixture_does_not_overwrite(tmp_path):
    path = tmp_path / "fixture.npz"
    save_camera_fixture(path, **_fixture_payload())

    with pytest.raises(FileExistsError):
        save_camera_fixture(path, **_fixture_payload())

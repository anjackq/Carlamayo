"""Private frozen-input fixtures for deterministic camera ablations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_FIXTURE_SCHEMA_VERSION = 1


class CameraFixtureError(ValueError):
    """Raised when a frozen camera fixture violates its local contract."""


def _json_bytes(metadata: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            metadata,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CameraFixtureError("fixture metadata must be finite and JSON-safe") from exc
    return text.encode("utf-8")


def _validated_fixture_arrays(
    images_array,
    history_xyz,
    history_rot,
    camera_ids,
    frame_ids,
    simulation_times_s,
    capture_pose_world,
):
    images = np.asarray(images_array)
    if images.shape != (4, 4, 1080, 1920, 3) or images.dtype != np.uint8:
        raise CameraFixtureError(
            "images_array must be uint8 with shape (4,4,1080,1920,3)"
        )
    # Store the exact model-facing tensor layout, not the CARLA decoder layout.
    image_frames = np.transpose(images, (0, 1, 4, 2, 3)).copy()
    xyz = np.asarray(history_xyz, dtype=np.float32)
    rotations = np.asarray(history_rot, dtype=np.float32)
    if xyz.shape != (16, 3):
        raise CameraFixtureError("history_xyz must have shape (16,3)")
    if rotations.shape != (16, 3, 3):
        raise CameraFixtureError("history_rot must have shape (16,3,3)")
    ids = np.asarray(camera_ids, dtype=np.int64)
    if tuple(ids.tolist()) != (0, 1, 2, 6):
        raise CameraFixtureError("camera IDs must be [0,1,2,6]")
    frames = np.asarray(frame_ids, dtype=np.int64)
    timestamps = np.asarray(simulation_times_s, dtype=np.float64)
    if frames.shape != (4,) or timestamps.shape != (4,):
        raise CameraFixtureError("frame IDs and timestamps must each contain four values")
    if np.any(np.diff(frames) != 1):
        raise CameraFixtureError("fixture camera frames must be consecutive")
    if np.any(np.diff(timestamps) <= 0.0):
        raise CameraFixtureError("fixture camera timestamps must be strictly increasing")
    pose = np.asarray(capture_pose_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise CameraFixtureError("capture_pose_world must be a finite 4x4 matrix")
    if not (np.isfinite(xyz).all() and np.isfinite(rotations).all()):
        raise CameraFixtureError("fixture history must be finite")
    return {
        "image_frames": image_frames,
        "ego_history_xyz": xyz,
        "ego_history_rot": rotations,
        "camera_ids": ids,
        "frame_ids": frames,
        "simulation_times_s": timestamps,
        "capture_pose_world": pose,
    }


def save_camera_fixture(
    path,
    *,
    images_array,
    history_xyz,
    history_rot,
    camera_ids,
    frame_ids,
    simulation_times_s,
    capture_pose_world,
    metadata,
) -> dict[str, str]:
    """Atomically write one permission-0600 synthetic frozen-input fixture."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"fixture already exists: {destination}")
    arrays = _validated_fixture_arrays(
        images_array,
        history_xyz,
        history_rot,
        camera_ids,
        frame_ids,
        simulation_times_s,
        capture_pose_world,
    )
    complete_metadata = {
        "schema_version": CAMERA_FIXTURE_SCHEMA_VERSION,
        **dict(metadata),
    }
    metadata_bytes = _json_bytes(complete_metadata)
    fixture_id = hashlib.sha256(metadata_bytes).hexdigest()[:16]
    complete_metadata["fixture_id"] = fixture_id
    metadata_bytes = _json_bytes(complete_metadata)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                **arrays,
                metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"fixture_id": fixture_id, "fixture_sha256": sha256}


def load_camera_fixture(path) -> dict[str, Any]:
    """Load a frozen fixture without allowing pickle-backed object arrays."""

    source = Path(path).expanduser().resolve()
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CameraFixtureError(f"cannot load fixture {source}: {exc}") from exc
    required = {
        "image_frames",
        "ego_history_xyz",
        "ego_history_rot",
        "camera_ids",
        "frame_ids",
        "simulation_times_s",
        "capture_pose_world",
        "metadata_json",
    }
    missing = required - set(archive.files)
    if missing:
        archive.close()
        raise CameraFixtureError(f"fixture is missing fields: {sorted(missing)}")
    try:
        result = {name: archive[name] for name in required - {"metadata_json"}}
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CameraFixtureError("fixture metadata is invalid") from exc
    finally:
        archive.close()
    if metadata.get("schema_version") != CAMERA_FIXTURE_SCHEMA_VERSION:
        raise CameraFixtureError("unsupported fixture schema_version")
    if result["image_frames"].shape != (4, 4, 3, 1080, 1920):
        raise CameraFixtureError("fixture image tensor has an invalid shape")
    if result["image_frames"].dtype != np.uint8:
        raise CameraFixtureError("fixture image tensor must be uint8")
    result["metadata"] = metadata
    result["fixture_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result

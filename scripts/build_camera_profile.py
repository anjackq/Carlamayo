#!/usr/bin/env python3
"""Build a private CarlaMayo camera profile from PhysicalAI-AV metadata only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from physical_ai_av import PhysicalAIAVDatasetInterface


CAMERA_MAPPING = (
    ("cam_front_left", 0, "camera_cross_left_120fov"),
    ("cam_front_wide", 1, "camera_front_wide_120fov"),
    ("cam_front_right", 2, "camera_cross_right_120fov"),
    ("cam_front_tele", 6, "camera_front_tele_30fov"),
)
RIG_FRAME = "rear_axle_ground_x_forward_y_left_z_up"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create a local, permission-0600 camera profile from one authorized "
            "PhysicalAI-AV clip. Camera video is never requested."
        )
    )
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-id",
        default="",
        help="Optional non-sensitive local profile identifier.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional PhysicalAI-AV dataset revision; defaults to resolved main.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local profile.",
    )
    return parser.parse_args(argv)


def _json_number_list(values):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("calibration contains non-finite values")
    return array.tolist()


def build_profile(avdi, clip_id, profile_id):
    features = avdi.features.CALIBRATION
    extrinsics = avdi.get_clip_feature(
        clip_id,
        features.SENSOR_EXTRINSICS,
        maybe_stream=True,
    )
    intrinsics = avdi.get_clip_feature(
        clip_id,
        features.CAMERA_INTRINSICS,
        maybe_stream=True,
    )
    dimensions = avdi.get_clip_feature(
        clip_id,
        features.VEHICLE_DIMENSIONS,
        maybe_stream=True,
    )
    platform_class = str(avdi.data_collection.loc[clip_id, "platform_class"])

    cameras = []
    for local_name, camera_id, dataset_name in CAMERA_MAPPING:
        if dataset_name not in extrinsics.sensor_poses:
            raise KeyError(f"clip has no extrinsics for {dataset_name}")
        if dataset_name not in intrinsics.camera_models:
            raise KeyError(f"clip has no intrinsics for {dataset_name}")
        sensor_pose = extrinsics.sensor_poses[dataset_name]
        camera_model = intrinsics.camera_models[dataset_name]
        if type(camera_model).__name__ != "FThetaCameraModel":
            raise TypeError(f"{dataset_name} is not an FThetaCameraModel")
        cameras.append(
            {
                "name": local_name,
                "alpamayo_id": camera_id,
                "resolution": [
                    int(camera_model.width),
                    int(camera_model.height),
                ],
                "sensor_to_rig_matrix": _json_number_list(sensor_pose.as_matrix()),
                "projection": {
                    "type": "ftheta",
                    "principal_point": _json_number_list(camera_model.principal_point),
                    "angle_to_radius_coefficients": _json_number_list(
                        camera_model.th2r.coef
                    ),
                    "radius_to_angle_coefficients": _json_number_list(
                        camera_model.r2th.coef
                    ),
                },
            }
        )

    return {
        "schema_version": 1,
        "profile_id": profile_id or f"{platform_class}-{clip_id[:8]}",
        "dataset_revision": str(avdi.revision),
        "source_clip_id": clip_id,
        "platform_class": platform_class,
        "rig_frame": RIG_FRAME,
        "vehicle": {
            "length": float(dimensions.length),
            "width": float(dimensions.width),
            "height": float(dimensions.height),
            "rear_axle_to_bbox_center": float(dimensions.rear_axle_to_bbox_center),
        },
        "cameras": cameras,
    }


def _write_private_json(path: Path, payload, *, overwrite: bool):
    destination = path.expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass --overwrite to replace it"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False, allow_nan=False)
            stream.write("\n")
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
    return destination


def main(argv=None):
    args = parse_args(argv)
    avdi = PhysicalAIAVDatasetInterface(revision=args.revision)
    profile = build_profile(avdi, args.clip_id, args.profile_id.strip())
    destination = _write_private_json(
        args.output,
        profile,
        overwrite=args.overwrite,
    )
    # Intentionally print only identifiers. Calibration coefficients and
    # transforms are private gated data and must not enter scheduler logs.
    print(f"Camera profile written: {destination}")
    print(f"Profile ID: {profile['profile_id']}")
    print(f"Dataset revision: {profile['dataset_revision']}")


if __name__ == "__main__":
    main()

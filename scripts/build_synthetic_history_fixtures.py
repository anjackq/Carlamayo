#!/usr/bin/env python3
"""Derive private ego-history sensitivity fixtures from one frozen image bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module import config as cfg  # noqa: E402
from module.camera_fixture import load_camera_fixture, save_camera_fixture  # noqa: E402


DEFAULT_CONDITIONS = (
    ("stationary", "stationary", 0.0),
    ("steady-0p5", "steady", 0.5),
    ("steady-1p0", "steady", 1.0),
    ("steady-2p0", "steady", 2.0),
    ("steady-5p0", "steady", 5.0),
    ("launch-0p5", "launch", 0.5),
    ("launch-1p0", "launch", 1.0),
    ("launch-2p0", "launch", 2.0),
)


def synthetic_history_xyz(kind: str, target_speed_mps: float) -> np.ndarray:
    """Return 16 model-frame poses ending at the capture origin."""

    kind = str(kind).strip().lower()
    speed = float(target_speed_mps)
    if kind not in {"stationary", "steady", "launch"}:
        raise ValueError("history kind must be stationary, steady, or launch")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("target speed must be finite and nonnegative")
    intervals = int(cfg.NUM_HISTORY) - 1
    if kind == "stationary":
        interval_speeds = np.zeros(intervals, dtype=np.float64)
    elif kind == "steady":
        interval_speeds = np.full(intervals, speed, dtype=np.float64)
    else:
        node_speeds = np.linspace(0.0, speed, int(cfg.NUM_HISTORY))
        interval_speeds = 0.5 * (node_speeds[:-1] + node_speeds[1:])
    distance = np.concatenate(
        [[0.0], np.cumsum(interval_speeds * float(cfg.CONTROL_DT))]
    )
    xyz = np.zeros((int(cfg.NUM_HISTORY), 3), dtype=np.float32)
    xyz[:, 0] = (distance - distance[-1]).astype(np.float32)
    return xyz


def build_fixtures(
    base_path: Path,
    output_dir: Path,
    *,
    navigation_text: str,
) -> dict[str, dict[str, str]]:
    if output_dir == REPOSITORY_ROOT or REPOSITORY_ROOT in output_dir.parents:
        raise ValueError("synthetic fixtures must remain outside the repository")
    base = load_camera_fixture(base_path)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    images_hwc = np.transpose(
        base["image_frames"],
        (0, 1, 3, 4, 2),
    )
    results = {}
    for label, kind, speed in DEFAULT_CONDITIONS:
        destination = output_dir / f"{label}.npz"
        metadata = {
            **{
                key: value
                for key, value in base["metadata"].items()
                if key not in {"fixture_id", "navigation_text", "navigation_context"}
            },
            "experiment": "synthetic_ego_history_sensitivity",
            "source_fixture_id": base["metadata"].get("fixture_id"),
            "source_fixture_sha256": base["fixture_sha256"],
            "history_kind": kind,
            "target_speed_mps": float(speed),
            "actual_speed_mps": float(speed),
            "navigation_text": str(navigation_text),
            "navigation_weight": 1.0,
            "synthetic_history_only": True,
        }
        results[label] = save_camera_fixture(
            destination,
            images_array=images_hwc,
            history_xyz=synthetic_history_xyz(kind, speed),
            history_rot=base["ego_history_rot"],
            camera_ids=base["camera_ids"],
            frame_ids=base["frame_ids"],
            simulation_times_s=base["simulation_times_s"],
            capture_pose_world=base["capture_pose_world"],
            metadata=metadata,
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_fixture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--navigation-text",
        default="Continue in the current lane.",
    )
    args = parser.parse_args(argv)
    args.base_fixture = args.base_fixture.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = build_fixtures(
        args.base_fixture,
        args.output_dir,
        navigation_text=args.navigation_text,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

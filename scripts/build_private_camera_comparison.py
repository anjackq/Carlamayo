#!/usr/bin/env python3
"""Build private PhysicalAI/CARLA camera comparison contact sheets.

The generated artifacts contain gated PhysicalAI frames and exact calibration
translations.  They must stay outside the repository and are written with
owner-only permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from physical_ai_av import PhysicalAIAVDatasetInterface

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.camera_fixture import load_camera_fixture  # noqa: E402


CAMERAS = (
    ("cross-left", 0, "CAMERA_CROSS_LEFT_120FOV"),
    ("front-wide", 1, "CAMERA_FRONT_WIDE_120FOV"),
    ("cross-right", 2, "CAMERA_CROSS_RIGHT_120FOV"),
    ("front-tele", 6, "CAMERA_FRONT_TELE_30FOV"),
)
CELL_SIZE = (480, 270)
HEADER_HEIGHT = 62


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create private, owner-readable comparison sheets from an authorized "
            "PhysicalAI clip and synthetic CARLA frozen-input fixtures."
        )
    )
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-profile", type=Path, required=True)
    parser.add_argument(
        "--physical-t0-us",
        type=int,
        nargs="+",
        default=(5_100_000, 15_100_000, 25_100_000),
    )
    parser.add_argument(
        "--fixture",
        action="append",
        required=True,
        metavar="MODE=PATH",
        help=(
            "CARLA camera fixture. Provide baseline, projection-only, and "
            "pose-projection for the complete comparison."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _parse_fixtures(values):
    fixtures = {}
    for value in values:
        mode, separator, raw_path = value.partition("=")
        mode = mode.strip()
        if not separator or not mode or not raw_path.strip():
            raise ValueError(f"invalid --fixture value: {value!r}")
        if mode in fixtures:
            raise ValueError(f"duplicate fixture mode: {mode}")
        fixtures[mode] = Path(raw_path).expanduser().resolve()
    expected = {"baseline", "projection-only", "pose-projection"}
    if set(fixtures) != expected:
        raise ValueError(
            f"fixtures must contain exactly {sorted(expected)}, got {sorted(fixtures)}"
        )
    return fixtures


def _private_output_dir(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    try:
        output.relative_to(REPOSITORY_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("comparison output must be outside the Git repository")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} is not empty; pass --overwrite to replace comparison files"
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    return output


def _save_private_image(image: Image.Image, path: Path, *, quality: int = 92):
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="JPEG", quality=quality, subsampling=1)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _save_private_json(payload, path: Path):
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _fit_rgb(array, size=CELL_SIZE):
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (10, 13, 18))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _contact_sheet(rows, column_labels, row_labels, title):
    width = CELL_SIZE[0] * len(column_labels)
    height = HEADER_HEIGHT + CELL_SIZE[1] * len(rows)
    sheet = Image.new("RGB", (width, height), (8, 11, 16))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22, bold=True)
    label_font = _font(16, bold=True)
    small_font = _font(14)
    draw.text((14, 8), title, fill=(245, 248, 252), font=title_font)
    for column, label in enumerate(column_labels):
        x = column * CELL_SIZE[0] + 12
        draw.text((x, 37), label, fill=(105, 205, 255), font=small_font)
    for row_index, row in enumerate(rows):
        y = HEADER_HEIGHT + row_index * CELL_SIZE[1]
        for column, frame in enumerate(row):
            sheet.paste(_fit_rgb(frame), (column * CELL_SIZE[0], y))
        row_label = row_labels[row_index]
        box = draw.textbbox((0, 0), row_label, font=label_font)
        box_width = box[2] - box[0] + 16
        draw.rectangle((5, y + 5, 5 + box_width, y + 33), fill=(0, 0, 0))
        draw.text((13, y + 9), row_label, fill=(255, 220, 90), font=label_font)
    return sheet


def _load_physical_frames(avdi, clip_id: str, requested_timestamps):
    requested = np.asarray(requested_timestamps, dtype=np.int64)
    frames_by_camera = {}
    actual_timestamps = {}
    for name, camera_id, feature_name in CAMERAS:
        feature = getattr(avdi.features.CAMERA, feature_name)
        camera = avdi.get_clip_feature(clip_id, feature, maybe_stream=True)
        frames, timestamps = camera.decode_images_from_timestamps(requested)
        frames_by_camera[camera_id] = np.asarray(frames, dtype=np.uint8)
        actual_timestamps[camera_id] = np.asarray(timestamps, dtype=np.int64)
    return frames_by_camera, actual_timestamps


def _load_fixture_frames(paths):
    result = {}
    metadata = {}
    hashes = {}
    for mode, path in paths.items():
        fixture = load_camera_fixture(path)
        ids = fixture["camera_ids"].tolist()
        result[mode] = {
            int(camera_id): np.transpose(
                fixture["image_frames"][index],
                (0, 2, 3, 1),
            )
            for index, camera_id in enumerate(ids)
        }
        metadata[mode] = fixture["metadata"]
        hashes[mode] = fixture["fixture_sha256"]
    return result, metadata, hashes


def _build_mode_comparison(physical, fixtures):
    rows = []
    row_labels = []
    for name, camera_id, _ in CAMERAS:
        rows.append(
            [
                physical[camera_id][0],
                fixtures["baseline"][camera_id][-1],
                fixtures["projection-only"][camera_id][-1],
                fixtures["pose-projection"][camera_id][-1],
            ]
        )
        row_labels.append(f"ID {camera_id} · {name}")
    return _contact_sheet(
        rows,
        (
            "PhysicalAI real · F-theta",
            "CARLA baseline · pinhole",
            "CARLA projection-only · F-theta",
            "CARLA aligned pose + F-theta",
        ),
        row_labels,
        "All camera positions · real-vs-synthetic input geometry",
    )


def _build_physical_temporal_sheet(physical, timestamps):
    camera_ids = [camera_id for _, camera_id, _ in CAMERAS]
    rows = [
        [physical[camera_id][time_index] for camera_id in camera_ids]
        for time_index in range(len(timestamps))
    ]
    return _contact_sheet(
        rows,
        tuple(f"ID {camera_id} · {name}" for name, camera_id, _ in CAMERAS),
        [f"requested t={timestamp / 1e6:.1f}s" for timestamp in timestamps],
        "PhysicalAI authorized clip · temporal samples across all camera positions",
    )


def _build_carla_temporal_sheet(fixtures):
    camera_ids = [camera_id for _, camera_id, _ in CAMERAS]
    aligned = fixtures["pose-projection"]
    rows = [
        [aligned[camera_id][time_index] for camera_id in camera_ids]
        for time_index in range(4)
    ]
    return _contact_sheet(
        rows,
        tuple(f"ID {camera_id} · {name}" for name, camera_id, _ in CAMERAS),
        [f"CARLA history frame {index - 3:+d}" for index in range(4)],
        "CARLA pose-projection fixture · consecutive model-facing frames",
    )


def _build_single_camera_sheet(name, camera_id, physical, fixtures):
    panels = (
        ("PhysicalAI real · F-theta", physical[camera_id][0]),
        ("CARLA aligned pose + F-theta", fixtures["pose-projection"][camera_id][-1]),
        ("CARLA baseline pose + pinhole", fixtures["baseline"][camera_id][-1]),
        (
            "CARLA baseline pose + F-theta",
            fixtures["projection-only"][camera_id][-1],
        ),
    )
    panel_size = (960, 540)
    header = 66
    sheet = Image.new("RGB", (1920, header + 1080), (8, 11, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 14),
        f"Camera ID {camera_id} · {name}",
        fill=(245, 248, 252),
        font=_font(28, bold=True),
    )
    panel_font = _font(18, bold=True)
    for index, (label, frame) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = column * panel_size[0]
        y = header + row * panel_size[1]
        sheet.paste(_fit_rgb(frame, panel_size), (x, y))
        label_box = draw.textbbox((0, 0), label, font=panel_font)
        label_width = label_box[2] - label_box[0] + 24
        draw.rectangle((x + 10, y + 10, x + 10 + label_width, y + 46), fill=(0, 0, 0))
        draw.text(
            (x + 22, y + 16),
            label,
            fill=(255, 220, 90),
            font=panel_font,
        )
    return sheet


def _build_rig_plot(profile, output: Path):
    dataset_positions = {
        int(camera["alpamayo_id"]): np.asarray(
            camera["sensor_to_rig_matrix"], dtype=np.float64
        )[:3, 3]
        for camera in profile["cameras"]
    }
    # Baseline CARLA y-right is converted to y-left for diagram orientation.
    baseline_positions = {
        0: np.array([1.0, 0.5, 2.4]),
        1: np.array([1.5, 0.0, 2.4]),
        2: np.array([1.0, -0.5, 2.4]),
        6: np.array([1.5, 0.0, 2.4]),
    }
    baseline_yaws_deg = {0: 60.0, 1: 0.0, 2: -60.0, 6: 0.0}
    dataset_forward = {}
    for camera in profile["cameras"]:
        camera_id = int(camera["alpamayo_id"])
        rotation = np.asarray(camera["sensor_to_rig_matrix"], dtype=np.float64)[:3, :3]
        forward = rotation @ np.array([0.0, 0.0, 1.0])
        dataset_forward[camera_id] = forward / np.linalg.norm(forward)
    labels = {camera_id: name for name, camera_id, _ in CAMERAS}
    colors = {"baseline": "#ffb44c", "aligned": "#55c8ff"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    annotation_offsets = {
        0: (-18, 10),
        1: (8, 10),
        2: (-18, -22),
        6: (8, -22),
    }

    def plot_views(top_axis, side_axis, positions, forward_vectors, group, origin):
        color = colors["baseline" if group == "CARLA baseline" else "aligned"]
        for camera_id, xyz in positions.items():
            top_axis.scatter(xyz[0], xyz[1], s=100, color=color)
            top_axis.annotate(
                f"{camera_id} {labels[camera_id]}",
                (xyz[0], xyz[1]),
                xytext=annotation_offsets[camera_id],
                textcoords="offset points",
                fontsize=9,
            )
            forward = forward_vectors[camera_id]
            top_axis.arrow(
                xyz[0],
                xyz[1],
                0.32 * forward[0],
                0.32 * forward[1],
                width=0.008,
                head_width=0.06,
                length_includes_head=True,
                color=color,
                alpha=0.9,
            )
            side_axis.scatter(xyz[0], xyz[2], s=100, color=color)
            side_axis.annotate(
                f"{camera_id} {labels[camera_id]}",
                (xyz[0], xyz[2]),
                xytext=annotation_offsets[camera_id],
                textcoords="offset points",
                fontsize=9,
            )
        top_axis.set_title(f"{group} top view · {origin}")
        top_axis.set_xlabel("x forward [m]")
        top_axis.set_ylabel("y left [m]")
        side_axis.set_title(f"{group} side view · {origin}")
        side_axis.set_xlabel("x forward [m]")
        side_axis.set_ylabel("z up [m]")
        for axis in (top_axis, side_axis):
            axis.grid(True, alpha=0.25)
            axis.axis("equal")

    baseline_forward = {
        camera_id: np.array(
            [
                np.cos(np.deg2rad(yaw)),
                np.sin(np.deg2rad(yaw)),
                0.0,
            ]
        )
        for camera_id, yaw in baseline_yaws_deg.items()
    }
    plot_views(
        axes[0, 0],
        axes[0, 1],
        baseline_positions,
        baseline_forward,
        "CARLA baseline",
        "actor origin",
    )
    plot_views(
        axes[1, 0],
        axes[1, 1],
        dataset_positions,
        dataset_forward,
        "PhysicalAI + aligned CARLA",
        "rear-axle origin",
    )
    figure.suptitle(
        "Camera mounting positions and optical headings\n"
        "Panels use different origins; aligned CARLA converts rear axle → actor at runtime",
        fontsize=16,
    )
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.chmod(temporary, 0o600)
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    return dataset_positions, baseline_positions


def main(argv=None):
    args = parse_args(argv)
    fixture_paths = _parse_fixtures(args.fixture)
    output = _private_output_dir(args.output_dir, overwrite=args.overwrite)

    profile_path = args.camera_profile.expanduser().resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("source_clip_id") != args.clip_id:
        raise ValueError("camera profile source_clip_id does not match --clip-id")

    avdi = PhysicalAIAVDatasetInterface(revision=profile.get("dataset_revision"))
    physical, actual_timestamps = _load_physical_frames(
        avdi,
        args.clip_id,
        args.physical_t0_us,
    )
    fixtures, fixture_metadata, fixture_hashes = _load_fixture_frames(fixture_paths)

    artifacts = {
        "all_positions_mode_comparison.jpg": _build_mode_comparison(
            physical, fixtures
        ),
        "physicalai_all_positions_temporal.jpg": _build_physical_temporal_sheet(
            physical, args.physical_t0_us
        ),
        "carla_aligned_all_positions_temporal.jpg": _build_carla_temporal_sheet(
            fixtures
        ),
    }
    for filename, image in artifacts.items():
        _save_private_image(image, output / filename)
    for name, camera_id, _ in CAMERAS:
        image = _build_single_camera_sheet(name, camera_id, physical, fixtures)
        _save_private_image(
            image,
            output / f"camera_{camera_id}_{name}_comparison.jpg",
            quality=94,
        )

    dataset_positions, baseline_positions = _build_rig_plot(
        profile, output / "camera_mounting_positions.png"
    )
    metadata = {
        "schema_version": 1,
        "warning": (
            "Private gated-data comparison. Do not commit, redistribute, or "
            "treat different real/synthetic scenes as pixel correspondences."
        ),
        "clip_id": args.clip_id,
        "dataset_revision": str(avdi.revision),
        "profile_id": profile.get("profile_id"),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "camera_order": [
            {"name": name, "alpamayo_id": camera_id}
            for name, camera_id, _ in CAMERAS
        ],
        "requested_physical_timestamps_us": list(args.physical_t0_us),
        "actual_physical_timestamps_us": {
            str(camera_id): timestamps.tolist()
            for camera_id, timestamps in actual_timestamps.items()
        },
        "fixture_metadata": fixture_metadata,
        "fixture_sha256": fixture_hashes,
        "mounting_positions": {
            "physicalai_rear_axle_frame_x_forward_y_left_z_up": {
                str(camera_id): xyz.tolist()
                for camera_id, xyz in dataset_positions.items()
            },
            "carla_baseline_actor_frame_converted_to_y_left": {
                str(camera_id): xyz.tolist()
                for camera_id, xyz in baseline_positions.items()
            },
        },
    }
    _save_private_json(metadata, output / "comparison_metadata.json")

    print(f"Private comparison written: {output}")
    for path in sorted(output.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()

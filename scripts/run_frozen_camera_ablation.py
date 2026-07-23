#!/usr/bin/env python3
"""Run repeatable Alpamayo inference over private frozen camera fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.camera_fixture import load_camera_fixture  # noqa: E402
from module.inference import (  # noqa: E402
    configure_cuda_linalg_library,
    extract_cot_texts,
    extract_trajectory_samples,
    load_model,
    run_inference,
)
from module.trajectory_runtime import (  # noqa: E402
    TrajectoryValidationError,
    classify_and_validate_model_trajectory,
)


def _fixture_argument(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("fixture must use LABEL=/path/to/fixture.npz")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label or not path.strip():
        raise argparse.ArgumentTypeError("fixture label and path must be non-empty")
    return label, Path(path).expanduser().resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a seeded frozen-input camera alignment ablation."
    )
    parser.add_argument(
        "--fixture",
        action="append",
        type=_fixture_argument,
        required=True,
        help="Repeat as --fixture baseline=/private/file.npz.",
    )
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--num-traj-samples", type=int, default=3)
    parser.add_argument("--diffusion-temperature", type=float, default=1.0)
    parser.add_argument("--navigation-weight", type=float, default=None)
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Private JSONL output. Default: "
            "~/.cache/carlamayo/ablations/camera-ablation-<time>.jsonl"
        ),
    )
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.fixture]
    if len(labels) != len(set(labels)):
        parser.error("fixture labels must be unique")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.num_traj_samples <= 0:
        parser.error("--num-traj-samples must be positive")
    if (
        not np.isfinite(args.diffusion_temperature)
        or args.diffusion_temperature <= 0.0
    ):
        parser.error("--diffusion-temperature must be finite and positive")
    output = args.output
    if output is None:
        output = Path(
            f"~/.cache/carlamayo/ablations/camera-ablation-"
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
        )
    args.output = output.expanduser().resolve()
    if args.output == REPOSITORY_ROOT or REPOSITORY_ROOT in args.output.parents:
        parser.error("--output must point outside the Git repository")
    return args


def _model_data(fixture, *, history_reference):
    return {
        "image_frames": torch.from_numpy(fixture["image_frames"].copy()),
        "camera_indices": torch.from_numpy(fixture["camera_ids"].copy()).long(),
        "ego_history_xyz": torch.from_numpy(
            history_reference["ego_history_xyz"].copy()
        ).float()[None, None],
        "ego_history_rot": torch.from_numpy(
            history_reference["ego_history_rot"].copy()
        ).float()[None, None],
    }


def _pairwise_ade(trajectories):
    values = []
    for left in range(len(trajectories)):
        for right in range(left + 1, len(trajectories)):
            values.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            trajectories[left, :, :2]
                            - trajectories[right, :, :2],
                            axis=1,
                        )
                    )
                )
            )
    return values


def _candidate_records(trajectories, cot_texts):
    records = []
    for index, (trajectory, cot_text) in enumerate(
        zip(trajectories, cot_texts, strict=True)
    ):
        validation_status = "valid"
        stop_intent = None
        validation_error = None
        try:
            _, stop_intent = classify_and_validate_model_trajectory(trajectory)
        except TrajectoryValidationError as exc:
            validation_status = "invalid"
            validation_error = str(exc)
        digest = hashlib.sha256()
        digest.update(np.asarray(trajectory, dtype=np.float32).tobytes())
        digest.update(str(cot_text).encode("utf-8"))
        records.append(
            {
                "candidate_index": index,
                "candidate_sha256": digest.hexdigest(),
                "coc_text": str(cot_text),
                "trajectory": np.asarray(trajectory, dtype=np.float64).tolist(),
                "stop_intent": stop_intent,
                "validation_status": validation_status,
                "validation_error": validation_error,
            }
        )
    return records


def _write_line(stream, payload):
    stream.write(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    stream.flush()


def main(argv=None):
    args = parse_args(argv)
    fixtures = {
        label: load_camera_fixture(path) for label, path in args.fixture
    }
    reference_label = "baseline" if "baseline" in fixtures else next(iter(fixtures))
    history_reference = fixtures[reference_label]
    reference_navigation = history_reference["metadata"].get("navigation_text")
    history_deltas = {}
    for label, fixture in fixtures.items():
        if not np.array_equal(fixture["camera_ids"], history_reference["camera_ids"]):
            raise ValueError(f"fixture {label!r} has different camera IDs")
        if fixture["metadata"].get("navigation_text") != reference_navigation:
            raise ValueError(f"fixture {label!r} has different navigation text")
        history_deltas[label] = {
            "captured_xyz_max_abs_delta_m": float(
                np.max(
                    np.abs(
                        fixture["ego_history_xyz"]
                        - history_reference["ego_history_xyz"]
                    )
                )
            ),
            "captured_rotation_max_abs_delta": float(
                np.max(
                    np.abs(
                        fixture["ego_history_rot"]
                        - history_reference["ego_history_rot"]
                    )
                )
            ),
        }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    stream = os.fdopen(descriptor, "w", encoding="utf-8")
    summaries = {}
    try:
        configure_cuda_linalg_library("magma")
        model, processor = load_model(
            args.quantization,
            device_map=args.device_map,
        )
        metrics = defaultdict(
            lambda: {
                "requests": 0,
                "candidates": 0,
                "stop_candidates": 0,
                "pairwise_ade": [],
                "latency_s": [],
            }
        )
        _write_line(
            stream,
            {
                "event_type": "ablation_start",
                "schema_version": 1,
                "fixtures": {
                    label: {
                        "fixture_id": fixture["metadata"].get("fixture_id"),
                        "fixture_sha256": fixture["fixture_sha256"],
                        "camera_alignment_mode": fixture["metadata"].get(
                            "camera_alignment_mode"
                        ),
                        "camera_profile_sha256": fixture["metadata"].get(
                            "camera_profile_sha256"
                        ),
                    }
                    for label, fixture in fixtures.items()
                },
                "repeats": args.repeats,
                "num_traj_samples": args.num_traj_samples,
                "diffusion_temperature": args.diffusion_temperature,
                "history_reference_fixture": reference_label,
                "captured_history_deltas": history_deltas,
            },
        )
        for label, fixture in fixtures.items():
            metadata = fixture["metadata"]
            navigation_text = str(metadata.get("navigation_text") or "")
            navigation_weight = (
                float(args.navigation_weight)
                if args.navigation_weight is not None
                else float(metadata.get("navigation_weight", 1.0))
            )
            for seed in range(args.repeats):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                started = time.monotonic()
                pred_xyz, extra = run_inference(
                    model,
                    processor,
                    _model_data(
                        fixture,
                        history_reference=history_reference,
                    ),
                    navigation_text=navigation_text,
                    navigation_weight=navigation_weight,
                    num_traj_samples=args.num_traj_samples,
                    diffusion_temperature=args.diffusion_temperature,
                )
                latency_s = time.monotonic() - started
                trajectories = extract_trajectory_samples(pred_xyz)
                cot_texts = extract_cot_texts(extra, len(trajectories))
                candidates = _candidate_records(trajectories, cot_texts)
                pairwise_ade = _pairwise_ade(trajectories)
                _write_line(
                    stream,
                    {
                        "event_type": "inference_result",
                        "schema_version": 1,
                        "fixture_label": label,
                        "fixture_id": metadata.get("fixture_id"),
                        "seed": seed,
                        "latency_s": latency_s,
                        "pairwise_ade_m": pairwise_ade,
                        "candidates": candidates,
                    },
                )
                group = metrics[label]
                group["requests"] += 1
                group["candidates"] += len(candidates)
                group["stop_candidates"] += sum(
                    candidate["stop_intent"] is True for candidate in candidates
                )
                group["pairwise_ade"].extend(pairwise_ade)
                group["latency_s"].append(latency_s)

        for label, values in metrics.items():
            candidates = max(1, values["candidates"])
            summaries[label] = {
                "requests": values["requests"],
                "candidates": values["candidates"],
                "stop_candidate_rate": values["stop_candidates"] / candidates,
                "pairwise_ade_median": (
                    float(np.median(values["pairwise_ade"]))
                    if values["pairwise_ade"]
                    else None
                ),
                "pairwise_ade_mean": (
                    float(np.mean(values["pairwise_ade"]))
                    if values["pairwise_ade"]
                    else None
                ),
                "latency_s_mean": float(np.mean(values["latency_s"])),
            }
        _write_line(
            stream,
            {
                "event_type": "ablation_summary",
                "schema_version": 1,
                "groups": summaries,
            },
        )
    finally:
        stream.close()
    os.chmod(args.output, 0o600)
    print(f"Ablation report: {args.output}")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

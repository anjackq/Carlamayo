#!/usr/bin/env python3
"""Aggregate the six-run synchronous reachability go/no-go experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.summarize_route_rollout import (  # noqa: E402
    _read_events,
    summarize_route_rollout,
)


ARMS = ("current", "reachability-first")
SEEDS = (0, 1, 2)


def _run_argument(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "run must use ARM:SEED=/path/to/runtime.jsonl"
        )
    identity, raw_path = value.split("=", 1)
    arm, raw_seed = identity.split(":", 1)
    if arm not in ARMS:
        raise argparse.ArgumentTypeError(f"unknown arm: {arm}")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("seed must be 0, 1, or 2")
    return arm, seed, Path(raw_path).expanduser().resolve()


def _outside_repository(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY_ROOT or REPOSITORY_ROOT in path.parents:
        raise argparse.ArgumentTypeError(
            "experiment output must remain outside the repository"
        )
    return path


def _ratio(counts: dict[str, Any], names: set[str]) -> float:
    total = sum(int(value) for value in counts.values())
    selected = sum(
        int(value)
        for key, value in counts.items()
        if str(key) in names
    )
    return selected / total if total else 0.0


def _active_coverage(summary: dict[str, Any]) -> tuple[int, int]:
    phase = summary["candidate_policy"].get("phase") or {}
    active = phase.get("RIGHT/ACTIVE") or {}
    return (
        int(active.get("full_turn_executable_available") or 0),
        int(active.get("requests") or 0),
    )


def _safety_clean(summary: dict[str, Any]) -> bool:
    route = summary["route"]
    safety = summary["safety"]
    return bool(
        safety["collisions"] == 0
        and safety["current_ego_road_unsafe_or_unknown_ticks"] == 0
        and safety["direct_override_ticks"] == 0
        and safety["fallback_ticks"] <= 3
        and route["accepted_unauthorized_prefix_count"] == 0
        and route["route_only_emergency_override_ticks"] == 0
        and route["route_unavailable_context_count"] == 0
        and route["prompt_truth_mismatch_count"] == 0
    )


def aggregate(
    summaries: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    patch = {
        seed: summaries[("reachability-first", seed)]
        for seed in SEEDS
    }
    control = {
        seed: summaries[("current", seed)] for seed in SEEDS
    }
    patch_active = {
        seed: _active_coverage(summary)
        for seed, summary in patch.items()
    }
    pooled_active_available = sum(
        available for available, _requests in patch_active.values()
    )
    pooled_active_requests = sum(
        requests for _available, requests in patch_active.values()
    )
    pooled_active_coverage = (
        pooled_active_available / pooled_active_requests
        if pooled_active_requests
        else 0.0
    )
    per_seed_active_coverage = {
        str(seed): (
            available / requests if requests else 0.0
        )
        for seed, (available, requests) in patch_active.items()
    }
    selector_mechanism_gate = all(
        summary["candidate_policy"][
            "reachability_profile_error_count"
        ]
        == 0
        and summary["candidate_policy"][
            "effective_near_conditional_misses"
        ]
        == 0
        and summary["candidate_policy"][
            "effective_full_conditional_misses"
        ]
        == 0
        and (
            summary["candidate_policy"][
                "reachability_compute_p95_ms"
            ]
            is not None
            and summary["candidate_policy"][
                "reachability_compute_p95_ms"
            ]
            < 2.0
        )
        and (
            summary["candidate_policy"]["selection_compute_p95_ms"]
            is not None
            and summary["candidate_policy"]["selection_compute_p95_ms"]
            < 100.0
        )
        for summary in patch.values()
    )
    motion_non_regression = all(
        _ratio(
            patch[seed]["candidate_policy"][
                "selected_motion_class_counts"
            ],
            {"EXPLICIT_STOP", "DELAYED_START"},
        )
        <= _ratio(
            control[seed]["candidate_policy"][
                "selected_motion_class_counts"
            ],
            {"EXPLICIT_STOP", "DELAYED_START"},
        )
        + 1e-12
        for seed in SEEDS
    )
    safety_gate = all(
        _safety_clean(summary)
        for summary in summaries.values()
    )
    synchronous_gate = all(
        summary["synchronous_contract"]["execution"] == "sync"
        and summary["synchronous_contract"][
            "maximum_inference_simulation_duration_s"
        ]
        in {None, 0.0}
        for summary in summaries.values()
    )
    direct_controller_gate = all(
        summary["route"]["completion_ratio"] is not None
        and summary["route"]["completion_ratio"] >= 0.95
        and summary["gates"]["destination_stop"]
        and not summary["motion"]["absorbing_stop"]
        for summary in patch.values()
    )
    coverage_gate = bool(
        pooled_active_requests > 0
        and pooled_active_coverage >= 0.70
        and all(
            coverage >= 0.50
            for coverage in per_seed_active_coverage.values()
        )
    )
    if not selector_mechanism_gate or not safety_gate or not synchronous_gate:
        decision = "SELECTOR_OR_SYSTEM_NO_GO_MODEL_INCONCLUSIVE"
    elif not coverage_gate:
        decision = "ALPAMAYO_DIRECT_LATERAL_AUTHORITY_NO_GO"
    elif not direct_controller_gate:
        decision = "INTEGRATION_INCONCLUSIVE"
    elif not motion_non_regression:
        decision = "SELECTOR_MOTION_REGRESSION_NO_GO"
    else:
        decision = "DIRECT_CONTROLLER_GO_EXPAND_VALIDATION"

    paired = {}
    for seed in SEEDS:
        paired[str(seed)] = {
            "current": control[seed],
            "reachability-first": patch[seed],
            "delta": {
                "route_completion": (
                    patch[seed]["route"]["completion_ratio"]
                    - control[seed]["route"]["completion_ratio"]
                    if patch[seed]["route"]["completion_ratio"] is not None
                    and control[seed]["route"]["completion_ratio"] is not None
                    else None
                ),
                "integrated_distance_m": (
                    patch[seed]["motion"]["integrated_distance_m"]
                    - control[seed]["motion"]["integrated_distance_m"]
                ),
                "stationary_suffix_ticks": (
                    patch[seed]["motion"]["stationary_suffix_ticks"]
                    - control[seed]["motion"]["stationary_suffix_ticks"]
                ),
            },
        }
    patch_completions = [
        summary["route"]["completion_ratio"]
        for summary in patch.values()
        if summary["route"]["completion_ratio"] is not None
    ]
    return {
        "schema_version": "carlamayo.reachability-gate.v1",
        "decision": decision,
        "gates": {
            "selector_mechanism": selector_mechanism_gate,
            "motion_non_regression": motion_non_regression,
            "safety": safety_gate,
            "synchronous": synchronous_gate,
            "active_full_turn_coverage": coverage_gate,
            "direct_controller": direct_controller_gate,
        },
        "active_full_turn_coverage": {
            "pooled_available": pooled_active_available,
            "pooled_requests": pooled_active_requests,
            "pooled_ratio": pooled_active_coverage,
            "per_seed_ratio": per_seed_active_coverage,
        },
        "patch_median_route_completion": (
            statistics.median(patch_completions)
            if patch_completions
            else None
        ),
        "paired_runs": paired,
        "conclusion_boundary": (
            "This engineering decision applies only to Alpamayo 1.5 as "
            "direct lateral trajectory authority in the tested CarlaMayo "
            "Town03 synchronous configuration."
        ),
    }


def _write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_run_argument,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=_outside_repository,
        required=True,
    )
    args = parser.parse_args(argv)
    run_paths = {
        (arm, seed): path for arm, seed, path in args.run
    }
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(run_paths) != expected or len(args.run) != len(expected):
        parser.error(
            "exactly one runtime JSONL is required for each "
            "current/reachability-first seed 0/1/2"
        )
    summaries = {
        identity: summarize_route_rollout(_read_events(path))
        for identity, path in run_paths.items()
    }
    aggregate_summary = aggregate(summaries)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_exclusive(
        args.output_dir / "paired-summary.json",
        json.dumps(
            aggregate_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    csv_path = args.output_dir / "paired-summary.csv"
    descriptor = os.open(
        csv_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "seed",
                "arm",
                "route_completion",
                "integrated_distance_m",
                "stationary_suffix_ticks",
                "near_coverage_at_k",
                "full_coverage_at_k",
                "near_conditional_misses",
                "full_conditional_misses",
            ),
        )
        writer.writeheader()
        for seed in SEEDS:
            for arm in ARMS:
                summary = summaries[(arm, seed)]
                writer.writerow(
                    {
                        "seed": seed,
                        "arm": arm,
                        "route_completion": summary["route"][
                            "completion_ratio"
                        ],
                        "integrated_distance_m": summary["motion"][
                            "integrated_distance_m"
                        ],
                        "stationary_suffix_ticks": summary["motion"][
                            "stationary_suffix_ticks"
                        ],
                        "near_coverage_at_k": summary[
                            "candidate_policy"
                        ]["near_executable_coverage_at_k"],
                        "full_coverage_at_k": summary[
                            "candidate_policy"
                        ]["full_turn_executable_coverage_at_k"],
                        "near_conditional_misses": summary[
                            "candidate_policy"
                        ]["effective_near_conditional_misses"],
                        "full_conditional_misses": summary[
                            "candidate_policy"
                        ]["effective_full_conditional_misses"],
                    }
                )
    print(json.dumps(aggregate_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

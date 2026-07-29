import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_frozen_camera_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("run_frozen_camera_ablation", SCRIPT_PATH)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_fixture_argument_and_output_contract(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    label, path = ablation._fixture_argument("baseline=/private/baseline.npz")

    assert label == "baseline"
    assert path == Path("/private/baseline.npz")


def test_fixture_argument_rejects_missing_label():
    with pytest.raises(Exception, match="LABEL="):
        ablation._fixture_argument("/private/baseline.npz")


def test_policy_conditioning_mode_is_opt_in():
    args = ablation.parse_args(
        [
            "--fixture",
            "moving=/private/moving.npz",
            "--conditioning-source",
            "per-fixture",
            "--output",
            "/private/output.jsonl",
        ]
    )

    assert args.conditioning_source == "per-fixture"


def test_pairwise_ade_and_candidate_hashes_are_deterministic():
    trajectories = np.zeros((3, 64, 3), dtype=np.float64)
    trajectories[1, :, 0] = 1.0
    trajectories[2, :, 0] = 2.0

    ade = ablation._pairwise_ade(trajectories)
    first = ablation._candidate_records(trajectories, ["a", "b", "c"])
    second = ablation._candidate_records(trajectories, ["a", "b", "c"])

    assert ade == pytest.approx([1.0, 2.0, 1.0])
    assert [item["candidate_sha256"] for item in first] == [
        item["candidate_sha256"] for item in second
    ]
    assert first[0]["stop_intent"] is True

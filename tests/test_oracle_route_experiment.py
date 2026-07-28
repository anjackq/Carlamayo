from pathlib import Path

import pytest

from scripts.run_oracle_route_experiment import parse_args


def _api(tmp_path: Path) -> Path:
    marker = tmp_path / "carla" / "agents" / "navigation"
    marker.mkdir(parents=True)
    (marker / "global_route_planner.py").write_text("", encoding="utf-8")
    return tmp_path / "carla"


def test_oracle_cli_requires_external_telemetry_and_valid_route_api(tmp_path):
    args = parse_args(
        [
            "--target-speed-mps",
            "2",
            "--camera-alignment",
            "baseline",
            "--telemetry-jsonl",
            str(tmp_path / "runtime.jsonl"),
            "--carla-python-api-path",
            str(_api(tmp_path)),
        ]
    )

    assert args.target_speed_mps == pytest.approx(2.0)
    assert args.capture_inference_fixture is None


def test_oracle_cli_capture_options_are_paired(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--target-speed-mps",
                "2",
                "--camera-alignment",
                "baseline",
                "--telemetry-jsonl",
                str(tmp_path / "runtime.jsonl"),
                "--capture-inference-fixture",
                str(tmp_path / "capture.npz"),
                "--carla-python-api-path",
                str(_api(tmp_path)),
            ]
        )

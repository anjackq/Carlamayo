import importlib
import sys
import types

import pytest


sys.modules.setdefault(
    "carla",
    types.SimpleNamespace(
        command=types.SimpleNamespace(DestroyActor=lambda actor_id: ("destroy", actor_id)),
        VehicleControl=lambda: types.SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0),
        Vector3D=lambda: types.SimpleNamespace(),
    ),
)
closed_loop = importlib.import_module("carlamayo_closed_loop")


def test_runtime_telemetry_and_episode_limit_arguments():
    args = closed_loop.parse_args(
        [
            "--telemetry-jsonl",
            "runs/episode.jsonl",
            "--max-episode-seconds",
            "12.5",
        ]
    )

    assert args.telemetry_jsonl == "runs/episode.jsonl"
    assert args.max_episode_seconds == pytest.approx(12.5)


def test_sampling_diagnostic_arguments():
    args = closed_loop.parse_args(
        [
            "--num-traj-samples",
            "3",
            "--diffusion-temperature",
            "0.6",
        ]
    )

    assert args.num_traj_samples == 3
    assert args.diffusion_temperature == pytest.approx(0.6)


def test_road_assessment_backend_defaults_remain_serial():
    args = closed_loop.parse_args([])

    assert args.road_assessment_backend == "serial"
    assert args.road_assessment_workers is None


def test_process_road_backend_resolves_default_workers_from_cpu_affinity(
    monkeypatch,
):
    monkeypatch.setattr(
        closed_loop.os,
        "sched_getaffinity",
        lambda _pid: set(range(8)),
    )

    args = closed_loop.parse_args(
        ["--road-assessment-backend", "process"]
    )

    assert args.road_assessment_backend == "process"
    assert args.road_assessment_workers == 6


def test_process_road_backend_preserves_explicit_worker_count(monkeypatch):
    monkeypatch.setattr(
        closed_loop.os,
        "sched_getaffinity",
        lambda _pid: set(range(8)),
    )

    args = closed_loop.parse_args(
        [
            "--road-assessment-backend",
            "process",
            "--road-assessment-workers",
            "3",
        ]
    )

    assert args.road_assessment_workers == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["--road-assessment-workers", "2"],
        [
            "--road-assessment-backend",
            "process",
            "--road-assessment-workers",
            "0",
        ],
    ],
)
def test_road_backend_worker_argument_validation(arguments):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(arguments)

    assert exc_info.value.code == 2


def test_process_road_backend_requires_three_available_cpus(monkeypatch):
    monkeypatch.setattr(
        closed_loop.os,
        "sched_getaffinity",
        lambda _pid: {0, 1},
    )

    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(
            ["--road-assessment-backend", "process"]
        )

    assert exc_info.value.code == 2


def test_camera_alignment_arguments_and_environment(monkeypatch):
    monkeypatch.setenv("CARLAMAYO_CAMERA_PROFILE", "/private/profile.json")

    args = closed_loop.parse_args(
        [
            "--camera-alignment",
            "pose-projection",
            "--capture-inference-fixture",
            "/private/fixture.npz",
            "--capture-only",
        ]
    )

    assert args.camera_alignment == "pose-projection"
    assert args.camera_profile == "/private/profile.json"
    assert args.capture_inference_fixture == "/private/fixture.npz"
    assert args.capture_only is True


def test_camera_profile_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("CARLAMAYO_CAMERA_PROFILE", "/private/environment.json")

    args = closed_loop.parse_args(
        [
            "--camera-alignment",
            "pose-only",
            "--camera-profile",
            "/private/cli.json",
        ]
    )

    assert args.camera_profile == "/private/cli.json"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--camera-alignment", "projection-only"],
        ["--capture-only"],
    ],
)
def test_camera_alignment_missing_required_inputs_fails_fast(monkeypatch, arguments):
    monkeypatch.delenv("CARLAMAYO_CAMERA_PROFILE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(arguments)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--num-traj-samples", "0"],
        ["--num-traj-samples", "17"],
        ["--diffusion-temperature", "0"],
        ["--diffusion-temperature", "nan"],
    ],
)
def test_sampling_diagnostic_arguments_are_bounded(arguments):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(arguments)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_episode_limit_must_be_positive(value):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(["--max-episode-seconds", value])

    assert exc_info.value.code == 2


def test_route_navigation_arguments_resolve_agents_path(tmp_path):
    api_path = tmp_path / "PythonAPI" / "carla"
    marker = api_path / "agents" / "navigation" / "global_route_planner.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# test marker\n", encoding="utf-8")

    args = closed_loop.parse_args(
        [
            "--mode",
            "navigation",
            "--navigation-source",
            "route",
            "--route-destination=-43.350975,-2.8402605,0",
            "--carla-python-api-path",
            str(api_path),
        ]
    )

    assert args.navigation_source == "route"
    assert args.route_destination == pytest.approx(
        (-43.350975, -2.8402605, 0.0)
    )
    assert args.carla_python_api_path == str(api_path.resolve())
    assert args.navigation_text == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--navigation-source", "route"],
        [
            "--mode",
            "normal",
            "--navigation-source",
            "route",
            "--route-destination=1,2,3",
        ],
        [
            "--mode",
            "navigation",
            "--navigation-source",
            "route",
            "--route-destination=1,2,3",
            "--navigation-text",
            "Turn right.",
        ],
        ["--route-destination=1,2,3"],
    ],
)
def test_route_navigation_invalid_argument_combinations_fail_fast(arguments):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(arguments)

    assert exc_info.value.code == 2

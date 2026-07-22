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


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_episode_limit_must_be_positive(value):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(["--max-episode-seconds", value])

    assert exc_info.value.code == 2

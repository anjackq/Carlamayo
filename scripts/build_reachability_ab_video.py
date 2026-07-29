#!/usr/bin/env python3
"""Build a simulation-time-aligned current/reachability review video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _outside_repository(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY_ROOT or REPOSITORY_ROOT in path.parents:
        raise argparse.ArgumentTypeError(
            "comparison video must remain outside the repository"
        )
    return path


def build_command(
    current_video: Path,
    reachability_video: Path,
    output: Path,
    *,
    seed: int,
) -> list[str]:
    filter_graph = (
        "[0:v]scale=960:540,"
        f"drawtext=text='CURRENT | SEED {seed}':"
        "x=24:y=24:fontsize=30:fontcolor=white:"
        "box=1:boxcolor=black@0.65[a];"
        "[1:v]scale=960:540,"
        f"drawtext=text='REACHABILITY-FIRST | SEED {seed}':"
        "x=24:y=24:fontsize=30:fontcolor=white:"
        "box=1:boxcolor=black@0.65[b];"
        "[a][b]hstack=inputs=2[v]"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(current_video),
        "-i",
        str(reachability_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-shortest",
        str(output),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=_outside_repository, required=True)
    args = parser.parse_args(argv)
    for video in (args.current, args.reachability):
        if not video.is_file():
            parser.error(f"video does not exist: {video}")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    subprocess.run(
        build_command(
            args.current.resolve(),
            args.reachability.resolve(),
            args.output,
            seed=args.seed,
        ),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

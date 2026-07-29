from pathlib import Path

from scripts.build_reachability_ab_video import build_command


def test_comparison_video_command_labels_both_arms():
    command = build_command(
        Path("/runs/current.mp4"),
        Path("/runs/reachability.mp4"),
        Path("/runs/comparison.mp4"),
        seed=2,
    )

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "CURRENT | SEED 2" in filter_graph
    assert "REACHABILITY-FIRST | SEED 2" in filter_graph
    assert command[-1] == "/runs/comparison.mp4"

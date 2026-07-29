import queue

from module.data_collection import (
    ExactFrameCollector,
    collect_synchronous_sensor_frame,
    frame_file_path,
    frame_is_complete,
)


def test_frame_is_complete_requires_all_sensor_outputs(tmp_path):
    (tmp_path / "cam_front_wide").mkdir()
    (tmp_path / "lidar_top").mkdir()
    (tmp_path / "cam_front_wide" / "000007.jpg").write_bytes(b"image")

    assert not frame_is_complete(tmp_path, ["cam_front_wide", "lidar_top"], 7)

    (tmp_path / "lidar_top" / "000007.ply").write_bytes(b"lidar")

    assert frame_is_complete(tmp_path, ["cam_front_wide", "lidar_top"], 7)
    assert frame_file_path(tmp_path, "lidar_top", 7).endswith("lidar_top/000007.ply")


def test_collect_synchronous_sensor_frame_keeps_only_exact_tick():
    sensor_queue = queue.Queue()
    sensor_queue.put((9, "camera", "old-camera"))
    sensor_queue.put((10, "camera", "current-camera"))
    sensor_queue.put((11, "lidar", "future-lidar"))
    sensor_queue.put((10, "lidar", "current-lidar"))

    frame = collect_synchronous_sensor_frame(
        sensor_queue,
        ["camera", "lidar"],
        frame_id=10,
        timeout=0.01,
    )

    assert frame == {"camera": "current-camera", "lidar": "current-lidar"}

    sensor_queue.put((11, "camera", "future-camera"))
    next_frame = collect_synchronous_sensor_frame(
        sensor_queue,
        ["camera", "lidar"],
        frame_id=11,
        timeout=0.01,
    )

    assert next_frame == {"camera": "future-camera", "lidar": "future-lidar"}


def test_exact_frame_collector_discards_old_packets_and_clears_pending_frames():
    collector = ExactFrameCollector(max_pending_frames=2)
    collector.put(4, "camera", "old")
    collector.put(6, "camera", "future")
    collector.put(5, "camera", "current")

    assert collector.collect(5, ["camera"], timeout=0.01) == {
        "camera": "current"
    }
    assert collector.stats()["dropped_old"] == 1
    assert collector.stats()["pending_frames"] == 1

    collector.clear()

    assert collector.stats()["pending_frames"] == 0
    assert collector.stats()["queued_packets"] == 0

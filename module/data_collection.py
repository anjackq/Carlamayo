"""Reusable helpers for CARLA data collection."""

import os
import queue
import time
from collections import defaultdict


class ExactFrameCollector:
    """Collect complete named sensor bundles without mixing CARLA frames.

    Future packets are retained for the next control tick, old packets are
    counted and discarded, and the pending cache is bounded so a sensor fault
    cannot grow memory without limit.
    """

    def __init__(self, *, sensor_queue=None, queue_size=256, max_pending_frames=8):
        self.queue = (
            sensor_queue
            if sensor_queue is not None
            else queue.Queue(maxsize=max(1, int(queue_size)))
        )
        self.max_pending_frames = max(1, int(max_pending_frames))
        self.pending = defaultdict(dict)
        self.dropped_old = 0
        self.dropped_overflow = 0
        self.duplicates = 0

    def put(self, frame_id, sensor_name, data):
        packet = (int(frame_id), str(sensor_name), data)
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_overflow += 1
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                # Multiple CARLA callback threads may refill the queue between
                # the eviction and retry. Dropping one packet is safer than
                # blocking a synchronous sensor callback indefinitely.
                self.dropped_overflow += 1

    def _store(self, frame_id, sensor_name, data):
        bundle = self.pending[int(frame_id)]
        if sensor_name in bundle:
            self.duplicates += 1
        bundle[str(sensor_name)] = data
        while len(self.pending) > self.max_pending_frames:
            # Preserve the nearest future frame and shed the frame farthest
            # from the current control boundary first.
            farthest = max(self.pending)
            self.dropped_overflow += len(self.pending.pop(farthest))

    def collect(self, frame_id, expected_sensor_names, timeout=5.0):
        target_frame = int(frame_id)
        expected = tuple(str(name) for name in expected_sensor_names)
        expected_set = set(expected)
        if len(expected_set) != len(expected):
            raise ValueError("expected_sensor_names must be unique")

        old_frames = [
            pending_frame
            for pending_frame in self.pending
            if pending_frame < target_frame
        ]
        for old_frame in old_frames:
            self.dropped_old += len(self.pending.pop(old_frame))

        frame_data = {
            name: data
            for name, data in self.pending.pop(target_frame, {}).items()
            if name in expected_set
        }
        timeout = float(timeout)
        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout

        while expected_set - set(frame_data):
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0.0:
                    sensor_frame, name, data = self.queue.get_nowait()
                else:
                    sensor_frame, name, data = self.queue.get(True, remaining)
            except queue.Empty:
                break

            sensor_frame = int(sensor_frame)
            name = str(name)
            if sensor_frame < target_frame:
                self.dropped_old += 1
                continue
            if sensor_frame > target_frame:
                self._store(sensor_frame, name, data)
                continue
            if name not in expected_set:
                continue
            if name in frame_data:
                self.duplicates += 1
            frame_data[name] = data

        ordered_frame_data = {
            name: frame_data[name] for name in expected if name in frame_data
        }
        if expected_set - set(frame_data):
            # A timeout is recoverable: retain the partial target bundle so a
            # late sensor packet can complete it on a retry for the same tick.
            self.pending[target_frame].update(ordered_frame_data)
            while len(self.pending) > self.max_pending_frames:
                future_frames = [
                    pending_frame
                    for pending_frame in self.pending
                    if pending_frame != target_frame
                ]
                evicted_frame = max(future_frames) if future_frames else target_frame
                self.dropped_overflow += len(self.pending.pop(evicted_frame))
        return ordered_frame_data

    def clear(self):
        self.pending.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def stats(self):
        """Return a snapshot of collector health counters."""

        return {
            "dropped_old": int(self.dropped_old),
            "dropped_overflow": int(self.dropped_overflow),
            "duplicates": int(self.duplicates),
            "pending_frames": len(self.pending),
            "queued_packets": self.queue.qsize(),
        }


def frame_file_path(output_dir, sensor_name, frame_id):
    """Return the on-disk path for one saved sensor frame."""

    ext = "ply" if "lidar" in sensor_name else "jpg"
    return os.path.join(output_dir, sensor_name, f"{frame_id:06d}.{ext}")


def frame_is_complete(output_dir, sensor_names, frame_id):
    """Return True when every expected sensor file exists for a frame."""

    return all(
        os.path.exists(frame_file_path(output_dir, sensor_name, frame_id))
        for sensor_name in sensor_names
    )


def collect_synchronous_sensor_frame(
    sensor_queue,
    expected_sensor_names,
    frame_id,
    timeout=5.0,
):
    """Collect one complete synchronous sensor packet for a world tick frame.

    CARLA sensors can leave older or newer frame messages in the shared queue,
    especially after map reloads or when image encoding is slower than the
    simulation tick. Filter by the exact frame returned from ``world.tick()``
    so trajectory poses and sensor files stay aligned.
    """

    if isinstance(sensor_queue, ExactFrameCollector):
        collector = sensor_queue
    else:
        # Keep a collector on the queue itself so packets from frame N+1 that
        # arrive while frame N is being assembled survive the next function
        # call. ``queue.Queue`` instances permit private attributes; the
        # fallback only matters for unusual queue-compatible implementations.
        collector = getattr(sensor_queue, "_carlamayo_exact_frame_collector", None)
        if collector is None:
            collector = ExactFrameCollector(sensor_queue=sensor_queue)
            try:
                setattr(sensor_queue, "_carlamayo_exact_frame_collector", collector)
            except (AttributeError, TypeError):
                pass

    return collector.collect(
        frame_id,
        expected_sensor_names,
        timeout=timeout,
    )

"""Simulator-free contracts for the exact spawn-process road backend."""

from __future__ import annotations

import dataclasses
import hashlib
import multiprocessing
import pickle
from collections import deque
from queue import Empty

import pytest

from module import road_assessment_backend as road_backend
from module.road_assessment_backend import (
    ExactRoadProcessBackend,
    FootprintQuery,
    FootprintQueryResult,
    ProcessRoadBatchStats,
    RoadProcessBackendCrashed,
    RoadProcessBackendTimeout,
    RoadQueryChunk,
    RoadQueryChunkResult,
    WaypointPrimitive,
)


def _opendrive_map_digest(opendrive: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"carla_opendrive")
    digest.update(b"\0")
    digest.update(opendrive.encode("utf-8"))
    return digest.hexdigest()


def _query(query_id: int, *, x: float = 10.0) -> FootprintQuery:
    return FootprintQuery(
        query_id=query_id,
        points_xyz=tuple(
            (x + longitudinal, 1.75 + lateral, 0.0)
            for longitudinal, lateral in (
                (0.0, 0.0),
                (-0.4, -0.3),
                (-0.4, 0.3),
                (0.4, -0.3),
                (0.4, 0.3),
            )
        ),
    )


def _waypoint(*, lane_id: int = -1) -> WaypointPrimitive:
    return WaypointPrimitive(
        found=True,
        road_id=1,
        lane_id=lane_id,
        is_junction=False,
        lane_center_x=10.0,
        lane_center_y=1.75,
        lane_yaw_deg=0.0,
        lane_width=3.5,
    )


def _result(
    query_id: int,
    *,
    worker_pid: int = 4001,
) -> FootprintQueryResult:
    return FootprintQueryResult(
        query_id=query_id,
        waypoints=tuple(_waypoint() for _ in range(5)),
        error_type=None,
        map_query_count=5,
        worker_query_ms=0.25,
        worker_pid=worker_pid,
    )


def _chunk_result(
    chunk_id: int,
    query_ids: tuple[int, ...],
    *,
    worker_pid: int = 4001,
) -> RoadQueryChunkResult:
    return RoadQueryChunkResult(
        chunk_id=chunk_id,
        results=tuple(
            _result(query_id, worker_pid=worker_pid)
            for query_id in query_ids
        ),
        worker_pid=worker_pid,
    )


class _FakeReadyQueue:
    def __init__(self, items):
        self.items = deque(items)
        self.closed_count = 0

    def get(self, timeout):
        assert timeout > 0.0
        if not self.items:
            raise Empty
        item = self.items.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed_count += 1


class _FakeAsyncResult:
    def __init__(self, value=None, exception=None):
        self.value = value
        self.exception = exception
        self.timeouts = []

    def get(self, timeout):
        self.timeouts.append(timeout)
        assert timeout > 0.0
        if self.exception is not None:
            raise self.exception
        return self.value


class _ApplyAsyncFailure:
    def __init__(self, exception):
        self.exception = exception


class _FakePool:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.apply_calls = []
        self.close_count = 0
        self.terminate_count = 0
        self.join_count = 0

    def apply_async(self, function, args):
        self.apply_calls.append((function, args))
        response = self.responses.popleft()
        if isinstance(response, _ApplyAsyncFailure):
            raise response.exception
        if isinstance(response, BaseException):
            return _FakeAsyncResult(exception=response)
        return _FakeAsyncResult(value=response)

    def close(self):
        self.close_count += 1

    def terminate(self):
        self.terminate_count += 1

    def join(self):
        self.join_count += 1


class _FakeContext:
    def __init__(
        self,
        *,
        worker_count,
        responses=(),
        readiness=None,
    ):
        self.ready_queue = _FakeReadyQueue(
            readiness
            if readiness is not None
            else tuple((5000 + index, None) for index in range(worker_count))
        )
        self.pool = _FakePool(responses)
        self.pool_kwargs = None

    def Queue(self):
        return self.ready_queue

    def Pool(self, **kwargs):
        self.pool_kwargs = kwargs
        return self.pool


def _backend_with_fake_context(
    *,
    worker_count=2,
    responses=(),
    readiness=None,
    chunk_pose_count=2,
):
    context = _FakeContext(
        worker_count=worker_count,
        responses=responses,
        readiness=readiness,
    )
    requested_start_methods = []

    def context_factory(start_method):
        requested_start_methods.append(start_method)
        return context

    backend = ExactRoadProcessBackend(
        map_name="TinyRoad",
        opendrive="<OpenDRIVE/>",
        map_digest=_opendrive_map_digest("<OpenDRIVE/>"),
        worker_count=worker_count,
        chunk_pose_count=chunk_pose_count,
        startup_timeout_s=1.0,
        batch_timeout_s=1.0,
        context_factory=context_factory,
    )
    return backend, context, requested_start_methods


def _assert_numeric_payload(value):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _assert_numeric_payload(getattr(value, field.name))
        return
    if isinstance(value, tuple):
        for item in value:
            _assert_numeric_payload(item)
        return
    assert value is None or type(value) in {bool, int, float, str}


def test_numeric_dtos_are_immutable_primitive_and_pickle_round_trip():
    query = _query(17)
    result = _result(17)
    chunk = RoadQueryChunk(chunk_id=4, queries=(query,))
    chunk_result = RoadQueryChunkResult(
        chunk_id=4,
        results=(result,),
        worker_pid=4001,
    )
    stats = ProcessRoadBatchStats(
        map_query_wall_ms=1.0,
        worker_query_sum_ms=0.25,
        map_query_count=5,
        worker_count=1,
        chunk_count=1,
    )

    for value in (query, result, chunk, chunk_result, stats):
        _assert_numeric_payload(value)
        assert pickle.loads(pickle.dumps(value)) == value
        with pytest.raises(dataclasses.FrozenInstanceError):
            value.__setattr__(
                dataclasses.fields(value)[0].name,
                getattr(value, dataclasses.fields(value)[0].name),
            )


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0, 0.0),) * 4,
        ((0.0, 0.0),) * 5,
        ((float("nan"), 0.0, 0.0),) * 5,
        ((float("inf"), 0.0, 0.0),) * 5,
    ],
)
def test_footprint_query_rejects_non_numeric_or_nonfinite_shapes(points):
    with pytest.raises(ValueError):
        FootprintQuery(query_id=0, points_xyz=points)


def test_backend_requests_spawn_and_keeps_opendrive_out_of_task_payloads():
    response = _chunk_result(0, (7,))
    backend, context, requested_methods = _backend_with_fake_context(
        worker_count=1,
        responses=(response,),
        chunk_pose_count=16,
    )

    results, stats = backend.query((_query(7),))

    assert requested_methods == ["spawn"]
    assert context.pool_kwargs["processes"] == 1
    assert callable(context.pool_kwargs["initializer"])
    assert context.pool_kwargs["initializer"].__module__ == road_backend.__name__
    initargs = context.pool_kwargs["initargs"]
    assert initargs[0] == "TinyRoad"
    assert initargs[1] == "<OpenDRIVE/>"
    assert len(initargs[2]) == 64
    assert initargs[3] is context.ready_queue
    assert len(context.pool.apply_calls) == 1
    function, args = context.pool.apply_calls[0]
    assert callable(function)
    assert function.__module__ == road_backend.__name__
    assert len(args) == 1 and isinstance(args[0], RoadQueryChunk)
    _assert_numeric_payload(args[0])
    assert "<OpenDRIVE/>" not in repr(args[0])
    assert [result.query_id for result in results] == [7]
    assert stats.worker_count == 1


def test_reverse_chunk_completion_is_reassembled_in_query_input_order():
    query_ids = (20, 3, 14, 1, 9)
    responses = (
        _chunk_result(2, (9,), worker_pid=4002),
        _chunk_result(1, (14, 1), worker_pid=4001),
        _chunk_result(0, (20, 3), worker_pid=4002),
    )
    backend, context, _requested_methods = _backend_with_fake_context(
        responses=responses,
        chunk_pose_count=2,
    )

    results, stats = backend.query(
        tuple(_query(query_id) for query_id in query_ids)
    )

    assert [result.query_id for result in results] == list(query_ids)
    assert stats.map_query_count == 25
    assert stats.worker_query_sum_ms == pytest.approx(1.25)
    assert stats.worker_count == 2
    assert stats.chunk_count == 3
    assert len(context.pool.apply_calls) == 3


@pytest.mark.parametrize(
    "responses",
    [
        (
            "not-a-chunk-result",
            _chunk_result(1, (11,)),
        ),
        (
            _chunk_result(0, (10,)),
            _chunk_result(0, (11,)),
        ),
        (
            _chunk_result(0, (10,)),
            _chunk_result(1, (99,)),
        ),
        (
            _chunk_result(0, (10,)),
            _chunk_result(1, (10,)),
        ),
        (
            _chunk_result(0, (10,)),
            _chunk_result(1, ()),
        ),
    ],
)
def test_malformed_worker_results_fail_closed_and_terminate_pool(responses):
    backend, context, _requested_methods = _backend_with_fake_context(
        responses=responses,
        chunk_pose_count=1,
    )

    with pytest.raises(RoadProcessBackendCrashed):
        backend.query((_query(10), _query(11)))

    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    with pytest.raises(RoadProcessBackendCrashed):
        backend.query((_query(12),))


@pytest.mark.parametrize(
    ("worker_failure", "expected_exception"),
    [
        (multiprocessing.TimeoutError(), RoadProcessBackendTimeout),
        (RuntimeError("synthetic worker crash"), RoadProcessBackendCrashed),
        (EOFError("synthetic worker pipe closed"), RoadProcessBackendCrashed),
    ],
)
def test_timeout_or_worker_crash_terminates_pool(
    worker_failure,
    expected_exception,
):
    backend, context, _requested_methods = _backend_with_fake_context(
        worker_count=1,
        responses=(worker_failure,),
        chunk_pose_count=1,
    )

    with pytest.raises(expected_exception):
        backend.query((_query(1),))

    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1


@pytest.mark.parametrize(
    ("responses", "queries"),
    [
        (
            (_ApplyAsyncFailure(RuntimeError("first dispatch failed")),),
            (_query(1),),
        ),
        (
            (
                _chunk_result(0, (1,)),
                _ApplyAsyncFailure(RuntimeError("mid dispatch failed")),
            ),
            (_query(1), _query(2)),
        ),
    ],
)
def test_apply_async_dispatch_failure_terminates_and_closes_backend(
    responses,
    queries,
):
    backend, context, _requested_methods = _backend_with_fake_context(
        worker_count=1,
        responses=responses,
        chunk_pose_count=1,
    )

    with pytest.raises(
        RoadProcessBackendCrashed,
        match="dispatch failure:RuntimeError",
    ):
        backend.query(queries)

    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1
    with pytest.raises(RoadProcessBackendCrashed, match="closed"):
        backend.query((_query(99),))


@pytest.mark.parametrize("failure_path", ["submit", "async_result"])
def test_query_keyboard_interrupt_cleans_up_and_propagates_unchanged(
    failure_path,
):
    interrupt = KeyboardInterrupt(f"synthetic {failure_path} interrupt")
    response = (
        _ApplyAsyncFailure(interrupt)
        if failure_path == "submit"
        else interrupt
    )
    backend, context, _requested_methods = _backend_with_fake_context(
        worker_count=1,
        responses=(response,),
        chunk_pose_count=1,
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        backend.query((_query(1),))

    assert exc_info.value is interrupt
    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1
    with pytest.raises(RoadProcessBackendCrashed, match="closed"):
        backend.query((_query(2),))


def test_duplicate_input_query_ids_are_rejected_before_dispatch():
    backend, context, _requested_methods = _backend_with_fake_context(
        worker_count=1,
    )

    with pytest.raises(ValueError, match="unique"):
        backend.query((_query(1), _query(1)))

    assert context.pool.apply_calls == []


def test_close_is_idempotent_and_joins_workers():
    backend, context, _requested_methods = _backend_with_fake_context(
        worker_count=1,
    )

    backend.close()
    backend.close()

    assert context.pool.close_count == 1
    assert context.pool.terminate_count == 0
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1
    with pytest.raises(RoadProcessBackendCrashed):
        backend.query((_query(1),))


def test_worker_initialization_failure_terminates_partial_pool():
    context = _FakeContext(
        worker_count=1,
        responses=(),
        readiness=((5000, "ValueError"),),
    )

    with pytest.raises(RoadProcessBackendCrashed, match="initialization"):
        ExactRoadProcessBackend(
            map_name="TinyRoad",
            opendrive="<OpenDRIVE/>",
            map_digest=_opendrive_map_digest("<OpenDRIVE/>"),
            worker_count=1,
            startup_timeout_s=1.0,
            context_factory=lambda _method: context,
        )

    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1


def test_worker_startup_timeout_terminates_partial_pool():
    context = _FakeContext(
        worker_count=1,
        responses=(),
        readiness=(),
    )

    with pytest.raises(RoadProcessBackendTimeout, match="startup"):
        ExactRoadProcessBackend(
            map_name="TinyRoad",
            opendrive="<OpenDRIVE/>",
            map_digest=_opendrive_map_digest("<OpenDRIVE/>"),
            worker_count=1,
            startup_timeout_s=1.0,
            context_factory=lambda _method: context,
        )

    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1


def test_worker_startup_keyboard_interrupt_cleans_up_and_propagates_unchanged():
    interrupt = KeyboardInterrupt("synthetic startup interrupt")
    context = _FakeContext(
        worker_count=1,
        responses=(),
        readiness=(interrupt,),
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        ExactRoadProcessBackend(
            map_name="TinyRoad",
            opendrive="<OpenDRIVE/>",
            map_digest=_opendrive_map_digest("<OpenDRIVE/>"),
            worker_count=1,
            startup_timeout_s=1.0,
            context_factory=lambda _method: context,
        )

    assert exc_info.value is interrupt
    assert context.pool.terminate_count == 1
    assert context.pool.join_count == 1
    assert context.ready_queue.closed_count == 1


def test_backend_rejects_opendrive_digest_mismatch_before_spawning():
    context_factory_calls = []

    with pytest.raises(
        ValueError,
        match="OpenDRIVE snapshot digest mismatch",
    ):
        ExactRoadProcessBackend(
            map_name="TinyRoad",
            opendrive="<OpenDRIVE/>",
            map_digest="0" * 64,
            worker_count=1,
            context_factory=lambda method: context_factory_calls.append(method),
        )

    assert context_factory_calls == []

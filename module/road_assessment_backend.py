"""Lightweight spawn-process backend for exact CARLA map queries.

Only numeric, immutable DTOs cross the process boundary.  Trajectory
densification, heading construction, footprint semantics, road aggregation,
and admission remain in the parent process.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable


class RoadProcessBackendError(RuntimeError):
    """Base class for infrastructure failures in the process query backend."""


class RoadProcessBackendTimeout(RoadProcessBackendError):
    """The ordered batch did not complete inside its global deadline."""


class RoadProcessBackendCrashed(RoadProcessBackendError):
    """A worker or the DTO protocol failed before a complete batch returned."""


@dataclass(frozen=True)
class FootprintQuery:
    """Five exact xyz locations belonging to one oriented footprint pose."""

    query_id: int
    points_xyz: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        query_id = int(self.query_id)
        if query_id < 0:
            raise ValueError("query_id must be nonnegative")
        points = tuple(
            tuple(float(value) for value in point)
            for point in self.points_xyz
        )
        if len(points) != 5 or any(len(point) != 3 for point in points):
            raise ValueError("one footprint query requires exactly five xyz points")
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError("footprint query coordinates must be finite")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "points_xyz", points)


@dataclass(frozen=True)
class WaypointPrimitive:
    """Pickle-safe subset of one exact CARLA Driving waypoint."""

    found: bool
    road_id: int | None = None
    lane_id: int | None = None
    is_junction: bool | None = None
    lane_center_x: float | None = None
    lane_center_y: float | None = None
    lane_yaw_deg: float | None = None
    lane_width: float | None = None


@dataclass(frozen=True)
class FootprintQueryResult:
    """Ordered primitive waypoint results for one footprint."""

    query_id: int
    waypoints: tuple[WaypointPrimitive, ...]
    error_type: str | None
    map_query_count: int
    worker_query_ms: float
    worker_pid: int


@dataclass(frozen=True)
class RoadQueryChunk:
    """Stable chunk of complete footprint poses."""

    chunk_id: int
    queries: tuple[FootprintQuery, ...]


@dataclass(frozen=True)
class RoadQueryChunkResult:
    """One worker's result for a stable input chunk."""

    chunk_id: int
    results: tuple[FootprintQueryResult, ...]
    worker_pid: int


@dataclass(frozen=True)
class ProcessRoadBatchStats:
    """Infrastructure and query workload for one process batch."""

    map_query_wall_ms: float
    worker_query_sum_ms: float
    map_query_count: int
    worker_count: int
    chunk_count: int
    commissioning_batch: bool = False
    query_deadline_ms: float = 0.0


def validate_footprint_query_result(
    result: FootprintQueryResult,
    *,
    expected_query_id: int | None = None,
) -> None:
    """Validate the immutable worker protocol before semantic aggregation."""

    if not isinstance(result, FootprintQueryResult):
        raise ValueError("road worker returned an invalid query result type")
    if (
        isinstance(result.query_id, bool)
        or not isinstance(result.query_id, int)
        or result.query_id < 0
    ):
        raise ValueError("road worker returned an invalid query ID")
    if (
        expected_query_id is not None
        and result.query_id != int(expected_query_id)
    ):
        raise ValueError("road worker returned a mismatched query ID")
    if not isinstance(result.waypoints, tuple):
        raise ValueError("road worker waypoints must be an immutable tuple")
    if (
        isinstance(result.map_query_count, bool)
        or not isinstance(result.map_query_count, int)
        or result.map_query_count < 1
        or result.map_query_count > 5
    ):
        raise ValueError("road worker returned an invalid map-query count")
    if (
        isinstance(result.worker_pid, bool)
        or not isinstance(result.worker_pid, int)
        or result.worker_pid <= 0
    ):
        raise ValueError("road worker returned an invalid worker PID")
    if (
        not isinstance(result.worker_query_ms, (int, float))
        or isinstance(result.worker_query_ms, bool)
        or not math.isfinite(float(result.worker_query_ms))
        or float(result.worker_query_ms) < 0.0
    ):
        raise ValueError("road worker returned invalid query timing")

    if result.error_type is None:
        if len(result.waypoints) != 5 or result.map_query_count != 5:
            raise ValueError(
                "successful footprint query requires five waypoint results"
            )
    else:
        if (
            not isinstance(result.error_type, str)
            or not result.error_type.strip()
            or len(result.waypoints) > 4
            or result.map_query_count != len(result.waypoints) + 1
        ):
            raise ValueError("malformed failed footprint query result")

    for waypoint in result.waypoints:
        if not isinstance(waypoint, WaypointPrimitive):
            raise ValueError("road worker returned an invalid waypoint type")
        if type(waypoint.found) is not bool:
            raise ValueError("road waypoint found flag must be boolean")
        fields = (
            waypoint.road_id,
            waypoint.lane_id,
            waypoint.is_junction,
            waypoint.lane_center_x,
            waypoint.lane_center_y,
            waypoint.lane_yaw_deg,
            waypoint.lane_width,
        )
        if not waypoint.found:
            if any(value is not None for value in fields):
                raise ValueError("missing road waypoint contains stale fields")
            continue
        if (
            isinstance(waypoint.road_id, bool)
            or not isinstance(waypoint.road_id, int)
            or isinstance(waypoint.lane_id, bool)
            or not isinstance(waypoint.lane_id, int)
            or type(waypoint.is_junction) is not bool
        ):
            raise ValueError("road waypoint identity fields are invalid")
        numeric_fields = (
            waypoint.lane_center_x,
            waypoint.lane_center_y,
            waypoint.lane_yaw_deg,
            waypoint.lane_width,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in numeric_fields
        ):
            raise ValueError("road waypoint geometry fields must be finite")
        if float(waypoint.lane_width) <= 0.0:
            raise ValueError("road waypoint lane width must be positive")


_WORKER_MAP: Any = None
_WORKER_CARLA: Any = None


def _raw_opendrive_digest(opendrive: str) -> str:
    return hashlib.sha256(opendrive.encode("utf-8")).hexdigest()


def _map_opendrive_digest(opendrive: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"carla_opendrive\0")
    digest.update(opendrive.encode("utf-8"))
    return digest.hexdigest()


def _worker_initialize(
    map_name: str,
    opendrive: str,
    expected_opendrive_digest: str,
    ready_queue: Any,
) -> None:
    """Construct one immutable CARLA map and report readiness."""

    global _WORKER_CARLA, _WORKER_MAP

    try:
        if _raw_opendrive_digest(opendrive) != expected_opendrive_digest:
            raise ValueError("worker OpenDRIVE digest mismatch")
        import carla as worker_carla

        worker_map = worker_carla.Map(str(map_name), opendrive)
        _WORKER_CARLA = worker_carla
        _WORKER_MAP = worker_map
    except BaseException as exc:
        ready_queue.put((int(os.getpid()), type(exc).__name__))
        raise
    ready_queue.put((int(os.getpid()), None))


def _primitive_from_waypoint(waypoint: Any) -> WaypointPrimitive:
    if waypoint is None:
        return WaypointPrimitive(found=False)
    transform = waypoint.transform
    return WaypointPrimitive(
        found=True,
        road_id=int(waypoint.road_id),
        lane_id=int(waypoint.lane_id),
        is_junction=bool(getattr(waypoint, "is_junction", False)),
        lane_center_x=float(transform.location.x),
        lane_center_y=float(transform.location.y),
        lane_yaw_deg=float(transform.rotation.yaw),
        lane_width=float(waypoint.lane_width),
    )


def _query_chunk(chunk: RoadQueryChunk) -> RoadQueryChunkResult:
    """Execute one chunk without allowing CARLA objects into the result."""

    if _WORKER_MAP is None or _WORKER_CARLA is None:
        raise RuntimeError("road worker map is not initialized")
    worker_pid = int(os.getpid())
    results = []
    for query in chunk.queries:
        query_started_s = time.perf_counter()
        query_count = 0
        primitives = []
        error_type = None
        for point in query.points_xyz:
            query_count += 1
            try:
                waypoint = _WORKER_MAP.get_waypoint(
                    _WORKER_CARLA.Location(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                    ),
                    project_to_road=False,
                    lane_type=_WORKER_CARLA.LaneType.Driving,
                )
                primitives.append(_primitive_from_waypoint(waypoint))
            except Exception as exc:
                error_type = type(exc).__name__
                break
        results.append(
            FootprintQueryResult(
                query_id=int(query.query_id),
                waypoints=tuple(primitives),
                error_type=error_type,
                map_query_count=query_count,
                worker_query_ms=(
                    time.perf_counter() - query_started_s
                )
                * 1000.0,
                worker_pid=worker_pid,
            )
        )
    return RoadQueryChunkResult(
        chunk_id=int(chunk.chunk_id),
        results=tuple(results),
        worker_pid=worker_pid,
    )


class ExactRoadProcessBackend:
    """Owned spawn pool for ordered exact-map footprint queries."""

    def __init__(
        self,
        *,
        map_name: str,
        opendrive: str,
        map_digest: str,
        worker_count: int,
        chunk_pose_count: int = 16,
        startup_timeout_s: float = 10.0,
        commissioning_timeout_s: float = 0.5,
        batch_timeout_s: float = 0.25,
        context_factory: Callable[[str], Any] = multiprocessing.get_context,
    ) -> None:
        self.worker_count = int(worker_count)
        self.chunk_pose_count = int(chunk_pose_count)
        self.commissioning_timeout_s = float(commissioning_timeout_s)
        self.batch_timeout_s = float(batch_timeout_s)
        if self.worker_count < 1:
            raise ValueError("worker_count must be positive")
        if self.chunk_pose_count < 1:
            raise ValueError("chunk_pose_count must be positive")
        if (
            not math.isfinite(self.commissioning_timeout_s)
            or self.commissioning_timeout_s <= 0.0
        ):
            raise ValueError(
                "commissioning_timeout_s must be finite and positive"
            )
        if (
            not math.isfinite(self.batch_timeout_s)
            or self.batch_timeout_s <= 0.0
        ):
            raise ValueError("batch_timeout_s must be finite and positive")
        startup_timeout = float(startup_timeout_s)
        if not math.isfinite(startup_timeout) or startup_timeout <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        if not isinstance(opendrive, str) or not opendrive:
            raise ValueError("process backend requires nonempty OpenDRIVE")
        map_identity_digest = _map_opendrive_digest(opendrive)
        if (
            not isinstance(map_digest, str)
            or map_digest != map_identity_digest
        ):
            raise ValueError(
                "process backend OpenDRIVE snapshot digest mismatch"
            )

        self._closed = False
        self._commissioning_pending = True
        self.last_query_commissioning_batch = False
        self.last_query_deadline_ms = 0.0
        self._context = context_factory("spawn")
        self._ready_queue = self._context.Queue()
        self._pool = None
        try:
            self._pool = self._context.Pool(
                processes=self.worker_count,
                initializer=_worker_initialize,
                initargs=(
                    str(map_name),
                    opendrive,
                    _raw_opendrive_digest(opendrive),
                    self._ready_queue,
                ),
            )
            self._wait_until_ready(startup_timeout)
        except BaseException:
            self._terminate_pool()
            raise

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        ready_pids = set()
        while len(ready_pids) < self.worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RoadProcessBackendTimeout(
                    "road worker startup readiness timed out"
                )
            try:
                worker_pid, error_type = self._ready_queue.get(
                    timeout=remaining
                )
            except Empty as exc:
                raise RoadProcessBackendTimeout(
                    "road worker startup readiness timed out"
                ) from exc
            if error_type is not None:
                raise RoadProcessBackendCrashed(
                    f"road worker initialization failed:{error_type}"
                )
            ready_pids.add(int(worker_pid))

    def query(
        self,
        queries: tuple[FootprintQuery, ...],
    ) -> tuple[tuple[FootprintQueryResult, ...], ProcessRoadBatchStats]:
        """Return results in query input order or raise an infrastructure error."""

        if self._closed or self._pool is None:
            raise RoadProcessBackendCrashed("road process backend is closed")
        ordered_queries = tuple(queries)
        expected_ids = tuple(query.query_id for query in ordered_queries)
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("footprint query IDs must be unique")
        if not ordered_queries:
            self.last_query_commissioning_batch = False
            self.last_query_deadline_ms = 0.0
            return (), ProcessRoadBatchStats(
                map_query_wall_ms=0.0,
                worker_query_sum_ms=0.0,
                map_query_count=0,
                worker_count=self.worker_count,
                chunk_count=0,
                commissioning_batch=False,
                query_deadline_ms=0.0,
            )

        chunks = tuple(
            RoadQueryChunk(
                chunk_id=chunk_id,
                queries=ordered_queries[start : start + self.chunk_pose_count],
            )
            for chunk_id, start in enumerate(
                range(0, len(ordered_queries), self.chunk_pose_count)
            )
        )
        batch_started_s = time.perf_counter()
        commissioning_batch = bool(self._commissioning_pending)
        query_timeout_s = (
            self.commissioning_timeout_s
            if commissioning_batch
            else self.batch_timeout_s
        )
        self.last_query_commissioning_batch = commissioning_batch
        self.last_query_deadline_ms = query_timeout_s * 1000.0
        deadline = time.monotonic() + query_timeout_s
        chunk_results = {}
        failure_phase = "dispatch"
        try:
            pending = []
            for chunk in chunks:
                pending.append(
                    self._pool.apply_async(_query_chunk, (chunk,))
                )
            failure_phase = "collection"
            for async_result in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RoadProcessBackendTimeout(
                        "road process batch timed out"
                    )
                result = async_result.get(timeout=remaining)
                if not isinstance(result, RoadQueryChunkResult):
                    raise RoadProcessBackendCrashed(
                        "road worker returned an invalid chunk type"
                    )
                if (
                    isinstance(result.chunk_id, bool)
                    or not isinstance(result.chunk_id, int)
                    or not isinstance(result.results, tuple)
                    or isinstance(result.worker_pid, bool)
                    or not isinstance(result.worker_pid, int)
                    or result.worker_pid <= 0
                ):
                    raise RoadProcessBackendCrashed(
                        "road worker returned a malformed chunk"
                    )
                try:
                    for query_result in result.results:
                        validate_footprint_query_result(query_result)
                        if query_result.worker_pid != result.worker_pid:
                            raise ValueError(
                                "query and chunk worker PIDs differ"
                            )
                except ValueError as exc:
                    raise RoadProcessBackendCrashed(
                        f"road worker protocol failure:{exc}"
                    ) from exc
                if result.chunk_id in chunk_results:
                    raise RoadProcessBackendCrashed(
                        "road worker returned a duplicate chunk"
                    )
                chunk_results[result.chunk_id] = result

            failure_phase = "protocol"
            if tuple(sorted(chunk_results)) != tuple(range(len(chunks))):
                raise RoadProcessBackendCrashed(
                    "road process batch returned missing chunk IDs"
                )
            flattened = tuple(
                result
                for chunk_id in range(len(chunks))
                for result in chunk_results[chunk_id].results
            )
            returned_ids = tuple(result.query_id for result in flattened)
            if (
                len(set(returned_ids)) != len(returned_ids)
                or set(returned_ids) != set(expected_ids)
            ):
                raise RoadProcessBackendCrashed(
                    "road process batch returned malformed query IDs"
                )
            by_id = {result.query_id: result for result in flattened}
            ordered_results = tuple(
                by_id[query_id] for query_id in expected_ids
            )
            for query_id, result in zip(expected_ids, ordered_results):
                validate_footprint_query_result(
                    result,
                    expected_query_id=query_id,
                )
        except multiprocessing.TimeoutError as exc:
            self._terminate_pool()
            raise RoadProcessBackendTimeout(
                "road process batch timed out"
            ) from exc
        except RoadProcessBackendError:
            self._terminate_pool()
            raise
        except ValueError as exc:
            self._terminate_pool()
            raise RoadProcessBackendCrashed(
                f"road worker protocol failure:{exc}"
            ) from exc
        except Exception as exc:
            self._terminate_pool()
            raise RoadProcessBackendCrashed(
                "road process "
                f"{failure_phase} failure:{type(exc).__name__}"
            ) from exc
        except BaseException:
            self._terminate_pool()
            raise
        self._commissioning_pending = False
        return ordered_results, ProcessRoadBatchStats(
            map_query_wall_ms=(
                time.perf_counter() - batch_started_s
            )
            * 1000.0,
            worker_query_sum_ms=sum(
                result.worker_query_ms for result in ordered_results
            ),
            map_query_count=sum(
                result.map_query_count for result in ordered_results
            ),
            worker_count=self.worker_count,
            chunk_count=len(chunks),
            commissioning_batch=commissioning_batch,
            query_deadline_ms=query_timeout_s * 1000.0,
        )

    def _terminate_pool(self) -> None:
        pool = self._pool
        self._pool = None
        self._closed = True
        if pool is not None:
            try:
                pool.terminate()
            finally:
                pool.join()
        ready_queue = getattr(self, "_ready_queue", None)
        if ready_queue is not None:
            try:
                ready_queue.close()
            except Exception:
                pass

    def close(self) -> None:
        """Close the pool idempotently without leaving worker processes."""

        if self._closed:
            return
        pool = self._pool
        self._pool = None
        self._closed = True
        if pool is not None:
            try:
                pool.close()
            finally:
                pool.join()
        try:
            self._ready_queue.close()
        except Exception:
            pass

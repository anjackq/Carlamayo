"""Dependency-free JSONL telemetry and bounded episode aggregation.

The control loop can emit full-fidelity events to :class:`JsonlWriter` while
keeping only a bounded reservoir of numeric observations in memory.  This file
deliberately imports neither CARLA, NumPy, nor Torch so telemetry remains usable
in simulator-free tests and failure-reporting paths.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import operator
import os
import random
import threading
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TextIO


RUNTIME_SCHEMA_VERSION = 2
SUPPORTED_RUNTIME_SCHEMA_VERSIONS = (1, RUNTIME_SCHEMA_VERSION)


def to_json_safe(value: Any) -> Any:
    """Recursively convert common runtime values to strict JSON-safe values.

    NumPy-like scalars and arrays are supported through their public ``item``
    and ``tolist`` protocols, without importing NumPy.  Non-finite floating
    point values become ``None`` so the result is valid with
    ``json.dumps(..., allow_nan=False)``.
    """

    return _to_json_safe(value, seen=set())


def _exact_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(normalized)


def _to_json_safe(value: Any, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _to_json_safe(value.value, seen)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")

    value_id = id(value)
    if value_id in seen:
        raise ValueError("Cannot convert a recursive value to JSON")

    # Runtime records can intentionally summarize opaque/high-volume fields
    # (for example camera arrays) instead of expanding their dataclass fields.
    to_json_dict_method = getattr(value, "to_json_dict", None)
    if callable(to_json_dict_method):
        seen.add(value_id)
        try:
            return _to_json_safe(to_json_dict_method(), seen)
        finally:
            seen.remove(value_id)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(value_id)
        try:
            return {
                field.name: _to_json_safe(getattr(value, field.name), seen)
                for field in dataclasses.fields(value)
            }
        finally:
            seen.remove(value_id)

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            converted = {}
            for key, item in value.items():
                safe_key = _to_json_safe(key, seen)
                if not isinstance(safe_key, str):
                    safe_key = str(safe_key)
                converted[safe_key] = _to_json_safe(item, seen)
            return converted
        finally:
            seen.remove(value_id)

    if isinstance(value, (set, frozenset)):
        seen.add(value_id)
        try:
            converted = [_to_json_safe(item, seen) for item in value]
            return sorted(converted, key=lambda item: repr(item))
        finally:
            seen.remove(value_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(value_id)
        try:
            return [_to_json_safe(item, seen) for item in value]
        finally:
            seen.remove(value_id)

    # NumPy scalar objects expose item(); zero-dimensional Torch tensors do too.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError, RuntimeError):
            item = value
        if item is not value:
            return _to_json_safe(item, seen)

    # NumPy arrays and tensor-like objects expose tolist().
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        converted = tolist_method()
        if converted is not value:
            return _to_json_safe(converted, seen)

    raise TypeError(f"Unsupported telemetry value: {type(value).__name__}")


class JsonlWriter:
    """Thread-safe append-only JSON Lines writer.

    The file is opened lazily, or immediately on entering a context manager.
    Existing content is never truncated.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        flush: bool = True,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.flush = bool(flush)
        self.fsync = bool(fsync)
        self._stream: TextIO | None = None
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        return self._stream is None or self._stream.closed

    def open(self) -> JsonlWriter:
        with self._lock:
            if self._stream is None or self._stream.closed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("a", encoding="utf-8")
        return self

    def append(self, event: Any) -> dict[str, Any] | list[Any] | Any:
        """Append one converted event and return the JSON-safe payload."""

        payload = to_json_safe(event)
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock:
            self.open()
            assert self._stream is not None
            self._stream.write(line)
            self._stream.write("\n")
            if self.flush or self.fsync:
                self._stream.flush()
            if self.fsync:
                os.fsync(self._stream.fileno())
        return payload

    write = append

    def write_event(
        self,
        event_type: str,
        event: Any | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Append a named event, merging an optional mapping/dataclass payload."""

        if event is None:
            payload: dict[str, Any] = {}
        else:
            converted = to_json_safe(event)
            if not isinstance(converted, Mapping):
                payload = {"payload": converted}
            else:
                payload = dict(converted)
        payload.update(to_json_safe(fields))
        payload["event_type"] = str(event_type)
        self.append(payload)
        return payload

    def close(self) -> None:
        with self._lock:
            if self._stream is not None and not self._stream.closed:
                self._stream.flush()
                self._stream.close()

    def __enter__(self) -> JsonlWriter:
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


# A conventional all-caps spelling is kept as a convenience for call sites.
JSONLWriter = JsonlWriter


class _BoundedCounter:
    """Counter with bounded key cardinality and an overflow bucket."""

    _OVERFLOW = "__other__"

    def __init__(self, max_keys: int) -> None:
        if max_keys < 1:
            raise ValueError("max_categories must be at least 1")
        self.max_keys = int(max_keys)
        self._counts: dict[str, int] = {}

    def add(self, key: Any, amount: int = 1) -> None:
        normalized = str(key) if key not in (None, "") else "unspecified"
        if normalized in self._counts:
            self._counts[normalized] += int(amount)
        elif len(self._counts) < self.max_keys:
            self._counts[normalized] = int(amount)
        else:
            self._counts[self._OVERFLOW] = self._counts.get(self._OVERFLOW, 0) + int(amount)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self._counts.items()))


class _ReservoirMetric:
    """Exact scalar statistics plus bounded-memory percentile samples."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("max_samples must be at least 1")
        self.capacity = int(capacity)
        self.samples: list[float] = []
        self.count = 0
        self.invalid_count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self._random = random.Random(seed)

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            self.invalid_count += 1
            return
        if not math.isfinite(number):
            self.invalid_count += 1
            return

        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

        if len(self.samples) < self.capacity:
            self.samples.append(number)
            return
        candidate = self._random.randrange(self.count)
        if candidate < self.capacity:
            self.samples[candidate] = number

    def summary(self) -> dict[str, int | float | None]:
        ordered = sorted(self.samples)
        return {
            "count": self.count,
            "invalid_count": self.invalid_count,
            "sample_count": len(ordered),
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count if self.count else None,
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
        }


def _percentile(ordered_values: list[float], quantile: float) -> float | None:
    if not ordered_values:
        return None
    position = (len(ordered_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered_values[lower]
    fraction = position - lower
    return ordered_values[lower] + fraction * (ordered_values[upper] - ordered_values[lower])


class RuntimeMetrics:
    """Bounded-memory telemetry aggregation for one simulation episode."""

    _LATENCY_KEYS = (
        "inference_latency_s",
        "inference_wall_latency_s",
        "wall_latency_s",
        "latency_s",
    )
    _AGE_KEYS = ("source_age_s", "plan_source_age_s", "plan_age_s", "age_s")
    _AGE_PROXY_KEYS = ("source_age_proxy_s", "plan_source_age_proxy_s")
    _SIMULATION_TIME_KEYS = ("simulation_time_s", "current_simulation_time_s")
    _INACTIVE_FALLBACK_STATES = {"", "none", "inactive", "off", "tracking"}

    def __init__(
        self,
        *,
        max_samples: int = 2048,
        max_categories: int = 128,
        reservoir_seed: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_samples = int(max_samples)
        self.max_categories = int(max_categories)
        self._clock = clock
        self._started_at = float(clock())
        self._lock = threading.RLock()

        self._latency = _ReservoirMetric(max_samples, reservoir_seed)
        self._age = _ReservoirMetric(max_samples, reservoir_seed + 1)
        self._age_proxy = _ReservoirMetric(max_samples, reservoir_seed + 2)
        self._event_counts = _BoundedCounter(max_categories)
        self._event_status_counts = _BoundedCounter(max_categories)
        self._controller_states = _BoundedCounter(max_categories)
        self._rejection_reasons = _BoundedCounter(max_categories)
        self._override_reasons = _BoundedCounter(max_categories)
        self._fallback_reasons = _BoundedCounter(max_categories)

        self._total_events = 0
        self._rejection_count = 0
        self._override_count = 0
        self._fallback_count = 0
        self._collision_total = 0
        self._last_collision_count: int | None = None
        self._first_simulation_time_s: float | None = None
        self._last_simulation_time_s: float | None = None

    def record_event(
        self,
        event_type: str,
        event: Any | None = None,
        *,
        latency_s: Any | None = None,
        age_s: Any | None = None,
        rejected: bool | None = None,
        rejection_reason: Any | None = None,
        overridden: bool | None = None,
        override_reason: Any | None = None,
        fallback: bool | None = None,
        fallback_reason: Any | None = None,
        collision_count: int | None = None,
        collision_delta: int | None = None,
        aggregate_age: bool = True,
        aggregate_rejection: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        """Aggregate one event and return its normalized JSON-safe payload.

        ``collision_count`` is preferably a cumulative episode counter. A legacy
        spawn-local counter may reset to zero after respawn; callers must record
        that reset sample (or use explicit ``collision_delta`` values) so no
        collisions are ambiguous. A single event must not provide both forms.
        """

        if not isinstance(aggregate_age, bool):
            raise TypeError("aggregate_age must be a bool")
        if not isinstance(aggregate_rejection, bool):
            raise TypeError("aggregate_rejection must be a bool")

        payload = self._event_payload(event_type, event, fields)
        explicit_fields = {
            "inference_latency_s": latency_s,
            "source_age_s": age_s,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "safety_override_applied": overridden,
            "override_reason": override_reason,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "collision_count": collision_count,
            "collision_delta": collision_delta,
        }
        payload.update(
            {
                key: to_json_safe(value)
                for key, value in explicit_fields.items()
                if value is not None
            }
        )

        event_collision_count = payload.get("collision_count")
        event_collision_delta = payload.get("collision_delta")
        if event_collision_count is not None and event_collision_delta is not None:
            raise ValueError("Provide collision_count or collision_delta, not both")

        with self._lock:
            self._total_events += 1
            self._event_counts.add(event_type)
            if payload.get("status") not in (None, ""):
                self._event_status_counts.add(f"{event_type}.{payload['status']}")
            if payload.get("controller_state") not in (None, ""):
                self._controller_states.add(payload["controller_state"])

            self._latency.add(
                latency_s if latency_s is not None else _first_present(payload, self._LATENCY_KEYS)
            )
            if aggregate_age:
                self._age.add(
                    age_s
                    if age_s is not None
                    else _first_present(payload, self._AGE_KEYS)
                )
                self._age_proxy.add(_first_present(payload, self._AGE_PROXY_KEYS))
            self._record_simulation_time(_first_present(payload, self._SIMULATION_TIME_KEYS))

            rejection_reason = _coalesce(rejection_reason, payload.get("rejection_reason"))
            if rejected is None:
                rejected = _truthy_field(payload, "rejected")
                if rejected is None and payload.get("valid") is False:
                    rejected = True
                if rejected is None and rejection_reason not in (None, ""):
                    rejected = True
            if rejected and aggregate_rejection:
                self._rejection_count += 1
                self._rejection_reasons.add(rejection_reason)

            override_reason = _coalesce(
                override_reason,
                payload.get("override_reason"),
                payload.get("safety_override_reason"),
            )
            if overridden is None:
                overridden = _first_truthy_field(
                    payload,
                    (
                        "overridden",
                        "override_applied",
                        "safety_override_applied",
                        "safety_override",
                    ),
                )
                if overridden is None and override_reason not in (None, ""):
                    overridden = True
            if overridden:
                self._override_count += 1
                self._override_reasons.add(override_reason)

            fallback_reason = _coalesce(fallback_reason, payload.get("fallback_reason"))
            if fallback is None:
                fallback = _first_truthy_field(payload, ("fallback", "fallback_applied"))
                fallback_state = payload.get("fallback_state")
                if fallback is None and isinstance(fallback_state, str):
                    fallback = fallback_state.strip().lower() not in self._INACTIVE_FALLBACK_STATES
                if fallback is None and fallback_reason not in (None, ""):
                    fallback = True
            if fallback:
                self._fallback_count += 1
                self._fallback_reasons.add(fallback_reason)

            if event_collision_count is not None:
                self._record_collision_counter(event_collision_count)
            elif event_collision_delta is not None:
                self._record_collision_delta(event_collision_delta)
            elif str(event_type).strip().lower() in {"collision", "collision_event"}:
                self._record_collision_delta(1)

        return payload

    record = record_event

    def record_collision_count(self, count: int) -> None:
        """Observe a collision counter that may reset when the ego respawns."""

        with self._lock:
            self._record_collision_counter(count)

    def record_collision(self, count: int = 1) -> None:
        """Add one or more explicit collision events to the episode total."""

        with self._lock:
            self._record_collision_delta(count)

    def final_summary(self, **metadata: Any) -> dict[str, Any]:
        """Return a JSON-safe final summary for the episode."""

        with self._lock:
            simulation_duration = None
            if (
                self._first_simulation_time_s is not None
                and self._last_simulation_time_s is not None
            ):
                simulation_duration = max(
                    0.0, self._last_simulation_time_s - self._first_simulation_time_s
                )
            summary = {
                "schema_version": RUNTIME_SCHEMA_VERSION,
                "event_type": "episode_summary",
                "wall_duration_s": max(0.0, float(self._clock()) - self._started_at),
                "simulation_duration_s": simulation_duration,
                "total_events": self._total_events,
                "event_counts": self._event_counts.as_dict(),
                "event_status_counts": self._event_status_counts.as_dict(),
                "controller_states": self._controller_states.as_dict(),
                "inference_latency_s": self._latency.summary(),
                "source_age_s": self._age.summary(),
                "source_age_proxy_s": self._age_proxy.summary(),
                "rejections": {
                    "total": self._rejection_count,
                    "reasons": self._rejection_reasons.as_dict(),
                },
                "safety_overrides": {
                    "total": self._override_count,
                    "reasons": self._override_reasons.as_dict(),
                },
                "fallbacks": {
                    "total": self._fallback_count,
                    "reasons": self._fallback_reasons.as_dict(),
                },
                "collisions": {"total": self._collision_total},
                "aggregation_limits": {
                    "max_samples_per_metric": self.max_samples,
                    "max_categories_per_counter": self.max_categories,
                },
            }
            summary.update(to_json_safe(metadata))
            return summary

    summary = final_summary

    def write_final_summary(self, writer: JsonlWriter, **metadata: Any) -> dict[str, Any]:
        summary = self.final_summary(**metadata)
        writer.append(summary)
        return summary

    def _event_payload(
        self,
        event_type: str,
        event: Any | None,
        fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        if event is None:
            payload: dict[str, Any] = {}
        else:
            converted = to_json_safe(event)
            if isinstance(converted, Mapping):
                payload = dict(converted)
            else:
                payload = {"payload": converted}
        payload.update(to_json_safe(fields))
        payload["event_type"] = str(event_type)
        return payload

    def _record_simulation_time(self, value: Any) -> None:
        if value is None:
            return
        try:
            simulation_time = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(simulation_time) or simulation_time < 0.0:
            return
        self._first_simulation_time_s = min(
            simulation_time,
            self._first_simulation_time_s
            if self._first_simulation_time_s is not None
            else simulation_time,
        )
        self._last_simulation_time_s = max(
            simulation_time,
            self._last_simulation_time_s
            if self._last_simulation_time_s is not None
            else simulation_time,
        )

    def _record_collision_counter(self, count: Any) -> None:
        normalized = _exact_nonnegative_int(count, "collision_count")
        if self._last_collision_count is None:
            delta = normalized
        elif normalized >= self._last_collision_count:
            delta = normalized - self._last_collision_count
        else:
            # The per-ego counter reset after respawn. Keep the episode total.
            delta = normalized
        self._collision_total += delta
        self._last_collision_count = normalized

    def _record_collision_delta(self, delta: Any) -> None:
        normalized = _exact_nonnegative_int(delta, "collision_delta")
        self._collision_total += normalized


EpisodeMetrics = RuntimeMetrics


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _truthy_field(payload: Mapping[str, Any], key: str) -> bool | None:
    if key not in payload or payload[key] is None:
        return None
    return bool(payload[key])


def _first_truthy_field(payload: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    for key in keys:
        result = _truthy_field(payload, key)
        if result is not None:
            return result
    return None


def _coalesce(*values: Any) -> Any | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


__all__ = [
    "EpisodeMetrics",
    "JSONLWriter",
    "JsonlWriter",
    "RuntimeMetrics",
    "to_json_safe",
]

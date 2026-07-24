"""No-server smoke tests for the real spawned CARLA map worker."""

from __future__ import annotations

import importlib
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from module.road_assessment_backend import (
    ExactRoadProcessBackend,
    FootprintQuery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CARLAMAYO_ROAD_WORKER_IMPORT="
TINY_STRAIGHT_XODR = """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="tiny" version="1.00"
          date="" north="10" south="-10" east="100" west="0" vendor="test"/>
  <road name="straight" length="100.0" id="1" junction="-1">
    <link/>
    <type s="0.0" type="town"><speed max="50" unit="km/h"/></type>
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="100.0">
        <line/>
      </geometry>
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneOffset s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="broken" material="standard"
                      color="white" width="0.12" laneChange="both"/>
          </lane>
        </left>
        <center>
          <lane id="0" type="none" level="false">
            <roadMark sOffset="0.0" type="solid" material="standard"
                      color="yellow" width="0.12" laneChange="none"/>
          </lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="broken" material="standard"
                      color="white" width="0.12" laneChange="both"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def _footprint(query_id: int, x: float, y: float) -> FootprintQuery:
    return FootprintQuery(
        query_id=query_id,
        points_xyz=(
            (x, y, 0.0),
            (x - 0.4, y - 0.3, 0.0),
            (x - 0.4, y + 0.3, 0.0),
            (x + 0.4, y - 0.3, 0.0),
            (x + 0.4, y + 0.3, 0.0),
        ),
    )


def test_backend_module_import_does_not_load_carla_or_model_runtime():
    probe = textwrap.dedent(
        f"""
        import json
        import sys

        before = set(sys.modules)
        import module.road_assessment_backend
        after = set(sys.modules)
        forbidden_roots = (
            "carla",
            "torch",
            "transformers",
            "module.inference",
            "carlamayo_closed_loop",
        )
        added = sorted(
            name
            for name in after - before
            if any(
                name == root or name.startswith(root + ".")
                for root in forbidden_roots
            )
        )
        print({RESULT_PREFIX!r} + json.dumps({{"added": added}}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(lines) == 1
    assert json.loads(lines[0][len(RESULT_PREFIX) :])["added"] == []


def test_real_spawn_worker_builds_carla_map_without_server_and_closes():
    # Other runtime tests intentionally install a tiny fake ``carla`` module
    # during collection. Probe the real PythonAPI without letting collection
    # order turn this process-backend smoke test into a false skip.
    previous_carla = sys.modules.pop("carla", None)
    try:
        try:
            real_carla = importlib.import_module("carla")
        except ImportError:
            real_carla = None
    finally:
        if previous_carla is not None:
            sys.modules["carla"] = previous_carla
        elif real_carla is not None:
            sys.modules["carla"] = real_carla
    if real_carla is None:
        pytest.skip("CARLA PythonAPI is not installed")
    if not hasattr(real_carla, "Map"):
        pytest.skip("CARLA PythonAPI does not expose carla.Map")

    parent_pid = os.getpid()
    child_pids_before = {
        child.pid for child in multiprocessing.active_children()
    }
    backend = ExactRoadProcessBackend(
        map_name="TinyStraight",
        opendrive=TINY_STRAIGHT_XODR,
        map_digest=hashlib.sha256(
            b"carla_opendrive\0" + TINY_STRAIGHT_XODR.encode("utf-8")
        ).hexdigest(),
        worker_count=1,
        chunk_pose_count=2,
        startup_timeout_s=15.0,
        batch_timeout_s=5.0,
    )
    worker_pid = None
    try:
        queries = (
            _footprint(9, 10.0, 1.75),
            _footprint(4, 20.0, 1.75),
            _footprint(7, 30.0, 10.0),
        )
        results, stats = backend.query(queries)
        worker_pid = results[0].worker_pid

        assert [result.query_id for result in results] == [9, 4, 7]
        assert worker_pid != parent_pid
        assert all(result.worker_pid == worker_pid for result in results)
        assert all(
            waypoint.found
            and waypoint.road_id == 1
            and waypoint.lane_id == -1
            and waypoint.lane_width == pytest.approx(3.5)
            for result in results[:2]
            for waypoint in result.waypoints
        )
        assert all(
            not waypoint.found
            for waypoint in results[2].waypoints
        )
        assert stats.map_query_count == 15
        assert stats.worker_count == 1
        assert stats.chunk_count == 2
    finally:
        backend.close()

    assert worker_pid is not None
    child_pids_after = {
        child.pid for child in multiprocessing.active_children()
    }
    assert worker_pid not in child_pids_after
    assert child_pids_after <= child_pids_before

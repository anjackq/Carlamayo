import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

from module import carla_safety_adapter
from module.carla_safety_adapter import (
    CarlaGroundTruthSafetyAdapter,
    PlanAdmissionStatus,
    RoadRecoveryMode,
    StoppingReserveStatus,
    decide_plan_admission,
    raw_physical_stopping_speed_cap_mps,
    road_stopping_speed_cap_mps,
)
from module.safety_shield import AssessmentStatus, EgoKinematics, SafetyPolicy
from module.trajectory_runtime import detect_terminal_stop_index

GEOMETRIC_STOPPING_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22862336_geometric_stopping.json"
)
NARROW_JUNCTION_GAP_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22863131_narrow_junction_gap.json"
)
BOUNDARY_RECOVERY_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22863138_boundary_recovery.json"
)


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeRotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)


class FakeTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = FakeLocation(x, y, z)
        self.rotation = FakeRotation(yaw)


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeBoundingBox:
    def __init__(
        self,
        *,
        half_length=0.4,
        half_width=0.3,
        offset_x=0.0,
        offset_y=0.0,
        yaw=0.0,
    ):
        self.extent = types.SimpleNamespace(
            x=float(half_length),
            y=float(half_width),
            z=0.5,
        )
        self.location = FakeLocation(offset_x, offset_y, 0.0)
        self.rotation = FakeRotation(yaw)


class FakeWaypoint:
    def __init__(
        self,
        location,
        *,
        road_id=1,
        lane_id=-1,
        lane_width=4.0,
        lane_yaw=0.0,
        is_junction=False,
    ):
        self.transform = FakeTransform(
            x=location.x,
            y=0.0,
            z=location.z,
            yaw=lane_yaw,
        )
        self.road_id = int(road_id)
        self.lane_id = int(lane_id)
        self.lane_width = float(lane_width)
        self.is_junction = bool(is_junction)


class RecordingMap:
    def __init__(self, resolver=None):
        self.resolver = resolver or (lambda location: FakeWaypoint(location))
        self.calls = []

    def get_waypoint(self, location, *, project_to_road, lane_type):
        self.calls.append(
            {
                "location": location,
                "project_to_road": project_to_road,
                "lane_type": lane_type,
            }
        )
        return self.resolver(location)


class FakeActor:
    def __init__(self, actor_id, type_id, bounding_box=None):
        self.id = int(actor_id)
        self.type_id = str(type_id)
        self.bounding_box = bounding_box or FakeBoundingBox()

    def get_transform(self):
        raise AssertionError("adapter must not read live actor transforms")

    def get_velocity(self):
        raise AssertionError("adapter must not read live actor velocities")


class FakeActorList:
    def __init__(self, actors):
        self.actors = tuple(actors)
        self.filter_patterns = []

    def filter(self, pattern):
        self.filter_patterns.append(pattern)
        prefix = pattern.removesuffix("*")
        return [actor for actor in self.actors if actor.type_id.startswith(prefix)]


class FakeWorld:
    def __init__(self, carla_map, actors=()):
        self.carla_map = carla_map
        self.actor_list = FakeActorList(actors)
        self.actor_query_count = 0

    def get_map(self):
        return self.carla_map

    def get_actors(self):
        self.actor_query_count += 1
        return self.actor_list


class FakeActorSnapshot:
    def __init__(self, transform, velocity):
        self.transform = transform
        self.velocity = velocity

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity


class FakeSnapshot:
    def __init__(self, frame, actor_snapshots):
        self.frame = int(frame)
        self.actor_snapshots = dict(actor_snapshots)

    def find(self, actor_id):
        return self.actor_snapshots.get(int(actor_id))


@pytest.fixture
def driving_lane_type(monkeypatch):
    driving = object()
    monkeypatch.setattr(
        carla_safety_adapter,
        "carla",
        types.SimpleNamespace(
            Location=FakeLocation,
            LaneType=types.SimpleNamespace(Driving=driving),
        ),
    )
    return driving


def _ego_kinematics(*, half_length=0.4, half_width=0.3):
    return EgoKinematics(
        frame_id=50,
        actor_id=1,
        center_xy=(0.0, 0.0),
        yaw_rad=0.0,
        velocity_xy=(0.0, 0.0),
        half_length_m=half_length,
        half_width_m=half_width,
    )


def _adapter(
    carla_map,
    *,
    actors=(),
    ego_bounding_box=None,
    policy=None,
):
    ego = FakeActor(
        1,
        "vehicle.ego",
        bounding_box=ego_bounding_box or FakeBoundingBox(),
    )
    world = FakeWorld(carla_map, (ego, *actors))
    return CarlaGroundTruthSafetyAdapter(world, ego, policy=policy), world, ego


def _tick_context(
    ego,
    *,
    frame=50,
    simulation_time_s=5.0,
    ego_x=0.0,
    ego_y=0.0,
    speed_x=0.0,
    actor_snapshots=None,
):
    ego_transform = FakeTransform(x=ego_x, y=ego_y)
    ego_velocity = FakeVector(x=speed_x)
    snapshots = {
        ego.id: FakeActorSnapshot(ego_transform, ego_velocity),
        **(actor_snapshots or {}),
    }
    return types.SimpleNamespace(
        frame_id=frame,
        simulation_time_s=simulation_time_s,
        snapshot=FakeSnapshot(frame, snapshots),
        ego_transform=ego_transform,
        ego_velocity=ego_velocity,
    )


def _plan(*, plan_id="plan-1", start_time_s=5.0):
    return types.SimpleNamespace(
        plan_id=plan_id,
        world_points=np.array(
            [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=np.float64,
        ),
        waypoint_times_s=np.array(
            [start_time_s + 0.1, start_time_s + 0.2, start_time_s + 0.3],
            dtype=np.float64,
        ),
    )


def _long_plan(*, plan_id="long-plan", start_time_s=5.0, step_m=0.1):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = step_m * np.arange(1, 65)
    return types.SimpleNamespace(
        plan_id=plan_id,
        world_points=points,
        waypoint_times_s=start_time_s + np.arange(1, 65, dtype=np.float64) * 0.1,
    )


def test_every_lane_query_is_exact_driving_without_projection(driving_lane_type):
    carla_map = RecordingMap()
    adapter, _, _ = _adapter(carla_map)

    road = adapter._assess_path_road(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        _ego_kinematics(),
    )

    assert road.status is AssessmentStatus.SAFE
    assert carla_map.calls
    assert all(call["project_to_road"] is False for call in carla_map.calls)
    assert all(call["lane_type"] is driving_lane_type for call in carla_map.calls)


def test_center_valid_but_corner_off_lane_is_rejected(driving_lane_type):
    def resolve(location):
        if abs(location.y) >= 0.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, _ = _adapter(RecordingMap(resolve))
    samples = adapter._query_footprint(
        center_xyz=np.array([0.0, 0.0, 0.0]),
        yaw_rad=0.0,
        half_length_m=0.4,
        half_width_m=0.6,
        sample_index_start=0,
    )
    road = carla_safety_adapter.assess_road_containment(samples)

    assert samples[0].contained is True
    assert road.status is AssessmentStatus.UNSAFE
    assert "footprint_corner_off_driving_lane" in road.reason_codes


def test_spawn_preflight_uses_actual_ego_bbox_and_exact_lane_queries(
    driving_lane_type,
):
    carla_map = RecordingMap()
    adapter, _, _ = _adapter(
        carla_map,
        ego_bounding_box=FakeBoundingBox(
            half_length=1.2,
            half_width=0.8,
            offset_x=0.2,
            offset_y=0.1,
            yaw=3.0,
        ),
    )

    road = adapter.assess_ego_transform(FakeTransform(x=4.0, y=0.0, yaw=7.0))

    assert road.status is AssessmentStatus.SAFE
    assert road.quality == "carla_ground_truth_spawn_preflight"
    assert road.sample_count == 5
    assert len(carla_map.calls) == 5
    assert all(call["project_to_road"] is False for call in carla_map.calls)
    assert all(call["lane_type"] is driving_lane_type for call in carla_map.calls)
    queried_center = carla_map.calls[0]["location"]
    expected_x = 4.0 + 0.2 * math.cos(math.radians(7.0)) - 0.1 * math.sin(
        math.radians(7.0)
    )
    expected_y = 0.2 * math.sin(math.radians(7.0)) + 0.1 * math.cos(
        math.radians(7.0)
    )
    assert queried_center.x == pytest.approx(expected_x)
    assert queried_center.y == pytest.approx(expected_y)


def test_spawn_preflight_rejects_footprint_that_crosses_driving_lane_edge(
    driving_lane_type,
):
    def resolve(location):
        if abs(location.y) >= 0.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, _ = _adapter(
        RecordingMap(resolve),
        ego_bounding_box=FakeBoundingBox(half_length=0.4, half_width=0.6),
    )

    road = adapter.assess_ego_transform(FakeTransform())

    assert road.status is AssessmentStatus.UNSAFE
    assert road.quality == "carla_ground_truth_spawn_preflight"
    assert "footprint_corner_off_driving_lane" in road.reason_codes


def test_spawn_preflight_map_error_fails_closed_to_unknown(driving_lane_type):
    def fail(_location):
        raise RuntimeError("map unavailable")

    adapter, _, _ = _adapter(RecordingMap(fail))

    road = adapter.assess_ego_transform(FakeTransform())

    assert road.status is AssessmentStatus.UNKNOWN
    assert road.quality == "carla_ground_truth_spawn_preflight"
    assert "carla_map_query_error:RuntimeError" in road.reason_codes


def test_path_is_densified_to_half_metre_and_catches_off_road_gap(driving_lane_type):
    def resolve(location):
        if 0.9 <= location.x <= 1.1:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    carla_map = RecordingMap(resolve)
    adapter, _, _ = _adapter(carla_map)
    road = adapter._assess_path_road(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        _ego_kinematics(half_length=0.05, half_width=0.05),
    )
    queried_centers = [call["location"] for call in carla_map.calls[::5]]

    assert road.status is AssessmentStatus.UNSAFE
    assert any(location.x == pytest.approx(1.0) for location in queried_centers)
    assert max(
        math.hypot(end.x - start.x, end.y - start.y)
        for start, end in zip(queried_centers, queried_centers[1:])
    ) <= 0.5


def test_path_heading_ignores_submillimetre_reverse_start_jitter(
    driving_lane_type,
):
    adapter, _, _ = _adapter(RecordingMap())
    road = adapter._assess_path_road(
        [
            (0.0, 0.0, 0.0),
            (-0.00003, 0.0, 0.0),
            (-0.00010, 0.0, 0.0),
            (0.0001, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        _ego_kinematics(half_length=0.05, half_width=0.05),
    )

    assert road.status is AssessmentStatus.SAFE
    assert "path_heading_opposes_lane" not in road.reason_codes


def test_junction_transition_accepts_any_exact_driving_lane(driving_lane_type):
    def resolve(location):
        if location.x > 0.0:
            return FakeWaypoint(
                location,
                road_id=22,
                lane_id=7,
                lane_width=0.1,
                is_junction=True,
            )
        return FakeWaypoint(location, road_id=11, lane_id=-1, is_junction=False)

    adapter, _, _ = _adapter(RecordingMap(resolve))
    samples = adapter._query_footprint(
        center_xyz=np.array([0.0, 0.0, 0.0]),
        yaw_rad=0.0,
        half_length_m=0.4,
        half_width_m=0.3,
        sample_index_start=0,
    )
    road = carla_safety_adapter.assess_road_containment(
        samples,
        quality="carla_ground_truth_drivable_only_at_junction",
    )

    assert any(sample.is_junction for sample in samples)
    assert road.status is AssessmentStatus.SAFE

    junction_center_samples = adapter._query_footprint(
        center_xyz=np.array([0.2, 0.0, 0.0]),
        yaw_rad=0.0,
        half_length_m=0.4,
        half_width_m=0.3,
        sample_index_start=0,
    )
    junction_center_road = carla_safety_adapter.assess_road_containment(
        junction_center_samples,
        quality="carla_ground_truth_drivable_only_at_junction",
    )
    assert junction_center_samples[0].is_junction is True
    assert junction_center_samples[0].margin_m is None
    assert junction_center_road.status is AssessmentStatus.SAFE


def test_map_query_exception_fails_closed_to_unknown(driving_lane_type):
    def fail(_location):
        raise RuntimeError("map unavailable")

    adapter, _, _ = _adapter(RecordingMap(fail))

    road = adapter._assess_path_road(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        _ego_kinematics(),
    )

    assert road.status is AssessmentStatus.UNKNOWN
    assert "carla_map_query_error:RuntimeError" in road.reason_codes


def test_actor_conversion_uses_same_snapshot_for_all_vehicles_and_walkers(
    driving_lane_type,
):
    vehicle = FakeActor(
        2,
        "vehicle.test",
        FakeBoundingBox(offset_x=1.0, yaw=10.0),
    )
    walker = FakeActor(
        3,
        "walker.pedestrian.test",
        FakeBoundingBox(half_length=0.2, half_width=0.2),
    )
    adapter, world, ego = _adapter(RecordingMap(), actors=(vehicle, walker))
    context = _tick_context(
        ego,
        actor_snapshots={
            2: FakeActorSnapshot(
                FakeTransform(x=10.0, y=5.0, yaw=90.0),
                FakeVector(x=2.0, y=3.0),
            ),
            3: FakeActorSnapshot(
                FakeTransform(x=4.0, y=-2.0, yaw=-20.0),
                FakeVector(x=-1.0, y=0.5),
            ),
        },
    )

    actors = adapter._actors_from_context(context)
    by_id = {actor.actor_id: actor for actor in actors}

    assert world.actor_list.filter_patterns == ["vehicle.*", "walker.pedestrian.*"]
    assert world.actor_query_count == 1
    assert set(by_id) == {2, 3}
    assert all(actor.frame_id == 50 for actor in actors)
    assert by_id[2].center_xy == pytest.approx((10.0, 6.0))
    assert by_id[2].yaw_rad == pytest.approx(math.radians(100.0))
    assert by_id[2].velocity_xy == pytest.approx((2.0, 3.0))
    assert by_id[3].center_xy == pytest.approx((4.0, -2.0))
    assert by_id[3].velocity_xy == pytest.approx((-1.0, 0.5))


def test_missing_actor_snapshot_makes_obstacle_assessment_unknown(driving_lane_type):
    vehicle = FakeActor(2, "vehicle.test")
    adapter, _, ego = _adapter(RecordingMap(), actors=(vehicle,))
    context = _tick_context(ego, actor_snapshots={})

    assessment = adapter.assess(tick_context=context, plan=_plan())

    assert assessment.road.status is AssessmentStatus.SAFE
    assert assessment.obstacles.status is AssessmentStatus.UNKNOWN
    assert "obstacle_data_unavailable" in assessment.obstacles.reason_codes


def test_far_future_road_violation_is_safe_prefix_not_immediate_override(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 4.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    context = _tick_context(ego, simulation_time_s=5.0)

    assessment = adapter.assess(tick_context=context, plan=_long_plan())
    envelope = assessment.road_envelope

    assert assessment.road.status is AssessmentStatus.SAFE
    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.time_to_first_bad_s > 4.0
    assert envelope.distance_to_first_bad_m > 3.0
    assert envelope.target_speed_cap_mps > 0.0
    assert envelope.emergency_required is False
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.ACCEPT_SAFE_PREFIX
    )


def test_near_term_unsafe_candidate_retains_executable_active_plan(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 1.7:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    context = _tick_context(ego, simulation_time_s=5.0)
    candidate = adapter.assess_plan_road(
        tick_context=context,
        plan=_long_plan(plan_id="candidate"),
    )
    safe_adapter, _, safe_ego = _adapter(RecordingMap())
    active = safe_adapter.assess_plan_road(
        tick_context=_tick_context(safe_ego, simulation_time_s=5.0),
        plan=_long_plan(plan_id="active"),
    )

    assert candidate.near_term_path_road.status is AssessmentStatus.UNSAFE
    assert (
        decide_plan_admission(candidate, active)
        is PlanAdmissionStatus.REJECT_RETAIN_ACTIVE
    )
    assert (
        decide_plan_admission(candidate)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_stopping_envelope_escalates_only_when_current_speed_exceeds_cap(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 4.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    plan = _long_plan()
    stopped = adapter.assess(
        tick_context=_tick_context(ego, simulation_time_s=5.0, speed_x=0.0),
        plan=plan,
    )
    fast = adapter.assess(
        tick_context=_tick_context(ego, simulation_time_s=5.0, speed_x=5.0),
        plan=plan,
    )

    assert stopped.road.status is AssessmentStatus.SAFE
    assert stopped.road_envelope.emergency_required is False
    assert fast.road_envelope.emergency_required is True
    assert fast.road.status is AssessmentStatus.UNSAFE
    assert "road_stopping_envelope_exhausted" in fast.road.reason_codes


def test_raw_physical_cap_is_not_clipped_by_controller_speed_limit():
    policy = SafetyPolicy()
    distance_to_bad = 30.0

    raw_cap = raw_physical_stopping_speed_cap_mps(distance_to_bad, policy)
    controller_cap = road_stopping_speed_cap_mps(distance_to_bad, policy)

    assert raw_cap > carla_safety_adapter.cfg.TRAJECTORY_MAX_SPEED_MPS
    assert controller_cap == pytest.approx(
        carla_safety_adapter.cfg.TRAJECTORY_MAX_SPEED_MPS
    )


def test_controller_cap_saturation_does_not_create_a_false_emergency(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 30.0:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            speed_x=10.0,
        ),
        plan=_long_plan(step_m=0.5),
    )
    reserve = envelope.stopping_reserve_profile

    assert reserve.raw_physical_stopping_cap_mps > 10.0
    assert envelope.target_speed_cap_mps == pytest.approx(
        carla_safety_adapter.cfg.TRAJECTORY_MAX_SPEED_MPS
    )
    assert envelope.emergency_required is False


def test_guarded_stopping_reserve_can_be_fragile_without_emergency(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 4.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            speed_x=0.0,
        ),
        plan=_long_plan(step_m=0.5),
    )
    reserve = envelope.stopping_reserve_profile

    assert reserve.guard_speed_mps == pytest.approx(5.5)
    assert reserve.required_stopping_distance_m > envelope.distance_to_first_bad_m
    assert reserve.stopping_reserve_m < 0.0
    assert reserve.status is StoppingReserveStatus.FRAGILE
    assert envelope.target_speed_cap_mps == pytest.approx(
        reserve.raw_physical_stopping_cap_mps
        - carla_safety_adapter.cfg.SAFETY_GUARDED_ACCELERATION_MPS2
        * carla_safety_adapter.cfg.CONTROL_DT
    )
    assert envelope.emergency_required is False
    assert envelope.stopping_reserve_compute_ms >= 0.0


def test_genuine_insufficient_physical_stopping_distance_remains_fail_closed(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 4.5:
            return None
        return FakeWaypoint(location, lane_width=4.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    assessment = adapter.assess(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            speed_x=5.0,
        ),
        plan=_long_plan(step_m=0.1),
    )

    assert assessment.road_envelope.emergency_required is True
    assert assessment.road.status is AssessmentStatus.UNSAFE


def test_full_safe_path_has_unbounded_stopping_reserve_and_additive_json(
    driving_lane_type,
):
    adapter, _, ego = _adapter(RecordingMap())
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(ego, simulation_time_s=5.0),
        plan=_long_plan(),
    )
    payload = envelope.to_json_dict()

    assert envelope.stopping_reserve_profile.status is StoppingReserveStatus.UNBOUNDED
    assert payload["stopping_reserve_status"] == "UNBOUNDED"
    assert payload["raw_physical_stopping_cap_mps"] is None
    assert payload["stopping_reserve_profile"]["status"] == "UNBOUNDED"
    assert payload["recovery_mode"] == "NONE"
    assert payload["near_term_end_clearance_road"]["status"] == "SAFE"
    assert payload["recovery_path_length_m"] > 0.0
    assert payload["recovery_displacement_m"] > 0.0


def test_current_ego_off_lane_remains_an_immediate_fail_closed_trigger(
    driving_lane_type,
):
    adapter, _, ego = _adapter(RecordingMap())

    assessment = adapter.assess(
        tick_context=_tick_context(ego, simulation_time_s=5.0, ego_y=3.0),
        plan=_long_plan(),
    )

    assert assessment.road_envelope.current_ego_road.status is AssessmentStatus.UNSAFE
    assert assessment.road.status is AssessmentStatus.UNSAFE


def test_current_ego_surface_is_separate_from_buffered_lane_clearance(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=_long_plan(),
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.current_ego_road.min_margin_m == pytest.approx(0.2)
    assert (
        envelope.current_ego_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.current_ego_clearance_road.min_margin_m == pytest.approx(-0.05)


def test_junction_transition_can_admit_low_speed_physical_recovery_prefix(
    driving_lane_type,
):
    def resolve(location):
        return FakeWaypoint(
            location,
            lane_width=2.0,
            is_junction=location.x > 0.0,
        )

    adapter, _, ego = _adapter(RecordingMap(resolve))
    plan = _long_plan()
    plan.world_points[:, 1] = 0.5
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert (
        envelope.current_ego_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.near_term_path_surface.status is AssessmentStatus.SAFE
    assert envelope.junction_context is True
    assert envelope.recovery_required is True
    assert envelope.recovery_mode is RoadRecoveryMode.JUNCTION_CLEARANCE
    assert envelope.target_speed_cap_mps == pytest.approx(
        carla_safety_adapter.cfg.SAFETY_JUNCTION_RECOVERY_SPEED_CAP_MPS
    )
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX
    )


def test_nonjunction_clearance_violation_does_not_receive_recovery_grace(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan()
    plan.world_points[:, 1] = 0.5
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert (
        envelope.current_ego_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.junction_context is False
    assert envelope.recovery_required is False
    assert envelope.recovery_mode is RoadRecoveryMode.NONE
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_nonjunction_boundary_recovery_requires_and_restores_clearance(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan(plan_id="boundary-recovery")
    plan.world_points[:, 1] = np.linspace(0.5, 0.0, len(plan.world_points))

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert (
        envelope.current_ego_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.near_term_path_surface.status is AssessmentStatus.SAFE
    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.SAFE
    )
    assert envelope.recovery_displacement_m >= (
        carla_safety_adapter.cfg.SAFETY_BOUNDARY_RECOVERY_MIN_DISPLACEMENT_M
    )
    assert envelope.junction_context is False
    assert envelope.recovery_required is True
    assert envelope.recovery_mode is RoadRecoveryMode.BOUNDARY_CLEARANCE
    assert envelope.target_speed_cap_mps == pytest.approx(
        carla_safety_adapter.cfg.SAFETY_JUNCTION_RECOVERY_SPEED_CAP_MPS
    )
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX
    )


def test_nonjunction_boundary_recovery_rejects_outward_motion(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan(plan_id="boundary-outward")
    plan.world_points[:, 1] = np.linspace(0.5, 0.6, len(plan.world_points))

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.near_term_path_surface.status is AssessmentStatus.SAFE
    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.recovery_required is False
    assert envelope.recovery_mode is RoadRecoveryMode.NONE
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_nonjunction_boundary_recovery_rejects_physical_excursion(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan(plan_id="boundary-physical-excursion")
    plan.world_points[:, 1] = 0.0
    plan.world_points[:8, 1] = np.linspace(0.5, 0.8, 8)
    plan.world_points[8:16, 1] = np.linspace(0.8, 0.0, 8)

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.near_term_path_surface.status is AssessmentStatus.UNSAFE
    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.SAFE
    )
    assert envelope.recovery_required is False
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_nonjunction_boundary_recovery_rejects_stationary_stop(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = np.linspace(0.001, 0.1, len(points))
    points[:, 1] = 0.5
    plan = types.SimpleNamespace(
        plan_id="boundary-stationary-stop",
        world_points=points,
        waypoint_times_s=5.0 + np.arange(1, 65, dtype=np.float64) * 0.1,
        terminal_stop_index=0,
    )

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.UNSAFE
    )
    assert envelope.recovery_displacement_m < (
        carla_safety_adapter.cfg.SAFETY_BOUNDARY_RECOVERY_MIN_DISPLACEMENT_M
    )
    assert envelope.recovery_required is False
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_nonjunction_boundary_recovery_does_not_authorize_explicit_stop(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan(plan_id="boundary-recenter-stop")
    plan.world_points[:, 1] = np.linspace(0.5, 0.0, len(plan.world_points))
    plan.terminal_stop_index = 15

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.near_term_path_surface.status is AssessmentStatus.SAFE
    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.SAFE
    )
    assert envelope.recovery_required is False
    assert envelope.recovery_mode is RoadRecoveryMode.NONE
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_nonjunction_boundary_recovery_end_query_failure_fails_closed(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 1.5:
            raise RuntimeError("endpoint map unavailable")
        return FakeWaypoint(location, lane_width=2.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    plan = _long_plan(plan_id="boundary-unknown-end")
    plan.world_points[:, 1] = np.linspace(0.5, 0.0, len(plan.world_points))

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.UNKNOWN
    )
    assert envelope.recovery_required is False
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_boundary_recovery_returns_to_buffered_semantics_after_horizon(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=2.0))
    )
    plan = _long_plan(plan_id="boundary-recovery-then-regress")
    plan.world_points[:, 1] = 0.0
    plan.world_points[:16, 1] = np.linspace(0.5, 0.0, 16)
    plan.world_points[16:, 1] = np.linspace(0.0, 0.6, 48)

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_y=0.5,
        ),
        plan=plan,
    )

    assert envelope.recovery_mode is RoadRecoveryMode.BOUNDARY_CLEARANCE
    assert envelope.near_term_path_surface.status is AssessmentStatus.SAFE
    assert (
        envelope.near_term_end_clearance_road.status
        is AssessmentStatus.SAFE
    )
    assert envelope.full_path_surface.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.last_safe_waypoint_index is not None
    assert envelope.last_safe_waypoint_index < len(plan.world_points) - 1
    assert envelope.target_speed_cap_mps is not None


def test_job_22863138_boundary_recovery_fixture_captures_source_gate():
    replay = json.loads(BOUNDARY_RECOVERY_FIXTURE.read_text())
    containment = replay["source_containment"]
    trajectory = replay["trajectory_facts"]
    expected = replay["expected_policy"]

    assert containment["current_physical_status"] == "SAFE"
    assert containment["current_clearance_status"] == "UNSAFE"
    assert containment["near_term_physical_status"] == "SAFE"
    assert containment["near_term_end_clearance_status"] == "SAFE"
    assert containment["near_term_end_clearance_margin_m"] > 0.0
    assert trajectory["terminal_stop_index"] is None
    assert trajectory["near_term_recovery_displacement_m"] >= (
        carla_safety_adapter.cfg.SAFETY_BOUNDARY_RECOVERY_MIN_DISPLACEMENT_M
    )
    assert expected["recovery_mode"] == RoadRecoveryMode.BOUNDARY_CLEARANCE.value
    assert (
        expected["admission_status"]
        == PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX.value
    )
    assert expected["maximum_authorized_waypoint_index"] == 16
    assert expected["maximum_speed_cap_mps"] == pytest.approx(
        carla_safety_adapter.cfg.SAFETY_JUNCTION_RECOVERY_SPEED_CAP_MPS
    )


def test_timed_profile_cache_reuses_queries_and_tracks_geometric_progress(
    driving_lane_type,
):
    carla_map = RecordingMap()
    adapter, _, ego = _adapter(carla_map)
    plan = _long_plan()

    first = adapter.assess_plan_road(
        tick_context=_tick_context(ego, simulation_time_s=5.0),
        plan=plan,
    )
    calls_after_first = len(carla_map.calls)
    time_only = adapter.assess_plan_road(
        tick_context=_tick_context(ego, simulation_time_s=6.0),
        plan=plan,
    )
    calls_after_time_only = len(carla_map.calls)
    advanced = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=6.0,
            ego_x=1.0,
        ),
        plan=plan,
    )

    assert calls_after_time_only == calls_after_first + 5
    assert time_only.full_path_road.sample_count == first.full_path_road.sample_count
    assert len(carla_map.calls) == calls_after_time_only + 5
    assert advanced.full_path_road.sample_count < first.full_path_road.sample_count
    assert advanced.ego_path_progress_m > first.ego_path_progress_m


def test_near_term_window_starts_at_geometric_cursor_when_ahead_of_schedule(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=20.0))
    )
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_x=4.0,
        ),
        plan=_long_plan(step_m=0.1),
    )

    assert envelope.execution_cursor_index is not None
    assert envelope.execution_cursor_index >= 39
    assert envelope.near_term_path_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.sample_count > 0
    assert envelope.full_path_road.status is AssessmentStatus.SAFE


def test_near_term_window_does_not_expand_when_behind_schedule(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 3.0:
            return None
        return FakeWaypoint(location, lane_width=20.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=9.0,
            ego_x=0.0,
        ),
        plan=_long_plan(step_m=0.1),
    )

    assert envelope.execution_cursor_index == 0
    assert envelope.near_term_path_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.time_to_first_bad_s == 0.0
    assert envelope.distance_to_first_bad_m == pytest.approx(2.5)


def test_quarter_metre_sampling_catches_recorded_narrow_junction_gap(
    driving_lane_type,
):
    replay = json.loads(
        NARROW_JUNCTION_GAP_FIXTURE.read_text(encoding="utf-8")
    )
    segment_length = float(replay["first_segment_length_m"])
    unsafe_start, unsafe_end = replay["unsafe_progress_interval_m"]
    runtime_spacing = float(
        carla_safety_adapter.cfg.SAFETY_PATH_SAMPLE_SPACING_M
    )
    assert runtime_spacing == pytest.approx(replay["required_spacing_m"])

    points = np.zeros((64, 3), dtype=np.float64)
    points[1:, 0] = segment_length + 0.5 * np.arange(63)
    plan = types.SimpleNamespace(
        plan_id="narrow-junction-gap",
        world_points=points,
        waypoint_times_s=5.0
        + np.arange(1, 65, dtype=np.float64) * 0.1,
    )

    def resolve(location):
        if unsafe_start <= location.x <= unsafe_end:
            return None
        return FakeWaypoint(
            location,
            lane_width=20.0,
            is_junction=True,
        )

    fine_adapter, _, fine_ego = _adapter(
        RecordingMap(resolve),
        ego_bounding_box=FakeBoundingBox(
            half_length=0.001,
            half_width=0.001,
        ),
        policy=SafetyPolicy(
            path_sample_spacing_m=runtime_spacing
        ),
    )
    coarse_adapter, _, coarse_ego = _adapter(
        RecordingMap(resolve),
        ego_bounding_box=FakeBoundingBox(
            half_length=0.001,
            half_width=0.001,
        ),
        policy=SafetyPolicy(
            path_sample_spacing_m=replay["legacy_spacing_m"]
        ),
    )

    fine = fine_adapter.assess_plan_road(
        tick_context=_tick_context(
            fine_ego,
            simulation_time_s=5.0,
        ),
        plan=plan,
    )
    coarse = coarse_adapter.assess_plan_road(
        tick_context=_tick_context(
            coarse_ego,
            simulation_time_s=5.0,
        ),
        plan=plan,
    )

    assert coarse.near_term_path_road.status is AssessmentStatus.SAFE
    assert fine.near_term_path_road.status is AssessmentStatus.UNSAFE
    assert fine.full_path_road.status is AssessmentStatus.UNSAFE
    assert fine.first_bad_path_progress_m == pytest.approx(
        replay["expected_fine_sample_progress_m"]
    )
    assert (
        decide_plan_admission(fine)
        is PlanAdmissionStatus.REJECT_FALLBACK_STOP
    )


def test_stopping_distance_uses_ego_geometric_progress_when_ahead_of_time(
    driving_lane_type,
):
    def resolve(location):
        if location.x >= 12.0:
            return None
        return FakeWaypoint(location, lane_width=20.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.5,
            ego_x=10.0,
        ),
        plan=_long_plan(step_m=0.5),
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.ego_path_progress_m == pytest.approx(9.5)
    assert envelope.ego_path_cross_track_m == pytest.approx(0.0)
    assert envelope.first_bad_path_progress_m == pytest.approx(11.5)
    assert envelope.distance_to_first_bad_m == pytest.approx(2.0)
    assert envelope.stopping_reserve_profile.raw_physical_stopping_cap_mps == 0.0
    assert envelope.target_speed_cap_mps == 0.0


def test_elapsed_bad_pose_ahead_of_ego_remains_in_execution_envelope(
    driving_lane_type,
):
    def resolve(location):
        if 1.9 <= location.x <= 2.6:
            return None
        return FakeWaypoint(location, lane_width=20.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=6.0,
            ego_x=0.5,
        ),
        plan=_long_plan(step_m=0.5),
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.full_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.time_to_first_bad_s == 0.0
    assert envelope.distance_to_first_bad_m == pytest.approx(1.0)
    assert envelope.last_safe_waypoint_index == 1


def test_bad_pose_behind_geometric_progress_is_not_reexecuted(
    driving_lane_type,
):
    def resolve(location):
        if 1.9 <= location.x <= 2.6:
            return None
        return FakeWaypoint(location, lane_width=20.0)

    adapter, _, ego = _adapter(RecordingMap(resolve))
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.1,
            ego_x=4.0,
        ),
        plan=_long_plan(step_m=0.5),
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.SAFE
    assert envelope.distance_to_first_bad_m is None
    assert (
        envelope.stopping_reserve_profile.status
        is StoppingReserveStatus.UNBOUNDED
    )


def test_ambiguous_self_intersection_projection_fails_closed(
    driving_lane_type,
):
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=20.0))
    )
    plan = types.SimpleNamespace(
        plan_id="self-intersection",
        world_points=np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [2.0, -2.0, 0.0],
                [3.0, -2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        waypoint_times_s=5.0 + np.arange(1, 6, dtype=np.float64) * 0.1,
    )

    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(ego, simulation_time_s=5.0),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.UNKNOWN
    assert envelope.target_speed_cap_mps == 0.0
    assert envelope.emergency_required is True


def test_near_overlap_with_separated_progress_fails_closed():
    vertices = np.array(
        [
            [0.1, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [8.0, 5.0],
            [4.0, 8.0],
            [0.0, 5.0],
            [0.0, 0.01],
            [4.0, 0.01],
            [10.0, 0.01],
        ],
        dtype=np.float64,
    )
    vertex_distance = np.concatenate(
        [
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1)),
        ]
    )
    sampled_distance = np.linspace(0.0, vertex_distance[-1], 64)
    points = np.column_stack(
        [
            np.interp(sampled_distance, vertex_distance, vertices[:, 0]),
            np.interp(sampled_distance, vertex_distance, vertices[:, 1]),
        ]
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )

    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 1.0
    with pytest.raises(ValueError, match="ambiguous path progress"):
        carla_safety_adapter._project_path_progress(
            points=points,
            cumulative_distance_m=cumulative,
            ego_xy=(2.0, 0.009),
            ego_yaw_rad=0.0,
        )


def test_heading_filter_cannot_select_segment_outside_tracking_corridor():
    points = np.array(
        [
            [1.0, 2.49],
            [-1.0, 2.49],
            [-1.0, 10.0],
            [-2.0, 10.0],
            [-2.0, 2.7],
            [-1.0, 2.7],
            [1.0, 2.7],
        ],
        dtype=np.float64,
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )

    with pytest.raises(ValueError, match="path progress projection"):
        carla_safety_adapter._project_path_progress(
            points=points,
            cumulative_distance_m=cumulative,
            ego_xy=(0.0, 0.0),
            ego_yaw_rad=0.0,
        )


def test_stationary_stop_jitter_remains_road_admissible(driving_lane_type):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:, 0] = (np.arange(64) % 2) * 0.02
    terminal_stop_index = detect_terminal_stop_index(points)
    assert terminal_stop_index == 0

    plan = types.SimpleNamespace(
        plan_id="stationary-stop-jitter",
        world_points=points,
        waypoint_times_s=5.0
        + np.arange(1, 65, dtype=np.float64) * 0.1,
        terminal_stop_index=terminal_stop_index,
    )
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=20.0))
    )
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(ego, simulation_time_s=5.0),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.SAFE
    assert envelope.execution_cursor_index == 0
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.ACCEPT_FULLY_SAFE
    )


def test_moving_plan_terminal_jitter_remains_road_admissible(
    driving_lane_type,
):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:10, 0] = np.linspace(0.5, 5.0, 10)
    points[10:, 0] = 5.0 + (np.arange(54) % 2) * 0.02
    terminal_stop_index = detect_terminal_stop_index(points)
    assert terminal_stop_index == 9

    plan = types.SimpleNamespace(
        plan_id="moving-terminal-stop-jitter",
        world_points=points,
        waypoint_times_s=5.0
        + np.arange(1, 65, dtype=np.float64) * 0.1,
        terminal_stop_index=terminal_stop_index,
    )
    adapter, _, ego = _adapter(
        RecordingMap(lambda location: FakeWaypoint(location, lane_width=20.0))
    )
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_x=5.02,
        ),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.near_term_path_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.SAFE
    assert envelope.execution_cursor_index is not None
    assert envelope.execution_cursor_index <= terminal_stop_index
    assert envelope.ego_path_cross_track_m == pytest.approx(0.02)
    assert (
        decide_plan_admission(envelope)
        is PlanAdmissionStatus.ACCEPT_FULLY_SAFE
    )


def test_terminal_jitter_does_not_add_stopping_headroom(driving_lane_type):
    points = np.zeros((64, 3), dtype=np.float64)
    points[:10, 0] = np.linspace(0.5, 5.0, 10)
    points[10:, 0] = 5.0 + (np.arange(54) % 2) * 0.02
    terminal_stop_index = detect_terminal_stop_index(points)
    assert terminal_stop_index == 9

    def resolve(location):
        if location.x >= 5.015:
            return None
        return FakeWaypoint(location, lane_width=20.0)

    plan = types.SimpleNamespace(
        plan_id="terminal-jitter-bad-point",
        world_points=points,
        waypoint_times_s=5.0
        + np.arange(1, 65, dtype=np.float64) * 0.1,
        terminal_stop_index=terminal_stop_index,
    )
    adapter, _, ego = _adapter(
        RecordingMap(resolve),
        ego_bounding_box=FakeBoundingBox(
            half_length=0.001,
            half_width=0.001,
        ),
    )
    envelope = adapter.assess_plan_road(
        tick_context=_tick_context(
            ego,
            simulation_time_s=5.0,
            ego_x=4.8,
        ),
        plan=plan,
    )

    assert envelope.current_ego_road.status is AssessmentStatus.SAFE
    assert envelope.full_path_road.status is AssessmentStatus.UNSAFE
    assert envelope.first_bad_path_progress_m == pytest.approx(4.52)
    assert (
        envelope.effective_stopping_boundary_progress_m
        == pytest.approx(4.5)
    )
    assert envelope.ego_path_progress_m == pytest.approx(4.3)
    assert envelope.distance_to_first_bad_m == pytest.approx(0.2)


def test_path_progress_heading_ignores_subcentimetre_reverse_jitter():
    points = np.array(
        [
            [0.0, 0.0],
            [-0.0001, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )

    projection = carla_safety_adapter._project_path_progress(
        points=points,
        cumulative_distance_m=cumulative,
        ego_xy=(0.0, 0.0),
        ego_yaw_rad=0.0,
    )

    assert projection.progress_m == pytest.approx(0.0)
    assert projection.cross_track_m == pytest.approx(0.0)
    assert projection.heading_error_deg == pytest.approx(0.0)


def test_path_progress_heading_uses_future_tangent_at_shared_turn_vertex():
    points = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )

    projection = carla_safety_adapter._project_path_progress(
        points=points,
        cumulative_distance_m=cumulative,
        ego_xy=(0.0, 0.0),
        ego_yaw_rad=math.pi / 2.0,
    )

    assert projection.progress_m == pytest.approx(1.0)
    assert projection.first_path_index == 1
    assert projection.cross_track_m == pytest.approx(0.0)
    assert projection.heading_error_deg == pytest.approx(0.0)


def test_job_22862336_geometric_stopping_distance_replay():
    replay = json.loads(
        GEOMETRIC_STOPPING_FIXTURE.read_text(encoding="utf-8")
    )
    points = np.asarray(
        replay["world_points_xy_through_first_bad"],
        dtype=np.float64,
    )
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    first_bad_progress = float(replay["first_bad_path_progress_m"])
    bbox_offset = float(replay["ego_bbox_center_offset_x_m"])
    policy = SafetyPolicy()

    assert cumulative[-1] == pytest.approx(first_bad_progress, abs=1e-5)
    for tick in replay["ticks"]:
        yaw_rad = math.radians(tick["ego_yaw_deg"])
        ego_center_xy = (
            tick["ego_xy"][0] + bbox_offset * math.cos(yaw_rad),
            tick["ego_xy"][1] + bbox_offset * math.sin(yaw_rad),
        )
        projection = carla_safety_adapter._project_path_progress(
            points=points,
            cumulative_distance_m=cumulative,
            ego_xy=ego_center_xy,
            ego_yaw_rad=yaw_rad,
        )
        distance = max(0.0, first_bad_progress - projection.progress_m)
        raw_cap = raw_physical_stopping_speed_cap_mps(distance, policy)
        target_cap = min(
            max(
                0.0,
                raw_cap
                - carla_safety_adapter.cfg.SAFETY_GUARDED_ACCELERATION_MPS2
                * carla_safety_adapter.cfg.CONTROL_DT,
            ),
            carla_safety_adapter.cfg.TRAJECTORY_MAX_SPEED_MPS,
        )
        emergency = (
            tick["speed_mps"]
            > raw_cap
            + carla_safety_adapter.cfg.SAFETY_SPEED_CAP_EPSILON_MPS
        )

        assert distance == pytest.approx(
            tick["expected_distance_to_bad_m"],
            abs=0.01,
        )
        assert distance < tick["legacy_distance_to_bad_m"] - 3.0
        assert raw_cap == pytest.approx(
            tick["expected_raw_cap_mps"],
            abs=0.01,
        )
        assert target_cap == pytest.approx(
            tick["expected_target_cap_mps"],
            abs=0.01,
        )
        assert emergency is tick["expected_emergency"]

import math
import types

import numpy as np
import pytest

from module import carla_safety_adapter
from module.carla_safety_adapter import CarlaGroundTruthSafetyAdapter
from module.safety_shield import AssessmentStatus, EgoKinematics


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


def _adapter(carla_map, *, actors=(), ego_bounding_box=None):
    ego = FakeActor(
        1,
        "vehicle.ego",
        bounding_box=ego_bounding_box or FakeBoundingBox(),
    )
    world = FakeWorld(carla_map, (ego, *actors))
    return CarlaGroundTruthSafetyAdapter(world, ego), world, ego


def _tick_context(ego, *, frame=50, simulation_time_s=5.0, actor_snapshots=None):
    ego_transform = FakeTransform()
    ego_velocity = FakeVector()
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

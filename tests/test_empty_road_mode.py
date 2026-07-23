import importlib
import sys
import types

import pytest


sys.modules.setdefault(
    "carla",
    types.SimpleNamespace(
        command=types.SimpleNamespace(DestroyActor=lambda actor_id: ("destroy", actor_id)),
        VehicleControl=lambda: types.SimpleNamespace(
            steer=0.0,
            throttle=0.0,
            brake=0.0,
        ),
        Vector3D=lambda: types.SimpleNamespace(),
    ),
)

closed_loop = importlib.import_module("carlamayo_closed_loop")
carla_interface_module = importlib.import_module("module.carla_interface")
CARLAInterface = carla_interface_module.CARLAInterface


def test_runtime_seed_helper_covers_host_and_cuda_rngs(monkeypatch):
    python_seeds = []
    numpy_seeds = []
    torch_seeds = []
    cuda_seeds = []
    monkeypatch.setattr(closed_loop.random, "seed", python_seeds.append)
    monkeypatch.setattr(closed_loop.np.random, "seed", numpy_seeds.append)
    monkeypatch.setattr(closed_loop.torch, "manual_seed", torch_seeds.append)
    monkeypatch.setattr(closed_loop.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(closed_loop.torch.cuda, "manual_seed_all", cuda_seeds.append)

    closed_loop.seed_runtime_randomness(17)

    assert python_seeds == [17]
    assert numpy_seeds == [17]
    assert torch_seeds == [17]
    assert cuda_seeds == [17]


def test_normal_mode_retains_random_scenario_defaults():
    args = closed_loop.parse_args([])

    assert args.empty_road is False
    assert args.scenario_seed is None
    assert args.ego_spawn_index is None


def test_empty_road_resolves_reproducible_scenario_defaults():
    args = closed_loop.parse_args(["--empty-road"])

    assert args.empty_road is True
    assert args.scenario_seed == closed_loop.cfg.EMPTY_ROAD_SCENARIO_SEED == 0
    assert args.ego_spawn_index == closed_loop.cfg.EMPTY_ROAD_EGO_SPAWN_INDEX == 0


def test_explicit_scenario_seed_and_spawn_index_override_empty_road_defaults():
    args = closed_loop.parse_args(
        [
            "--empty-road",
            "--scenario-seed",
            "17",
            "--ego-spawn-index",
            "23",
        ]
    )

    assert args.scenario_seed == 17
    assert args.ego_spawn_index == 23


def test_scenario_seed_and_spawn_index_can_pin_a_normal_run():
    args = closed_loop.parse_args(
        [
            "--scenario-seed",
            "5",
            "--ego-spawn-index",
            "9",
        ]
    )

    assert args.empty_road is False
    assert args.scenario_seed == 5
    assert args.ego_spawn_index == 9


@pytest.mark.parametrize("option", ["--scenario-seed", "--ego-spawn-index"])
def test_scenario_seed_and_spawn_index_must_be_nonnegative(option):
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args([option, "-1"])

    assert exc_info.value.code == 2


def test_scenario_seed_rejects_values_outside_carla_seed_range():
    with pytest.raises(SystemExit) as exc_info:
        closed_loop.parse_args(
            [
                "--scenario-seed",
                str(closed_loop.cfg.MAX_SCENARIO_SEED + 1),
            ]
        )

    assert exc_info.value.code == 2


class _SpawnPoint:
    def __init__(self, label):
        self.label = label
        self.location = types.SimpleNamespace(label=label)


class _Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)


class _Transform:
    def __init__(self, *, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = _Location(x, y, z)
        self.rotation = _Rotation(yaw)


class _SpawnMap:
    def __init__(self, spawn_points):
        self._spawn_points = list(spawn_points)

    def get_spawn_points(self):
        return list(self._spawn_points)


def _interface_with_spawn_points(*spawn_points):
    carla_if = CARLAInterface()
    actor_list = types.SimpleNamespace(filter=lambda _pattern: [])
    carla_if.world = types.SimpleNamespace(
        get_map=lambda: _SpawnMap(spawn_points),
        get_actors=lambda: actor_list,
    )
    return carla_if


def test_indexed_ego_spawn_selection_returns_the_exact_map_spawn(monkeypatch):
    spawn_points = [_SpawnPoint("zero"), _SpawnPoint("one"), _SpawnPoint("two")]
    carla_if = _interface_with_spawn_points(*spawn_points)
    monkeypatch.setattr(
        carla_interface_module.random,
        "shuffle",
        lambda _points: pytest.fail("fixed spawn selection must not shuffle"),
    )

    selected = carla_if._select_ego_spawn_point(spawn_index=1)

    assert selected is spawn_points[1]


@pytest.mark.parametrize("spawn_index", [-1, 3])
def test_indexed_ego_spawn_selection_rejects_invalid_indices(spawn_index):
    carla_if = _interface_with_spawn_points(
        _SpawnPoint("zero"),
        _SpawnPoint("one"),
        _SpawnPoint("two"),
    )

    with pytest.raises(ValueError, match="spawn index"):
        carla_if._select_ego_spawn_point(spawn_index=spawn_index)


def test_indexed_ego_spawn_selection_rejects_an_occupied_point():
    spawn_point = _SpawnPoint("occupied")
    actor_location = types.SimpleNamespace(distance=lambda _location: 1.0)
    actor = types.SimpleNamespace(id=99, get_location=lambda: actor_location)
    actor_list = types.SimpleNamespace(
        filter=lambda pattern: [actor] if pattern == "vehicle.*" else []
    )
    carla_if = CARLAInterface()
    carla_if.world = types.SimpleNamespace(
        get_map=lambda: _SpawnMap([spawn_point]),
        get_actors=lambda: actor_list,
    )

    with pytest.raises(RuntimeError, match="spawn point 0 is not clear"):
        carla_if._select_ego_spawn_point(spawn_index=0)


def test_empty_road_spawn_is_centered_on_exact_driving_waypoint(monkeypatch):
    driving_lane = object()
    authored_spawn = _Transform(x=-6.446, y=-79.055, z=0.75, yaw=92.0)
    lane_center = _Transform(x=-5.532, y=-79.032, z=0.0, yaw=91.414)
    waypoint = types.SimpleNamespace(transform=lane_center)
    waypoint_queries = []

    class _Map:
        def get_spawn_points(self):
            return [authored_spawn]

        def get_waypoint(self, location, *, project_to_road, lane_type):
            waypoint_queries.append((location, project_to_road, lane_type))
            return waypoint

    carla_if = CARLAInterface()
    carla_if.world = types.SimpleNamespace(
        get_map=lambda: _Map(),
        get_actors=lambda: types.SimpleNamespace(filter=lambda _pattern: []),
    )
    monkeypatch.setattr(
        carla_interface_module.carla,
        "LaneType",
        types.SimpleNamespace(Driving=driving_lane),
        raising=False,
    )

    selected = carla_if._select_ego_spawn_point(
        spawn_index=0,
        center_on_driving_lane=True,
    )

    assert waypoint_queries == [(authored_spawn.location, True, driving_lane)]
    assert selected is lane_center
    assert selected.location.x == pytest.approx(-5.532)
    assert selected.location.y == pytest.approx(-79.032)
    assert selected.location.z == pytest.approx(authored_spawn.location.z)
    assert selected.rotation.yaw == pytest.approx(91.414)


def test_empty_road_spawn_centering_fails_if_no_driving_lane_exists(monkeypatch):
    authored_spawn = _Transform(x=1.0, y=2.0, z=0.5)

    class _Map:
        def get_spawn_points(self):
            return [authored_spawn]

        def get_waypoint(self, _location, *, project_to_road, lane_type):
            assert project_to_road is True
            assert lane_type is driving_lane
            return None

    driving_lane = object()
    carla_if = CARLAInterface()
    carla_if.world = types.SimpleNamespace(
        get_map=lambda: _Map(),
        get_actors=lambda: types.SimpleNamespace(filter=lambda _pattern: []),
    )
    monkeypatch.setattr(
        carla_interface_module.carla,
        "LaneType",
        types.SimpleNamespace(Driving=driving_lane),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="no nearby driving lane"):
        carla_if._select_ego_spawn_point(
            spawn_index=0,
            center_on_driving_lane=True,
        )


def test_zero_npc_spawn_performs_no_carla_operations():
    class _UnexpectedAccess:
        def __getattr__(self, name):
            pytest.fail(f"zero-NPC path unexpectedly accessed {name}")

    carla_if = CARLAInterface()
    carla_if.client = _UnexpectedAccess()
    carla_if.world = _UnexpectedAccess()

    carla_if.spawn_npcs(num_vehicles=0, num_walkers=0)

    assert carla_if.npc_vehicle_ids == []
    assert carla_if.npc_walker_ids == []
    assert carla_if.npc_walker_controller_ids == []


def test_carla_scenario_seed_reaches_local_traffic_and_pedestrian_rngs(monkeypatch):
    python_seeds = []
    numpy_seeds = []
    traffic_seeds = []
    pedestrian_seeds = []
    traffic_manager = types.SimpleNamespace(
        set_random_device_seed=traffic_seeds.append
    )
    carla_if = CARLAInterface()
    carla_if.client = types.SimpleNamespace(
        get_trafficmanager=lambda port: (
            traffic_manager
            if port == carla_if.tm_port
            else pytest.fail(f"unexpected TrafficManager port {port}")
        )
    )
    carla_if.world = types.SimpleNamespace(
        set_pedestrians_seed=pedestrian_seeds.append
    )
    monkeypatch.setattr(carla_interface_module.random, "seed", python_seeds.append)
    monkeypatch.setattr(carla_interface_module.np.random, "seed", numpy_seeds.append)

    carla_if.set_scenario_seed(17)

    assert python_seeds == [17]
    assert numpy_seeds == [17]
    assert traffic_seeds == [17]
    assert pedestrian_seeds == [17]


def test_force_map_reload_clears_an_existing_same_named_world(monkeypatch):
    class _World:
        def __init__(self, name):
            self.map = types.SimpleNamespace(name=name)
            self.wait_timeouts = []

        def get_map(self):
            return self.map

        def wait_for_tick(self, timeout):
            self.wait_timeouts.append(timeout)

    old_world = _World("/Game/Carla/Maps/Town03")
    fresh_world = _World("/Game/Carla/Maps/Town03")
    loaded_maps = []
    carla_if = CARLAInterface()
    carla_if.world = old_world
    carla_if.client = types.SimpleNamespace(
        load_world=lambda map_name: loaded_maps.append(map_name) or fresh_world
    )
    monkeypatch.setattr(carla_interface_module.time, "sleep", lambda _seconds: None)

    carla_if.load_map("Town03", force_reload=True)

    assert loaded_maps == ["Town03"]
    assert carla_if.world is fresh_world
    assert fresh_world.wait_timeouts == [20.0]


def test_dynamic_actor_census_excludes_ego_and_counts_empty_road_threat_classes():
    ego = types.SimpleNamespace(id=1)
    other_vehicle = types.SimpleNamespace(id=2)
    walker = types.SimpleNamespace(id=3)
    walker_controller = types.SimpleNamespace(id=4)
    actors_by_pattern = {
        "vehicle.*": [ego, other_vehicle],
        "walker.pedestrian.*": [walker],
        "controller.ai.walker": [walker_controller],
    }
    actor_list = types.SimpleNamespace(
        filter=lambda pattern: actors_by_pattern[pattern]
    )
    carla_if = CARLAInterface()
    carla_if.ego_vehicle = ego
    carla_if.world = types.SimpleNamespace(get_actors=lambda: actor_list)

    census = carla_if.get_non_ego_dynamic_actor_census()

    assert census == {
        "non_ego_vehicle_count": 1,
        "walker_count": 1,
        "walker_controller_count": 1,
    }

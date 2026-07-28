from types import SimpleNamespace

from module.carla_scene_truth import capture_scene_truth_snapshot


class _Actor:
    def __init__(self, actor_id, type_id, x, y, *, state=None):
        self.id = actor_id
        self.type_id = type_id
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0)
        )
        self._state = state

    def get_transform(self):
        return self._transform

    def get_state(self):
        return self._state


class _Map:
    def get_waypoint(self, location, **_kwargs):
        return SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )


class _World:
    def __init__(self, actors):
        self._actors = actors
        self._map = _Map()

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


def test_scene_truth_captures_only_nearby_non_ego_source_facts():
    ego = _Actor(1, "vehicle.tesla.model3", 0, 0)
    actors = [
        ego,
        _Actor(2, "vehicle.audi.tt", 10, 0),
        _Actor(3, "walker.pedestrian.0001", 70, 0),
        _Actor(
            4,
            "traffic.traffic_light",
            12,
            0,
            state=SimpleNamespace(name="Red"),
        ),
    ]
    snapshot = capture_scene_truth_snapshot(
        world=_World(actors),
        ego_vehicle=ego,
        source_frame_id=100,
        source_simulation_time_s=10.0,
        navigation_context=None,
        route_plan=None,
        route_index=0,
        carla_module=SimpleNamespace(
            LaneType=SimpleNamespace(Driving="Driving")
        ),
    )

    assert [actor.actor_id for actor in snapshot.dynamic_actors] == [2]
    assert snapshot.dynamic_actors[0].actor_class == "vehicle"
    assert len(snapshot.traffic_controls) == 1
    assert snapshot.traffic_controls[0].state == "red"
    assert snapshot.traffic_controls[0].route_relevant
    serialized = snapshot.to_json_dict()
    assert serialized["source_frame_id"] == 100
    assert serialized["dynamic_actors"][0]["actor_id"] == 2

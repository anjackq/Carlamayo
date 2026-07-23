import importlib
import queue
import sys
import types

import numpy as np
import pytest


fake_carla = types.SimpleNamespace(
    command=types.SimpleNamespace(DestroyActor=lambda actor_id: ("destroy", actor_id)),
    VehicleControl=lambda: types.SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0),
    Vector3D=lambda: types.SimpleNamespace(),
)
sys.modules.setdefault("carla", fake_carla)

carla_interface_module = importlib.import_module("module.carla_interface")
CARLAInterface = carla_interface_module.CARLAInterface
CARLATickContext = carla_interface_module.CARLATickContext


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeRotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)


class FakeTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.location = FakeLocation(x, y, z)
        self.rotation = FakeRotation(roll, pitch, yaw)


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeImage:
    width = 2
    height = 1

    def __init__(self, frame, *, bgr=(1, 2, 3), timestamp=None):
        self.frame = int(frame)
        pixel = bytes((*bgr, 255))
        self.raw_data = pixel * (self.width * self.height)
        if timestamp is not None:
            self.timestamp = float(timestamp)


def _state(frame_id, simulation_time_s, *, x=0.0, y=0.0, z=0.0, yaw=0.0):
    return {
        "frame_id": frame_id,
        "simulation_time_s": simulation_time_s,
        "x": x,
        "y": y,
        "z": z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": yaw,
        "speed": 0.0,
    }


def _tick_context(frame_id=20, simulation_time_s=2.0):
    return CARLATickContext(
        frame_id=frame_id,
        simulation_time_s=simulation_time_s,
        delta_seconds=0.1,
        snapshot=object(),
        actor_snapshot=object(),
        ego_transform=FakeTransform(x=4.0, y=5.0, yaw=10.0),
        ego_velocity=FakeVector(x=3.0, y=4.0),
    )


def test_environment_ports_configure_rpc_and_traffic_manager(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            clients.append(self)

        def set_timeout(self, timeout):
            self.timeout = timeout

        def get_world(self):
            return object()

    monkeypatch.setenv("CARLAMAYO_CARLA_HOST", "127.0.0.1")
    monkeypatch.setenv("CARLAMAYO_CARLA_PORT", "23456")
    monkeypatch.setenv("CARLAMAYO_TRAFFIC_MANAGER_PORT", "23458")
    monkeypatch.setattr(
        carla_interface_module,
        "carla",
        types.SimpleNamespace(Client=FakeClient),
    )

    carla_if = CARLAInterface()
    carla_if.connect()

    assert clients[0].host == "127.0.0.1"
    assert clients[0].port == 23456
    assert clients[0].timeout == pytest.approx(20.0)
    assert carla_if.tm_port == 23458


class EmptyCameraQueue:
    def get(self, timeout):
        raise queue.Empty


def test_get_camera_images_raises_when_camera_frame_missing():
    carla_if = CARLAInterface()
    carla_if.camera_order = ["cam_front_wide"]
    carla_if.sensor_queues = {"cam_front_wide": EmptyCameraQueue()}

    with pytest.raises(TimeoutError, match=r"Missing camera frames: \['cam_front_wide'\]"):
        carla_if.get_camera_images()


def test_camera_specs_use_alpamayo_order_and_training_fovs():
    carla_if = CARLAInterface()

    assert carla_if.camera_order == [
        "cam_front_left",
        "cam_front_wide",
        "cam_front_right",
        "cam_front_tele",
    ]
    assert carla_if.camera_ids == (0, 1, 2, 6)
    assert carla_if.camera_configs["cam_front_wide"]["fov"] == 120.0


def test_exact_camera_collection_retains_future_packets_and_decodes_rgb():
    carla_if = CARLAInterface()
    for name in carla_if.camera_order:
        carla_if.camera_collector.put(11, name, FakeImage(11, bgr=(7, 8, 9)))
    for name in carla_if.camera_order:
        carla_if.camera_collector.put(10, name, FakeImage(10, bgr=(1, 2, 3)))

    current = carla_if.get_camera_images(10, timeout=0.01)
    future = carla_if.get_camera_images(11, timeout=0.01)

    assert current.shape == (4, 1, 2, 3)
    assert current[0, 0, 0].tolist() == [3, 2, 1]
    assert future[0, 0, 0].tolist() == [9, 8, 7]
    assert carla_if.get_camera_sync_stats()["accepted_bundles"] == 2
    assert carla_if.get_camera_sync_stats()["frame_mismatches"] == 0


def test_camera_packet_frame_mismatch_is_rejected():
    carla_if = CARLAInterface()
    for name in carla_if.camera_order:
        packet_frame = 21 if name == "cam_front_wide" else 20
        carla_if.camera_collector.put(20, name, FakeImage(packet_frame))

    with pytest.raises(RuntimeError, match="returned frame 21; expected 20"):
        carla_if.get_camera_images(20, timeout=0.01)

    assert carla_if.get_camera_sync_stats()["frame_mismatches"] == 1


def test_synchronized_observation_updates_history_only_after_complete_bundle():
    carla_if = CARLAInterface()
    context = _tick_context()
    for name in carla_if.camera_order[:-1]:
        carla_if.camera_collector.put(context.frame_id, name, FakeImage(context.frame_id))

    with pytest.raises(TimeoutError, match="cam_front_tele"):
        carla_if.get_synchronized_observation(context, timeout=0.0)

    assert carla_if.history_buffer == []

    carla_if.camera_collector.put(
        context.frame_id,
        "cam_front_tele",
        FakeImage(context.frame_id),
    )
    observation = carla_if.get_synchronized_observation(context, timeout=0.01)

    assert observation.frame_id == 20
    assert observation.simulation_time_s == pytest.approx(2.0)
    assert observation.camera_ids == (0, 1, 2, 6)
    assert observation.camera_images.shape == (4, 1, 2, 3)
    assert observation.camera_intrinsics.shape == (4, 3, 3)
    assert observation.camera_extrinsics.shape == (4, 4, 4)
    assert observation.camera_intrinsics[1, 0, 0] == pytest.approx(1 / np.sqrt(3))
    assert observation.camera_extrinsics[1, 0, 3] == pytest.approx(1.5)
    assert observation.ego_pose_world[0, 3] == pytest.approx(4.0)
    assert observation.ego_velocity_world.tolist() == pytest.approx([3.0, 4.0, 0.0])
    assert observation.ego_history_frame_ids == (20,)
    assert len(carla_if.history_buffer) == 1


def test_synchronized_observation_rejects_camera_timestamp_mismatch():
    carla_if = CARLAInterface()
    context = _tick_context()
    for name in carla_if.camera_order:
        timestamp = 1.9 if name == "cam_front_left" else 2.0
        carla_if.camera_collector.put(
            context.frame_id,
            name,
            FakeImage(context.frame_id, timestamp=timestamp),
        )

    with pytest.raises(RuntimeError, match="does not match snapshot"):
        carla_if.get_synchronized_observation(context, timeout=0.01)

    assert carla_if.history_buffer == []
    assert carla_if.get_camera_sync_stats()["timestamp_mismatches"] == 1


def test_history_conversion_reflects_carla_right_and_does_not_mutate_padding():
    carla_if = CARLAInterface()
    carla_if.update_history(_state(1, 0.1, y=2.0, yaw=90.0))
    carla_if.update_history(_state(2, 0.2))
    buffer_length = len(carla_if.history_buffer)

    history_xyz, history_rot = carla_if.get_history_in_local_frame()

    assert len(carla_if.history_buffer) == buffer_length
    assert history_xyz.shape == (16, 3)
    assert history_xyz[-2].tolist() == pytest.approx([0.0, -2.0, 0.0])
    assert history_xyz[-1].tolist() == pytest.approx([0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        history_rot[-2],
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(history_rot[-1], np.eye(3), atol=1e-6)


def test_history_restarts_after_a_missing_exact_frame():
    carla_if = CARLAInterface()
    carla_if.update_history(_state(10, 1.0, x=10.0))
    carla_if.update_history(_state(12, 1.2, x=12.0))

    history_xyz, _ = carla_if.get_history_in_local_frame()

    assert [state["frame_id"] for state in carla_if.history_buffer] == [12]
    np.testing.assert_allclose(history_xyz, np.zeros((16, 3)), atol=1e-6)
    assert carla_if.get_camera_sync_stats()["history_sequence_resets"] == 1


def test_tick_returns_matching_snapshot_state_instead_of_live_actor_state():
    snapshot_transform = FakeTransform(x=10.0, y=2.0, yaw=15.0)
    snapshot_velocity = FakeVector(x=6.0)

    class ActorSnapshot:
        def get_transform(self):
            return snapshot_transform

        def get_velocity(self):
            return snapshot_velocity

    class Snapshot:
        frame = 37
        timestamp = types.SimpleNamespace(elapsed_seconds=3.7, delta_seconds=0.1)

        def find(self, actor_id):
            assert actor_id == 99
            return ActorSnapshot()

    class World:
        def tick(self):
            return 37

        def get_snapshot(self):
            return Snapshot()

    carla_if = CARLAInterface()
    carla_if.world = World()
    carla_if.ego_vehicle = types.SimpleNamespace(
        id=99,
        get_transform=lambda: FakeTransform(x=999.0),
        get_velocity=lambda: FakeVector(x=999.0),
    )

    context = carla_if.tick()
    state = carla_if.get_ego_state()

    assert context.frame_id == 37
    assert context.simulation_time_s == pytest.approx(3.7)
    assert state["frame_id"] == 37
    assert state["simulation_time_s"] == pytest.approx(3.7)
    assert state["x"] == pytest.approx(10.0)
    assert state["speed"] == pytest.approx(6.0)


def test_tick_rejects_a_snapshot_from_a_different_frame():
    snapshot = types.SimpleNamespace(frame=41)
    carla_if = CARLAInterface()
    carla_if.world = types.SimpleNamespace(
        tick=lambda: 42,
        get_snapshot=lambda: snapshot,
    )

    with pytest.raises(RuntimeError, match="tick returned frame 42.*snapshot.*frame 41"):
        carla_if.tick()


def test_episode_collision_count_survives_spawn_history_reset():
    carla_if = CARLAInterface()
    event = types.SimpleNamespace(
        frame=42,
        normal_impulse=types.SimpleNamespace(x=3.0, y=4.0, z=0.0),
        other_actor=types.SimpleNamespace(type_id="vehicle.test", id=7),
    )

    carla_if._collision_callback(event)
    assert carla_if.get_collision_count() == 1
    assert carla_if.get_episode_collision_count() == 1

    carla_if.reset_collision_history()
    assert carla_if.get_collision_count() == 0
    assert carla_if.get_episode_collision_count() == 1


def test_episode_collision_count_is_not_capped_by_detail_history():
    carla_if = CARLAInterface()
    for frame in range(25):
        carla_if._collision_callback(
            types.SimpleNamespace(
                frame=frame,
                normal_impulse=types.SimpleNamespace(x=1.0, y=0.0, z=0.0),
                other_actor=types.SimpleNamespace(type_id="vehicle.test", id=frame),
            )
        )

    assert len(carla_if.collision_events) == 20
    assert carla_if.get_collision_count() == 25
    assert carla_if.get_episode_collision_count() == 25


def test_cleanup_reports_recoverable_teardown_failures(capsys):
    class FailingTrafficManager:
        def set_synchronous_mode(self, _enabled):
            raise RuntimeError("traffic manager failed")

    class FailingClient:
        def get_trafficmanager(self, _port):
            return FailingTrafficManager()

        def apply_batch(self, _commands):
            raise RuntimeError("destroy batch failed")

    class FailingController:
        id = 10

        def stop(self):
            raise RuntimeError("controller stop failed")

    class FailingSettings:
        synchronous_mode = True
        fixed_delta_seconds = 0.1

    class FailingWorld:
        def get_actors(self, _actor_ids):
            return [FailingController()]

        def get_settings(self):
            return FailingSettings()

        def apply_settings(self, _settings):
            raise RuntimeError("world settings failed")

    class FailingSensor:
        id = 20

        def stop(self):
            raise RuntimeError("sensor stop failed")

        def destroy(self):
            raise RuntimeError("sensor destroy failed")

    class FailingEgoVehicle:
        def destroy(self):
            raise RuntimeError("ego destroy failed")

    carla_if = CARLAInterface()
    carla_if.client = FailingClient()
    carla_if.world = FailingWorld()
    carla_if.npc_walker_controller_ids = [10]
    carla_if.npc_walker_ids = [11]
    carla_if.npc_vehicle_ids = [12]
    carla_if.sensors = {"cam_front_wide": FailingSensor()}
    carla_if.ego_vehicle = FailingEgoVehicle()

    carla_if.cleanup()

    output = capsys.readouterr().out
    assert "Warning: failed to disable TrafficManager synchronous mode" in output
    assert "Warning: failed to stop walker controller 10" in output
    assert "Warning: failed to destroy walker controllers" in output
    assert "Warning: failed to destroy NPC walkers" in output
    assert "Warning: failed to destroy NPC vehicles" in output
    assert "Warning: failed to stop sensor cam_front_wide" in output
    assert "Warning: failed to destroy sensor cam_front_wide" in output
    assert "Warning: failed to destroy ego vehicle" in output
    assert "Warning: failed to restore world asynchronous mode" in output

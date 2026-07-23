"""CARLA environment interface and lifecycle management."""

import math
import os
import queue
import random
import time
from dataclasses import dataclass, field
from typing import Any

import carla
import cv2
import numpy as np

from . import config as cfg
from .data_collection import ExactFrameCollector
from .geometry import (
    camera_intrinsic_matrix,
    carla_relative_rotation_to_model,
    pose_matrix_from_components,
    pose_matrix_from_state,
    pose_matrix_from_transform,
    world_points_to_model_ego,
)
from .runtime_types import SynchronizedObservation


EXPECTED_ALPAMAYO_CAMERA_IDS = (0, 1, 2, 6)
EXPECTED_ALPAMAYO_CAMERA_NAMES = (
    "cam_front_left",
    "cam_front_wide",
    "cam_front_right",
    "cam_front_tele",
)


def _environment_port(name, default):
    raw = os.environ.get(name)
    value = int(default if raw in (None, "") else raw)
    if not 1024 <= value <= 65535:
        raise ValueError(f"{name} must be within [1024, 65535]")
    return value
EXPECTED_ALPAMAYO_CAMERA_FOVS = (120.0, 120.0, 120.0, 30.0)


@dataclass(frozen=True)
class CARLATickContext:
    """Snapshot-backed state for one successful synchronous CARLA tick."""

    frame_id: int
    simulation_time_s: float
    delta_seconds: float
    snapshot: Any = field(repr=False, compare=False)
    actor_snapshot: Any = field(repr=False, compare=False)
    ego_transform: Any = field(repr=False, compare=False)
    ego_velocity: Any = field(repr=False, compare=False)


def _validated_camera_specs():
    specs = tuple(dict(spec) for spec in cfg.CAMERA_SPECS)
    names = tuple(spec["name"] for spec in specs)
    camera_ids = tuple(int(spec["alpamayo_id"]) for spec in specs)
    camera_fovs = tuple(float(spec["fov"]) for spec in specs)
    if len(specs) != cfg.NUM_CAMERAS:
        raise ValueError("CAMERA_SPECS length must match NUM_CAMERAS")
    if len(set(names)) != len(names):
        raise ValueError("CAMERA_SPECS camera names must be unique")
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("CAMERA_SPECS Alpamayo IDs must be unique")
    if camera_ids != EXPECTED_ALPAMAYO_CAMERA_IDS:
        raise ValueError(
            "CAMERA_SPECS must use Alpamayo camera IDs [0, 1, 2, 6] in left/wide/right/tele order"
        )
    if names != EXPECTED_ALPAMAYO_CAMERA_NAMES:
        raise ValueError("CAMERA_SPECS must use left/wide/right/tele camera names in model order")
    if camera_fovs != EXPECTED_ALPAMAYO_CAMERA_FOVS:
        raise ValueError("CAMERA_SPECS must use nominal Alpamayo FOVs [120, 120, 120, 30]")
    return specs


def is_allowed_npc_vehicle_blueprint(blueprint):
    """Return True for regular passenger-car NPC vehicle blueprints."""

    if not blueprint.has_attribute("number_of_wheels"):
        return False
    if int(blueprint.get_attribute("number_of_wheels")) != 4:
        return False

    blueprint_id = getattr(blueprint, "id", "").lower()
    return not any(keyword in blueprint_id for keyword in cfg.NPC_EXCLUDED_VEHICLE_KEYWORDS)


class CARLAInterface:
    """Interface for CARLA simulation."""

    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        # ``sensor_queues`` is retained for compatibility with callers that
        # inject legacy per-camera queues. Normal operation uses one shared,
        # frame-aware collector so camera packets can never be mixed by dequeue
        # order.
        self.sensor_queues = {}
        self.camera_collector = ExactFrameCollector(
            queue_size=max(16, cfg.NUM_CAMERAS * 4),
            max_pending_frames=4,
        )
        self.collision_events = []
        self.spawn_collision_count = 0
        # The spawn-local counter drives RespawnMonitor, the detail list is a
        # bounded diagnostic ring, and the episode total deliberately survives
        # ego respawns for run-level telemetry.
        self.episode_collision_count = 0
        self.history_buffer = []
        self.npc_vehicle_ids = []
        self.npc_walker_ids = []
        self.npc_walker_controller_ids = []
        self.tm_port = _environment_port("CARLAMAYO_TRAFFIC_MANAGER_PORT", 8000)
        self.camera_specs = _validated_camera_specs()
        self.camera_configs = {spec["name"]: dict(spec) for spec in self.camera_specs}
        self.camera_order = [spec["name"] for spec in self.camera_specs]
        self.camera_ids = tuple(int(spec["alpamayo_id"]) for spec in self.camera_specs)
        self._last_tick_context = None
        self._last_camera_packets = None
        self._accepted_camera_bundles = 0
        self._missing_camera_bundles = 0
        self._camera_frame_mismatches = 0
        self._camera_timestamp_mismatches = 0
        self._history_sequence_resets = 0

    def connect(self, host=None, port=None):
        host = host or os.environ.get("CARLAMAYO_CARLA_HOST", "localhost")
        port = (
            _environment_port("CARLAMAYO_CARLA_PORT", 2000)
            if port is None
            else int(port)
        )
        if not 1024 <= port <= 65535:
            raise ValueError("CARLA port must be within [1024, 65535]")
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        print("Connected to CARLA")

    def load_map(self, map_name, *, force_reload=False):
        current_map = self.world.get_map().name
        if force_reload or map_name not in current_map:
            print(f"Loading map: {map_name}...")
            self.world = self.client.load_world(map_name)
            print("Map load requested. Waiting for world tick...")
            self.world.wait_for_tick(20.0)
            time.sleep(1.0)
            print(f"Map loaded: {map_name}")
        else:
            print(f"Already on map: {current_map}")

    def set_scenario_seed(self, seed):
        """Seed CARLA traffic, pedestrians, and local spawn selection."""

        seed = int(seed)
        if seed < 0 or seed > cfg.MAX_SCENARIO_SEED:
            raise ValueError(f"scenario seed must be within [0, {cfg.MAX_SCENARIO_SEED}]")
        random.seed(seed)
        np.random.seed(seed)
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        set_tm_seed = getattr(traffic_manager, "set_random_device_seed", None)
        if callable(set_tm_seed):
            set_tm_seed(seed)
        set_pedestrian_seed = getattr(self.world, "set_pedestrians_seed", None)
        if callable(set_pedestrian_seed):
            set_pedestrian_seed(seed)
        print(f"CARLA scenario seed: {seed}")

    def enable_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        tm = self.client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)
        print("Synchronous mode enabled.")

    def spawn_npcs(self, num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT):
        num_vehicles = int(num_vehicles)
        num_walkers = int(num_walkers)
        if num_vehicles < 0 or num_walkers < 0:
            raise ValueError("NPC counts must be nonnegative")
        if num_vehicles == 0 and num_walkers == 0:
            print("Empty-road mode: NPC vehicle and pedestrian spawning skipped.")
            return

        bp_lib = self.world.get_blueprint_library()
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.global_percentage_speed_difference(0.0)

        vehicle_bps = [
            bp for bp in bp_lib.filter("vehicle.*") if is_allowed_npc_vehicle_blueprint(bp)
        ]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_count = min(num_vehicles, len(spawn_points))
        vehicle_batch = []
        for i in range(vehicle_count):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            transform = spawn_points[i]
            vehicle_batch.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)
                )
            )
        vehicle_results = self.client.apply_batch_sync(vehicle_batch, True)
        for res in vehicle_results:
            if not res.error:
                self.npc_vehicle_ids.append(res.actor_id)
        print(f"Spawned NPC vehicles: {len(self.npc_vehicle_ids)}/{num_vehicles}")

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_spawn_points = []
        attempts = 0
        max_attempts = max(num_walkers * 5, 100)
        while len(walker_spawn_points) < num_walkers and attempts < max_attempts:
            loc = self.world.get_random_location_from_navigation()
            attempts += 1
            if loc is None:
                continue
            walker_spawn_points.append(carla.Transform(loc))

        walker_batch = []
        walker_speeds = []
        for transform in walker_spawn_points:
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            speed = 1.4
            if bp.has_attribute("speed"):
                speed_values = bp.get_attribute("speed").recommended_values
                if len(speed_values) > 1:
                    speed = float(speed_values[1])
            walker_speeds.append(speed)
            walker_batch.append(carla.command.SpawnActor(bp, transform))

        walker_results = self.client.apply_batch_sync(walker_batch, True)
        spawned_walker_ids = []
        spawned_walker_speeds = []
        for idx, res in enumerate(walker_results):
            if not res.error:
                spawned_walker_ids.append(res.actor_id)
                spawned_walker_speeds.append(walker_speeds[idx])
        self.npc_walker_ids = spawned_walker_ids

        walker_controller_bp = bp_lib.find("controller.ai.walker")
        controller_batch = [
            carla.command.SpawnActor(walker_controller_bp, carla.Transform(), wid)
            for wid in self.npc_walker_ids
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True)
        self.npc_walker_controller_ids = [
            res.actor_id for res in controller_results if not res.error
        ]
        controller_actors = self.world.get_actors(self.npc_walker_controller_ids)

        for i, controller in enumerate(controller_actors):
            controller.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(
                float(spawned_walker_speeds[i] if i < len(spawned_walker_speeds) else 1.4)
            )

        print(f"Spawned NPC walkers: {len(self.npc_walker_ids)}/{num_walkers}")

    def get_non_ego_dynamic_actor_census(self):
        """Count non-ego dynamic road users currently present in the world."""

        actors = self.world.get_actors()
        ego_id = int(self.ego_vehicle.id) if self.ego_vehicle is not None else None
        vehicle_count = sum(int(actor.id) != ego_id for actor in actors.filter("vehicle.*"))
        walker_count = len(actors.filter("walker.pedestrian.*"))
        walker_controller_count = len(actors.filter("controller.ai.walker"))
        return {
            "non_ego_vehicle_count": int(vehicle_count),
            "walker_count": int(walker_count),
            "walker_controller_count": int(walker_controller_count),
        }

    def _is_spawn_point_clear(self, spawn_point, min_distance=8.0):
        if self.world is None:
            return True
        spawn_location = spawn_point.location
        for actor in self.world.get_actors().filter("vehicle.*"):
            if self.ego_vehicle is not None and actor.id == self.ego_vehicle.id:
                continue
            if actor.get_location().distance(spawn_location) < min_distance:
                return False
        return True

    def _center_spawn_point_on_driving_lane(self, spawn_point):
        """Return the exact driving-lane center near a CARLA spawn point."""

        waypoint = self.world.get_map().get_waypoint(
            spawn_point.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError("configured spawn point has no nearby driving lane")
        centered = waypoint.transform
        # Preserve CARLA's authored spawn height while replacing its lateral
        # offset and heading with the OpenDRIVE lane center. Some Town03 spawn
        # points are close enough to a lane edge that a Tesla footprint does
        # not fit inside the lane even though the actor origin is on the road.
        centered.location.z = float(spawn_point.location.z)
        return centered

    def _select_ego_spawn_point(self, spawn_index=None, *, center_on_driving_lane=False):
        print("Selecting spawn point...")
        spawn_points = list(self.world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no ego spawn points")
        if spawn_index is not None:
            spawn_index = int(spawn_index)
            if spawn_index < 0 or spawn_index >= len(spawn_points):
                raise ValueError(
                    f"ego spawn index {spawn_index} is outside [0, {len(spawn_points) - 1}]"
                )
            spawn_point = spawn_points[spawn_index]
            if center_on_driving_lane:
                spawn_point = self._center_spawn_point_on_driving_lane(spawn_point)
            if not self._is_spawn_point_clear(spawn_point):
                raise RuntimeError(f"configured ego spawn point {spawn_index} is not clear")
            print(f"Selected configured spawn point {spawn_index}.")
            return spawn_point
        shuffled_points = list(spawn_points)
        random.shuffle(shuffled_points)
        for spawn_point in shuffled_points:
            if self._is_spawn_point_clear(spawn_point):
                print("Selected clear random spawn point.")
                return spawn_point
        spawn_point = random.choice(spawn_points)
        print("No clear spawn point found; selected random spawn point.")
        return spawn_point

    def spawn_ego_vehicle(self, *, spawn_index=None, center_on_driving_lane=False):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_point = self._select_ego_spawn_point(
            spawn_index,
            center_on_driving_lane=center_on_driving_lane,
        )
        print("Spawning ego vehicle...")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego vehicle at {spawn_point.location}")
        return self.ego_vehicle

    def respawn_ego_vehicle(self, *, spawn_index=None, center_on_driving_lane=False):
        """Teleport the ego vehicle to a clear spawn point and reset ego-local state."""

        if self.ego_vehicle is None:
            return self.spawn_ego_vehicle(
                spawn_index=spawn_index,
                center_on_driving_lane=center_on_driving_lane,
            )

        spawn_point = self._select_ego_spawn_point(
            spawn_index,
            center_on_driving_lane=center_on_driving_lane,
        )
        print(f"Respawning ego vehicle at {spawn_point.location}")
        self.apply_control(0.0, 0.0, 1.0)
        self.ego_vehicle.set_target_velocity(carla.Vector3D())
        self.ego_vehicle.set_target_angular_velocity(carla.Vector3D())
        self.ego_vehicle.set_transform(spawn_point)
        self.ego_vehicle.set_target_velocity(carla.Vector3D())
        self.ego_vehicle.set_target_angular_velocity(carla.Vector3D())
        self.apply_control(0.0, 0.0, 1.0)
        self.history_buffer.clear()
        self._last_tick_context = None
        self._last_camera_packets = None
        self.reset_collision_history()
        self.flush_camera_queues()
        return spawn_point

    def setup_cameras(self):
        print("Setting up cameras...")
        bp_lib = self.world.get_blueprint_library()
        for name in self.camera_order:
            cfg_cam = self.camera_configs[name]
            print(f"  - spawning {name}")
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.IMG_WIDTH))
            cam_bp.set_attribute("image_size_y", str(cfg.IMG_HEIGHT))
            cam_bp.set_attribute("fov", str(cfg_cam["fov"]))
            cam_bp.set_attribute(
                "enable_postprocess_effects", str(cfg.CAMERA_ENABLE_POSTPROCESS_EFFECTS)
            )
            cam_bp.set_attribute("sensor_tick", "0.0")

            transform = carla.Transform(
                carla.Location(x=cfg_cam["x"], y=cfg_cam["y"], z=cfg_cam["z"]),
                carla.Rotation(
                    roll=cfg_cam.get("roll", 0.0),
                    pitch=cfg_cam["pitch"],
                    yaw=cfg_cam["yaw"],
                ),
            )
            sensor = self.world.spawn_actor(cam_bp, transform, attach_to=self.ego_vehicle)
            sensor.listen(lambda data, n=name: self._camera_callback(data, n))
            self.sensors[name] = sensor
            time.sleep(0.2)

        print(f"Setup {len(self.sensors)} cameras ({cfg.IMG_WIDTH}x{cfg.IMG_HEIGHT})")

    def setup_collision_sensor(self):
        print("Setting up collision sensor...")
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find("sensor.other.collision")
        sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        sensor.listen(self._collision_callback)
        self.sensors["collision"] = sensor

    def _collision_callback(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        other_actor = getattr(event, "other_actor", None)
        collision_event = {
            "frame": int(getattr(event, "frame", 0)),
            "intensity": float(intensity),
            "other_actor": getattr(other_actor, "type_id", "unknown"),
            "other_actor_id": int(getattr(other_actor, "id", 0)),
        }
        self.collision_events.append(collision_event)
        self.spawn_collision_count += 1
        self.episode_collision_count += 1
        if len(self.collision_events) > 20:
            self.collision_events = self.collision_events[-20:]
        print(
            "Collision detected: "
            f"actor={collision_event['other_actor']} "
            f"impulse={collision_event['intensity']:.1f}"
        )

    def get_collision_count(self):
        return self.spawn_collision_count

    def get_episode_collision_count(self):
        """Return the collision count for the complete interface lifetime."""

        return self.episode_collision_count

    def get_last_collision_event(self):
        return self.collision_events[-1] if self.collision_events else None

    def reset_collision_history(self):
        self.collision_events.clear()
        self.spawn_collision_count = 0

    def flush_camera_queues(self):
        self.camera_collector.clear()
        for sensor_queue in self.sensor_queues.values():
            while True:
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break
        self._last_camera_packets = None

    def _camera_callback(self, image, name):
        self.camera_collector.put(image.frame, name, image)

    @staticmethod
    def _decode_camera_image(data):
        height = int(getattr(data, "height", cfg.IMG_HEIGHT))
        width = int(getattr(data, "width", cfg.IMG_WIDTH))
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        expected_size = height * width * 4
        if array.size != expected_size:
            raise ValueError(
                f"Camera frame contains {array.size} bytes; expected {expected_size} "
                f"for {width}x{height} BGRA"
            )
        array = array.reshape((height, width, 4))[:, :, :3]
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    def _collect_camera_packets(self, frame_id, timeout):
        target_frame = int(frame_id)
        if self._last_camera_packets is not None:
            cached_frame, cached_packets = self._last_camera_packets
            if cached_frame == target_frame:
                return cached_packets

        packets = self.camera_collector.collect(
            target_frame,
            self.camera_order,
            timeout=timeout,
        )
        missing = [name for name in self.camera_order if name not in packets]
        if missing:
            self._missing_camera_bundles += 1
            raise TimeoutError(f"Missing camera frames for CARLA frame {target_frame}: {missing}")

        for name, packet in packets.items():
            packet_frame = int(getattr(packet, "frame", target_frame))
            if packet_frame != target_frame:
                self._camera_frame_mismatches += 1
                raise RuntimeError(
                    f"Camera {name} returned frame {packet_frame}; expected {target_frame}"
                )

        ordered = {name: packets[name] for name in self.camera_order}
        self._last_camera_packets = (target_frame, ordered)
        self._accepted_camera_bundles += 1
        return ordered

    def _get_legacy_camera_images(self, timeout):
        """Read old per-camera queues when no tick context is available."""

        images = []
        missing = []
        for name in self.camera_order:
            sensor_queue = self.sensor_queues.get(name)
            if sensor_queue is None:
                missing.append(name)
                continue
            try:
                images.append(self._decode_camera_image(sensor_queue.get(timeout=timeout)))
            except queue.Empty:
                missing.append(name)
        if missing:
            raise TimeoutError(f"Missing camera frames: {missing}")
        return np.stack(images, axis=0)

    def get_camera_images(self, frame_id=None, timeout=1.0):
        """Return canonical RGB cameras for one exact CARLA frame.

        Omitting ``frame_id`` targets the most recent :meth:`tick` context. The
        legacy independent-queue path is retained only for callers that have not
        ticked this interface and explicitly populated ``sensor_queues``.
        """

        if isinstance(frame_id, CARLATickContext):
            frame_id = frame_id.frame_id
        if frame_id is None and self._last_tick_context is not None:
            frame_id = self._last_tick_context.frame_id
        if frame_id is None:
            return self._get_legacy_camera_images(timeout)

        packets = self._collect_camera_packets(frame_id, timeout)
        return np.stack(
            [self._decode_camera_image(packets[name]) for name in self.camera_order],
            axis=0,
        )

    @staticmethod
    def _state_from_transform(transform, velocity, *, frame_id=None, simulation_time_s=None):
        velocity_world = np.array(
            [velocity.x, velocity.y, velocity.z],
            dtype=np.float64,
        )
        state = {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "roll": transform.rotation.roll,
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "speed": float(np.linalg.norm(velocity_world)),
            "pose_world": pose_matrix_from_transform(transform),
            "velocity_world": velocity_world,
        }
        if frame_id is not None:
            state["frame_id"] = int(frame_id)
        if simulation_time_s is not None:
            state["simulation_time_s"] = float(simulation_time_s)
        return state

    def get_ego_state(self, tick_context=None):
        context = tick_context
        if context is None:
            context = self._last_tick_context
        if context is not None:
            if not isinstance(context, CARLATickContext):
                raise TypeError("tick_context must be a CARLATickContext")
            return self._state_from_transform(
                context.ego_transform,
                context.ego_velocity,
                frame_id=context.frame_id,
                simulation_time_s=context.simulation_time_s,
            )

        return self._state_from_transform(
            self.ego_vehicle.get_transform(),
            self.ego_vehicle.get_velocity(),
        )

    def update_history(self, state):
        normalized = dict(state)
        frame_id = normalized.get("frame_id")
        if frame_id is not None and self.history_buffer:
            previous_frame_id = self.history_buffer[-1].get("frame_id")
            if previous_frame_id is not None:
                if int(frame_id) < int(previous_frame_id):
                    raise ValueError("ego history frame IDs must be monotonic")
                if int(frame_id) == int(previous_frame_id):
                    self.history_buffer[-1] = normalized
                    return
                previous_time = self.history_buffer[-1].get("simulation_time_s")
                simulation_time = normalized.get("simulation_time_s")
                frame_gap = int(frame_id) != int(previous_frame_id) + 1
                time_gap = (
                    previous_time is not None
                    and simulation_time is not None
                    and not math.isclose(
                        float(simulation_time) - float(previous_time),
                        float(cfg.CONTROL_DT),
                        rel_tol=0.0,
                        abs_tol=1e-4,
                    )
                )
                if frame_gap or time_gap:
                    # Alpamayo expects history on a contiguous 10 Hz grid. A
                    # missing bundle starts a new sequence that will be padded
                    # from its first complete observation.
                    self.history_buffer.clear()
                    self._history_sequence_resets += 1
        self.history_buffer.append(normalized)
        if len(self.history_buffer) > cfg.NUM_HISTORY:
            self.history_buffer.pop(0)

    def _padded_history_states(self):
        states = list(self.history_buffer[-cfg.NUM_HISTORY :])
        if not states:
            states = [self.get_ego_state()]
        return [states[0]] * (cfg.NUM_HISTORY - len(states)) + states

    def get_history_in_local_frame(self):
        states = self._padded_history_states()
        poses = [pose_matrix_from_state(state) for state in states]
        current_pose = poses[-1]
        positions_world = np.stack([pose[:3, 3] for pose in poses], axis=0)
        history_xyz = world_points_to_model_ego(current_pose, positions_world)

        current_rotation_inv = current_pose[:3, :3].T
        history_rot = np.stack(
            [
                carla_relative_rotation_to_model(current_rotation_inv @ pose[:3, :3])
                for pose in poses
            ],
            axis=0,
        )
        return history_xyz.astype(np.float32), history_rot.astype(np.float32)

    def _camera_calibration(self, packets, ego_pose_world):
        intrinsics = []
        extrinsics = []
        world_to_ego = np.linalg.inv(ego_pose_world)
        for spec in self.camera_specs:
            packet = packets[spec["name"]]
            width = int(getattr(packet, "width", cfg.IMG_WIDTH))
            height = int(getattr(packet, "height", cfg.IMG_HEIGHT))
            intrinsics.append(camera_intrinsic_matrix(width, height, spec["fov"]))

            capture_transform = getattr(packet, "transform", None)
            if capture_transform is not None:
                sensor_to_ego = world_to_ego @ pose_matrix_from_transform(capture_transform)
            else:
                sensor_to_ego = pose_matrix_from_components(
                    spec["x"],
                    spec["y"],
                    spec["z"],
                    spec.get("roll", 0.0),
                    spec.get("pitch", 0.0),
                    spec.get("yaw", 0.0),
                )
            extrinsics.append(sensor_to_ego)
        return np.stack(intrinsics, axis=0), np.stack(extrinsics, axis=0)

    def get_synchronized_observation(
        self,
        tick_context=None,
        timeout=1.0,
        *,
        update_history=True,
    ):
        """Build one complete, exact-frame observation from a tick snapshot."""

        context = tick_context if tick_context is not None else self._last_tick_context
        if not isinstance(context, CARLATickContext):
            raise RuntimeError("tick() must succeed before collecting an observation")

        packets = self._collect_camera_packets(context.frame_id, timeout)
        for name, packet in packets.items():
            packet_timestamp = getattr(packet, "timestamp", None)
            if packet_timestamp is not None and not math.isclose(
                float(packet_timestamp),
                context.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1e-4,
            ):
                self._camera_timestamp_mismatches += 1
                raise RuntimeError(
                    f"Camera {name} timestamp {float(packet_timestamp):.6f}s does not "
                    f"match snapshot {context.simulation_time_s:.6f}s"
                )
        images = np.stack(
            [self._decode_camera_image(packets[name]) for name in self.camera_order],
            axis=0,
        )
        state = self.get_ego_state(context)
        if update_history:
            self.update_history(state)

        ego_pose_world = state["pose_world"].copy()
        intrinsics, extrinsics = self._camera_calibration(packets, ego_pose_world)
        identified_history = [
            history_state
            for history_state in self.history_buffer
            if "frame_id" in history_state and "simulation_time_s" in history_state
        ]
        if not identified_history:
            identified_history = [state]
        history_poses = np.stack(
            [pose_matrix_from_state(history_state) for history_state in identified_history],
            axis=0,
        )

        return SynchronizedObservation(
            frame_id=context.frame_id,
            simulation_time_s=context.simulation_time_s,
            ego_pose_world=ego_pose_world,
            ego_velocity_world=state["velocity_world"].copy(),
            camera_images=images,
            camera_ids=self.camera_ids,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            ego_history=history_poses,
            ego_history_frame_ids=tuple(
                int(history_state["frame_id"]) for history_state in identified_history
            ),
            ego_history_simulation_times_s=tuple(
                float(history_state["simulation_time_s"]) for history_state in identified_history
            ),
        )

    def get_camera_sync_stats(self):
        """Return exact-frame bundle and packet health counters."""

        return {
            "accepted_bundles": int(self._accepted_camera_bundles),
            "missing_bundles": int(self._missing_camera_bundles),
            "frame_mismatches": int(self._camera_frame_mismatches),
            "timestamp_mismatches": int(self._camera_timestamp_mismatches),
            "history_sequence_resets": int(self._history_sequence_resets),
            **self.camera_collector.stats(),
        }

    def apply_control(self, steering, throttle, brake):
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.ego_vehicle.apply_control(control)

    def get_applied_control(self):
        """Return CARLA's echoed control and gear for launch/actuator telemetry.

        The ego runs an automatic gearbox, so the deadlock and its fix are only
        observable by logging the gear CARLA actually selected against the
        throttle we commanded.  Returns ``None`` if the echo is unavailable.
        """

        if self.ego_vehicle is None:
            return None
        try:
            control = self.ego_vehicle.get_control()
        except Exception:
            return None
        return {
            "echoed_steer": float(getattr(control, "steer", 0.0)),
            "echoed_throttle": float(getattr(control, "throttle", 0.0)),
            "echoed_brake": float(getattr(control, "brake", 0.0)),
            "gear": int(getattr(control, "gear", 0)),
        }

    def tick(self):
        tick_frame = self.world.tick()
        snapshot = self.world.get_snapshot()
        snapshot_frame = int(snapshot.frame)
        frame_id = snapshot_frame if tick_frame is None else int(tick_frame)
        if snapshot_frame != frame_id:
            raise RuntimeError(
                f"CARLA tick returned frame {frame_id}, but world snapshot is "
                f"frame {snapshot_frame}"
            )

        actor_snapshot = snapshot.find(self.ego_vehicle.id)
        if actor_snapshot is None:
            raise RuntimeError(
                f"Ego vehicle {self.ego_vehicle.id} is missing from CARLA frame {frame_id}"
            )
        timestamp = snapshot.timestamp
        context = CARLATickContext(
            frame_id=frame_id,
            simulation_time_s=float(timestamp.elapsed_seconds),
            delta_seconds=float(getattr(timestamp, "delta_seconds", cfg.CONTROL_DT)),
            snapshot=snapshot,
            actor_snapshot=actor_snapshot,
            ego_transform=actor_snapshot.get_transform(),
            ego_velocity=actor_snapshot.get_velocity(),
        )
        self._last_tick_context = context
        return context

    def cleanup(self):
        print("\nCleaning up...")
        if self.client is None or self.world is None:
            return
        try:
            self.client.get_trafficmanager(self.tm_port).set_synchronous_mode(False)
        except Exception as exc:
            print(f"Warning: failed to disable TrafficManager synchronous mode: {exc}")
        if self.npc_walker_controller_ids:
            try:
                controllers = self.world.get_actors(self.npc_walker_controller_ids)
                for controller in controllers:
                    try:
                        controller.stop()
                    except Exception as exc:
                        controller_id = getattr(controller, "id", "unknown")
                        print(f"Warning: failed to stop walker controller {controller_id}: {exc}")
            except Exception as exc:
                print(f"Warning: failed to fetch walker controllers for cleanup: {exc}")
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.npc_walker_controller_ids]
                )
            except Exception as exc:
                print(f"Warning: failed to destroy walker controllers: {exc}")
        if self.npc_walker_ids:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.npc_walker_ids]
                )
            except Exception as exc:
                print(f"Warning: failed to destroy NPC walkers: {exc}")
        if self.npc_vehicle_ids:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.npc_vehicle_ids]
                )
            except Exception as exc:
                print(f"Warning: failed to destroy NPC vehicles: {exc}")
        for sensor_name, sensor in self.sensors.items():
            try:
                sensor.stop()
            except Exception as exc:
                print(f"Warning: failed to stop sensor {sensor_name}: {exc}")
            try:
                sensor.destroy()
            except Exception as exc:
                print(f"Warning: failed to destroy sensor {sensor_name}: {exc}")
        if self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except Exception as exc:
                print(f"Warning: failed to destroy ego vehicle: {exc}")
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception as exc:
            print(f"Warning: failed to restore world asynchronous mode: {exc}")

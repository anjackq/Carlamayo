"""Modular entrypoint for CARLA closed-loop control with Alpamayo."""

import argparse
import math
import os
import queue
import random
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from module import config as cfg
from module.active_plan_availability import (
    ActivePlanAvailabilityStatus,
    decide_active_plan_availability,
)
from module.carla_safety_adapter import (
    CarlaGroundTruthSafetyAdapter,
    PlanAdmissionStatus,
    RoadExecutionEnvelope,
    decide_plan_admission,
)
from module.camera_fixture import save_camera_fixture
from module.candidate_selector import (
    CandidateEvaluation,
    rank_candidate_evaluations,
)
from module.geometry import pose_matrix_from_state
from module.navigation_control import NavigationControlState
from module.pid_controller import OfficialPIDFollower
from module.plan_handoff import (
    PlanHandoffStatus,
    decide_plan_handoff,
)
from module.proposal_audit import coc_audit_fields
from module.respawn_control import RespawnMonitor
from module.runtime_metrics import JsonlWriter, RUNTIME_SCHEMA_VERSION, RuntimeMetrics
from module.safety_shield import (
    AssessmentStatus,
    ControlCommand,
    ObstacleAssessment,
    RoadContainmentAssessment,
    SafetyDecision,
    SafetyPolicy,
    StopOnlySafetyShield,
)
from module.trajectory_runtime import (
    TrajectoryValidationError,
    build_fixed_world_trajectory,
    compute_trajectory_motion_profile,
    validate_plan_alignment,
    validate_plan_for_execution,
)
from module.vlm_generate_optimization import VlmGenerateTiming
from module.visualization import VideoRecorder, create_visualization_frame
from module.carla_interface import CARLAInterface
from module.inference import (
    configure_cuda_linalg_library,
    extract_answer_text,
    extract_cot_text,
    extract_cot_texts,
    extract_trajectory_samples,
    load_model,
    prepare_model_input,
    run_inference,
    run_vqa,
    select_trajectory_by_prev_similarity,
)


def derive_pygame_ui_video_path(output_video_path):
    """Return the companion video path for recorded Pygame UI frames."""

    root, ext = os.path.splitext(output_video_path)
    return f"{root}_pygame_ui{ext or '.mp4'}"


def format_vqa_answer_preview(answer, limit=160):
    """Return a non-misleading one-line VQA answer preview for logs."""

    answer = str(answer or "").strip()
    if not answer:
        return "(empty answer)"
    suffix = "..." if len(answer) > limit else ""
    return f"{answer[:limit]}{suffix}"


def seed_runtime_randomness(seed):
    """Seed host and accelerator RNGs used by the scenario and Alpamayo."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_safety_policy():
    """Build the one policy shared by spawn preflight and runtime arbitration."""

    return SafetyPolicy(
        path_sample_spacing_m=cfg.SAFETY_PATH_SAMPLE_SPACING_M,
        lateral_clearance_m=cfg.SAFETY_LATERAL_CLEARANCE_M,
        longitudinal_clearance_m=cfg.SAFETY_LONGITUDINAL_CLEARANCE_M,
        reaction_time_s=cfg.SAFETY_REACTION_TIME_S,
        assumed_deceleration_mps2=cfg.SAFETY_ASSUMED_DECELERATION_MPS2,
        stop_buffer_m=cfg.SAFETY_STOP_BUFFER_M,
        hard_gap_m=cfg.SAFETY_HARD_GAP_M,
        ttc_threshold_s=cfg.SAFETY_TTC_THRESHOLD_S,
        minimum_closing_speed_mps=cfg.SAFETY_MINIMUM_CLOSING_SPEED_MPS,
        prediction_horizon_s=cfg.SAFETY_PREDICTION_HORIZON_S,
        prediction_time_step_s=cfg.SAFETY_PREDICTION_TIME_STEP_S,
        emergency_hold_ticks=cfg.SAFETY_EMERGENCY_HOLD_TICKS,
        clear_ticks_to_release=cfg.SAFETY_CLEAR_TICKS_TO_RELEASE,
    )


def smooth_controller_control(
    *,
    steering_raw,
    throttle_raw,
    brake_raw,
    previous_nominal,
    alpha,
    bypass_smoothing=False,
    constrained_deceleration=False,
):
    """Smooth controller intent without feeding safety overrides back into it."""

    if bypass_smoothing:
        return (
            {
                "steering": float(np.clip(steering_raw, -1.0, 1.0)),
                "throttle": 0.0,
                "brake": 1.0,
            },
            ("emergency_brake_bypass_ema",),
        )

    previous_steering = previous_nominal.get(
        "steering",
        previous_nominal.get("steer", 0.0),
    )
    steering = (1.0 - alpha) * previous_steering + alpha * steering_raw
    if constrained_deceleration and float(brake_raw) > 0.0:
        return (
            {
                "steering": float(np.clip(steering, -1.0, 1.0)),
                "throttle": 0.0,
                "brake": float(np.clip(brake_raw, 0.0, 1.0)),
            },
            (
                "steering_ema_smoothing",
                "road_deceleration_bypass_longitudinal_ema",
            ),
        )
    throttle = (1.0 - alpha) * previous_nominal["throttle"] + alpha * throttle_raw
    brake = (1.0 - alpha) * previous_nominal["brake"] + alpha * brake_raw
    if throttle >= brake:
        brake = 0.0
    else:
        throttle = 0.0
    return (
        {
            "steering": float(np.clip(steering, -1.0, 1.0)),
            "throttle": float(np.clip(throttle, 0.0, 1.0)),
            "brake": float(np.clip(brake, 0.0, 1.0)),
        },
        ("ema_smoothing", "throttle_brake_arbitration"),
    )


def resolve_tick_identity(tick_context, loop_tick_id, simulation_tick_seconds):
    """Return exact CARLA identity when available, otherwise an explicit proxy."""

    if isinstance(tick_context, dict):
        frame_id = tick_context.get("frame_id")
        simulation_time_s = tick_context.get("simulation_time_s")
    else:
        frame_id = getattr(tick_context, "frame_id", None)
        simulation_time_s = getattr(tick_context, "simulation_time_s", None)
    if frame_id is not None and simulation_time_s is not None:
        return int(frame_id), float(simulation_time_s), "exact_carla_snapshot"
    return (
        int(loop_tick_id),
        float(loop_tick_id) * float(simulation_tick_seconds),
        "loop_counter_proxy",
    )


def capture_control_observation(
    carla_if,
    tick_context,
    *,
    loop_tick_id,
    simulation_tick_seconds,
):
    """Capture one complete observation while retaining legacy test-double support."""

    frame_id, simulation_time_s, frame_id_quality = resolve_tick_identity(
        tick_context,
        loop_tick_id,
        simulation_tick_seconds,
    )
    get_synchronized = getattr(carla_if, "get_synchronized_observation", None)
    if callable(get_synchronized):
        observation = get_synchronized(tick_context, timeout=1.0, update_history=True)
        state = carla_if.get_ego_state(tick_context)
        return {
            "images": observation.camera_images,
            "camera_ids": tuple(observation.camera_ids),
            "frame_id": int(observation.frame_id),
            "simulation_time_s": float(observation.simulation_time_s),
            "frame_id_quality": "exact_carla_snapshot",
            "capture_pose_world": np.asarray(observation.ego_pose_world, dtype=np.float64).copy(),
            "state": state,
            "observation": observation,
        }

    state = carla_if.get_ego_state()
    images = carla_if.get_camera_images()
    carla_if.update_history(state)
    camera_ids = tuple(int(spec["alpamayo_id"]) for spec in cfg.CAMERA_SPECS[: len(images)])
    if len(camera_ids) != len(images):
        camera_ids = tuple(range(len(images)))
    capture_pose_world = state.get("pose_world") if isinstance(state, dict) else None
    if capture_pose_world is None:
        required_pose_keys = {"x", "y", "z"}
        capture_pose_world = (
            pose_matrix_from_state(state)
            if isinstance(state, dict) and required_pose_keys <= state.keys()
            else np.eye(4, dtype=np.float64)
        )
    return {
        "images": images,
        "camera_ids": camera_ids,
        "frame_id": frame_id,
        "simulation_time_s": simulation_time_s,
        "frame_id_quality": frame_id_quality,
        "capture_pose_world": np.asarray(capture_pose_world, dtype=np.float64).copy(),
        "state": state,
        "observation": None,
    }


def capture_initial_ui_frame(carla_if, frame_count):
    """Tick once so paused pygame starts with a real camera frame."""

    tick_context = carla_if.tick()
    frame_count += 1
    observation = capture_control_observation(
        carla_if,
        tick_context,
        loop_tick_id=frame_count,
        simulation_tick_seconds=float(cfg.CONTROL_DT),
    )
    state = observation["state"]
    images = observation["images"]
    ui_frame = None
    if len(images) > 1:
        ui_frame = images[1]
    elif len(images) > 0:
        ui_frame = images[0]

    telemetry = {
        "frame": frame_count,
        "speed_kmh": state["speed"] * 3.6,
        "steering": 0.0,
        "inference_time": 0.0,
    }
    return frame_count, ui_frame, telemetry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CARLA closed-loop control with Alpamayo (modular)."
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action="store_true",
        default=False,
        help="Use 4-bit quantized model instead of the default full-precision model.",
    )
    parser.add_argument(
        "--oom-free",
        dest="oom_free",
        action="store_true",
        default=False,
        help=(
            "Use OOM-free CPU<->GPU demand layering (third_party/oom-free-alpamayo) "
            "instead of loading the whole model on the GPU. Streams full-precision "
            "VLM/ViT/Expert layers on demand so Alpamayo fits alongside a running "
            "CARLA server. The model is loaded after CARLA is fully spawned so the "
            "residency plan reflects the VRAM CARLA actually uses. Mutually "
            "exclusive with --quantization."
        ),
    )
    parser.add_argument(
        "--oom-free-headroom-gb",
        type=float,
        default=None,
        help="OOM-free: VRAM (GB) reserved for activation spikes. Default ~3.5.",
    )
    parser.add_argument(
        "--oom-free-margin",
        type=int,
        default=None,
        help="OOM-free: safety margin subtracted from the max resident VLM layer count.",
    )
    parser.add_argument(
        "--oom-free-resident",
        type=int,
        default=None,
        help="OOM-free: force this many GPU-resident VLM layers (default: auto from free VRAM).",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run internal async inference mode (non-blocking world tick).",
    )
    parser.add_argument(
        "--empty-road",
        action="store_true",
        help=(
            "Run a diagnostic scene with a freshly loaded map, fixed ego spawn, "
            "and zero NPC vehicles or pedestrians."
        ),
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=None,
        help=(
            "Seed CARLA, Python, NumPy, and Torch scenario/model randomness. "
            "Empty-road default: 0; normal traffic default: unset."
        ),
    )
    parser.add_argument(
        "--ego-spawn-index",
        type=int,
        default=None,
        help=(
            "Use this CARLA map spawn-point index for the ego. Empty-road "
            "default: 0; normal traffic default: random clear spawn."
        ),
    )
    parser.add_argument(
        "--pygame-ui",
        action="store_true",
        help="Show a pygame camera UI with prompt input and pause/resume controls.",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "navigation", "vqa"),
        default="normal",
        help="Closed-loop inference mode. Default: normal.",
    )
    parser.add_argument(
        "--navigation-text",
        default="",
        help='Initial navigation instruction, e.g. "Turn right in 30m".',
    )
    parser.add_argument(
        "--navigation-weight",
        type=float,
        default=1.0,
        help="Navigation CFG weight. 1.0 uses normal nav conditioning; other values use CFG nav.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=cfg.NUM_TRAJ_SAMPLES,
        help=(
            "Number of Alpamayo CoC/trajectory samples generated per inference. "
            f"Default: {cfg.NUM_TRAJ_SAMPLES}; use 3 for the multi-sample diagnostic."
        ),
    )
    parser.add_argument(
        "--diffusion-temperature",
        type=float,
        default=1.0,
        help=(
            "Initial diffusion-noise temperature. Default: 1.0; Alpamayo's "
            "navigation notebook uses 0.6 as a lower-diversity diagnostic."
        ),
    )
    parser.add_argument(
        "--vqa-question",
        default="",
        help='Initial VQA question for --mode vqa, e.g. "Describe the scene.".',
    )
    parser.add_argument(
        "--keep-generate-logits",
        dest="disable_unused_generate_logits",
        action="store_false",
        default=True,
        help=(
            "Keep Alpamayo VLM returned logits during trajectory generation. "
            "Default disables these unused returned logits to reduce CUDA memory "
            "without changing image tokens or sampling."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Model device_map passed to from_pretrained. Default: "auto".',
    )
    parser.add_argument(
        "--cuda-linalg-library",
        choices=("default", "cusolver", "magma"),
        default="magma",
        help=(
            "Preferred CUDA linalg backend for torch.linalg calls. "
            'Default: "magma" to avoid cuSOLVER cholesky handle failures.'
        ),
    )
    parser.add_argument(
        "--debug-worker-traceback",
        action="store_true",
        help="Print async inference worker tracebacks when worker requests fail.",
    )
    parser.add_argument(
        "--telemetry-jsonl",
        metavar="PATH",
        default=None,
        help=(
            "Append schema-versioned tick, inference, respawn, and summary events "
            "to this JSONL file. Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help=("Stop after this many seconds of successful simulation ticks. Disabled by default."),
    )
    parser.add_argument(
        "--camera-alignment",
        choices=("baseline", "pose-only", "projection-only", "pose-projection"),
        default="baseline",
        help=(
            "Model-facing camera alignment mode. Default: baseline. "
            "Every non-baseline mode requires --camera-profile."
        ),
    )
    parser.add_argument(
        "--camera-profile",
        default=os.environ.get("CARLAMAYO_CAMERA_PROFILE", ""),
        metavar="PATH",
        help=(
            "Local gated camera profile. CLI takes precedence over "
            "CARLAMAYO_CAMERA_PROFILE."
        ),
    )
    parser.add_argument(
        "--capture-inference-fixture",
        default=None,
        metavar="PATH",
        help="Capture one private frozen-input fixture after 16 real history ticks.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Hold the ego stopped, save the requested fixture, and exit before inference.",
    )
    args = parser.parse_args(argv)
    if args.oom_free and args.quantization:
        parser.error("--oom-free and --quantization are mutually exclusive.")
    if args.max_episode_seconds is not None and (
        not math.isfinite(args.max_episode_seconds) or args.max_episode_seconds <= 0.0
    ):
        parser.error("--max-episode-seconds must be finite and greater than zero.")
    if args.scenario_seed is not None and not (0 <= args.scenario_seed <= cfg.MAX_SCENARIO_SEED):
        parser.error(f"--scenario-seed must be within [0, {cfg.MAX_SCENARIO_SEED}].")
    if args.ego_spawn_index is not None and args.ego_spawn_index < 0:
        parser.error("--ego-spawn-index must be nonnegative.")
    if args.num_traj_samples < 1 or args.num_traj_samples > 16:
        parser.error("--num-traj-samples must be within [1, 16].")
    if not math.isfinite(args.diffusion_temperature) or args.diffusion_temperature <= 0.0:
        parser.error("--diffusion-temperature must be finite and greater than zero.")
    if args.camera_alignment != "baseline" and not args.camera_profile:
        parser.error(
            f"--camera-alignment {args.camera_alignment} requires --camera-profile "
            "or CARLAMAYO_CAMERA_PROFILE."
        )
    if args.capture_only and not args.capture_inference_fixture:
        parser.error("--capture-only requires --capture-inference-fixture.")
    repository_root = Path(__file__).resolve().parent
    for option_name, path_value in (
        ("--camera-profile", args.camera_profile),
        ("--capture-inference-fixture", args.capture_inference_fixture),
    ):
        if not path_value:
            continue
        resolved_path = Path(path_value).expanduser().resolve()
        if resolved_path == repository_root or repository_root in resolved_path.parents:
            parser.error(f"{option_name} must point outside the Git repository.")
    if args.empty_road:
        if args.scenario_seed is None:
            args.scenario_seed = cfg.EMPTY_ROAD_SCENARIO_SEED
        if args.ego_spawn_index is None:
            args.ego_spawn_index = cfg.EMPTY_ROAD_EGO_SPAWN_INDEX
        if args.mode == "navigation" and not args.navigation_text.strip():
            args.navigation_text = cfg.EMPTY_ROAD_NAVIGATION_TEXT
    args.start_paused = bool(args.pygame_ui)
    args.pygame_ui_video = derive_pygame_ui_video_path(cfg.OUTPUT_VIDEO) if args.pygame_ui else None
    return args


def main():
    args = parse_args()
    inference_interval_sec = 1.0
    seed_runtime_randomness(args.scenario_seed)

    print("=" * 60)
    print("CARLA Real-time Control with Alpamayo")
    print("=" * 60)
    if args.oom_free:
        print("Model loading: OOM-free CPU<->GPU demand layering (full-precision)")
    else:
        print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    print(f"Execution: {'ASYNC' if args.async_mode else 'SYNC'}")
    print(f"Inference mode: {args.mode}")
    print(f"Pygame UI: {'ON' if args.pygame_ui else 'OFF'}")
    print(f"CARLA map: {cfg.CARLA_MAP}")
    print(f"Scenario: {'EMPTY ROAD' if args.empty_road else 'NORMAL TRAFFIC'}")
    print(f"Scenario seed: {args.scenario_seed if args.scenario_seed is not None else 'random'}")
    print(
        "Ego spawn index: "
        f"{args.ego_spawn_index if args.ego_spawn_index is not None else 'random clear'}"
    )
    print(f"Device map: {args.device_map}")
    print(f"CUDA linalg library: {args.cuda_linalg_library}")
    print(f"Trajectory samples per inference: {args.num_traj_samples}")
    print(f"Diffusion temperature: {args.diffusion_temperature:.2f}")
    print(f"Camera alignment: {args.camera_alignment}")
    print("Auto respawn: ON after collisions")
    print(f"Runtime telemetry: {args.telemetry_jsonl or 'OFF'}")
    if args.max_episode_seconds is not None:
        print(f"Episode limit: {args.max_episode_seconds:.1f}s of simulation ticks")

    nav_state = NavigationControlState(
        args.navigation_text,
        args.navigation_weight,
        mode=args.mode,
        vqa_question=args.vqa_question,
    )
    nav_state.paused = bool(args.start_paused)
    if args.mode == "navigation" and nav_state.navigation_text:
        print(
            f"Initial navigation: {nav_state.navigation_text} "
            f"(weight={nav_state.navigation_weight:.2f})"
        )
    if args.mode == "vqa" and nav_state.vqa_question:
        print(f"Initial VQA question: {nav_state.vqa_question}")

    model = None
    processor = None
    carla_if = (
        CARLAInterface()
        if args.camera_alignment == "baseline"
        else CARLAInterface(
            camera_alignment=args.camera_alignment,
            camera_profile=args.camera_profile or None,
        )
    )

    def camera_alignment_metadata():
        getter = getattr(carla_if, "get_camera_alignment_metadata", None)
        if callable(getter):
            return getter()
        return {
            "alignment_mode": args.camera_alignment,
            "profile": None,
            "source_fov_deg": {},
            "output_projection_type": {},
            "vehicle_dimension_relative_error": {},
            "remap_valid_ratio": {},
            "preprocessing_latency": {
                "last_ms": 0.0,
                "sample_count": 0,
            },
            "camera_model_validation_status": "legacy_interface",
        }
    video_recorder = None
    pygame_ui = None
    pygame_ui_recorder = None
    latest_ui_frame = None
    latest_telemetry = {}
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    run_started_monotonic_s = None
    runtime_metrics = None
    telemetry_writer = None
    telemetry_write_failed = False
    frame_count = 0
    respawn_count = 0
    request_sequence = 0
    simulation_tick_seconds = float(cfg.CONTROL_DT)
    stop_reason = "unknown"
    run_error = None
    pending_inference = False
    inference_request_q = None
    inference_result_q = None
    inference_stop = None
    worker_thread = None
    exact_observation_count = 0
    active_sync_request = None
    fixture_capture_complete = False
    captured_fixture_identity = None
    scenario_actor_census = None
    empty_road_preflight = None
    safety_policy = build_safety_policy()
    safety_adapter = None

    def apply_vehicle_control(command):
        """The single low-level gateway for all loop-owned vehicle commands."""

        carla_if.apply_control(*command.as_tuple())

    def emit_runtime_event(
        event_type,
        *,
        aggregate_age=True,
        aggregate_rejection=True,
        **fields,
    ):
        """Aggregate an event and append it to JSONL when telemetry is enabled."""

        nonlocal telemetry_write_failed, run_started_monotonic_s, runtime_metrics
        if run_started_monotonic_s is None:
            run_started_monotonic_s = time.monotonic()
        if runtime_metrics is None:
            runtime_metrics = RuntimeMetrics()
        event = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "run_id": run_id,
            "wall_time_unix_s": time.time(),
            "wall_elapsed_s": max(0.0, time.monotonic() - run_started_monotonic_s),
            **fields,
        }
        payload = runtime_metrics.record_event(
            event_type,
            event,
            aggregate_age=aggregate_age,
            aggregate_rejection=aggregate_rejection,
        )
        if telemetry_writer is not None and not telemetry_write_failed:
            try:
                telemetry_writer.append(payload)
            except Exception as exc:
                telemetry_write_failed = True
                print(f"Warning: runtime telemetry disabled after write failure: {exc}")
        return payload

    def draw_pygame_ui(frame_rgb, telemetry):
        if pygame_ui is None:
            return
        pygame_ui.draw(frame_rgb, nav_state, telemetry)
        if pygame_ui_recorder is not None:
            pygame_ui_recorder.add_frame(pygame_ui.capture_frame())

    try:
        if args.telemetry_jsonl:
            telemetry_writer = JsonlWriter(args.telemetry_jsonl)
        if cfg.SAVE_VIDEO:
            video_recorder = VideoRecorder(
                cfg.OUTPUT_VIDEO,
                fps=cfg.VIDEO_FPS,
                preview_path=cfg.LIVE_PREVIEW_IMAGE,
                preview_interval_frames=max(1, cfg.VIDEO_FPS // 2),
            )
        if args.pygame_ui:
            from module.pygame_ui import ClosedLoopPygameUI

            pygame_ui = ClosedLoopPygameUI(
                width=cfg.PYGAME_WINDOW_WIDTH,
                height=cfg.PYGAME_WINDOW_HEIGHT,
                mode=args.mode,
            )
            pygame_ui_recorder = VideoRecorder(
                args.pygame_ui_video,
                fps=cfg.VIDEO_FPS,
            )

        print("\nLoading model...")
        configure_cuda_linalg_library(args.cuda_linalg_library)
        if args.capture_only:
            print("Capture-only mode: model loading skipped.")
        elif not args.oom_free:
            model, processor = load_model(
                args.quantization,
                device_map=args.device_map,
            )
            seed_runtime_randomness(args.scenario_seed)
            print("Model loaded!")
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
        else:
            # Defer loading until CARLA has spawned its cameras/NPCs so the
            # OOM-free plan reflects the VRAM CARLA actually leaves free.
            print("OOM-free mode: Alpamayo loads after CARLA is fully spawned.")

        carla_if.connect()
        carla_if.load_map(cfg.CARLA_MAP, force_reload=args.empty_road)
        if args.scenario_seed is not None:
            carla_if.set_scenario_seed(args.scenario_seed)
        carla_if.spawn_ego_vehicle(
            spawn_index=args.ego_spawn_index,
            center_on_driving_lane=args.empty_road,
        )
        carla_if.enable_synchronous_mode()
        fixed_delta_seconds = carla_if.world.get_settings().fixed_delta_seconds
        if fixed_delta_seconds is not None and float(fixed_delta_seconds) > 0.0:
            simulation_tick_seconds = float(fixed_delta_seconds)
        npc_vehicle_count = 0 if args.empty_road else cfg.NPC_VEHICLE_COUNT
        npc_walker_count = 0 if args.empty_road else cfg.NPC_WALKER_COUNT
        carla_if.spawn_npcs(
            num_vehicles=npc_vehicle_count,
            num_walkers=npc_walker_count,
        )
        scenario_actor_census = carla_if.get_non_ego_dynamic_actor_census()
        print(f"Dynamic actor census: {scenario_actor_census}")
        if args.empty_road and any(scenario_actor_census.values()):
            raise RuntimeError(
                f"empty-road preflight found unexpected dynamic actors: {scenario_actor_census}"
            )
        try:
            safety_adapter = CarlaGroundTruthSafetyAdapter(
                carla_if.world,
                carla_if.ego_vehicle,
                safety_policy,
            )
        except Exception as exc:
            safety_adapter = None
            if args.empty_road:
                raise RuntimeError(
                    "empty-road footprint preflight is unavailable"
                ) from exc
            print(
                "Warning: CARLA ground-truth safety adapter unavailable; "
                f"movement will fail closed ({exc})."
            )
        if args.empty_road:
            empty_road_preflight = safety_adapter.assess_ego_transform(
                carla_if.ego_vehicle.get_transform()
            )
            print(
                "Empty-road footprint preflight: "
                f"{empty_road_preflight.to_json_dict()}"
            )
            if empty_road_preflight.status is not AssessmentStatus.SAFE:
                raise RuntimeError(
                    "empty-road ego footprint is not safely contained: "
                    f"{empty_road_preflight.to_json_dict()}"
                )
        carla_if.setup_cameras()
        alignment_metadata = camera_alignment_metadata()
        print(
            "Camera model validation: "
            f"{alignment_metadata['camera_model_validation_status']}"
        )
        if alignment_metadata["profile"] is not None:
            print(f"Camera profile: {alignment_metadata['profile']}")
        carla_if.setup_collision_sensor()
        if args.capture_inference_fixture:
            # Capture must observe a stationary ego with a real, echoed brake
            # command. The loop keeps this hold until all 16 history ticks exist.
            carla_if.apply_control(0.0, 0.0, 1.0)
        time.sleep(1.0)

        if args.oom_free and not args.capture_only:
            from module.oom_offload import load_offloaded_model

            print("Loading model (OOM-free CPU<->GPU demand layering)...")
            oom_kwargs = {}
            if args.oom_free_headroom_gb is not None:
                oom_kwargs["headroom_gb"] = args.oom_free_headroom_gb
            if args.oom_free_margin is not None:
                oom_kwargs["margin"] = args.oom_free_margin
            if args.oom_free_resident is not None:
                oom_kwargs["resident_override"] = args.oom_free_resident
            model, processor = load_offloaded_model(**oom_kwargs)
            # The third-party OOM-free loader currently seeds Torch internally.
            # Restore the requested run seed before the first sampled proposal.
            seed_runtime_randomness(args.scenario_seed)
            print("Model loaded!")
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
        safety_shield = StopOnlySafetyShield(safety_policy)

        current_plan = None
        current_trajectory = None
        current_pred_xyz = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        vlm_generate_timing = VlmGenerateTiming()
        respawn_monitor = RespawnMonitor(cooldown_frames=cfg.RESPAWN_COLLISION_COOLDOWN_FRAMES)
        frame_buffer = []
        current_trajectory_ts = None
        current_plan_id = None
        current_plan_source_loop_tick_id = None
        current_plan_source_elapsed_proxy_s = None
        current_plan_admission_status = None
        latest_candidate_admission_status = None
        latest_handoff_status = None
        active_plan_availability = decide_active_plan_availability(
            active_plan_present=False,
            standard_validity=None,
            bridge_validity=None,
            alignment_validity=None,
            road_envelope=None,
            bridge_deadline_age_s=float(
                cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
            ),
            bridge_min_remaining_horizon_s=float(
                cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
            ),
        )
        current_road_envelope = None
        latest_proposal_plan_id = None
        current_carla_frame_id = None
        current_simulation_time_s = None
        current_frame_id_quality = "loop_counter_proxy"
        current_observation_entry = None
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        prev_nominal_control = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
        }
        last_applied_control_echo = None

        pending_inference = False
        pending_request_id = None
        outstanding_async_requests = {}
        last_inference_submit_simulation_time_s = None
        respawn_revision = 0
        inference_request_q = None
        inference_result_q = None
        inference_stop = None
        worker_thread = None

        def _emit_async_terminal(request_id, status, rejection_reason, result=None):
            if request_id is None:
                return False
            request = outstanding_async_requests.pop(int(request_id), None)
            if request is None:
                return False
            has_result = result is not None
            result = result or {}
            source_loop_tick_id = request["source_loop_tick_id"]
            source_age_proxy_s = max(
                0.0,
                float(frame_count - source_loop_tick_id) * simulation_tick_seconds,
            )
            request_lifetime_s = max(
                0.0,
                time.monotonic() - request["submission_monotonic_s"],
            )
            source_simulation_time = request.get("source_simulation_time_s")
            exact_source_age_s = None
            if source_simulation_time is not None and current_simulation_time_s is not None:
                exact_source_age_s = max(
                    0.0,
                    float(current_simulation_time_s) - float(source_simulation_time),
                )
            emit_runtime_event(
                "inference_result",
                aggregate_age=False,
                request_id=int(request_id),
                mode=result.get("mode", request["mode"]),
                status=status,
                rejected=rejection_reason is not None,
                rejection_reason=rejection_reason,
                source_loop_tick_id=source_loop_tick_id,
                source_elapsed_proxy_s=request["source_elapsed_proxy_s"],
                source_carla_frame_id=request.get("source_carla_frame_id"),
                source_simulation_time_s=source_simulation_time,
                arrival_loop_tick_id=int(frame_count),
                arrival_carla_frame_id=current_carla_frame_id,
                arrival_simulation_time_s=current_simulation_time_s,
                frame_id_quality=request.get("frame_id_quality", "loop_counter_proxy"),
                source_age_s=exact_source_age_s,
                source_age_proxy_s=source_age_proxy_s,
                inference_wall_latency_s=request_lifetime_s if has_result else None,
                request_lifetime_s=request_lifetime_s,
                model_inference_latency_s=result.get("model_inference_time"),
                worker_compute_latency_s=result.get("inference_time"),
                prompt_revision=request["prompt_revision"],
                respawn_revision=request["respawn_revision"],
                source_plan_id=result.get("accepted_plan_id"),
                coc_sha256=result.get("coc_sha256"),
            )
            return True

        def _emit_sync_terminal(
            *,
            request_id,
            mode,
            status,
            rejection_reason,
            submission_monotonic_s,
            model_inference_latency_s=None,
            source_plan_id=None,
            source_loop_tick_id=None,
            source_carla_frame_id=None,
            source_simulation_time_s=None,
            frame_id_quality=None,
            coc_sha256=None,
        ):
            nonlocal active_sync_request

            request_lifetime_s = max(0.0, time.monotonic() - submission_monotonic_s)
            if current_observation_entry is not None:
                if source_carla_frame_id is None:
                    source_carla_frame_id = current_observation_entry["frame_id"]
                if source_simulation_time_s is None:
                    source_simulation_time_s = current_observation_entry["simulation_time_s"]
                if frame_id_quality is None:
                    frame_id_quality = current_observation_entry["frame_id_quality"]
            source_loop_tick_id = (
                int(frame_count) if source_loop_tick_id is None else int(source_loop_tick_id)
            )
            source_elapsed_proxy_s = float(source_loop_tick_id) * simulation_tick_seconds
            source_age_s = None
            if source_simulation_time_s is not None and current_simulation_time_s is not None:
                source_age_s = max(
                    0.0,
                    float(current_simulation_time_s) - float(source_simulation_time_s),
                )
            payload = emit_runtime_event(
                "inference_result",
                aggregate_age=False,
                request_id=request_id,
                mode=mode,
                status=status,
                rejected=rejection_reason is not None,
                rejection_reason=rejection_reason,
                source_loop_tick_id=source_loop_tick_id,
                source_elapsed_proxy_s=source_elapsed_proxy_s,
                source_carla_frame_id=source_carla_frame_id,
                source_simulation_time_s=source_simulation_time_s,
                arrival_loop_tick_id=int(frame_count),
                arrival_carla_frame_id=current_carla_frame_id,
                arrival_simulation_time_s=current_simulation_time_s,
                frame_id_quality=frame_id_quality or current_frame_id_quality,
                source_age_s=source_age_s,
                source_age_proxy_s=max(
                    0.0,
                    float(frame_count - source_loop_tick_id) * simulation_tick_seconds,
                ),
                inference_wall_latency_s=request_lifetime_s,
                request_lifetime_s=request_lifetime_s,
                model_inference_latency_s=model_inference_latency_s,
                prompt_revision=nav_state.revision,
                respawn_revision=respawn_revision,
                source_plan_id=source_plan_id,
                coc_sha256=coc_sha256,
            )
            if active_sync_request is not None and int(active_sync_request["request_id"]) == int(
                request_id
            ):
                active_sync_request = None
            return payload

        def _clear_async_queues(reason):
            if inference_request_q is not None:
                while True:
                    try:
                        request = inference_request_q.get_nowait()
                    except queue.Empty:
                        break
                    if request is not None:
                        _emit_async_terminal(
                            request.get("request_id"),
                            "cancelled",
                            f"request_queue_cleared:{reason}",
                        )
            if inference_result_q is not None:
                while True:
                    try:
                        result = inference_result_q.get_nowait()
                    except queue.Empty:
                        break
                    for request_id in result.get("superseded_request_ids", ()):
                        _emit_async_terminal(
                            request_id,
                            "discarded",
                            "result_queue_superseded",
                        )
                    _emit_async_terminal(
                        result.get("request_id"),
                        "discarded",
                        f"result_queue_cleared:{reason}",
                        result=result,
                    )

        def _simulation_elapsed_proxy_s():
            return float(frame_count) * simulation_tick_seconds

        def _extract_and_audit_proposal(result, *, model_inference_latency_s):
            """Extract all generated candidates before any one is selected."""

            proposal_id = f"{run_id}:{int(result['request_id'])}"
            try:
                traj_samples = extract_trajectory_samples(result["pred_xyz"])
                if len(traj_samples) == 0:
                    raise ValueError("model returned zero trajectory candidates")
                selected_idx, similarity_scores = select_trajectory_by_prev_similarity(
                    traj_samples,
                    prev_selected_trajectory,
                )
            except Exception as exc:
                cot_text = extract_cot_text(result.get("extra"))
                audit = coc_audit_fields(cot_text)
                result["coc_sha256"] = audit["coc_sha256"]
                emit_runtime_event(
                    "alpamayo_proposal",
                    layer="ALPAMAYO_PROPOSAL",
                    proposal_id=proposal_id,
                    request_id=int(result["request_id"]),
                    mode=result.get("mode", args.mode),
                    source_loop_tick_id=int(result["source_loop_tick_id"]),
                    source_carla_frame_id=int(result["source_carla_frame_id"]),
                    source_simulation_time_s=float(result["source_simulation_time_s"]),
                    frame_id_quality=result.get("frame_id_quality", "loop_counter_proxy"),
                    camera_ids=list(result.get("camera_ids", tuple())),
                    prompt_revision=int(result.get("prompt_revision", nav_state.revision)),
                    respawn_revision=int(result.get("respawn_revision", respawn_revision)),
                    selected_candidate_index=None,
                    candidate_count=0,
                    candidate_similarity_scores=[],
                    candidate_trajectories_model=None,
                    model_inference_latency_s=float(model_inference_latency_s),
                    extraction_error=str(exc),
                    **audit,
                )
                raise TrajectoryValidationError(f"trajectory_extraction_error:{exc}") from exc
            candidate_cot_texts = extract_cot_texts(
                result.get("extra"),
                candidate_count=len(traj_samples),
            )
            candidate_coc_audits = [coc_audit_fields(text) for text in candidate_cot_texts]
            return {
                "proposal_id": proposal_id,
                "trajectory_samples": traj_samples,
                "preselected_index": int(selected_idx),
                "similarity_scores": tuple(similarity_scores),
                "candidate_cot_texts": tuple(candidate_cot_texts),
                "candidate_coc_audits": tuple(candidate_coc_audits),
                "model_inference_latency_s": float(model_inference_latency_s),
            }

        def _emit_ranked_proposal_audit(result, proposal, selection):
            """Log model outputs once, after road-aware candidate selection."""

            selected_idx = int(selection.selected_index)
            cot_text = proposal["candidate_cot_texts"][selected_idx]
            audit = proposal["candidate_coc_audits"][selected_idx]
            result["coc_sha256"] = audit["coc_sha256"]
            emit_runtime_event(
                "alpamayo_proposal",
                layer="ALPAMAYO_PROPOSAL",
                proposal_id=proposal["proposal_id"],
                request_id=int(result["request_id"]),
                mode=result.get("mode", args.mode),
                source_loop_tick_id=int(result["source_loop_tick_id"]),
                source_carla_frame_id=int(result["source_carla_frame_id"]),
                source_simulation_time_s=float(result["source_simulation_time_s"]),
                frame_id_quality=result.get("frame_id_quality", "loop_counter_proxy"),
                camera_ids=list(result.get("camera_ids", tuple())),
                prompt_revision=int(result.get("prompt_revision", nav_state.revision)),
                respawn_revision=int(result.get("respawn_revision", respawn_revision)),
                selected_candidate_index=int(selected_idx),
                preselected_candidate_index=int(proposal["preselected_index"]),
                candidate_count=int(len(proposal["trajectory_samples"])),
                candidate_similarity_scores=[
                    None if value is None else float(value)
                    for value in proposal["similarity_scores"]
                ],
                candidate_trajectories_model=np.asarray(
                    proposal["trajectory_samples"],
                    dtype=np.float64,
                ),
                candidate_coc_texts_full=list(proposal["candidate_cot_texts"]),
                candidate_coc_sha256=[
                    candidate_audit["coc_sha256"]
                    for candidate_audit in proposal["candidate_coc_audits"]
                ],
                candidate_selection=selection.to_json_dict(),
                model_inference_latency_s=proposal["model_inference_latency_s"],
                **audit,
            )
            return {
                **proposal,
                "selected_index": int(selected_idx),
                "selected_points": proposal["trajectory_samples"][selected_idx],
                "coc_text": cot_text,
                "coc_sha256": audit["coc_sha256"],
            }

        def _audit_discarded_generated_result(result, discard_reason):
            """Preserve generated reasoning even when revisions make a result stale."""

            if "error" in result or result.get("mode") == "vqa":
                return
            cot_text = extract_cot_text(result.get("extra"))
            audit = coc_audit_fields(cot_text)
            proposal_id = f"{run_id}:{int(result['request_id'])}"
            result["coc_sha256"] = audit["coc_sha256"]
            emit_runtime_event(
                "alpamayo_proposal",
                layer="ALPAMAYO_PROPOSAL",
                proposal_id=proposal_id,
                proposal_status="discarded_before_trajectory_validation",
                discard_reason=discard_reason,
                request_id=int(result["request_id"]),
                mode=result.get("mode", args.mode),
                source_loop_tick_id=int(result["source_loop_tick_id"]),
                source_carla_frame_id=int(result["source_carla_frame_id"]),
                source_simulation_time_s=float(result["source_simulation_time_s"]),
                frame_id_quality=result.get("frame_id_quality", "loop_counter_proxy"),
                camera_ids=list(result.get("camera_ids", tuple())),
                prompt_revision=int(result.get("prompt_revision", nav_state.revision)),
                respawn_revision=int(result.get("respawn_revision", respawn_revision)),
                selected_candidate_index=None,
                candidate_count=None,
                candidate_similarity_scores=None,
                candidate_trajectories_model=None,
                model_inference_latency_s=float(
                    result.get("model_inference_time") or result.get("inference_time") or 0.0
                ),
                **audit,
            )

        def _build_and_validate_fixed_plan(result, proposal):
            plan = build_fixed_world_trajectory(
                plan_id=proposal["proposal_id"],
                source_frame_id=int(result["source_carla_frame_id"]),
                source_simulation_time_s=float(result["source_simulation_time_s"]),
                capture_pose_world=result["capture_pose_world"],
                model_points=proposal["selected_points"],
                coc_text=proposal["coc_text"],
                prompt_revision=int(result.get("prompt_revision", nav_state.revision)),
                respawn_revision=int(result.get("respawn_revision", respawn_revision)),
                selected_candidate_index=int(proposal["selected_index"]),
            )
            validity = validate_plan_for_execution(
                plan,
                float(current_simulation_time_s),
                current_prompt_revision=nav_state.revision,
                current_respawn_revision=respawn_revision,
            )
            alignment = validate_plan_alignment(
                plan,
                float(current_simulation_time_s),
                current_observation_entry["capture_pose_world"],
            )
            emit_runtime_event(
                "plan_validation",
                aggregate_age=False,
                aggregate_rejection=False,
                layer="ALPAMAYO_PROPOSAL",
                proposal_id=plan.plan_id,
                source_carla_frame_id=plan.source_frame_id,
                source_simulation_time_s=plan.source_simulation_time_s,
                arrival_carla_frame_id=current_carla_frame_id,
                arrival_simulation_time_s=current_simulation_time_s,
                valid=bool(validity.valid and alignment.valid),
                rejection_reason=(validity.rejection_reason or alignment.rejection_reason),
                source_age_s=validity.source_age_s,
                remaining_horizon_s=validity.remaining_horizon_s,
                tracking_error_m=alignment.tracking_error_m,
                heading_error_deg=alignment.heading_error_deg,
                stop_requested=bool(plan.stop_requested),
                terminal_stop_index=plan.terminal_stop_index,
                point_count=int(len(plan.world_points)),
                capture_pose_world=plan.capture_pose_world,
                selected_trajectory_model=plan.model_points,
                selected_trajectory_world=plan.world_points,
                waypoint_times_s=plan.waypoint_times_s,
                coc_sha256=proposal["coc_sha256"],
            )
            if not validity.valid:
                raise TrajectoryValidationError(str(validity.rejection_reason))
            if not alignment.valid:
                raise TrajectoryValidationError(str(alignment.rejection_reason))
            return plan, validity

        def _clear_active_plan_state():
            nonlocal current_plan, current_trajectory, current_plan_id
            nonlocal current_plan_source_loop_tick_id
            nonlocal current_plan_source_elapsed_proxy_s
            nonlocal current_road_envelope
            nonlocal current_plan_admission_status
            nonlocal active_plan_availability

            current_plan = None
            current_trajectory = None
            current_plan_id = None
            current_plan_source_loop_tick_id = None
            current_plan_source_elapsed_proxy_s = None
            current_road_envelope = None
            current_plan_admission_status = None
            active_plan_availability = decide_active_plan_availability(
                active_plan_present=False,
                standard_validity=None,
                bridge_validity=None,
                alignment_validity=None,
                road_envelope=None,
                bridge_deadline_age_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                ),
                bridge_min_remaining_horizon_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                ),
            )
            pid_follower.reset_plan_progress()

        def _assess_plan_road_envelope(plan):
            if safety_adapter is None or tick_context is None:
                raise RuntimeError("carla_safety_adapter_unavailable")
            assess_plan_road = getattr(safety_adapter, "assess_plan_road", None)
            if callable(assess_plan_road):
                return assess_plan_road(tick_context=tick_context, plan=plan)
            legacy = safety_adapter.assess(tick_context=tick_context, plan=plan)
            envelope = getattr(legacy, "road_envelope", None)
            if envelope is not None:
                return envelope
            current = getattr(legacy, "current_ego_road", None) or legacy.road
            path = getattr(legacy, "proposed_path_road", None) or legacy.road
            return RoadExecutionEnvelope(
                current_ego_road=current,
                near_term_path_road=path,
                full_path_road=path,
                last_safe_waypoint_index=(
                    len(plan.world_points) - 1
                    if legacy.road.status is AssessmentStatus.SAFE
                    else None
                ),
                time_to_first_bad_s=None,
                distance_to_first_bad_m=None,
                target_speed_cap_mps=None,
                emergency_required=legacy.road.status is not AssessmentStatus.SAFE,
            )

        def _active_plan_validity_window(plan):
            standard = validate_plan_for_execution(
                plan,
                float(current_simulation_time_s),
                current_prompt_revision=nav_state.revision,
                current_respawn_revision=respawn_revision,
            )
            alignment = validate_plan_alignment(
                plan,
                float(current_simulation_time_s),
                current_observation_entry["capture_pose_world"],
            )
            bridge = None
            if not standard.valid:
                bridge = validate_plan_for_execution(
                    plan,
                    float(current_simulation_time_s),
                    current_prompt_revision=nav_state.revision,
                    current_respawn_revision=respawn_revision,
                    maximum_plan_age_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                    ),
                    minimum_remaining_horizon_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                    ),
                )
            return standard, bridge, alignment

        def _active_plan_road_envelope():
            if (
                current_plan is None
                or safety_adapter is None
                or tick_context is None
            ):
                return None
            standard, bridge, active_alignment = _active_plan_validity_window(
                current_plan
            )
            if not active_alignment.valid:
                return None
            try:
                envelope = _assess_plan_road_envelope(current_plan)
            except Exception:
                return None
            availability = decide_active_plan_availability(
                active_plan_present=True,
                standard_validity=standard,
                bridge_validity=bridge,
                alignment_validity=active_alignment,
                road_envelope=envelope,
                bridge_deadline_age_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                ),
                bridge_min_remaining_horizon_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                ),
            )
            return envelope if availability.execution_allowed else None

        def _emit_selected_plan_admission(
            candidate_plan,
            admission,
            candidate_envelope,
            active_envelope,
            admission_error,
        ):
            nonlocal latest_candidate_admission_status, latest_proposal_plan_id

            latest_proposal_plan_id = candidate_plan.plan_id
            latest_candidate_admission_status = admission
            emit_runtime_event(
                "plan_admission",
                aggregate_age=False,
                aggregate_rejection=False,
                layer="ALPAMAYO_PROPOSAL",
                proposal_id=candidate_plan.plan_id,
                selected_candidate_index=candidate_plan.selected_candidate_index,
                active_plan_id=(current_plan.plan_id if current_plan is not None else None),
                admission_status=admission.value,
                admitted=admission in (
                    PlanAdmissionStatus.ACCEPT_FULLY_SAFE,
                    PlanAdmissionStatus.ACCEPT_SAFE_PREFIX,
                    PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX,
                ),
                admission_error=admission_error,
                candidate_road_envelope=(
                    candidate_envelope.to_json_dict()
                    if candidate_envelope is not None
                    else None
                ),
                retained_active_road_envelope=(
                    active_envelope.to_json_dict()
                    if admission is PlanAdmissionStatus.REJECT_RETAIN_ACTIVE
                    and active_envelope is not None
                    else None
                ),
            )

        def _candidate_path_features(points):
            array = np.asarray(points, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
                return 0.0, 0.0
            progress = float(np.max(array[:, 0], initial=0.0))
            lateral_values = array[:, 1]
            lateral_index = int(np.argmax(np.abs(lateral_values)))
            return progress, float(lateral_values[lateral_index])

        def _verified_empty_road():
            if not args.empty_road or safety_adapter is None or tick_context is None:
                return False
            try:
                actors = safety_adapter._actors_from_context(tick_context)
            except Exception:
                return False
            return actors is not None and len(actors) == 0

        def _select_road_aware_candidate(result, proposal):
            """Validate and road-rank every sample before choosing one."""

            nonlocal latest_candidate_admission_status, latest_proposal_plan_id

            selection_started_s = time.perf_counter()
            active_envelope = _active_plan_road_envelope()
            candidate_records = []
            candidate_count = len(proposal["trajectory_samples"])

            for candidate_index in range(candidate_count):
                plan_id = (
                    proposal["proposal_id"]
                    if candidate_count == 1
                    else f"{proposal['proposal_id']}/candidate-{candidate_index}"
                )
                points = proposal["trajectory_samples"][candidate_index]
                coc_text = proposal["candidate_cot_texts"][candidate_index]
                coc_audit = proposal["candidate_coc_audits"][candidate_index]
                candidate_proposal = {
                    **proposal,
                    "proposal_id": plan_id,
                    "selected_index": candidate_index,
                    "selected_points": points,
                    "coc_text": coc_text,
                    "coc_sha256": coc_audit["coc_sha256"],
                }
                plan = None
                envelope = None
                admission = None
                admission_error = None
                rejection_reason = None
                motion_profile = None
                motion_profile_compute_ms = None
                try:
                    plan, _validity = _build_and_validate_fixed_plan(
                        result,
                        candidate_proposal,
                    )
                except TrajectoryValidationError as exc:
                    rejection_reason = exc.reason
                except Exception as exc:
                    rejection_reason = (
                        f"candidate_validation_error:{type(exc).__name__}"
                    )
                else:
                    motion_profile_started_s = time.perf_counter()
                    try:
                        motion_profile = compute_trajectory_motion_profile(
                            plan,
                            float(current_simulation_time_s),
                        )
                    except TrajectoryValidationError as exc:
                        rejection_reason = exc.reason
                    finally:
                        motion_profile_compute_ms = (
                            time.perf_counter() - motion_profile_started_s
                        ) * 1000.0
                    try:
                        if rejection_reason is None:
                            envelope = _assess_plan_road_envelope(plan)
                    except Exception as exc:
                        admission_error = (
                            "carla_safety_adapter_unavailable"
                            if str(exc) == "carla_safety_adapter_unavailable"
                            else f"candidate_road_assessment_error:{type(exc).__name__}"
                        )
                        admission = PlanAdmissionStatus.REJECT_FALLBACK_STOP
                    else:
                        if envelope is None:
                            admission = None
                        else:
                            admission = decide_plan_admission(
                                envelope,
                                active_envelope,
                            )
                    if admission not in (
                        PlanAdmissionStatus.ACCEPT_FULLY_SAFE,
                        PlanAdmissionStatus.ACCEPT_SAFE_PREFIX,
                        PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX,
                    ):
                        rejection_reason = f"road_admission:{admission.value}"

                progress_m, lateral_m = _candidate_path_features(points)
                margin_m = None
                if envelope is not None:
                    margin_m = envelope.full_path_road.min_margin_m
                continuity_m = proposal["similarity_scores"][candidate_index]
                if continuity_m is not None:
                    try:
                        continuity_m = float(continuity_m)
                    except (TypeError, ValueError):
                        continuity_m = None
                if continuity_m is not None and not math.isfinite(continuity_m):
                    continuity_m = None
                reserve = (
                    envelope.stopping_reserve_profile
                    if envelope is not None
                    else None
                )
                evaluation = CandidateEvaluation(
                    candidate_index=candidate_index,
                    plan_id=plan_id,
                    admission_status=(
                        admission.value if admission is not None else None
                    ),
                    rejection_reason=rejection_reason,
                    stop_requested=bool(plan.stop_requested) if plan is not None else False,
                    forward_progress_m=progress_m,
                    representative_lateral_m=lateral_m,
                    full_path_margin_m=margin_m,
                    continuity_m=continuity_m,
                    motion_class=(
                        motion_profile.motion_class.value
                        if motion_profile is not None
                        else None
                    ),
                    initial_target_speed_mps=(
                        motion_profile.initial_target_speed_mps
                        if motion_profile is not None
                        else None
                    ),
                    stopping_reserve_status=(
                        reserve.status.value if reserve is not None else None
                    ),
                    stopping_reserve_m=(
                        reserve.stopping_reserve_m if reserve is not None else None
                    ),
                    time_to_first_bad_s=(
                        envelope.time_to_first_bad_s
                        if envelope is not None
                        else None
                    ),
                )
                candidate_records.append(
                    {
                        "candidate_proposal": candidate_proposal,
                        "plan": plan,
                        "envelope": envelope,
                        "admission": admission,
                        "admission_error": admission_error,
                        "evaluation": evaluation,
                        "motion_profile": motion_profile,
                        "motion_profile_compute_ms": motion_profile_compute_ms,
                    }
                )

            prefer_moving = _verified_empty_road()
            selection = rank_candidate_evaluations(
                [record["evaluation"] for record in candidate_records],
                navigation_text=(
                    result.get("navigation_text")
                    if result.get("mode", args.mode) == "navigation"
                    else None
                ),
                prefer_moving=prefer_moving,
                current_speed_mps=float(state["speed"]),
            )
            ranks_by_index = {
                ranked.evaluation.candidate_index: ranked
                for ranked in selection.ranked_candidates
            }
            for record in candidate_records:
                candidate_index = record["evaluation"].candidate_index
                ranked = ranks_by_index[candidate_index]
                emit_runtime_event(
                    "candidate_evaluation",
                    aggregate_age=False,
                    aggregate_rejection=False,
                    layer="ALPAMAYO_PROPOSAL",
                    proposal_id=proposal["proposal_id"],
                    candidate_plan_id=record["evaluation"].plan_id,
                    selected=candidate_index == selection.selected_index,
                    candidate_road_envelope=(
                        record["envelope"].to_json_dict()
                        if record["envelope"] is not None
                        else None
                    ),
                    admission_error=record["admission_error"],
                    trajectory_motion_profile=(
                        record["motion_profile"].to_json_dict()
                        if record["motion_profile"] is not None
                        else None
                    ),
                    motion_profile_compute_ms=record[
                        "motion_profile_compute_ms"
                    ],
                    stopping_reserve_compute_ms=(
                        record["envelope"].stopping_reserve_compute_ms
                        if record["envelope"] is not None
                        else None
                    ),
                    motion_reserve_compute_ms=(
                        record["motion_profile_compute_ms"]
                        + record["envelope"].stopping_reserve_compute_ms
                        if record["motion_profile_compute_ms"] is not None
                        and record["envelope"] is not None
                        and record["envelope"].stopping_reserve_compute_ms
                        is not None
                        else None
                    ),
                    verified_empty_road=prefer_moving,
                    **ranked.to_json_dict(),
                )
            emit_runtime_event(
                "candidate_selection",
                aggregate_age=False,
                aggregate_rejection=False,
                layer="ALPAMAYO_PROPOSAL",
                proposal_id=proposal["proposal_id"],
                preselected_candidate_index=proposal["preselected_index"],
                selection_latency_ms=(
                    (time.perf_counter() - selection_started_s) * 1000.0
                ),
                **selection.to_json_dict(),
            )

            selected_record = next(
                record
                for record in candidate_records
                if record["evaluation"].candidate_index == selection.selected_index
            )
            ranked_proposal = _emit_ranked_proposal_audit(
                result,
                proposal,
                selection,
            )
            latest_proposal_plan_id = selected_record["evaluation"].plan_id
            if selected_record["admission"] is not None:
                latest_candidate_admission_status = selected_record["admission"]
            if (
                selected_record["plan"] is not None
                and selected_record["admission"] is not None
            ):
                _emit_selected_plan_admission(
                    selected_record["plan"],
                    selected_record["admission"],
                    selected_record["envelope"],
                    active_envelope,
                    selected_record["admission_error"],
                )
            return {
                "proposal": ranked_proposal,
                "plan": selected_record["plan"],
                "envelope": selected_record["envelope"],
                "admission": selected_record["admission"],
                "motion_profile": selected_record["motion_profile"],
                "verified_empty_road": prefer_moving,
                "active_envelope": active_envelope,
                "rejection_reason": selected_record["evaluation"].rejection_reason,
            }

        def _road_envelope_executable(envelope):
            if envelope is None or envelope.emergency_required:
                return False
            clearance = (
                envelope.current_ego_clearance_road
                or envelope.current_ego_road
            )
            return bool(
                envelope.current_ego_road.status is AssessmentStatus.SAFE
                and (
                    clearance.status is AssessmentStatus.SAFE
                    or envelope.recovery_required
                )
                and envelope.last_safe_waypoint_index is not None
            )

        def _apply_selected_plan_outcome(
            outcome,
            *,
            inference_time_s,
            trajectory_timestamp_s,
            source_loop_tick_id,
            source_elapsed_proxy_s,
        ):
            """Apply one sync/async selection through the same bounded handoff."""

            nonlocal current_plan, current_trajectory, current_plan_id
            nonlocal current_selected_traj_idx, prev_selected_trajectory
            nonlocal current_pred_xyz, current_cot, current_inference_time
            nonlocal current_trajectory_ts, current_road_envelope
            nonlocal current_plan_source_loop_tick_id
            nonlocal current_plan_source_elapsed_proxy_s
            nonlocal current_plan_admission_status
            nonlocal latest_handoff_status

            proposal = outcome["proposal"]
            candidate_plan = outcome["plan"]
            candidate_envelope = outcome["envelope"]
            candidate_admission = outcome["admission"]
            accepted = candidate_plan is not None and candidate_admission in (
                PlanAdmissionStatus.ACCEPT_FULLY_SAFE,
                PlanAdmissionStatus.ACCEPT_SAFE_PREFIX,
                PlanAdmissionStatus.ACCEPT_RECOVERY_PREFIX,
            )
            current_inference_time = float(inference_time_s)
            if not accepted:
                rejection_reason = (
                    outcome["rejection_reason"]
                    or "all_candidates_failed_validation"
                )
                if candidate_admission is PlanAdmissionStatus.REJECT_FALLBACK_STOP:
                    _clear_active_plan_state()
                return {
                    "status": "rejected_plan",
                    "rejection_reason": rejection_reason,
                    "activated": False,
                    "retained": (
                        candidate_admission
                        is PlanAdmissionStatus.REJECT_RETAIN_ACTIVE
                    ),
                    "handoff": None,
                }

            active_envelope = outcome.get("active_envelope")
            active_motion = None
            active_executable = bool(
                current_plan is not None
                and _road_envelope_executable(active_envelope)
            )
            if active_executable:
                try:
                    active_motion = compute_trajectory_motion_profile(
                        current_plan,
                        float(current_simulation_time_s),
                    )
                except TrajectoryValidationError:
                    active_executable = False

            candidate_motion = outcome.get("motion_profile")
            candidate_reserve = (
                candidate_envelope.stopping_reserve_profile
                if candidate_envelope is not None
                else None
            )
            active_reserve = (
                active_envelope.stopping_reserve_profile
                if active_envelope is not None
                else None
            )
            handoff = decide_plan_handoff(
                candidate_plan_id=candidate_plan.plan_id,
                candidate_admission_status=candidate_admission,
                candidate_motion_class=(
                    candidate_motion.motion_class
                    if candidate_motion is not None
                    else None
                ),
                candidate_reserve_status=(
                    candidate_reserve.status
                    if candidate_reserve is not None
                    else None
                ),
                candidate_explicit_stop=bool(candidate_plan.stop_requested),
                active_plan_id=(
                    current_plan.plan_id if current_plan is not None else None
                ),
                active_admission_status=current_plan_admission_status,
                active_motion_class=(
                    active_motion.motion_class if active_motion is not None else None
                ),
                active_reserve_status=(
                    active_reserve.status if active_reserve is not None else None
                ),
                active_remaining_horizon_s=(
                    active_motion.remaining_horizon_s
                    if active_motion is not None
                    else None
                ),
                active_executable=active_executable,
                verified_empty_road=bool(outcome.get("verified_empty_road")),
                retention_deadline_s=(
                    float(cfg.TRAJECTORY_MIN_REMAINING_HORIZON_S)
                    + float(inference_interval_sec)
                ),
            )
            latest_handoff_status = handoff.status
            emit_runtime_event(
                "plan_handoff",
                aggregate_age=False,
                aggregate_rejection=False,
                layer="CONTROLLER_EXECUTION",
                **handoff.to_json_dict(),
                candidate_motion_profile=(
                    candidate_motion.to_json_dict()
                    if candidate_motion is not None
                    else None
                ),
                active_motion_profile=(
                    active_motion.to_json_dict()
                    if active_motion is not None
                    else None
                ),
                candidate_road_envelope=(
                    candidate_envelope.to_json_dict()
                    if candidate_envelope is not None
                    else None
                ),
                active_road_envelope=(
                    active_envelope.to_json_dict()
                    if active_envelope is not None
                    else None
                ),
            )
            if not handoff.activate_candidate:
                return {
                    "status": "retained_active_plan",
                    "rejection_reason": None,
                    "activated": False,
                    "retained": True,
                    "handoff": handoff,
                }

            current_plan = candidate_plan
            current_selected_traj_idx = proposal["selected_index"]
            current_trajectory = proposal["selected_points"]
            prev_selected_trajectory = current_trajectory.copy()
            current_pred_xyz = proposal["trajectory_samples"]
            current_cot = proposal["coc_text"]
            current_trajectory_ts = float(trajectory_timestamp_s)
            current_plan_id = candidate_plan.plan_id
            current_road_envelope = candidate_envelope
            current_plan_source_loop_tick_id = int(source_loop_tick_id)
            current_plan_source_elapsed_proxy_s = float(source_elapsed_proxy_s)
            current_plan_admission_status = candidate_admission
            return {
                "status": "accepted_plan",
                "rejection_reason": None,
                "activated": True,
                "retained": False,
                "handoff": handoff,
            }

        def _current_plan_timing_proxies():
            if current_plan is not None and current_simulation_time_s is not None:
                validity = validate_plan_for_execution(
                    current_plan,
                    current_simulation_time_s,
                    current_prompt_revision=nav_state.revision,
                    current_respawn_revision=respawn_revision,
                )
                return validity.source_age_s, validity.remaining_horizon_s
            if current_plan_source_loop_tick_id is None:
                return None, None
            age_s = max(
                0.0,
                float(frame_count - current_plan_source_loop_tick_id) * simulation_tick_seconds,
            )
            horizon_s = None
            if current_trajectory is not None:
                horizon_s = max(
                    0.0,
                    float(len(current_trajectory)) * simulation_tick_seconds - age_s,
                )
            return age_s, horizon_s

        def _emit_control_tick(
            *,
            state,
            controller_state,
            requested_control,
            applied_control,
            nominal_control=None,
            fallback_state="NONE",
            fallback_reason=None,
            rejection_reason=None,
            control_origin="nominal_controller",
            postprocessing=(),
            event_type="tick",
            safety_override_applied=False,
            safety_override_type=None,
            safety_override_reason=None,
            safety_reason_codes=(),
            safety_assessment=None,
            applied_control_source=None,
            controller_debug=None,
            applied_control_echo=None,
            direct_safety_trigger=False,
            latch_only=False,
            identity_available=True,
            control_timing="command_applied_after_observation",
        ):
            plan_age_s, remaining_horizon_s = _current_plan_timing_proxies()
            if nominal_control is None:
                nominal_control = requested_control
            if applied_control_source is None:
                applied_control_source = (
                    "SAFETY_OVERRIDE"
                    if safety_override_applied
                    else ("CONTROLLER_EXECUTION" if requested_control is not None else "FALLBACK")
                )
            return emit_runtime_event(
                event_type,
                loop_tick_id=int(frame_count),
                carla_frame_id=(current_carla_frame_id if identity_available else None),
                simulation_time_s=(current_simulation_time_s if identity_available else None),
                simulation_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                frame_id_quality=(
                    current_frame_id_quality if identity_available else "unavailable"
                ),
                source_carla_frame_id=(
                    current_plan.source_frame_id if current_plan is not None else None
                ),
                source_simulation_time_s=(
                    current_plan.source_simulation_time_s if current_plan is not None else None
                ),
                source_loop_tick_id=current_plan_source_loop_tick_id,
                source_elapsed_proxy_s=current_plan_source_elapsed_proxy_s,
                source_age_s=plan_age_s if current_plan is not None else None,
                plan_source_age_proxy_s=(
                    plan_age_s if current_frame_id_quality == "loop_counter_proxy" else None
                ),
                remaining_horizon_s=(remaining_horizon_s if current_plan is not None else None),
                remaining_horizon_proxy_s=(
                    remaining_horizon_s
                    if current_frame_id_quality == "loop_counter_proxy"
                    else None
                ),
                speed_mps=float(state["speed"]),
                ego_position_world=(
                    {
                        "x": float(state["x"]),
                        "y": float(state["y"]),
                        "z": float(state["z"]),
                    }
                    if all(axis in state for axis in ("x", "y", "z"))
                    else None
                ),
                ego_yaw_deg=(
                    float(state["yaw"]) if state.get("yaw") is not None else None
                ),
                ego_velocity_world=state.get("velocity_world"),
                controller_state=controller_state,
                control_timing=control_timing,
                control_origin=control_origin,
                requested_control=requested_control,
                nominal_control=nominal_control,
                applied_control=applied_control,
                applied_control_source=applied_control_source,
                postprocessing=list(postprocessing),
                fallback_state=fallback_state,
                fallback_reason=fallback_reason,
                safety_override_applied=bool(safety_override_applied),
                safety_override_type=safety_override_type,
                safety_override_reason=safety_override_reason,
                safety_reason_codes=list(safety_reason_codes),
                direct_safety_trigger=bool(direct_safety_trigger),
                latch_only=bool(latch_only),
                safety_assessment=safety_assessment,
                rejection_reason=rejection_reason,
                proposal_plan_id=latest_proposal_plan_id,
                active_plan_id=current_plan_id,
                plan_admission_status=(
                    current_plan_admission_status.value
                    if isinstance(current_plan_admission_status, PlanAdmissionStatus)
                    else current_plan_admission_status
                ),
                active_plan_admission_status=(
                    current_plan_admission_status.value
                    if isinstance(current_plan_admission_status, PlanAdmissionStatus)
                    else current_plan_admission_status
                ),
                latest_candidate_admission_status=(
                    latest_candidate_admission_status.value
                    if isinstance(
                        latest_candidate_admission_status,
                        PlanAdmissionStatus,
                    )
                    else latest_candidate_admission_status
                ),
                latest_handoff_status=(
                    latest_handoff_status.value
                    if isinstance(latest_handoff_status, PlanHandoffStatus)
                    else latest_handoff_status
                ),
                active_plan_availability_status=(
                    active_plan_availability.status.value
                    if active_plan_availability is not None
                    else ActivePlanAvailabilityStatus.NO_ACTIVE_PLAN.value
                ),
                active_plan_availability=(
                    active_plan_availability.to_json_dict()
                    if active_plan_availability is not None
                    else None
                ),
                availability_bridge_active=bool(
                    active_plan_availability is not None
                    and active_plan_availability.bridge_active
                ),
                availability_bridge_original_rejection_reason=(
                    active_plan_availability.original_rejection_reason
                    if active_plan_availability is not None
                    else None
                ),
                availability_bridge_deadline_age_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                ),
                availability_bridge_min_remaining_horizon_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                ),
                road_execution_envelope=(
                    current_road_envelope.to_json_dict()
                    if current_road_envelope is not None
                    else None
                ),
                echoed_control=applied_control_echo,
                gear=(
                    applied_control_echo.get("gear")
                    if isinstance(applied_control_echo, dict)
                    else None
                ),
                source_plan_id=current_plan_id,
                source_plan_frame_id=(
                    current_plan.source_frame_id if current_plan is not None else None
                ),
                source_plan_loop_tick_id=current_plan_source_loop_tick_id,
                prompt_revision=nav_state.revision,
                respawn_revision=respawn_revision,
                inference_pending=bool(pending_inference),
                collision_count=carla_if.get_episode_collision_count(),
                controller_debug=controller_debug,
                camera_alignment_mode=args.camera_alignment,
                camera_preprocessing_latency=camera_alignment_metadata()[
                    "preprocessing_latency"
                ],
            )

        def _normalize_control_command(value):
            """Return a finite normalized command, raising on malformed input."""

            if value is None or isinstance(value, ControlCommand):
                return value
            if isinstance(value, dict):
                steering = value.get("steering", value.get("steer"))
                return ControlCommand(
                    steering=steering,
                    throttle=value.get("throttle"),
                    brake=value.get("brake"),
                )
            steering, throttle, brake = value
            return ControlCommand(steering, throttle, brake)

        def _apply_arbitrated_control(
            *,
            state,
            tick_context,
            plan,
            controller_state,
            requested_control,
            nominal_control,
            fallback_state="NONE",
            fallback_reason=None,
            control_origin="nominal_controller",
            postprocessing=(),
            controller_debug=None,
            adapter_assessment=None,
            event_type="tick",
            emit_event=True,
            identity_available=True,
            control_timing="command_applied_after_observation",
        ):
            """Validate, safety-arbitrate, apply, and audit one vehicle command."""

            nonlocal prev_control, current_road_envelope
            nonlocal last_applied_control_echo

            arbitration_errors = []
            try:
                requested_command = _normalize_control_command(requested_control)
            except (TypeError, ValueError) as exc:
                requested_command = None
                arbitration_errors.append(f"invalid_controller_request:{exc}")
            try:
                nominal_command = _normalize_control_command(nominal_control)
            except (TypeError, ValueError) as exc:
                nominal_command = None
                arbitration_errors.append(f"invalid_nominal_control:{exc}")

            try:
                if plan is None and nominal_command is not None and not arbitration_errors:
                    current_road_envelope = None
                    safety_decision = safety_shield.decide_fallback(
                        controller_requested_control=requested_command,
                        nominal_control=nominal_command,
                        reason=fallback_reason or "no_executable_plan_context",
                    )
                    safety_source = "not_evaluated_stop_fallback"
                else:
                    try:
                        if tick_context is None or plan is None:
                            raise RuntimeError("no_executable_plan_context")
                        if safety_adapter is None:
                            raise RuntimeError("carla_safety_adapter_unavailable")
                        if adapter_assessment is None:
                            adapter_assessment = safety_adapter.assess(
                                tick_context=tick_context,
                                plan=plan,
                            )
                        road_assessment = adapter_assessment.road
                        obstacle_assessment = adapter_assessment.obstacles
                        current_road_envelope = getattr(
                            adapter_assessment,
                            "road_envelope",
                            None,
                        )
                    except Exception as exc:
                        current_road_envelope = None
                        adapter_error = f"safety_adapter_error:{type(exc).__name__}"
                        road_assessment = RoadContainmentAssessment.unknown(
                            (adapter_error,),
                            quality="carla_ground_truth",
                        )
                        obstacle_assessment = ObstacleAssessment.unknown((adapter_error,))
                    safety_decision = safety_shield.decide(
                        road=road_assessment,
                        obstacles=obstacle_assessment,
                        controller_requested_control=requested_command,
                        nominal_control=nominal_command,
                    )
                    safety_source = "carla_ground_truth"
            except Exception as exc:
                arbitration_error = f"safety_arbitration_error:{type(exc).__name__}"
                arbitration_errors.append(f"{arbitration_error}:{exc}")
                unknown_road = RoadContainmentAssessment.unknown(
                    (arbitration_error,),
                    quality="fail_closed_arbitration_guard",
                )
                unknown_obstacles = ObstacleAssessment.unknown((arbitration_error,))
                safety_decision = SafetyDecision(
                    controller_requested_control=requested_command,
                    nominal_control=nominal_command or ControlCommand.full_brake(),
                    applied_control=ControlCommand.full_brake(),
                    applied_control_source="SAFETY_OVERRIDE",
                    safety_override_applied=True,
                    override_type="EMERGENCY_BRAKE",
                    primary_reason="safety_arbitration_error",
                    reason_codes=("safety_arbitration_error", arbitration_error),
                    latched=True,
                    road_containment=unknown_road,
                    obstacle_assessment=unknown_obstacles,
                )
                safety_source = "fail_closed_arbitration_guard"
            steering, throttle, brake = safety_decision.applied_control.as_tuple()
            prev_control = {
                "steer": safety_decision.applied_control.steering,
                "throttle": safety_decision.applied_control.throttle,
                "brake": safety_decision.applied_control.brake,
            }
            applied_postprocessing = tuple(postprocessing)
            if safety_decision.safety_override_applied:
                applied_postprocessing = (
                    *applied_postprocessing,
                    "stop_only_safety_override",
                )
            if arbitration_errors:
                fallback_state = "INVALID_CONTROLLER_OUTPUT"
                fallback_reason = ";".join(arbitration_errors)
                controller_state = "INVALID_CONTROLLER_OUTPUT"
                control_origin = "fail_closed_controller_validation"

            # Every control-loop command reaches CARLA through this one call.
            apply_vehicle_control(safety_decision.applied_control)
            get_applied_control = getattr(carla_if, "get_applied_control", None)
            last_applied_control_echo = (
                get_applied_control() if callable(get_applied_control) else None
            )
            latch_only = bool(
                safety_decision.safety_override_applied
                and "emergency_brake_latched" in safety_decision.reason_codes
            )
            direct_safety_trigger = bool(
                safety_decision.safety_override_applied and not latch_only
            )
            if emit_event:
                _emit_control_tick(
                    state=state,
                    controller_state=controller_state,
                    requested_control=(
                        requested_command.to_json_dict() if requested_command is not None else None
                    ),
                    nominal_control=(
                        nominal_command.to_json_dict() if nominal_command is not None else None
                    ),
                    applied_control=safety_decision.applied_control.to_json_dict(),
                    fallback_state=fallback_state,
                    fallback_reason=fallback_reason,
                    control_origin=control_origin,
                    postprocessing=applied_postprocessing,
                    event_type=event_type,
                    controller_debug=controller_debug,
                    safety_override_applied=safety_decision.safety_override_applied,
                    safety_override_type=safety_decision.override_type,
                    safety_override_reason=safety_decision.primary_reason,
                    safety_reason_codes=safety_decision.reason_codes,
                    direct_safety_trigger=direct_safety_trigger,
                    latch_only=latch_only,
                    safety_assessment={
                        "safety_source": safety_source,
                        "adapter_assessment_available": adapter_assessment is not None,
                        "arbitration_errors": arbitration_errors,
                        "current_ego_road_containment": (
                            getattr(
                                adapter_assessment,
                                "current_ego_road",
                                None,
                            ).to_json_dict()
                            if adapter_assessment is not None
                            and getattr(
                                adapter_assessment,
                                "current_ego_road",
                                None,
                            )
                            is not None
                            else None
                        ),
                        "proposed_path_road_containment": (
                            getattr(
                                adapter_assessment,
                                "proposed_path_road",
                                None,
                            ).to_json_dict()
                            if adapter_assessment is not None
                            and getattr(
                                adapter_assessment,
                                "proposed_path_road",
                                None,
                            )
                            is not None
                            else None
                        ),
                        "road_execution_envelope": (
                            getattr(
                                adapter_assessment,
                                "road_envelope",
                                None,
                            ).to_json_dict()
                            if adapter_assessment is not None
                            and getattr(
                                adapter_assessment,
                                "road_envelope",
                                None,
                            )
                            is not None
                            else None
                        ),
                        **safety_decision.to_json_dict(),
                    },
                    applied_control_source=safety_decision.applied_control_source,
                    applied_control_echo=last_applied_control_echo,
                    identity_available=identity_available,
                    control_timing=control_timing,
                )
            return safety_decision

        def _auto_respawn(reason):
            nonlocal current_plan
            nonlocal current_trajectory, current_pred_xyz, prev_selected_trajectory
            nonlocal current_selected_traj_idx, current_cot, current_inference_time
            nonlocal current_trajectory_ts, prev_control, prev_nominal_control
            nonlocal pending_inference, pid_follower
            nonlocal pending_request_id
            nonlocal respawn_revision, last_vqa_submitted_revision, last_vqa_completed_revision
            nonlocal current_plan_id, current_plan_source_loop_tick_id
            nonlocal current_plan_source_elapsed_proxy_s, respawn_count
            nonlocal current_plan_admission_status, current_road_envelope
            nonlocal latest_proposal_plan_id, latest_candidate_admission_status
            nonlocal latest_handoff_status
            nonlocal last_applied_control_echo
            nonlocal active_plan_availability

            print(f"[Frame {frame_count}] Auto-respawn: {reason}")
            carla_if.respawn_ego_vehicle(
                spawn_index=args.ego_spawn_index,
                center_on_driving_lane=args.empty_road,
            )
            respawn_count += 1
            pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
            safety_shield.reset()
            if safety_adapter is not None:
                safety_adapter.reset(carla_if.ego_vehicle)
            current_trajectory = None
            current_plan = None
            current_pred_xyz = None
            prev_selected_trajectory = None
            current_selected_traj_idx = 0
            current_cot = ""
            current_inference_time = 0.0
            current_trajectory_ts = None
            current_plan_id = None
            current_plan_source_loop_tick_id = None
            current_plan_source_elapsed_proxy_s = None
            current_plan_admission_status = None
            current_road_envelope = None
            latest_proposal_plan_id = None
            latest_candidate_admission_status = None
            latest_handoff_status = None
            active_plan_availability = decide_active_plan_availability(
                active_plan_present=False,
                standard_validity=None,
                bridge_validity=None,
                alignment_validity=None,
                road_envelope=None,
                bridge_deadline_age_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                ),
                bridge_min_remaining_horizon_s=float(
                    cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                ),
            )
            last_applied_control_echo = None
            prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
            prev_nominal_control = {
                "steering": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
            }
            pending_inference = False
            pending_request_id = None
            respawn_revision += 1
            last_vqa_submitted_revision = None
            last_vqa_completed_revision = None
            respawn_monitor.mark_respawn(
                frame_count=frame_count,
                collision_count=carla_if.get_collision_count(),
            )
            frame_buffer.clear()
            _clear_async_queues("respawn")
            emit_runtime_event(
                "respawn",
                loop_tick_id=int(frame_count),
                simulation_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                reason=reason,
                respawn_count=respawn_count,
                respawn_revision=respawn_revision,
                collision_count=carla_if.get_episode_collision_count(),
            )

        def _run_inference_with_nav_fallback(model_data, navigation_text, navigation_weight):
            def _run_once(weight):
                return run_inference(
                    model,
                    processor,
                    model_data,
                    navigation_text=navigation_text,
                    navigation_weight=weight,
                    num_traj_samples=args.num_traj_samples,
                    diffusion_temperature=args.diffusion_temperature,
                    vlm_generate_timing=vlm_generate_timing,
                    disable_unused_generate_logits=args.disable_unused_generate_logits,
                    vlm_image_pixels=cfg.VLM_IMAGE_PIXELS,
                )

            try:
                return _run_once(navigation_weight)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if navigation_text and abs(float(navigation_weight) - 1.0) > 1e-6:
                    message = (
                        "Navigation CFG ran out of CUDA memory; "
                        "falling back to normal nav conditioning with weight 1.0."
                    )
                    print(message)
                    nav_state.set_error(message)
                    return _run_once(1.0)
                raise
            except RuntimeError as exc:
                if "CUSOLVER_STATUS_INTERNAL_ERROR" not in str(exc):
                    raise
                message = (
                    "cuSOLVER linalg backend failed; switching CUDA linalg backend "
                    "to MAGMA and retrying inference once."
                )
                print(message)
                nav_state.set_error(message)
                configure_cuda_linalg_library("magma")
                return _run_once(navigation_weight)

        def _run_vqa_with_linalg_fallback(model_data, question):
            try:
                return run_vqa(model, processor, model_data, question=question)
            except RuntimeError as exc:
                if "CUSOLVER_STATUS_INTERNAL_ERROR" not in str(exc):
                    raise
                message = (
                    "cuSOLVER linalg backend failed; switching CUDA linalg backend "
                    "to MAGMA and retrying VQA once."
                )
                print(message)
                nav_state.set_error(message)
                configure_cuda_linalg_library("magma")
                return run_vqa(model, processor, model_data, question=question)

        def _current_input_arrays():
            """Build exact conditioned arrays from the current temporal buffer."""

            images_array = np.zeros(
                (
                    cfg.NUM_CAMERAS,
                    cfg.NUM_FRAMES,
                    cfg.IMG_HEIGHT,
                    cfg.IMG_WIDTH,
                    cfg.IMG_CHANNELS,
                ),
                dtype=np.uint8,
            )
            for temporal_index, frame_entry in enumerate(frame_buffer):
                frame_images = frame_entry["images"]
                for camera_index in range(cfg.NUM_CAMERAS):
                    images_array[camera_index, temporal_index] = frame_images[camera_index]
            history_xyz, history_rot = carla_if.get_history_in_local_frame()
            source_entry = frame_buffer[-1]
            return images_array, history_xyz, history_rot, source_entry

        def _prepare_current_model_input():
            """Build model input from the current complete temporal frame buffer."""

            images_array, history_xyz, history_rot, source_entry = _current_input_arrays()
            return (
                prepare_model_input(
                    images_array,
                    history_xyz,
                    history_rot,
                    camera_indices=source_entry["camera_ids"],
                ),
                source_entry,
            )

        if args.async_mode:
            inference_request_q = queue.Queue(maxsize=1)
            inference_result_q = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def _build_inference_request():
                nonlocal request_sequence
                (
                    images_array,
                    history_xyz,
                    history_rot,
                    source_entry,
                ) = _current_input_arrays()
                request_sequence += 1
                return {
                    "request_id": request_sequence,
                    "mode": args.mode,
                    "images_array": images_array,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "camera_ids": tuple(source_entry["camera_ids"]),
                    "capture_pose_world": np.asarray(
                        source_entry["capture_pose_world"], dtype=np.float64
                    ).copy(),
                    "navigation_text": nav_state.navigation_text,
                    "navigation_weight": nav_state.navigation_weight,
                    "vqa_question": nav_state.vqa_question,
                    "prompt_revision": nav_state.revision,
                    "respawn_revision": respawn_revision,
                    "source_loop_tick_id": int(frame_count),
                    "source_elapsed_proxy_s": _simulation_elapsed_proxy_s(),
                    "source_carla_frame_id": int(source_entry["frame_id"]),
                    "source_simulation_time_s": float(source_entry["simulation_time_s"]),
                    "frame_id_quality": source_entry["frame_id_quality"],
                    "submission_wall_time_s": time.time(),
                    "submission_monotonic_s": time.monotonic(),
                }

            def _inference_worker():
                while not inference_stop.is_set():
                    try:
                        req = inference_request_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if req is None:
                        break
                    req_frame = int(req["frame"])
                    model_t0 = None
                    try:
                        t0 = time.monotonic()
                        model_data = prepare_model_input(
                            req["images_array"],
                            req["history_xyz"],
                            req["history_rot"],
                            camera_indices=req["camera_ids"],
                        )
                        model_t0 = time.monotonic()
                        if req["mode"] == "vqa":
                            extra = _run_vqa_with_linalg_fallback(
                                model_data,
                                question=req["vqa_question"],
                            )
                            completed_monotonic_s = time.monotonic()
                            result = {
                                "mode": "vqa",
                                "frame_submitted": req_frame,
                                "extra": extra,
                                "answer": extract_answer_text(extra),
                                "inference_time": completed_monotonic_s - t0,
                                "model_inference_time": completed_monotonic_s - model_t0,
                                "result_ts": time.monotonic(),
                                "vqa_question": req["vqa_question"],
                                "prompt_revision": req["prompt_revision"],
                                "respawn_revision": req["respawn_revision"],
                                "request_id": req["request_id"],
                                "source_loop_tick_id": req["source_loop_tick_id"],
                                "source_elapsed_proxy_s": req["source_elapsed_proxy_s"],
                                "source_carla_frame_id": req["source_carla_frame_id"],
                                "source_simulation_time_s": req["source_simulation_time_s"],
                                "frame_id_quality": req["frame_id_quality"],
                                "capture_pose_world": req["capture_pose_world"],
                                "camera_ids": req["camera_ids"],
                                "submission_wall_time_s": req["submission_wall_time_s"],
                                "submission_monotonic_s": req["submission_monotonic_s"],
                            }
                        else:
                            navigation_text = (
                                req["navigation_text"] if req["mode"] == "navigation" else ""
                            )
                            navigation_weight = (
                                req["navigation_weight"] if req["mode"] == "navigation" else 1.0
                            )
                            pred_xyz, extra = _run_inference_with_nav_fallback(
                                model_data,
                                navigation_text=navigation_text,
                                navigation_weight=navigation_weight,
                            )
                            completed_monotonic_s = time.monotonic()
                            result = {
                                "mode": req["mode"],
                                "frame_submitted": req_frame,
                                "pred_xyz": pred_xyz,
                                "extra": extra,
                                "inference_time": completed_monotonic_s - t0,
                                "model_inference_time": completed_monotonic_s - model_t0,
                                "result_ts": time.monotonic(),
                                "navigation_text": navigation_text,
                                "navigation_weight": navigation_weight,
                                "prompt_revision": req["prompt_revision"],
                                "respawn_revision": req["respawn_revision"],
                                "request_id": req["request_id"],
                                "source_loop_tick_id": req["source_loop_tick_id"],
                                "source_elapsed_proxy_s": req["source_elapsed_proxy_s"],
                                "source_carla_frame_id": req["source_carla_frame_id"],
                                "source_simulation_time_s": req["source_simulation_time_s"],
                                "frame_id_quality": req["frame_id_quality"],
                                "capture_pose_world": req["capture_pose_world"],
                                "camera_ids": req["camera_ids"],
                                "submission_wall_time_s": req["submission_wall_time_s"],
                                "submission_monotonic_s": req["submission_monotonic_s"],
                            }
                    except Exception as e:
                        result = {
                            "frame_submitted": req_frame,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "inference_time": time.monotonic() - t0,
                            "model_inference_time": (
                                time.monotonic() - model_t0 if model_t0 is not None else None
                            ),
                            "result_ts": time.monotonic(),
                            "request_id": req.get("request_id"),
                            "source_loop_tick_id": req.get("source_loop_tick_id"),
                            "source_elapsed_proxy_s": req.get("source_elapsed_proxy_s"),
                            "source_carla_frame_id": req.get("source_carla_frame_id"),
                            "source_simulation_time_s": req.get("source_simulation_time_s"),
                            "frame_id_quality": req.get("frame_id_quality"),
                            "capture_pose_world": req.get("capture_pose_world"),
                            "camera_ids": req.get("camera_ids"),
                            "submission_wall_time_s": req.get("submission_wall_time_s"),
                            "submission_monotonic_s": req.get("submission_monotonic_s"),
                            "prompt_revision": req.get("prompt_revision"),
                            "respawn_revision": req.get("respawn_revision"),
                        }

                    superseded_request_ids = []
                    while True:
                        try:
                            superseded = inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                        superseded_request_ids.extend(superseded.get("superseded_request_ids", ()))
                        if superseded.get("request_id") is not None:
                            superseded_request_ids.append(superseded["request_id"])
                    if superseded_request_ids:
                        result["superseded_request_ids"] = superseded_request_ids
                    inference_result_q.put_nowait(result)

            worker_thread = threading.Thread(
                target=_inference_worker,
                name="alpamayo-inference-worker",
                daemon=True,
            )
            worker_thread.start()

        print("\nStarting control loop...")
        if cfg.SAVE_VIDEO:
            print(f"Recording video to: {cfg.OUTPUT_VIDEO}")
        if pygame_ui_recorder is not None:
            print(f"Recording Pygame UI to: {args.pygame_ui_video}")
        print("-" * 60)

        last_seen_nav_revision = nav_state.revision
        last_vqa_submitted_revision = None
        last_vqa_completed_revision = None
        pause_brake_active = False
        if pygame_ui is not None and nav_state.paused:
            _apply_arbitrated_control(
                state=carla_if.get_ego_state(),
                tick_context=None,
                plan=None,
                controller_state="PAUSED",
                requested_control=None,
                nominal_control=ControlCommand.full_brake(),
                fallback_state="PAUSED_BRAKE",
                fallback_reason="pygame_started_paused",
                control_origin="paused_ui",
                emit_event=False,
            )
            frame_count, latest_ui_frame, latest_telemetry = capture_initial_ui_frame(
                carla_if,
                frame_count,
            )
        emit_runtime_event(
            "episode_start",
            mode=args.mode,
            scenario="empty_road" if args.empty_road else "normal_traffic",
            scenario_seed=args.scenario_seed,
            ego_spawn_index=args.ego_spawn_index,
            requested_npc_vehicle_count=(0 if args.empty_road else cfg.NPC_VEHICLE_COUNT),
            requested_npc_walker_count=(0 if args.empty_road else cfg.NPC_WALKER_COUNT),
            dynamic_actor_census=scenario_actor_census,
            empty_road_footprint_preflight=(
                empty_road_preflight.to_json_dict()
                if empty_road_preflight is not None
                else None
            ),
            execution="async" if args.async_mode else "sync",
            num_traj_samples=args.num_traj_samples,
            diffusion_temperature=args.diffusion_temperature,
            navigation_text=nav_state.navigation_text if args.mode == "navigation" else None,
            navigation_weight=nav_state.navigation_weight if args.mode == "navigation" else None,
            camera_alignment=camera_alignment_metadata(),
            frame_id_quality=(
                "exact_carla_snapshot"
                if callable(getattr(carla_if, "get_synchronized_observation", None))
                else "loop_counter_proxy"
            ),
            exact_sensor_frame_ids_available=callable(
                getattr(carla_if, "get_synchronized_observation", None)
            ),
            simulation_tick_seconds=simulation_tick_seconds,
            initial_loop_tick_id=int(frame_count),
            max_episode_seconds=args.max_episode_seconds,
        )
        if pygame_ui is not None and nav_state.paused:
            _emit_control_tick(
                state=carla_if.get_ego_state(),
                controller_state="PAUSED",
                requested_control=None,
                applied_control={"steering": 0.0, "throttle": 0.0, "brake": 1.0},
                fallback_state="PAUSED_BRAKE",
                fallback_reason="pygame_started_paused",
                control_origin="paused_ui",
                identity_available=False,
                control_timing="command_applied_before_initial_observation",
            )
            pause_brake_active = True
            draw_pygame_ui(latest_ui_frame, latest_telemetry)

        while True:
            if (
                args.max_episode_seconds is not None
                and _simulation_elapsed_proxy_s() >= args.max_episode_seconds
            ):
                stop_reason = "max_episode_seconds"
                print(
                    "\nEpisode duration limit reached: "
                    f"{_simulation_elapsed_proxy_s():.1f}s of simulation ticks."
                )
                break
            if pygame_ui is not None:
                if not pygame_ui.process_events(nav_state):
                    print("\nPygame UI requested shutdown.")
                    stop_reason = "pygame_shutdown"
                    break
                if nav_state.revision != last_seen_nav_revision:
                    if args.mode == "navigation":
                        print(
                            f"Navigation updated: {nav_state.navigation_text or '(none)'} "
                            f"(weight={nav_state.navigation_weight:.2f})"
                        )
                    elif args.mode == "vqa":
                        print(f"VQA question updated: {nav_state.vqa_question or '(none)'}")
                    prev_selected_trajectory = None
                    current_plan = None
                    current_trajectory = None
                    current_pred_xyz = None
                    current_trajectory_ts = None
                    current_plan_id = None
                    current_plan_source_loop_tick_id = None
                    current_plan_source_elapsed_proxy_s = None
                    current_plan_admission_status = None
                    current_road_envelope = None
                    latest_proposal_plan_id = None
                    latest_candidate_admission_status = None
                    latest_handoff_status = None
                    active_plan_availability = decide_active_plan_availability(
                        active_plan_present=False,
                        standard_validity=None,
                        bridge_validity=None,
                        alignment_validity=None,
                        road_envelope=None,
                        bridge_deadline_age_s=float(
                            cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                        ),
                        bridge_min_remaining_horizon_s=float(
                            cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                        ),
                    )
                    pending_inference = False
                    pending_request_id = None
                    safety_shield.reset()
                    if safety_adapter is not None:
                        safety_adapter.reset(carla_if.ego_vehicle)
                    emit_runtime_event(
                        "prompt_revision_changed",
                        loop_tick_id=int(frame_count),
                        prompt_revision=nav_state.revision,
                        mode=args.mode,
                    )
                    last_seen_nav_revision = nav_state.revision
                if nav_state.paused:
                    if not pause_brake_active:
                        _apply_arbitrated_control(
                            state=carla_if.get_ego_state(),
                            tick_context=None,
                            plan=None,
                            controller_state="PAUSED",
                            requested_control=None,
                            nominal_control=ControlCommand.full_brake(),
                            fallback_state="PAUSED_BRAKE",
                            fallback_reason="pygame_paused",
                            control_origin="paused_ui",
                            event_type="control_command",
                        )
                        pause_brake_active = True
                    latest_telemetry = {
                        **latest_telemetry,
                        "frame": frame_count,
                        "inference_time": current_inference_time,
                    }
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)
                    continue
                pause_brake_active = False

            tick_context = carla_if.tick()
            frame_count += 1
            (
                current_carla_frame_id,
                current_simulation_time_s,
                current_frame_id_quality,
            ) = resolve_tick_identity(
                tick_context,
                frame_count,
                simulation_tick_seconds,
            )

            try:
                state = carla_if.get_ego_state(tick_context)
            except TypeError:
                state = carla_if.get_ego_state()

            collision_decision = respawn_monitor.check_collision(
                frame_count=frame_count,
                collision_count=carla_if.get_collision_count(),
                last_collision_event=carla_if.get_last_collision_event(),
            )
            if collision_decision.should_respawn:
                _auto_respawn(collision_decision.reason)
                respawn_state = carla_if.get_ego_state()
                _apply_arbitrated_control(
                    state=respawn_state,
                    tick_context=None,
                    plan=None,
                    controller_state="RESPAWNING",
                    requested_control=None,
                    nominal_control=ControlCommand.full_brake(),
                    fallback_state="RESPAWN_BRAKE",
                    fallback_reason=collision_decision.reason,
                    control_origin="respawn_reset",
                    identity_available=False,
                    control_timing="post_respawn_command_without_snapshot",
                )
                continue

            try:
                observation_entry = capture_control_observation(
                    carla_if,
                    tick_context,
                    loop_tick_id=frame_count,
                    simulation_tick_seconds=simulation_tick_seconds,
                )
            except (TimeoutError, RuntimeError, ValueError) as exc:
                print(f"[Frame {frame_count}] Warning: {exc}; braking and skipping this tick.")
                _apply_arbitrated_control(
                    state=state,
                    tick_context=tick_context,
                    plan=None,
                    controller_state="CAMERA_TIMEOUT",
                    requested_control=None,
                    nominal_control=ControlCommand.full_brake(),
                    fallback_state="CAMERA_TIMEOUT_BRAKE",
                    fallback_reason=str(exc),
                    control_origin="fallback_missing_camera",
                )
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": 0.0,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)
                continue
            current_observation_entry = observation_entry
            state = observation_entry["state"]
            images = observation_entry["images"]
            current_carla_frame_id = observation_entry["frame_id"]
            current_simulation_time_s = observation_entry["simulation_time_s"]
            current_frame_id_quality = observation_entry["frame_id_quality"]
            if current_frame_id_quality == "exact_carla_snapshot":
                exact_observation_count += 1
            if len(images) > 1:
                latest_ui_frame = images[1]
            latest_telemetry = {
                "frame": frame_count,
                "speed_kmh": state["speed"] * 3.6,
                "steering": prev_control["steer"],
                "inference_time": current_inference_time,
            }
            if (
                frame_buffer
                and current_frame_id_quality == "exact_carla_snapshot"
                and frame_buffer[-1]["frame_id_quality"] == "exact_carla_snapshot"
                and int(observation_entry["frame_id"]) != int(frame_buffer[-1]["frame_id"]) + 1
            ):
                emit_runtime_event(
                    "sensor_sequence_reset",
                    loop_tick_id=int(frame_count),
                    previous_carla_frame_id=int(frame_buffer[-1]["frame_id"]),
                    current_carla_frame_id=int(observation_entry["frame_id"]),
                    reason="non_consecutive_camera_bundle",
                )
                frame_buffer.clear()
            frame_buffer.append(observation_entry)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)

            if (
                args.capture_inference_fixture
                and not fixture_capture_complete
                and len(frame_buffer) >= cfg.NUM_FRAMES
                and carla_if.has_complete_ego_history()
            ):
                applied_echo = carla_if.get_applied_control()
                brake_is_held = (
                    isinstance(applied_echo, dict)
                    and float(applied_echo.get("echoed_brake", 0.0)) >= 0.99
                    and float(applied_echo.get("echoed_throttle", 1.0)) <= 0.01
                )
                if not brake_is_held:
                    carla_if.apply_control(0.0, 0.0, 1.0)
                else:
                    (
                        fixture_images,
                        fixture_history_xyz,
                        fixture_history_rot,
                        fixture_source,
                    ) = _current_input_arrays()
                    alignment_metadata = camera_alignment_metadata()
                    profile_metadata = alignment_metadata.get("profile") or {}
                    captured_fixture_identity = save_camera_fixture(
                        args.capture_inference_fixture,
                        images_array=fixture_images,
                        history_xyz=fixture_history_xyz,
                        history_rot=fixture_history_rot,
                        camera_ids=fixture_source["camera_ids"],
                        frame_ids=tuple(
                            int(entry["frame_id"]) for entry in frame_buffer
                        ),
                        simulation_times_s=tuple(
                            float(entry["simulation_time_s"])
                            for entry in frame_buffer
                        ),
                        capture_pose_world=fixture_source["capture_pose_world"],
                        metadata={
                            "camera_alignment_mode": args.camera_alignment,
                            "camera_profile_sha256": profile_metadata.get(
                                "profile_sha256"
                            ),
                            "camera_profile_id": profile_metadata.get("profile_id"),
                            "dataset_revision": profile_metadata.get(
                                "dataset_revision"
                            ),
                            "navigation_text": (
                                nav_state.navigation_text
                                if args.mode == "navigation"
                                else ""
                            ),
                            "navigation_weight": (
                                nav_state.navigation_weight
                                if args.mode == "navigation"
                                else 1.0
                            ),
                            "map": cfg.CARLA_MAP,
                            "spawn_index": args.ego_spawn_index,
                            "scenario_seed": args.scenario_seed,
                            "synthetic_scene": {
                                "empty_road": bool(args.empty_road),
                                "npc_vehicle_count": (
                                    0
                                    if args.empty_road
                                    else cfg.NPC_VEHICLE_COUNT
                                ),
                                "npc_walker_count": (
                                    0
                                    if args.empty_road
                                    else cfg.NPC_WALKER_COUNT
                                ),
                            },
                        },
                    )
                    fixture_capture_complete = True
                    emit_runtime_event(
                        "camera_fixture_captured",
                        fixture_id=captured_fixture_identity["fixture_id"],
                        fixture_sha256=captured_fixture_identity[
                            "fixture_sha256"
                        ],
                        camera_alignment_mode=args.camera_alignment,
                        source_carla_frame_id=int(fixture_source["frame_id"]),
                        source_simulation_time_s=float(
                            fixture_source["simulation_time_s"]
                        ),
                        ego_history_real_tick_count=len(
                            carla_if.history_buffer
                        ),
                        brake_echo=applied_echo,
                    )
                    print(
                        "Frozen camera fixture captured: "
                        f"id={captured_fixture_identity['fixture_id']} "
                        f"sha256={captured_fixture_identity['fixture_sha256']}"
                    )
                    if args.capture_only:
                        stop_reason = "capture_only_complete"
                        break

            camera_capture_gate_open = (
                not args.capture_inference_fixture or fixture_capture_complete
            )
            if args.async_mode:
                if args.mode == "vqa":
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
                        and camera_capture_gate_open
                        and bool(nav_state.vqa_question)
                        and not pending_inference
                        and nav_state.revision != last_vqa_submitted_revision
                        and nav_state.revision != last_vqa_completed_revision
                    )
                else:
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
                        and camera_capture_gate_open
                        and not pending_inference
                        and (
                            last_inference_submit_simulation_time_s is None
                            or float(current_simulation_time_s)
                            - float(last_inference_submit_simulation_time_s)
                            >= inference_interval_sec
                        )
                    )
                if should_submit_inference:
                    req = _build_inference_request()
                    req["frame"] = int(frame_count)
                    while True:
                        try:
                            superseded_request = inference_request_q.get_nowait()
                        except queue.Empty:
                            break
                        if superseded_request is not None:
                            _emit_async_terminal(
                                superseded_request.get("request_id"),
                                "cancelled",
                                "request_queue_superseded",
                            )
                    inference_request_q.put_nowait(req)
                    pending_inference = True
                    pending_request_id = req["request_id"]
                    outstanding_async_requests[req["request_id"]] = {
                        key: req[key]
                        for key in (
                            "mode",
                            "source_loop_tick_id",
                            "source_elapsed_proxy_s",
                            "source_carla_frame_id",
                            "source_simulation_time_s",
                            "frame_id_quality",
                            "submission_monotonic_s",
                            "prompt_revision",
                            "respawn_revision",
                        )
                    }
                    last_inference_submit_simulation_time_s = float(current_simulation_time_s)
                    emit_runtime_event(
                        "inference_submitted",
                        request_id=req["request_id"],
                        mode=req["mode"],
                        source_loop_tick_id=req["source_loop_tick_id"],
                        source_elapsed_proxy_s=req["source_elapsed_proxy_s"],
                        source_carla_frame_id=req["source_carla_frame_id"],
                        source_simulation_time_s=req["source_simulation_time_s"],
                        frame_id_quality=req["frame_id_quality"],
                        prompt_revision=req["prompt_revision"],
                        respawn_revision=req["respawn_revision"],
                    )
                    if args.mode == "vqa":
                        last_vqa_submitted_revision = nav_state.revision

                latest_result = None
                while True:
                    try:
                        latest_result = inference_result_q.get_nowait()
                    except queue.Empty:
                        break
                if latest_result is not None:
                    for superseded_request_id in latest_result.get("superseded_request_ids", ()):
                        _emit_async_terminal(
                            superseded_request_id,
                            "discarded",
                            "result_queue_superseded",
                        )
                    if latest_result.get("request_id") == pending_request_id:
                        pending_inference = False
                        pending_request_id = None
                    result_status = "completed"
                    result_rejection_reason = None
                    try:
                        proposal = None
                        if (
                            latest_result.get("prompt_revision", nav_state.revision)
                            != nav_state.revision
                        ):
                            _audit_discarded_generated_result(
                                latest_result,
                                "stale_prompt_revision",
                            )
                            print(
                                f"[Frame {frame_count}] Discarded stale inference result for "
                                f"prompt revision {latest_result.get('prompt_revision')}"
                            )
                            result_status = "discarded"
                            result_rejection_reason = "stale_prompt_revision"
                        elif (
                            latest_result.get("respawn_revision", respawn_revision)
                            != respawn_revision
                        ):
                            _audit_discarded_generated_result(
                                latest_result,
                                "stale_respawn_revision",
                            )
                            print(
                                f"[Frame {frame_count}] Discarded stale inference result from "
                                f"respawn revision {latest_result.get('respawn_revision')}"
                            )
                            result_status = "discarded"
                            result_rejection_reason = "stale_respawn_revision"
                        elif "error" not in latest_result and latest_result.get("mode") == "vqa":
                            answer = latest_result.get("answer") or extract_answer_text(
                                latest_result.get("extra")
                            )
                            nav_state.set_vqa_answer(answer)
                            current_inference_time = float(latest_result["inference_time"])
                            last_vqa_completed_revision = latest_result.get("prompt_revision")
                            print(
                                f"[Frame {frame_count}] VQA done: "
                                f"{current_inference_time:.2f}s "
                                f"(submitted at frame {latest_result['frame_submitted']})"
                            )
                            print(f"    Q: {latest_result.get('vqa_question') or '(none)'}")
                            print(f"    A: {format_vqa_answer_preview(answer)}")
                            result_status = "completed_vqa"
                        elif "error" not in latest_result:
                            proposal = _extract_and_audit_proposal(
                                latest_result,
                                model_inference_latency_s=latest_result.get(
                                    "model_inference_time",
                                    latest_result["inference_time"],
                                ),
                            )
                            inference_time = float(latest_result["inference_time"])
                            current_inference_time = inference_time
                            outcome = _select_road_aware_candidate(
                                latest_result,
                                proposal,
                            )
                            proposal = outcome["proposal"]
                            admission = outcome["admission"]
                            application = _apply_selected_plan_outcome(
                                outcome,
                                inference_time_s=inference_time,
                                trajectory_timestamp_s=float(
                                    latest_result["result_ts"]
                                ),
                                source_loop_tick_id=int(
                                    latest_result["source_loop_tick_id"]
                                ),
                                source_elapsed_proxy_s=float(
                                    latest_result["source_elapsed_proxy_s"]
                                ),
                            )
                            result_status = application["status"]
                            result_rejection_reason = application[
                                "rejection_reason"
                            ]
                            if result_status == "rejected_plan":
                                print(
                                    f"[Frame {frame_count}] Rejected Alpamayo proposal: "
                                    f"{result_rejection_reason}"
                                )
                            elif result_status == "retained_active_plan":
                                latest_result["retained_plan_id"] = current_plan_id
                                print(
                                    f"[Frame {frame_count}] Retained active plan "
                                    f"{current_plan_id}: "
                                    f"{application['handoff'].status.value}"
                                )
                            else:
                                latest_result["accepted_plan_id"] = current_plan_id
                                print(
                                    f"[Frame {frame_count}] Inference done: "
                                    f"{inference_time:.2f}s "
                                    f"(submitted at frame {latest_result['frame_submitted']})"
                                )
                                print(f"    CoT: {current_cot[:60]}...")
                                print(
                                    f"    Admission: {admission.value} | Nav: "
                                    f"{latest_result.get('navigation_text') or '(none)'} "
                                    f"(weight="
                                    f"{latest_result.get('navigation_weight', 1.0):.2f})"
                                )
                                print(
                                    f"    Selected traj sample: "
                                    f"{current_selected_traj_idx}/"
                                    f"{args.num_traj_samples - 1}"
                                )
                                print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")
                        else:
                            print(
                                f"[Frame {frame_count}] Inference error: {latest_result['error']}"
                            )
                            result_status = "error"
                            result_rejection_reason = str(latest_result["error"])
                            if args.debug_worker_traceback and latest_result.get("traceback"):
                                print(latest_result["traceback"].rstrip())
                    except Exception as exc:
                        _emit_async_terminal(
                            latest_result.get("request_id"),
                            "error",
                            f"result_processing_error:{exc}",
                            result=latest_result,
                        )
                        print(f"[Frame {frame_count}] Rejected malformed inference result: {exc}")
                    else:
                        _emit_async_terminal(
                            latest_result.get("request_id"),
                            result_status,
                            result_rejection_reason,
                            result=latest_result,
                        )
            else:
                if args.mode == "vqa":
                    should_prepare_sync_input = (
                        bool(nav_state.vqa_question)
                        and nav_state.revision != last_vqa_submitted_revision
                        and nav_state.revision != last_vqa_completed_revision
                    )
                else:
                    should_prepare_sync_input = (
                        last_inference_submit_simulation_time_s is None
                        or float(current_simulation_time_s)
                        - float(last_inference_submit_simulation_time_s)
                        >= inference_interval_sec
                    )
                if (
                    len(frame_buffer) >= cfg.NUM_FRAMES
                    and camera_capture_gate_open
                    and should_prepare_sync_input
                ):
                    source_entry = frame_buffer[-1]
                    model_input_error = None
                    try:
                        model_data, source_entry = _prepare_current_model_input()
                    except Exception as exc:
                        model_data = None
                        model_input_error = f"{type(exc).__name__}:{exc}"
                    if args.mode == "vqa":
                        should_run_vqa = (
                            bool(nav_state.vqa_question)
                            and nav_state.revision != last_vqa_submitted_revision
                            and nav_state.revision != last_vqa_completed_revision
                        )
                        if should_run_vqa:
                            request_sequence += 1
                            request_id = request_sequence
                            submission_monotonic_s = time.monotonic()
                            last_vqa_submitted_revision = nav_state.revision
                            active_sync_request = {
                                "request_id": request_id,
                                "mode": "vqa",
                                "submission_monotonic_s": submission_monotonic_s,
                                "source_loop_tick_id": int(frame_count),
                                "source_carla_frame_id": int(source_entry["frame_id"]),
                                "source_simulation_time_s": float(
                                    source_entry["simulation_time_s"]
                                ),
                                "frame_id_quality": source_entry["frame_id_quality"],
                            }
                            emit_runtime_event(
                                "inference_submitted",
                                request_id=request_id,
                                mode="vqa",
                                source_loop_tick_id=int(frame_count),
                                source_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                                source_carla_frame_id=int(source_entry["frame_id"]),
                                source_simulation_time_s=float(source_entry["simulation_time_s"]),
                                frame_id_quality=source_entry["frame_id_quality"],
                                prompt_revision=nav_state.revision,
                                respawn_revision=respawn_revision,
                            )
                            model_start_time = time.monotonic()
                            stage = "model_inference"
                            try:
                                if model_input_error is not None:
                                    raise RuntimeError(f"model_input_error:{model_input_error}")
                                extra = _run_vqa_with_linalg_fallback(
                                    model_data,
                                    question=nav_state.vqa_question,
                                )
                                model_inference_time = time.monotonic() - model_start_time
                                stage = "result_processing"
                                answer = extract_answer_text(extra)
                                nav_state.set_vqa_answer(answer)
                                current_inference_time = model_inference_time
                                last_vqa_completed_revision = nav_state.revision
                                print(f"[Frame {frame_count}] VQA: {model_inference_time:.2f}s")
                                print(f"    Q: {nav_state.vqa_question}")
                                print(f"    A: {format_vqa_answer_preview(answer)}")
                            except Exception as exc:
                                _emit_sync_terminal(
                                    request_id=request_id,
                                    mode="vqa",
                                    status="error",
                                    rejection_reason=f"{stage}_error:{exc}",
                                    submission_monotonic_s=submission_monotonic_s,
                                    model_inference_latency_s=(
                                        model_inference_time
                                        if stage == "result_processing"
                                        else time.monotonic() - model_start_time
                                    ),
                                )
                                message = f"VQA {stage} failed; retaining safe fallback: {exc}"
                                nav_state.set_error(message)
                                print(f"[Frame {frame_count}] {message}")
                            else:
                                _emit_sync_terminal(
                                    request_id=request_id,
                                    mode="vqa",
                                    status="completed_vqa",
                                    rejection_reason=None,
                                    submission_monotonic_s=submission_monotonic_s,
                                    model_inference_latency_s=model_inference_time,
                                )
                    elif (
                        last_inference_submit_simulation_time_s is None
                        or float(current_simulation_time_s)
                        - float(last_inference_submit_simulation_time_s)
                        >= inference_interval_sec
                    ):
                        navigation_text = (
                            nav_state.navigation_text if args.mode == "navigation" else ""
                        )
                        navigation_weight = (
                            nav_state.navigation_weight if args.mode == "navigation" else 1.0
                        )
                        request_sequence += 1
                        request_id = request_sequence
                        last_inference_submit_simulation_time_s = float(current_simulation_time_s)
                        submission_monotonic_s = time.monotonic()
                        active_sync_request = {
                            "request_id": request_id,
                            "mode": args.mode,
                            "submission_monotonic_s": submission_monotonic_s,
                            "source_loop_tick_id": int(frame_count),
                            "source_carla_frame_id": int(source_entry["frame_id"]),
                            "source_simulation_time_s": float(source_entry["simulation_time_s"]),
                            "frame_id_quality": source_entry["frame_id_quality"],
                        }
                        emit_runtime_event(
                            "inference_submitted",
                            request_id=request_id,
                            mode=args.mode,
                            source_loop_tick_id=int(frame_count),
                            source_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                            source_carla_frame_id=int(source_entry["frame_id"]),
                            source_simulation_time_s=float(source_entry["simulation_time_s"]),
                            frame_id_quality=source_entry["frame_id_quality"],
                            prompt_revision=nav_state.revision,
                            respawn_revision=respawn_revision,
                        )
                        model_start_time = time.monotonic()
                        stage = "model_inference"
                        try:
                            if model_input_error is not None:
                                raise RuntimeError(f"model_input_error:{model_input_error}")
                            pred_xyz, extra = _run_inference_with_nav_fallback(
                                model_data,
                                navigation_text=navigation_text,
                                navigation_weight=navigation_weight,
                            )
                            model_inference_time = time.monotonic() - model_start_time
                            stage = "result_processing"
                            sync_result = {
                                "request_id": request_id,
                                "mode": args.mode,
                                "pred_xyz": pred_xyz,
                                "extra": extra,
                                "source_loop_tick_id": int(frame_count),
                                "source_elapsed_proxy_s": _simulation_elapsed_proxy_s(),
                                "source_carla_frame_id": int(source_entry["frame_id"]),
                                "source_simulation_time_s": float(
                                    source_entry["simulation_time_s"]
                                ),
                                "frame_id_quality": source_entry["frame_id_quality"],
                                "capture_pose_world": source_entry["capture_pose_world"],
                                "camera_ids": source_entry["camera_ids"],
                                "navigation_text": navigation_text,
                                "navigation_weight": navigation_weight,
                                "prompt_revision": nav_state.revision,
                                "respawn_revision": respawn_revision,
                            }
                            proposal = _extract_and_audit_proposal(
                                sync_result,
                                model_inference_latency_s=model_inference_time,
                            )
                            outcome = _select_road_aware_candidate(
                                sync_result,
                                proposal,
                            )
                            proposal = outcome["proposal"]
                            admission = outcome["admission"]
                        except Exception as exc:
                            rejected_output = (
                                isinstance(exc, TrajectoryValidationError)
                                or stage == "result_processing"
                            )
                            rejection_reason = (
                                exc.reason
                                if isinstance(exc, TrajectoryValidationError)
                                else f"{stage}_error:{exc}"
                            )
                            _emit_sync_terminal(
                                request_id=request_id,
                                mode=args.mode,
                                status=("rejected_plan" if rejected_output else "error"),
                                rejection_reason=rejection_reason,
                                submission_monotonic_s=submission_monotonic_s,
                                model_inference_latency_s=(
                                    model_inference_time
                                    if stage == "result_processing"
                                    else time.monotonic() - model_start_time
                                ),
                            )
                            failure_kind = (
                                "Rejected Alpamayo proposal"
                                if rejected_output
                                else "Alpamayo inference failed; retaining prior safe plan"
                            )
                            nav_state.set_error(str(rejection_reason))
                            print(f"[Frame {frame_count}] {failure_kind}: {rejection_reason}")
                        else:
                            application = _apply_selected_plan_outcome(
                                outcome,
                                inference_time_s=model_inference_time,
                                trajectory_timestamp_s=time.monotonic(),
                                source_loop_tick_id=int(frame_count),
                                source_elapsed_proxy_s=(
                                    _simulation_elapsed_proxy_s()
                                ),
                            )
                            if application["status"] == "rejected_plan":
                                rejection_reason = application[
                                    "rejection_reason"
                                ]
                                nav_state.set_error(rejection_reason)
                                print(
                                    f"[Frame {frame_count}] Rejected Alpamayo proposal: "
                                    f"{rejection_reason}"
                                )
                                _emit_sync_terminal(
                                    request_id=request_id,
                                    mode=args.mode,
                                    status="rejected_plan",
                                    rejection_reason=rejection_reason,
                                    submission_monotonic_s=submission_monotonic_s,
                                    model_inference_latency_s=model_inference_time,
                                    coc_sha256=proposal["coc_sha256"],
                                )
                            elif application["status"] == "retained_active_plan":
                                print(
                                    f"[Frame {frame_count}] Retained active plan "
                                    f"{current_plan_id}: "
                                    f"{application['handoff'].status.value}"
                                )
                                _emit_sync_terminal(
                                    request_id=request_id,
                                    mode=args.mode,
                                    status="retained_active_plan",
                                    rejection_reason=None,
                                    submission_monotonic_s=submission_monotonic_s,
                                    model_inference_latency_s=model_inference_time,
                                    source_plan_id=current_plan_id,
                                    coc_sha256=proposal["coc_sha256"],
                                )
                            else:
                                print(
                                    f"[Frame {frame_count}] Inference: "
                                    f"{model_inference_time:.2f}s"
                                )
                                print(f"    CoT: {current_cot[:60]}...")
                                print(f"    Admission: {admission.value}")
                                if args.mode == "navigation":
                                    print(
                                        f"    Nav: {nav_state.navigation_text or '(none)'} "
                                        f"(weight={nav_state.navigation_weight:.2f})"
                                    )
                                print(
                                    f"    Selected traj sample: {current_selected_traj_idx}/"
                                    f"{args.num_traj_samples - 1}"
                                )
                                print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")
                                _emit_sync_terminal(
                                    request_id=request_id,
                                    mode=args.mode,
                                    status="accepted_plan",
                                    rejection_reason=None,
                                    submission_monotonic_s=submission_monotonic_s,
                                    model_inference_latency_s=model_inference_time,
                                    source_plan_id=current_plan_id,
                                    coc_sha256=proposal["coc_sha256"],
                                )

            expired_plan_reason = None
            bridge_adapter_assessment = None
            if current_plan is None:
                active_plan_availability = decide_active_plan_availability(
                    active_plan_present=False,
                    standard_validity=None,
                    bridge_validity=None,
                    alignment_validity=None,
                    road_envelope=None,
                    bridge_deadline_age_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                    ),
                    bridge_min_remaining_horizon_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                    ),
                )
            if current_plan is not None:
                (
                    execution_validity,
                    bridge_validity,
                    alignment_validity,
                ) = _active_plan_validity_window(current_plan)
                bridge_road_envelope = None
                if (
                    not execution_validity.valid
                    and bridge_validity is not None
                    and bridge_validity.valid
                    and alignment_validity.valid
                    and safety_adapter is not None
                ):
                    try:
                        bridge_adapter_assessment = safety_adapter.assess(
                            tick_context=tick_context,
                            plan=current_plan,
                        )
                        bridge_road_envelope = getattr(
                            bridge_adapter_assessment,
                            "road_envelope",
                            None,
                        )
                    except Exception:
                        bridge_adapter_assessment = None
                        bridge_road_envelope = None
                active_plan_availability = decide_active_plan_availability(
                    active_plan_present=True,
                    standard_validity=execution_validity,
                    bridge_validity=bridge_validity,
                    alignment_validity=alignment_validity,
                    road_envelope=bridge_road_envelope,
                    bridge_deadline_age_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MAX_PLAN_AGE_S
                    ),
                    bridge_min_remaining_horizon_s=float(
                        cfg.TRAJECTORY_ACTIVE_BRIDGE_MIN_REMAINING_HORIZON_S
                    ),
                )
                if not execution_validity.valid or not alignment_validity.valid:
                    expired_plan_reason = (
                        execution_validity.rejection_reason or alignment_validity.rejection_reason
                    )
                    emit_runtime_event(
                        "plan_validation",
                        aggregate_rejection=not active_plan_availability.bridge_active,
                        layer="CONTROLLER_EXECUTION",
                        proposal_id=current_plan.plan_id,
                        source_carla_frame_id=current_plan.source_frame_id,
                        source_simulation_time_s=current_plan.source_simulation_time_s,
                        arrival_carla_frame_id=current_carla_frame_id,
                        arrival_simulation_time_s=current_simulation_time_s,
                        valid=False,
                        rejection_reason=expired_plan_reason,
                        source_age_s=execution_validity.source_age_s,
                        remaining_horizon_s=execution_validity.remaining_horizon_s,
                        tracking_error_m=alignment_validity.tracking_error_m,
                        heading_error_deg=alignment_validity.heading_error_deg,
                        execution_allowed_by_bridge=bool(
                            active_plan_availability.bridge_active
                        ),
                        active_plan_availability=(
                            active_plan_availability.to_json_dict()
                        ),
                    )
                if (
                    active_plan_availability.status
                    is not ActivePlanAvailabilityStatus.STANDARD_EXECUTION
                ):
                    emit_runtime_event(
                        "active_plan_availability",
                        aggregate_age=False,
                        aggregate_rejection=False,
                        layer="CONTROLLER_EXECUTION",
                        active_plan_id=current_plan.plan_id,
                        alignment_valid=bool(alignment_validity.valid),
                        tracking_error_m=alignment_validity.tracking_error_m,
                        heading_error_deg=alignment_validity.heading_error_deg,
                        road_envelope=(
                            bridge_road_envelope.to_json_dict()
                            if bridge_road_envelope is not None
                            else None
                        ),
                        **active_plan_availability.to_json_dict(),
                    )
                if not active_plan_availability.execution_allowed:
                    denied_availability = active_plan_availability
                    expired_plan_reason = (
                        expired_plan_reason
                        or denied_availability.original_rejection_reason
                        or denied_availability.denial_reason
                        or "active_plan_execution_denied"
                    )
                    _clear_active_plan_state()
                    active_plan_availability = denied_availability

            if current_plan is not None:
                adapter_assessment = bridge_adapter_assessment
                road_speed_cap_mps = None
                maximum_authorized_waypoint_index = None
                if adapter_assessment is not None:
                    current_road_envelope = getattr(
                        adapter_assessment,
                        "road_envelope",
                        None,
                    )
                    if current_road_envelope is not None:
                        road_speed_cap_mps = (
                            current_road_envelope.target_speed_cap_mps
                        )
                        maximum_authorized_waypoint_index = (
                            current_road_envelope.last_safe_waypoint_index
                        )
                elif safety_adapter is not None:
                    try:
                        adapter_assessment = safety_adapter.assess(
                            tick_context=tick_context,
                            plan=current_plan,
                        )
                        current_road_envelope = getattr(
                            adapter_assessment,
                            "road_envelope",
                            None,
                        )
                        if current_road_envelope is not None:
                            road_speed_cap_mps = (
                                current_road_envelope.target_speed_cap_mps
                            )
                            maximum_authorized_waypoint_index = (
                                current_road_envelope.last_safe_waypoint_index
                            )
                    except Exception:
                        adapter_assessment = None
                        current_road_envelope = None
                try:
                    steering_raw, throttle_raw, brake_raw, ctrl_debug = (
                        pid_follower.compute_world_control(
                            plan_id=current_plan.plan_id,
                            wp_world=current_plan.world_points,
                            waypoint_times_s=current_plan.waypoint_times_s,
                            current_simulation_time_s=float(current_simulation_time_s),
                            speed_mps=float(state["speed"]),
                            stop_requested=bool(current_plan.stop_requested),
                            terminal_stop_index=current_plan.terminal_stop_index,
                            capture_origin_world=current_plan.capture_pose_world[:3, 3],
                            target_speed_cap_mps=road_speed_cap_mps,
                            maximum_authorized_waypoint_index=(
                                maximum_authorized_waypoint_index
                            ),
                        )
                    )
                    requested_control = {
                        "steering": float(steering_raw),
                        "throttle": float(throttle_raw),
                        "brake": float(brake_raw),
                    }

                    emergency_stop_requested = bool(ctrl_debug.get("bypass_smoothing", False))
                    nominal_control, postprocessing = smooth_controller_control(
                        steering_raw=steering_raw,
                        throttle_raw=throttle_raw,
                        brake_raw=brake_raw,
                        previous_nominal=prev_nominal_control,
                        alpha=cfg.CONTROL_SMOOTH_ALPHA,
                        bypass_smoothing=emergency_stop_requested,
                        constrained_deceleration=(
                            ctrl_debug.get("controller_state")
                            == "ROAD_CONSTRAINED_DECELERATING"
                        ),
                    )
                except Exception as exc:
                    requested_control = None
                    nominal_control = None
                    ctrl_debug = {
                        "controller_state": "INVALID_CONTROLLER_OUTPUT",
                        "controller_error": f"{type(exc).__name__}:{exc}",
                    }
                    postprocessing = ("controller_output_validation_failed",)

                safety_decision = _apply_arbitrated_control(
                    state=state,
                    tick_context=tick_context,
                    plan=current_plan,
                    controller_state=ctrl_debug.get("controller_state", "TRACKING"),
                    requested_control=requested_control,
                    nominal_control=nominal_control,
                    postprocessing=postprocessing,
                    controller_debug=ctrl_debug,
                    adapter_assessment=adapter_assessment,
                )
                if nominal_control is not None:
                    prev_nominal_control = dict(nominal_control)
                steering, throttle, brake = safety_decision.applied_control.as_tuple()

                if current_pred_xyz is not None:
                    cam_img = images[1]
                    plan_age_s = max(
                        0.0,
                        float(current_simulation_time_s)
                        - float(current_plan.source_simulation_time_s),
                    )
                    observation = current_observation_entry.get("observation")
                    camera_pose_world = None
                    camera_intrinsic = None
                    if observation is not None:
                        camera_pose_world = np.asarray(
                            observation.ego_pose_world, dtype=np.float64
                        ) @ np.asarray(
                            observation.camera_extrinsics[1],
                            dtype=np.float64,
                        )
                        camera_intrinsic = observation.camera_output_models[1]
                    first_future_index = int(
                        np.searchsorted(
                            current_plan.waypoint_times_s,
                            float(current_simulation_time_s),
                            side="right",
                        )
                    )
                    relative_last_safe_index = None
                    if current_road_envelope is not None:
                        if current_road_envelope.last_safe_waypoint_index is None:
                            relative_last_safe_index = -1
                        else:
                            relative_last_safe_index = (
                                int(current_road_envelope.last_safe_waypoint_index)
                                - first_future_index
                            )
                    vis_frame = create_visualization_frame(
                        cam_img,
                        current_pred_xyz,
                        current_selected_traj_idx,
                        frame_count,
                        current_inference_time,
                        current_cot,
                        state["speed"] * 3.6,
                        steering,
                        navigation_text=nav_state.navigation_text,
                        navigation_weight=nav_state.navigation_weight,
                        paused=nav_state.paused,
                        world_trajectory=current_plan.world_points[first_future_index:],
                        camera_pose_world=camera_pose_world,
                        camera_intrinsic=camera_intrinsic,
                        source_frame_id=current_plan.source_frame_id,
                        source_age_s=plan_age_s,
                        controller_state=ctrl_debug.get(
                            "controller_state",
                            "TRACKING",
                        ),
                        requested_control=(
                            safety_decision.controller_requested_control.to_json_dict()
                            if safety_decision.controller_requested_control is not None
                            else None
                        ),
                        applied_control=safety_decision.applied_control.to_json_dict(),
                        applied_control_source=(safety_decision.applied_control_source),
                        safety_override_applied=(safety_decision.safety_override_applied),
                        safety_override_reason=safety_decision.primary_reason,
                        plan_admission_status=(
                            current_plan_admission_status.value
                            if isinstance(
                                current_plan_admission_status,
                                PlanAdmissionStatus,
                            )
                            else current_plan_admission_status
                        ),
                        near_term_road_status=(
                            current_road_envelope.near_term_path_road.status.value
                            if current_road_envelope is not None
                            else None
                        ),
                        full_path_road_status=(
                            current_road_envelope.full_path_road.status.value
                            if current_road_envelope is not None
                            else None
                        ),
                        road_speed_cap_mps=(
                            current_road_envelope.target_speed_cap_mps
                            if current_road_envelope is not None
                            else None
                        ),
                        last_safe_waypoint_index=relative_last_safe_index,
                        camera_alignment_mode=args.camera_alignment,
                    )
                    latest_ui_frame = vis_frame
                    if cfg.SAVE_VIDEO:
                        video_recorder.add_frame(vis_frame)

                applied_echo = last_applied_control_echo
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": steering,
                    "throttle": throttle,
                    "brake": brake,
                    "inference_time": current_inference_time,
                    "controller_state": ctrl_debug.get(
                        "controller_state",
                        "TRACKING",
                    ),
                    "target_speed_mps": ctrl_debug.get("target_speed_mps"),
                    "launch_floor_mps": ctrl_debug.get("launch_floor_mps"),
                    "echoed_throttle": (
                        applied_echo["echoed_throttle"] if applied_echo else None
                    ),
                    "echoed_brake": (
                        applied_echo["echoed_brake"] if applied_echo else None
                    ),
                    "echoed_steering": (
                        applied_echo["echoed_steer"] if applied_echo else None
                    ),
                    "gear": applied_echo["gear"] if applied_echo else None,
                    "plan_admission_status": (
                        current_plan_admission_status.value
                        if isinstance(
                            current_plan_admission_status,
                            PlanAdmissionStatus,
                        )
                        else current_plan_admission_status
                    ),
                    "near_term_road_status": (
                        current_road_envelope.near_term_path_road.status.value
                        if current_road_envelope is not None
                        else None
                    ),
                    "full_path_road_status": (
                        current_road_envelope.full_path_road.status.value
                        if current_road_envelope is not None
                        else None
                    ),
                    "road_speed_cap_mps": (
                        current_road_envelope.target_speed_cap_mps
                        if current_road_envelope is not None
                        else None
                    ),
                    "applied_control_source": (safety_decision.applied_control_source),
                    "safety_override_applied": (safety_decision.safety_override_applied),
                    "safety_override_reason": safety_decision.primary_reason,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

                gear_display = "?" if applied_echo is None else applied_echo["gear"]
                print(
                    f"[Frame {frame_count}] Speed: {state['speed'] * 3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}, "
                    f"Gear: {gear_display}"
                )
                if current_plan is not None:
                    print(f"    Trajectory source age (CARLA simulation time): {plan_age_s:.2f}s")
            else:
                waiting_reason = expired_plan_reason or (
                    "vqa_mode_has_no_trajectory_control"
                    if args.mode == "vqa"
                    else "waiting_for_valid_plan"
                )
                safety_decision = _apply_arbitrated_control(
                    state=state,
                    tick_context=tick_context,
                    plan=None,
                    controller_state="WAITING_FOR_PLAN",
                    requested_control=None,
                    nominal_control=ControlCommand.full_brake(),
                    fallback_state="WAITING_FOR_PLAN",
                    fallback_reason=waiting_reason,
                    control_origin="waiting_for_plan",
                )
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": safety_decision.applied_control.steering,
                    "throttle": safety_decision.applied_control.throttle,
                    "brake": safety_decision.applied_control.brake,
                    "inference_time": current_inference_time,
                    "controller_state": "WAITING_FOR_PLAN",
                    "applied_control_source": safety_decision.applied_control_source,
                    "safety_override_applied": safety_decision.safety_override_applied,
                    "safety_override_reason": safety_decision.primary_reason,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        emergency_brake_attempted = carla_if.ego_vehicle is not None
        emergency_brake_applied = False
        emergency_brake_error = None
        if emergency_brake_attempted:
            try:
                apply_vehicle_control(ControlCommand.full_brake())
                emergency_brake_applied = True
            except Exception as exc:
                emergency_brake_error = f"{type(exc).__name__}:{exc}"
        sync_terminal_error = None
        if active_sync_request is not None:
            request = dict(active_sync_request)
            try:
                _emit_sync_terminal(
                    request_id=request["request_id"],
                    mode=request["mode"],
                    status="interrupted",
                    rejection_reason="keyboard_interrupt_during_sync_inference",
                    submission_monotonic_s=request["submission_monotonic_s"],
                    source_loop_tick_id=request["source_loop_tick_id"],
                    source_carla_frame_id=request["source_carla_frame_id"],
                    source_simulation_time_s=request["source_simulation_time_s"],
                    frame_id_quality=request["frame_id_quality"],
                )
            except Exception as exc:
                sync_terminal_error = f"{type(exc).__name__}:{exc}"
        emit_runtime_event(
            "episode_stop_requested",
            status=stop_reason,
            emergency_brake_attempted=emergency_brake_attempted,
            emergency_brake_applied=emergency_brake_applied,
            emergency_brake_error=emergency_brake_error,
            sync_terminal_error=sync_terminal_error,
        )
        print("\n\nInterrupted by user.")
    except Exception as e:
        stop_reason = "error"
        run_error = str(e)
        emergency_brake_attempted = carla_if.ego_vehicle is not None
        emergency_brake_applied = False
        fail_closed_apply_error = None
        if emergency_brake_attempted:
            try:
                apply_vehicle_control(ControlCommand.full_brake())
                emergency_brake_applied = True
            except Exception as apply_exc:
                fail_closed_apply_error = f"{type(apply_exc).__name__}:{apply_exc}"
        sync_terminal_error = None
        if active_sync_request is not None:
            request = dict(active_sync_request)
            try:
                _emit_sync_terminal(
                    request_id=request["request_id"],
                    mode=request["mode"],
                    status="error",
                    rejection_reason=f"runtime_error:{type(e).__name__}:{e}",
                    submission_monotonic_s=request["submission_monotonic_s"],
                    source_loop_tick_id=request["source_loop_tick_id"],
                    source_carla_frame_id=request["source_carla_frame_id"],
                    source_simulation_time_s=request["source_simulation_time_s"],
                    frame_id_quality=request["frame_id_quality"],
                )
            except Exception as terminal_exc:
                sync_terminal_error = f"{type(terminal_exc).__name__}:{terminal_exc}"
        emit_runtime_event(
            "runtime_error",
            status="error",
            error=run_error,
            emergency_brake_attempted=emergency_brake_attempted,
            fail_closed_brake_applied=emergency_brake_applied,
            fail_closed_apply_error=fail_closed_apply_error,
            sync_terminal_error=sync_terminal_error,
        )
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        async_shutdown_error = None
        try:
            if args.async_mode and inference_stop is not None:
                inference_stop.set()
                try:
                    inference_request_q.put_nowait(None)
                except Exception:
                    pass
                if worker_thread is not None:
                    worker_thread.join(timeout=2.0)
                worker_is_alive = bool(
                    worker_thread is not None
                    and callable(getattr(worker_thread, "is_alive", None))
                    and worker_thread.is_alive()
                )
                _clear_async_queues("episode_end")
                for outstanding_request_id in list(outstanding_async_requests):
                    _emit_async_terminal(
                        outstanding_request_id,
                        "abandoned" if worker_is_alive else "cancelled",
                        (
                            "worker_still_running_at_episode_end"
                            if worker_is_alive
                            else "episode_ended_before_result_consumption"
                        ),
                    )
        except Exception as exc:
            async_shutdown_error = f"{type(exc).__name__}:{exc}"
            print(f"Warning: async shutdown telemetry failed: {exc}")

        cleanup_error = None
        try:
            carla_if.cleanup()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}:{exc}"
            print(f"Warning: CARLA cleanup failed: {exc}")

        if runtime_metrics is None:
            runtime_metrics = RuntimeMetrics()
        runtime_metrics.record_collision_count(carla_if.get_episode_collision_count())
        exact_timing_available = exact_observation_count > 0
        get_camera_sync_stats = getattr(carla_if, "get_camera_sync_stats", None)
        camera_sync_stats = get_camera_sync_stats() if callable(get_camera_sync_stats) else None
        summary = runtime_metrics.final_summary(
            run_id=run_id,
            stop_reason=stop_reason,
            error=run_error,
            mode=args.mode,
            execution="async" if args.async_mode else "sync",
            loop_tick_count=int(frame_count),
            simulation_duration_proxy_s=float(frame_count) * simulation_tick_seconds,
            simulation_tick_seconds=simulation_tick_seconds,
            simulation_duration_quality="successful_tick_count_times_world_fixed_delta",
            exact_source_frame_ids_available=exact_timing_available,
            exact_plan_age_available=False,
            camera_sync_stats=camera_sync_stats,
            camera_alignment=camera_alignment_metadata(),
            camera_fixture=captured_fixture_identity,
            respawn_count=int(respawn_count),
            episode_collision_count=carla_if.get_episode_collision_count(),
            telemetry_path=args.telemetry_jsonl,
            telemetry_write_failed=telemetry_write_failed,
            async_shutdown_error=async_shutdown_error,
            cleanup_error=cleanup_error,
        )
        summary["exact_plan_age_available"] = bool(summary["source_age_s"]["count"] > 0)
        if telemetry_writer is not None and not telemetry_write_failed:
            try:
                telemetry_writer.append(summary)
            except Exception as exc:
                telemetry_write_failed = True
                print(f"Warning: failed to write final runtime summary: {exc}")
        if telemetry_writer is not None:
            try:
                telemetry_writer.close()
            except Exception as exc:
                print(f"Warning: failed to close runtime telemetry: {exc}")

        latency_summary = summary["inference_latency_s"]
        source_age_summary = summary["source_age_s"]
        source_age_proxy_summary = summary["source_age_proxy_s"]
        print("\nRuntime summary:")
        print(
            f"  stop={stop_reason}, ticks={frame_count}, respawns={respawn_count}, "
            f"collisions={carla_if.get_episode_collision_count()}"
        )
        print(
            "  inference latency p50/p95/p99: "
            f"{latency_summary['p50']!r} / {latency_summary['p95']!r} / "
            f"{latency_summary['p99']!r} s"
        )
        if exact_timing_available:
            print(
                "  exact plan source-age p50/p95/p99: "
                f"{source_age_summary['p50']!r} / "
                f"{source_age_summary['p95']!r} / "
                f"{source_age_summary['p99']!r} s"
            )
        else:
            print("  exact source-frame and plan-age metrics: unavailable")
        print(
            "  source-age proxy p50/p95/p99: "
            f"{source_age_proxy_summary['p50']!r} / "
            f"{source_age_proxy_summary['p95']!r} / "
            f"{source_age_proxy_summary['p99']!r} s"
        )
        if args.telemetry_jsonl:
            print(f"  telemetry: {args.telemetry_jsonl}")
        if cfg.SAVE_VIDEO and video_recorder:
            try:
                video_recorder.save()
            except Exception as exc:
                print(f"Warning: failed to save closed-loop video: {exc}")
        if pygame_ui_recorder is not None:
            try:
                pygame_ui_recorder.save()
            except Exception as exc:
                print(f"Warning: failed to save Pygame UI video: {exc}")
        if pygame_ui is not None:
            try:
                pygame_ui.close()
            except Exception as exc:
                print(f"Warning: failed to close Pygame UI: {exc}")

    print("\nStopped.")
    if stop_reason == "keyboard_interrupt":
        return 130
    return 1 if stop_reason == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())

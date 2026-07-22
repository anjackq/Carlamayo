"""Modular entrypoint for CARLA closed-loop control with Alpamayo."""

import argparse
import math
import os
import queue
import threading
import time
import traceback

import numpy as np
import torch

from module import config as cfg
from module.carla_safety_adapter import CarlaGroundTruthSafetyAdapter
from module.geometry import pose_matrix_from_state
from module.navigation_control import NavigationControlState
from module.pid_controller import OfficialPIDFollower
from module.proposal_audit import coc_audit_fields
from module.respawn_control import RespawnMonitor
from module.runtime_metrics import JsonlWriter, RuntimeMetrics
from module.safety_shield import (
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
    camera_ids = tuple(
        int(spec["alpamayo_id"])
        for spec in cfg.CAMERA_SPECS[: len(images)]
    )
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
        help=(
            "Stop after this many seconds of successful simulation ticks. "
            "Disabled by default."
        ),
    )
    args = parser.parse_args(argv)
    if args.oom_free and args.quantization:
        parser.error("--oom-free and --quantization are mutually exclusive.")
    if args.max_episode_seconds is not None and (
        not math.isfinite(args.max_episode_seconds) or args.max_episode_seconds <= 0.0
    ):
        parser.error("--max-episode-seconds must be finite and greater than zero.")
    args.start_paused = bool(args.pygame_ui)
    args.pygame_ui_video = (
        derive_pygame_ui_video_path(cfg.OUTPUT_VIDEO) if args.pygame_ui else None
    )
    return args


def main():
    args = parse_args()
    inference_interval_sec = 1.0

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
    print(f"Device map: {args.device_map}")
    print(f"CUDA linalg library: {args.cuda_linalg_library}")
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
    carla_if = CARLAInterface()
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
            "schema_version": 1,
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
        if not args.oom_free:
            model, processor = load_model(
                args.quantization,
                device_map=args.device_map,
            )
            print("Model loaded!")
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
        else:
            # Defer loading until CARLA has spawned its cameras/NPCs so the
            # OOM-free plan reflects the VRAM CARLA actually leaves free.
            print("OOM-free mode: Alpamayo loads after CARLA is fully spawned.")

        carla_if.connect()
        carla_if.load_map(cfg.CARLA_MAP)
        carla_if.spawn_ego_vehicle()
        carla_if.enable_synchronous_mode()
        fixed_delta_seconds = carla_if.world.get_settings().fixed_delta_seconds
        if fixed_delta_seconds is not None and float(fixed_delta_seconds) > 0.0:
            simulation_tick_seconds = float(fixed_delta_seconds)
        carla_if.spawn_npcs(num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT)
        carla_if.setup_cameras()
        carla_if.setup_collision_sensor()
        time.sleep(1.0)

        if args.oom_free:
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
            print("Model loaded!")
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
        safety_policy = SafetyPolicy(
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
        safety_shield = StopOnlySafetyShield(safety_policy)
        try:
            safety_adapter = CarlaGroundTruthSafetyAdapter(
                carla_if.world,
                carla_if.ego_vehicle,
                safety_policy,
            )
        except Exception as exc:
            safety_adapter = None
            print(
                "Warning: CARLA ground-truth safety adapter unavailable; "
                f"movement will fail closed ({exc})."
            )

        current_plan = None
        current_trajectory = None
        current_pred_xyz = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        vlm_generate_timing = VlmGenerateTiming()
        respawn_monitor = RespawnMonitor(
            cooldown_frames=cfg.RESPAWN_COLLISION_COOLDOWN_FRAMES
        )
        frame_buffer = []
        current_trajectory_ts = None
        current_plan_id = None
        current_plan_source_loop_tick_id = None
        current_plan_source_elapsed_proxy_s = None
        current_carla_frame_id = None
        current_simulation_time_s = None
        current_frame_id_quality = "loop_counter_proxy"
        current_observation_entry = None
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

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
                    source_simulation_time_s = current_observation_entry[
                        "simulation_time_s"
                    ]
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
            if (
                active_sync_request is not None
                and int(active_sync_request["request_id"]) == int(request_id)
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
            """Extract one generated proposal and log its complete reasoning once."""

            proposal_id = f"{run_id}:{int(result['request_id'])}"
            try:
                traj_samples = extract_trajectory_samples(result["pred_xyz"])
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
                    respawn_revision=int(
                        result.get("respawn_revision", respawn_revision)
                    ),
                    selected_candidate_index=None,
                    candidate_count=0,
                    candidate_similarity_scores=[],
                    candidate_trajectories_model=None,
                    model_inference_latency_s=float(model_inference_latency_s),
                    extraction_error=str(exc),
                    **audit,
                )
                raise TrajectoryValidationError(f"trajectory_extraction_error:{exc}") from exc
            cot_text = extract_cot_text(
                result.get("extra"),
                candidate_index=int(selected_idx),
            )
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
                selected_candidate_index=int(selected_idx),
                candidate_count=int(len(traj_samples)),
                candidate_similarity_scores=[
                    None if value is None else float(value)
                    for value in similarity_scores
                ],
                candidate_trajectories_model=np.asarray(traj_samples, dtype=np.float64),
                model_inference_latency_s=float(model_inference_latency_s),
                **audit,
            )
            return {
                "proposal_id": proposal_id,
                "trajectory_samples": traj_samples,
                "selected_index": int(selected_idx),
                "selected_points": traj_samples[selected_idx],
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
                    result.get("model_inference_time")
                    or result.get("inference_time")
                    or 0.0
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
                rejection_reason=(
                    validity.rejection_reason or alignment.rejection_reason
                ),
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
                simulation_time_s=(
                    current_simulation_time_s if identity_available else None
                ),
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
                remaining_horizon_s=(
                    remaining_horizon_s if current_plan is not None else None
                ),
                remaining_horizon_proxy_s=(
                    remaining_horizon_s
                    if current_frame_id_quality == "loop_counter_proxy"
                    else None
                ),
                speed_mps=float(state["speed"]),
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
                safety_assessment=safety_assessment,
                rejection_reason=rejection_reason,
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
            event_type="tick",
            emit_event=True,
            identity_available=True,
            control_timing="command_applied_after_observation",
        ):
            """Validate, safety-arbitrate, apply, and audit one vehicle command."""

            nonlocal prev_control

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

            adapter_assessment = None
            try:
                if (
                    plan is None
                    and nominal_command is not None
                    and not arbitration_errors
                ):
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
                        adapter_assessment = safety_adapter.assess(
                            tick_context=tick_context,
                            plan=plan,
                        )
                        road_assessment = adapter_assessment.road
                        obstacle_assessment = adapter_assessment.obstacles
                    except Exception as exc:
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
            if emit_event:
                _emit_control_tick(
                    state=state,
                    controller_state=controller_state,
                    requested_control=(
                        requested_command.to_json_dict()
                        if requested_command is not None
                        else None
                    ),
                    nominal_control=(
                        nominal_command.to_json_dict()
                        if nominal_command is not None
                        else None
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
                    safety_assessment={
                        "safety_source": safety_source,
                        "adapter_assessment_available": adapter_assessment is not None,
                        "arbitration_errors": arbitration_errors,
                        **safety_decision.to_json_dict(),
                    },
                    applied_control_source=safety_decision.applied_control_source,
                    identity_available=identity_available,
                    control_timing=control_timing,
                )
            return safety_decision

        def _auto_respawn(reason):
            nonlocal current_plan
            nonlocal current_trajectory, current_pred_xyz, prev_selected_trajectory
            nonlocal current_selected_traj_idx, current_cot, current_inference_time
            nonlocal current_trajectory_ts, prev_control, pending_inference, pid_follower
            nonlocal pending_request_id
            nonlocal respawn_revision, last_vqa_submitted_revision, last_vqa_completed_revision
            nonlocal current_plan_id, current_plan_source_loop_tick_id
            nonlocal current_plan_source_elapsed_proxy_s, respawn_count

            print(f"[Frame {frame_count}] Auto-respawn: {reason}")
            carla_if.respawn_ego_vehicle()
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
            prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
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

        def _prepare_current_model_input():
            """Build model input from the current complete temporal frame buffer."""

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
                    images_array[camera_index, temporal_index] = frame_images[
                        camera_index
                    ]
            history_xyz, history_rot = carla_if.get_history_in_local_frame()
            source_entry = frame_buffer[-1]
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
                for t, frame_entry in enumerate(frame_buffer):
                    frame_images = frame_entry["images"]
                    for c in range(cfg.NUM_CAMERAS):
                        images_array[c, t] = frame_images[c]
                history_xyz, history_rot = carla_if.get_history_in_local_frame()
                source_entry = frame_buffer[-1]
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
                        superseded_request_ids.extend(
                            superseded.get("superseded_request_ids", ())
                        )
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
            execution="async" if args.async_mode else "sync",
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
                and int(observation_entry["frame_id"])
                != int(frame_buffer[-1]["frame_id"]) + 1
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

            if args.async_mode:
                if args.mode == "vqa":
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
                        and bool(nav_state.vqa_question)
                        and not pending_inference
                        and nav_state.revision != last_vqa_submitted_revision
                        and nav_state.revision != last_vqa_completed_revision
                    )
                else:
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
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
                    last_inference_submit_simulation_time_s = float(
                        current_simulation_time_s
                    )
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
                    for superseded_request_id in latest_result.get(
                        "superseded_request_ids", ()
                    ):
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
                        elif (
                            "error" not in latest_result
                            and latest_result.get("mode") == "vqa"
                        ):
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
                            current_trajectory_ts = float(latest_result["result_ts"])
                            try:
                                candidate_plan, _validity = _build_and_validate_fixed_plan(
                                    latest_result,
                                    proposal,
                                )
                            except TrajectoryValidationError as exc:
                                result_status = "rejected_plan"
                                result_rejection_reason = exc.reason
                                print(
                                    f"[Frame {frame_count}] Rejected Alpamayo proposal: "
                                    f"{exc.reason}"
                                )
                            else:
                                current_plan = candidate_plan
                                current_selected_traj_idx = proposal["selected_index"]
                                current_trajectory = proposal["selected_points"]
                                prev_selected_trajectory = current_trajectory.copy()
                                current_pred_xyz = proposal["trajectory_samples"]
                                current_cot = proposal["coc_text"]
                                current_plan_id = candidate_plan.plan_id
                                latest_result["accepted_plan_id"] = current_plan_id
                                current_plan_source_loop_tick_id = int(
                                    latest_result["source_loop_tick_id"]
                                )
                                current_plan_source_elapsed_proxy_s = float(
                                    latest_result["source_elapsed_proxy_s"]
                                )
                                result_status = "accepted_plan"

                                print(
                                    f"[Frame {frame_count}] Inference done: "
                                    f"{inference_time:.2f}s "
                                    f"(submitted at frame {latest_result['frame_submitted']})"
                                )
                                print(f"    CoT: {current_cot[:60]}...")
                                print(
                                    f"    Nav: "
                                    f"{latest_result.get('navigation_text') or '(none)'} "
                                    f"(weight="
                                    f"{latest_result.get('navigation_weight', 1.0):.2f})"
                                )
                                print(
                                    f"    Selected traj sample: "
                                    f"{current_selected_traj_idx}/"
                                    f"{cfg.NUM_TRAJ_SAMPLES - 1}"
                                )
                                print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")
                        else:
                            print(
                                f"[Frame {frame_count}] Inference error: "
                                f"{latest_result['error']}"
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
                        print(
                            f"[Frame {frame_count}] Rejected malformed inference result: {exc}"
                        )
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
                                source_simulation_time_s=float(
                                    source_entry["simulation_time_s"]
                                ),
                                frame_id_quality=source_entry["frame_id_quality"],
                                prompt_revision=nav_state.revision,
                                respawn_revision=respawn_revision,
                            )
                            model_start_time = time.monotonic()
                            stage = "model_inference"
                            try:
                                if model_input_error is not None:
                                    raise RuntimeError(
                                        f"model_input_error:{model_input_error}"
                                    )
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
                                print(
                                    f"[Frame {frame_count}] VQA: "
                                    f"{model_inference_time:.2f}s"
                                )
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
                        last_inference_submit_simulation_time_s = float(
                            current_simulation_time_s
                        )
                        submission_monotonic_s = time.monotonic()
                        active_sync_request = {
                            "request_id": request_id,
                            "mode": args.mode,
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
                                raise RuntimeError(
                                    f"model_input_error:{model_input_error}"
                                )
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
                                "prompt_revision": nav_state.revision,
                                "respawn_revision": respawn_revision,
                            }
                            proposal = _extract_and_audit_proposal(
                                sync_result,
                                model_inference_latency_s=model_inference_time,
                            )
                            candidate_plan, _validity = _build_and_validate_fixed_plan(
                                sync_result,
                                proposal,
                            )
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
                                status=(
                                    "rejected_plan"
                                    if rejected_output
                                    else "error"
                                ),
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
                            current_plan = candidate_plan
                            current_selected_traj_idx = proposal["selected_index"]
                            current_trajectory = proposal["selected_points"]
                            prev_selected_trajectory = current_trajectory.copy()
                            current_pred_xyz = proposal["trajectory_samples"]
                            current_cot = proposal["coc_text"]
                            current_inference_time = model_inference_time
                            current_trajectory_ts = time.monotonic()
                            current_plan_id = candidate_plan.plan_id
                            current_plan_source_loop_tick_id = int(frame_count)
                            current_plan_source_elapsed_proxy_s = (
                                _simulation_elapsed_proxy_s()
                            )

                            print(
                                f"[Frame {frame_count}] Inference: "
                                f"{model_inference_time:.2f}s"
                            )
                            print(f"    CoT: {current_cot[:60]}...")
                            if args.mode == "navigation":
                                print(
                                    f"    Nav: {nav_state.navigation_text or '(none)'} "
                                    f"(weight={nav_state.navigation_weight:.2f})"
                                )
                            print(
                                f"    Selected traj sample: {current_selected_traj_idx}/"
                                f"{cfg.NUM_TRAJ_SAMPLES - 1}"
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
            if current_plan is not None:
                execution_validity = validate_plan_for_execution(
                    current_plan,
                    float(current_simulation_time_s),
                    current_prompt_revision=nav_state.revision,
                    current_respawn_revision=respawn_revision,
                )
                alignment_validity = validate_plan_alignment(
                    current_plan,
                    float(current_simulation_time_s),
                    current_observation_entry["capture_pose_world"],
                )
                if not execution_validity.valid or not alignment_validity.valid:
                    expired_plan_reason = (
                        execution_validity.rejection_reason
                        or alignment_validity.rejection_reason
                    )
                    emit_runtime_event(
                        "plan_validation",
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
                    )
                    current_plan = None
                    current_trajectory = None
                    current_plan_id = None
                    current_plan_source_loop_tick_id = None
                    current_plan_source_elapsed_proxy_s = None
                    pid_follower.reset_plan_progress()

            if current_plan is not None:
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
                        )
                    )
                    requested_control = {
                        "steering": float(steering_raw),
                        "throttle": float(throttle_raw),
                        "brake": float(brake_raw),
                    }

                    alpha = cfg.CONTROL_SMOOTH_ALPHA
                    emergency_stop_requested = bool(
                        ctrl_debug.get("bypass_smoothing", False)
                    )
                    if emergency_stop_requested:
                        steering = float(steering_raw)
                        throttle = 0.0
                        brake = 1.0
                        postprocessing = ("emergency_brake_bypass_ema",)
                    else:
                        steering = (
                            (1.0 - alpha) * prev_control["steer"]
                            + alpha * steering_raw
                        )
                        throttle = (
                            (1.0 - alpha) * prev_control["throttle"]
                            + alpha * throttle_raw
                        )
                        brake = (
                            (1.0 - alpha) * prev_control["brake"]
                            + alpha * brake_raw
                        )
                        postprocessing = (
                            "ema_smoothing",
                            "throttle_brake_arbitration",
                        )

                    if throttle >= brake:
                        brake = 0.0
                    else:
                        throttle = 0.0

                    nominal_control = {
                        "steering": float(np.clip(steering, -1.0, 1.0)),
                        "throttle": float(np.clip(throttle, 0.0, 1.0)),
                        "brake": float(np.clip(brake, 0.0, 1.0)),
                    }
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
                )
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
                        camera_pose_world = (
                            np.asarray(observation.ego_pose_world, dtype=np.float64)
                            @ np.asarray(
                                observation.camera_extrinsics[1],
                                dtype=np.float64,
                            )
                        )
                        camera_intrinsic = observation.camera_intrinsics[1]
                    first_future_index = int(
                        np.searchsorted(
                            current_plan.waypoint_times_s,
                            float(current_simulation_time_s),
                            side="right",
                        )
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
                        world_trajectory=current_plan.world_points[
                            first_future_index:
                        ],
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
                        applied_control_source=(
                            safety_decision.applied_control_source
                        ),
                        safety_override_applied=(
                            safety_decision.safety_override_applied
                        ),
                        safety_override_reason=safety_decision.primary_reason,
                    )
                    latest_ui_frame = vis_frame
                    if cfg.SAVE_VIDEO:
                        video_recorder.add_frame(vis_frame)

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
                    "applied_control_source": (
                        safety_decision.applied_control_source
                    ),
                    "safety_override_applied": (
                        safety_decision.safety_override_applied
                    ),
                    "safety_override_reason": safety_decision.primary_reason,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

                print(
                    f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}"
                )
                if current_plan is not None:
                    print(
                        "    Trajectory source age (CARLA simulation time): "
                        f"{plan_age_s:.2f}s"
                    )
            else:
                waiting_reason = (
                    expired_plan_reason
                    or (
                        "vqa_mode_has_no_trajectory_control"
                        if args.mode == "vqa"
                        else "waiting_for_valid_plan"
                    )
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
                sync_terminal_error = (
                    f"{type(terminal_exc).__name__}:{terminal_exc}"
                )
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
        camera_sync_stats = (
            get_camera_sync_stats() if callable(get_camera_sync_stats) else None
        )
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
            respawn_count=int(respawn_count),
            episode_collision_count=carla_if.get_episode_collision_count(),
            telemetry_path=args.telemetry_jsonl,
            telemetry_write_failed=telemetry_write_failed,
            async_shutdown_error=async_shutdown_error,
            cleanup_error=cleanup_error,
        )
        summary["exact_plan_age_available"] = bool(
            summary["source_age_s"]["count"] > 0
        )
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

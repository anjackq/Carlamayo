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
from module.navigation_control import NavigationControlState
from module.pid_controller import OfficialPIDFollower
from module.respawn_control import RespawnMonitor
from module.runtime_metrics import JsonlWriter, RuntimeMetrics
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


def capture_initial_ui_frame(carla_if, frame_count):
    """Tick once so paused pygame starts with a real camera frame."""

    carla_if.apply_control(0.0, 0.0, 1.0)
    carla_if.tick()
    frame_count += 1
    state = carla_if.get_ego_state()
    carla_if.update_history(state)
    images = carla_if.get_camera_images()
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

    print("\nLoading model...")
    configure_cuda_linalg_library(args.cuda_linalg_library)
    model = None
    processor = None
    if not args.oom_free:
        model, processor = load_model(args.quantization, device_map=args.device_map)
        print("Model loaded!")
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
    else:
        # Defer loading until CARLA has spawned its cameras/NPCs so the OOM-free
        # residency plan is computed against the VRAM CARLA actually leaves free.
        print("OOM-free mode: Alpamayo loads after CARLA is fully spawned.")

    carla_if = CARLAInterface()
    video_recorder = (
        VideoRecorder(
            cfg.OUTPUT_VIDEO,
            fps=cfg.VIDEO_FPS,
            preview_path=cfg.LIVE_PREVIEW_IMAGE,
            preview_interval_frames=max(1, cfg.VIDEO_FPS // 2),
        )
        if cfg.SAVE_VIDEO
        else None
    )
    pygame_ui = None
    pygame_ui_recorder = None
    latest_ui_frame = None
    latest_telemetry = {}
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    run_started_monotonic_s = None
    runtime_metrics = None
    telemetry_writer = JsonlWriter(args.telemetry_jsonl) if args.telemetry_jsonl else None
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

    def emit_runtime_event(event_type, **fields):
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
        )
        if telemetry_writer is not None and not telemetry_write_failed:
            try:
                telemetry_writer.append(payload)
            except Exception as exc:
                telemetry_write_failed = True
                print(f"Warning: runtime telemetry disabled after write failure: {exc}")
        return payload

    if args.pygame_ui:
        from module.pygame_ui import ClosedLoopPygameUI

        pygame_ui = ClosedLoopPygameUI(
            width=cfg.PYGAME_WINDOW_WIDTH,
            height=cfg.PYGAME_WINDOW_HEIGHT,
            mode=args.mode,
        )
        pygame_ui_recorder = VideoRecorder(args.pygame_ui_video, fps=cfg.VIDEO_FPS)

    def draw_pygame_ui(frame_rgb, telemetry):
        if pygame_ui is None:
            return
        pygame_ui.draw(frame_rgb, nav_state, telemetry)
        if pygame_ui_recorder is not None:
            pygame_ui_recorder.add_frame(pygame_ui.capture_frame())

    try:
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
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

        pending_inference = False
        pending_request_id = None
        outstanding_async_requests = {}
        last_inference_submit_ts = 0.0
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
            emit_runtime_event(
                "inference_result",
                request_id=int(request_id),
                mode=result.get("mode", request["mode"]),
                status=status,
                rejected=rejection_reason is not None,
                rejection_reason=rejection_reason,
                source_loop_tick_id=source_loop_tick_id,
                source_elapsed_proxy_s=request["source_elapsed_proxy_s"],
                source_carla_frame_id=None,
                source_simulation_time_s=None,
                arrival_loop_tick_id=int(frame_count),
                arrival_carla_frame_id=None,
                arrival_simulation_time_s=None,
                frame_id_quality="loop_counter_proxy",
                source_age_s=None,
                source_age_proxy_s=source_age_proxy_s,
                inference_wall_latency_s=request_lifetime_s if has_result else None,
                request_lifetime_s=request_lifetime_s,
                model_inference_latency_s=result.get("model_inference_time"),
                worker_compute_latency_s=result.get("inference_time"),
                prompt_revision=request["prompt_revision"],
                respawn_revision=request["respawn_revision"],
                source_plan_id=current_plan_id if status == "accepted_plan" else None,
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
        ):
            request_lifetime_s = max(0.0, time.monotonic() - submission_monotonic_s)
            emit_runtime_event(
                "inference_result",
                request_id=request_id,
                mode=mode,
                status=status,
                rejected=rejection_reason is not None,
                rejection_reason=rejection_reason,
                source_loop_tick_id=int(frame_count),
                source_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                source_carla_frame_id=None,
                source_simulation_time_s=None,
                arrival_loop_tick_id=int(frame_count),
                arrival_carla_frame_id=None,
                arrival_simulation_time_s=None,
                frame_id_quality="loop_counter_proxy",
                source_age_s=None,
                source_age_proxy_s=0.0,
                inference_wall_latency_s=request_lifetime_s,
                request_lifetime_s=request_lifetime_s,
                model_inference_latency_s=model_inference_latency_s,
                prompt_revision=nav_state.revision,
                respawn_revision=respawn_revision,
                source_plan_id=source_plan_id,
            )

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

        def _current_plan_timing_proxies():
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
            fallback_state="NONE",
            fallback_reason=None,
            rejection_reason=None,
            control_origin="nominal_controller",
            postprocessing=(),
            event_type="tick",
        ):
            plan_age_proxy_s, remaining_horizon_proxy_s = _current_plan_timing_proxies()
            return emit_runtime_event(
                event_type,
                loop_tick_id=int(frame_count),
                carla_frame_id=None,
                simulation_time_s=None,
                simulation_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                frame_id_quality="loop_counter_proxy",
                source_carla_frame_id=None,
                source_simulation_time_s=None,
                source_loop_tick_id=current_plan_source_loop_tick_id,
                source_elapsed_proxy_s=current_plan_source_elapsed_proxy_s,
                source_age_s=None,
                plan_source_age_proxy_s=plan_age_proxy_s,
                remaining_horizon_s=None,
                remaining_horizon_proxy_s=remaining_horizon_proxy_s,
                speed_mps=float(state["speed"]),
                controller_state=controller_state,
                control_timing="command_applied_after_observation",
                control_origin=control_origin,
                requested_control=requested_control,
                applied_control=applied_control,
                postprocessing=list(postprocessing),
                fallback_state=fallback_state,
                fallback_reason=fallback_reason,
                safety_override_applied=False,
                safety_override_type=None,
                safety_override_reason=None,
                rejection_reason=rejection_reason,
                source_plan_id=current_plan_id,
                source_plan_frame_id=None,
                source_plan_loop_tick_id=current_plan_source_loop_tick_id,
                prompt_revision=nav_state.revision,
                respawn_revision=respawn_revision,
                inference_pending=bool(pending_inference),
                collision_count=carla_if.get_episode_collision_count(),
            )

        def _auto_respawn(reason):
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
            current_trajectory = None
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
                for t, frame_images in enumerate(frame_buffer):
                    for c in range(cfg.NUM_CAMERAS):
                        images_array[c, t] = frame_images[c]
                history_xyz, history_rot = carla_if.get_history_in_local_frame()
                request_sequence += 1
                return {
                    "request_id": request_sequence,
                    "mode": args.mode,
                    "images_array": images_array,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "navigation_text": nav_state.navigation_text,
                    "navigation_weight": nav_state.navigation_weight,
                    "vqa_question": nav_state.vqa_question,
                    "prompt_revision": nav_state.revision,
                    "respawn_revision": respawn_revision,
                    "source_loop_tick_id": int(frame_count),
                    "source_elapsed_proxy_s": _simulation_elapsed_proxy_s(),
                    "source_carla_frame_id": None,
                    "source_simulation_time_s": None,
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
            frame_count, latest_ui_frame, latest_telemetry = capture_initial_ui_frame(
                carla_if,
                frame_count,
            )
        emit_runtime_event(
            "episode_start",
            mode=args.mode,
            execution="async" if args.async_mode else "sync",
            frame_id_quality="loop_counter_proxy",
            exact_sensor_frame_ids_available=False,
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
                    current_trajectory = None
                    current_pred_xyz = None
                    current_trajectory_ts = None
                    current_plan_id = None
                    current_plan_source_loop_tick_id = None
                    current_plan_source_elapsed_proxy_s = None
                    pending_inference = False
                    pending_request_id = None
                    emit_runtime_event(
                        "prompt_revision_changed",
                        loop_tick_id=int(frame_count),
                        prompt_revision=nav_state.revision,
                        mode=args.mode,
                    )
                    last_seen_nav_revision = nav_state.revision
                if nav_state.paused:
                    if not pause_brake_active:
                        carla_if.apply_control(0.0, 0.0, 1.0)
                        _emit_control_tick(
                            state=carla_if.get_ego_state(),
                            controller_state="PAUSED",
                            requested_control=None,
                            applied_control={
                                "steering": 0.0,
                                "throttle": 0.0,
                                "brake": 1.0,
                            },
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

            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)

            collision_decision = respawn_monitor.check_collision(
                frame_count=frame_count,
                collision_count=carla_if.get_collision_count(),
                last_collision_event=carla_if.get_last_collision_event(),
            )
            if collision_decision.should_respawn:
                _auto_respawn(collision_decision.reason)
                _emit_control_tick(
                    state=state,
                    controller_state="RESPAWNING",
                    requested_control=None,
                    applied_control={"steering": 0.0, "throttle": 0.0, "brake": 1.0},
                    fallback_state="RESPAWN_BRAKE",
                    fallback_reason=collision_decision.reason,
                    control_origin="respawn_reset",
                )
                continue

            try:
                images = carla_if.get_camera_images()
            except TimeoutError as exc:
                print(f"[Frame {frame_count}] Warning: {exc}; braking and skipping this tick.")
                carla_if.apply_control(0.0, 0.0, 1.0)
                prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
                _emit_control_tick(
                    state=state,
                    controller_state="CAMERA_TIMEOUT",
                    requested_control=None,
                    applied_control={"steering": 0.0, "throttle": 0.0, "brake": 1.0},
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
            if len(images) > 1:
                latest_ui_frame = images[1]
            latest_telemetry = {
                "frame": frame_count,
                "speed_kmh": state["speed"] * 3.6,
                "steering": prev_control["steer"],
                "inference_time": current_inference_time,
            }
            frame_buffer.append(images)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)

            if args.async_mode:
                now_ts = time.monotonic()
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
                        and (now_ts - last_inference_submit_ts) >= inference_interval_sec
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
                            "submission_monotonic_s",
                            "prompt_revision",
                            "respawn_revision",
                        )
                    }
                    last_inference_submit_ts = now_ts
                    emit_runtime_event(
                        "inference_submitted",
                        request_id=req["request_id"],
                        mode=req["mode"],
                        source_loop_tick_id=req["source_loop_tick_id"],
                        source_elapsed_proxy_s=req["source_elapsed_proxy_s"],
                        source_carla_frame_id=None,
                        source_simulation_time_s=None,
                        frame_id_quality="loop_counter_proxy",
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
                        if (
                            latest_result.get("prompt_revision", nav_state.revision)
                            != nav_state.revision
                        ):
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
                            pred_xyz = latest_result["pred_xyz"]
                            extra = latest_result["extra"]
                            inference_time = float(latest_result["inference_time"])
                            traj_samples = extract_trajectory_samples(pred_xyz)
                            selected_idx, _similarity_scores = (
                                select_trajectory_by_prev_similarity(
                                    traj_samples,
                                    prev_selected_trajectory,
                                )
                            )
                            current_selected_traj_idx = selected_idx
                            current_trajectory = traj_samples[selected_idx]
                            prev_selected_trajectory = current_trajectory.copy()
                            current_pred_xyz = traj_samples
                            current_cot = extract_cot_text(extra)
                            current_inference_time = inference_time
                            current_trajectory_ts = float(latest_result["result_ts"])
                            current_plan_id = f"{run_id}:{latest_result['request_id']}"
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
                                f"    Nav: {latest_result.get('navigation_text') or '(none)'} "
                                f"(weight={latest_result.get('navigation_weight', 1.0):.2f})"
                            )
                            print(
                                f"    Selected traj sample: {current_selected_traj_idx}/"
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
                        raise
                    else:
                        _emit_async_terminal(
                            latest_result.get("request_id"),
                            result_status,
                            result_rejection_reason,
                            result=latest_result,
                        )
            else:
                if len(frame_buffer) >= cfg.NUM_FRAMES:
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
                    for t, frame_images in enumerate(frame_buffer):
                        for c in range(cfg.NUM_CAMERAS):
                            images_array[c, t] = frame_images[c]

                    history_xyz, history_rot = carla_if.get_history_in_local_frame()

                    model_data = prepare_model_input(images_array, history_xyz, history_rot)
                    if args.mode == "vqa":
                        should_run_vqa = (
                            bool(nav_state.vqa_question)
                            and nav_state.revision != last_vqa_completed_revision
                        )
                        if should_run_vqa:
                            request_sequence += 1
                            request_id = request_sequence
                            submission_monotonic_s = time.monotonic()
                            emit_runtime_event(
                                "inference_submitted",
                                request_id=request_id,
                                mode="vqa",
                                source_loop_tick_id=int(frame_count),
                                source_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                                source_carla_frame_id=None,
                                source_simulation_time_s=None,
                                frame_id_quality="loop_counter_proxy",
                                prompt_revision=nav_state.revision,
                                respawn_revision=respawn_revision,
                            )
                            model_start_time = time.monotonic()
                            stage = "model_inference"
                            try:
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
                                raise
                            _emit_sync_terminal(
                                request_id=request_id,
                                mode="vqa",
                                status="completed_vqa",
                                rejection_reason=None,
                                submission_monotonic_s=submission_monotonic_s,
                                model_inference_latency_s=model_inference_time,
                            )
                    else:
                        navigation_text = (
                            nav_state.navigation_text if args.mode == "navigation" else ""
                        )
                        navigation_weight = (
                            nav_state.navigation_weight if args.mode == "navigation" else 1.0
                        )
                        request_sequence += 1
                        request_id = request_sequence
                        submission_monotonic_s = time.monotonic()
                        emit_runtime_event(
                            "inference_submitted",
                            request_id=request_id,
                            mode=args.mode,
                            source_loop_tick_id=int(frame_count),
                            source_elapsed_proxy_s=_simulation_elapsed_proxy_s(),
                            source_carla_frame_id=None,
                            source_simulation_time_s=None,
                            frame_id_quality="loop_counter_proxy",
                            prompt_revision=nav_state.revision,
                            respawn_revision=respawn_revision,
                        )
                        model_start_time = time.monotonic()
                        stage = "model_inference"
                        try:
                            pred_xyz, extra = _run_inference_with_nav_fallback(
                                model_data,
                                navigation_text=navigation_text,
                                navigation_weight=navigation_weight,
                            )
                            model_inference_time = time.monotonic() - model_start_time
                            stage = "result_processing"

                            traj_samples = extract_trajectory_samples(pred_xyz)
                            selected_idx, _similarity_scores = (
                                select_trajectory_by_prev_similarity(
                                    traj_samples,
                                    prev_selected_trajectory,
                                )
                            )
                            current_selected_traj_idx = selected_idx
                            current_trajectory = traj_samples[selected_idx]
                            prev_selected_trajectory = current_trajectory.copy()
                            current_pred_xyz = traj_samples
                            current_cot = extract_cot_text(extra)
                            current_inference_time = model_inference_time
                            current_trajectory_ts = time.monotonic()
                            current_plan_id = f"{run_id}:{request_id}"
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
                        except Exception as exc:
                            _emit_sync_terminal(
                                request_id=request_id,
                                mode=args.mode,
                                status="error",
                                rejection_reason=f"{stage}_error:{exc}",
                                submission_monotonic_s=submission_monotonic_s,
                                model_inference_latency_s=(
                                    model_inference_time
                                    if stage == "result_processing"
                                    else time.monotonic() - model_start_time
                                ),
                            )
                            raise
                        _emit_sync_terminal(
                            request_id=request_id,
                            mode=args.mode,
                            status="accepted_plan",
                            rejection_reason=None,
                            submission_monotonic_s=submission_monotonic_s,
                            model_inference_latency_s=model_inference_time,
                            source_plan_id=current_plan_id,
                        )

            if current_trajectory is not None:
                vehicle_tf = carla_if.ego_vehicle.get_transform()
                steering_raw, throttle_raw, brake_raw, _ctrl_debug = pid_follower.compute_control(
                    vehicle_tf,
                    current_trajectory[:, :3],
                    float(state["speed"]),
                )

                alpha = cfg.CONTROL_SMOOTH_ALPHA
                steering = (1.0 - alpha) * prev_control["steer"] + alpha * steering_raw
                throttle = (1.0 - alpha) * prev_control["throttle"] + alpha * throttle_raw
                brake = (1.0 - alpha) * prev_control["brake"] + alpha * brake_raw

                if throttle >= brake:
                    brake = 0.0
                else:
                    throttle = 0.0

                prev_control = {"steer": steering, "throttle": throttle, "brake": brake}
                carla_if.apply_control(steering, throttle, brake)
                _emit_control_tick(
                    state=state,
                    controller_state="TRACKING",
                    requested_control={
                        "steering": float(steering_raw),
                        "throttle": float(throttle_raw),
                        "brake": float(brake_raw),
                    },
                    applied_control={
                        "steering": float(steering),
                        "throttle": float(throttle),
                        "brake": float(brake),
                    },
                    postprocessing=("ema_smoothing", "throttle_brake_arbitration"),
                )

                if current_pred_xyz is not None:
                    cam_img = images[1]
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
                    )
                    latest_ui_frame = vis_frame
                    if cfg.SAVE_VIDEO:
                        video_recorder.add_frame(vis_frame)

                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": steering,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

                print(
                    f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}"
                )
                if current_trajectory_ts is not None and args.async_mode:
                    print(
                        "    Trajectory result age (wall-clock, excludes inference): "
                        f"{time.monotonic() - current_trajectory_ts:.2f}s"
                    )
            else:
                waiting_reason = (
                    "vqa_mode_has_no_trajectory_control"
                    if args.mode == "vqa"
                    else "waiting_for_first_plan"
                )
                carla_if.apply_control(0.0, 0.0, 1.0)
                _emit_control_tick(
                    state=state,
                    controller_state="WAITING_FOR_PLAN",
                    requested_control=None,
                    applied_control={"steering": 0.0, "throttle": 0.0, "brake": 1.0},
                    fallback_state="WAITING_FOR_PLAN",
                    fallback_reason=waiting_reason,
                    control_origin="waiting_for_plan",
                )
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": 0.0,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        emit_runtime_event("episode_stop_requested", status=stop_reason)
        print("\n\nInterrupted by user.")
    except Exception as e:
        stop_reason = "error"
        run_error = str(e)
        emit_runtime_event("runtime_error", status="error", error=run_error)
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if args.async_mode and inference_stop is not None:
            inference_stop.set()
            try:
                inference_request_q.put_nowait(None)
            except Exception:
                pass
            if worker_thread is not None:
                worker_thread.join(timeout=2.0)
            _clear_async_queues("episode_end")
            for outstanding_request_id in list(outstanding_async_requests):
                _emit_async_terminal(
                    outstanding_request_id,
                    "cancelled",
                    "episode_ended_before_result_consumption",
                )
        if runtime_metrics is None:
            runtime_metrics = RuntimeMetrics()
        runtime_metrics.record_collision_count(carla_if.get_episode_collision_count())
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
            exact_source_frame_ids_available=False,
            exact_plan_age_available=False,
            respawn_count=int(respawn_count),
            episode_collision_count=carla_if.get_episode_collision_count(),
            telemetry_path=args.telemetry_jsonl,
            telemetry_write_failed=telemetry_write_failed,
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
        print("  exact source-frame and plan-age metrics: unavailable until PR2")
        print(
            "  source-age proxy p50/p95/p99: "
            f"{source_age_proxy_summary['p50']!r} / "
            f"{source_age_proxy_summary['p95']!r} / "
            f"{source_age_proxy_summary['p99']!r} s"
        )
        if args.telemetry_jsonl:
            print(f"  telemetry: {args.telemetry_jsonl}")
        try:
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
        finally:
            carla_if.cleanup()

    print("\nStopped.")


if __name__ == "__main__":
    main()

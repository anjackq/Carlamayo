"""Visualization and video recording helpers."""

import os
import shutil
import subprocess
import textwrap

import cv2
import numpy as np


def project_world_points_to_camera(
    world_points,
    camera_pose_world,
    camera_intrinsic,
    *,
    minimum_depth_m=0.5,
):
    """Project CARLA world points using an exact sensor pose and calibration.

    CARLA camera coordinates are ``x`` forward, ``y`` right, ``z`` up.  The
    returned boolean mask identifies finite points in front of the camera.
    """

    points = np.asarray(world_points, dtype=np.float64)
    camera_pose = np.asarray(camera_pose_world, dtype=np.float64)
    projection_model = (
        camera_intrinsic
        if callable(getattr(camera_intrinsic, "ray_to_pixel", None))
        else None
    )
    intrinsic = (
        None
        if projection_model is not None
        else np.asarray(camera_intrinsic, dtype=np.float64)
    )
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"world_points must have shape (N, >=3), got {points.shape}")
    if camera_pose.shape != (4, 4):
        raise ValueError("camera pose matrix must be 4x4")
    if projection_model is None and intrinsic.shape != (3, 3):
        raise ValueError("camera intrinsic matrix must be 3x3")
    if not (
        np.isfinite(points[:, :3]).all()
        and np.isfinite(camera_pose).all()
        and (projection_model is not None or np.isfinite(intrinsic).all())
    ):
        raise ValueError("projection inputs must be finite")

    homogeneous = np.column_stack([points[:, :3], np.ones(len(points))])
    camera_points = (np.linalg.inv(camera_pose) @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 0]
    valid = depth > float(minimum_depth_m)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if projection_model is not None:
        # CARLA camera x-forward/y-right/z-up -> OpenCV
        # x-right/y-down/z-forward.
        optical_rays = np.column_stack(
            [camera_points[:, 1], -camera_points[:, 2], camera_points[:, 0]]
        )
        projected, projection_valid = projection_model.ray_to_pixel(optical_rays)
        pixels[:] = projected
        valid &= np.asarray(projection_valid, dtype=bool)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels[valid, 0] = (
                intrinsic[0, 0] * camera_points[valid, 1] / depth[valid]
                + intrinsic[0, 2]
            )
            pixels[valid, 1] = (
                intrinsic[1, 2]
                - intrinsic[1, 1] * camera_points[valid, 2] / depth[valid]
            )
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, valid


def project_world_trajectory_to_image(
    cam_img,
    world_points,
    camera_pose_world,
    camera_intrinsic,
    *,
    last_safe_waypoint_index=None,
):
    """Draw an authorized prefix and advisory future path on the current image."""

    result = np.asarray(cam_img).copy()
    pixels, valid = project_world_points_to_camera(
        world_points,
        camera_pose_world,
        camera_intrinsic,
    )
    height, width = result.shape[:2]
    inside = (
        valid
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    for index in range(len(pixels) - 1):
        if inside[index] and inside[index + 1]:
            start = tuple(np.rint(pixels[index]).astype(np.int32))
            end = tuple(np.rint(pixels[index + 1]).astype(np.int32))
            segment_is_safe = (
                last_safe_waypoint_index is None
                or index + 1 <= int(last_safe_waypoint_index)
            )
            color = (0, 255, 80) if segment_is_safe else (255, 210, 0)
            cv2.line(result, start, end, color, 8, cv2.LINE_AA)
    for index, pixel in enumerate(pixels):
        if not inside[index]:
            continue
        point_is_safe = (
            last_safe_waypoint_index is None
            or index <= int(last_safe_waypoint_index)
        )
        cv2.circle(
            result,
            tuple(np.rint(pixel).astype(np.int32)),
            5,
            (80, 255, 120) if point_is_safe else (255, 225, 40),
            -1,
            cv2.LINE_AA,
        )
    if last_safe_waypoint_index is not None:
        first_bad_index = int(last_safe_waypoint_index) + 1
        if 0 <= first_bad_index < len(pixels) and inside[first_bad_index]:
            cv2.circle(
                result,
                tuple(np.rint(pixels[first_bad_index]).astype(np.int32)),
                10,
                (255, 0, 0),
                -1,
                cv2.LINE_AA,
            )
    return result


def _project_one_trajectory(
    result,
    points_3d,
    img_width,
    img_height,
    focal_length_px,
    camera_height,
    line_color,
    point_color,
    line_thickness,
):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_cam = z + camera_height
    valid = x > 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        u = img_width / 2 - (y / x) * focal_length_px
        v_temp = img_height / 2 - (z_cam / x) * focal_length_px
        v = img_height - v_temp

    u = np.clip(u, 0, img_width - 1).astype(np.int32)
    v = np.clip(v, 0, img_height - 1).astype(np.int32)
    points_2d = np.column_stack([u[valid], v[valid]])
    if len(points_2d) <= 1:
        return

    for i in range(len(points_2d) - 1):
        cv2.line(
            result,
            tuple(points_2d[i]),
            tuple(points_2d[i + 1]),
            line_color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    for pt in points_2d:
        cv2.circle(result, tuple(pt), max(4, line_thickness), point_color, -1, cv2.LINE_AA)


def project_trajectory_to_image(cam_img, pred_xyz, selected_idx=0, camera_height=2.4, fov=120):
    """Project one or multiple trajectories onto image."""
    img_height, img_width = cam_img.shape[:2]
    focal_length_px = img_width / (2 * np.tan(np.radians(fov / 2)))

    result = cam_img.copy()
    detach = getattr(pred_xyz, "detach", None)
    if callable(detach):
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        host_value = cpu() if callable(cpu) else detached
        to_numpy = getattr(host_value, "numpy", None)
        arr = to_numpy() if callable(to_numpy) else np.asarray(host_value)
    else:
        arr = np.asarray(pred_xyz)

    if arr.ndim == 2:
        traj_samples = arr[None, :, :3]
    elif arr.ndim == 3:
        traj_samples = arr[:, :, :3]
    else:
        raise ValueError(f"Expected trajectory with ndim 2 or 3, got shape {arr.shape}")

    num_samples = traj_samples.shape[0]
    selected_idx = int(np.clip(selected_idx, 0, max(0, num_samples - 1)))

    for i in range(num_samples):
        if i == selected_idx:
            continue
        _project_one_trajectory(
            result=result,
            points_3d=traj_samples[i],
            img_width=img_width,
            img_height=img_height,
            focal_length_px=focal_length_px,
            camera_height=camera_height,
            line_color=(255, 255, 255),
            point_color=(255, 255, 255),
            line_thickness=4,
        )

    _project_one_trajectory(
        result=result,
        points_3d=traj_samples[selected_idx],
        img_width=img_width,
        img_height=img_height,
        focal_length_px=focal_length_px,
        camera_height=camera_height,
        line_color=(255, 0, 0),
        point_color=(255, 100, 100),
        line_thickness=8,
    )

    return result


def create_visualization_frame(
    cam_img,
    pred_xyz,
    selected_idx,
    frame_count,
    inference_time,
    cot_text,
    speed_kmh,
    steering,
    navigation_text="",
    navigation_weight=1.0,
    paused=False,
    world_trajectory=None,
    camera_pose_world=None,
    camera_intrinsic=None,
    source_frame_id=None,
    source_age_s=None,
    controller_state="WAITING",
    requested_control=None,
    applied_control=None,
    applied_control_source="FALLBACK",
    safety_override_applied=False,
    safety_override_reason=None,
    plan_admission_status=None,
    near_term_road_status=None,
    full_path_road_status=None,
    road_speed_cap_mps=None,
    last_safe_waypoint_index=None,
    camera_alignment_mode="baseline",
):
    """Create a single visualization frame with all overlays."""
    if (
        world_trajectory is not None
        and camera_pose_world is not None
        and camera_intrinsic is not None
    ):
        vis_img = project_world_trajectory_to_image(
            cam_img,
            world_trajectory,
            camera_pose_world,
            camera_intrinsic,
            last_safe_waypoint_index=last_safe_waypoint_index,
        )
    else:
        vis_img = project_trajectory_to_image(cam_img, pred_xyz, selected_idx=selected_idx)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    h, w = vis_img.shape[:2]

    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, h - 190), (w - 10, h - 10), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    alignment_labels = {
        "baseline": "BASELINE PINHOLE",
        "pose-only": "POSE ONLY",
        "projection-only": "FTHETA ONLY",
        "pose-projection": "POSE + FTHETA",
    }
    camera_input_label = alignment_labels.get(
        camera_alignment_mode,
        str(camera_alignment_mode).upper(),
    )
    info_text = (
        f"Frame: {frame_count} | Inference: {inference_time:.2f}s | "
        f"Speed: {speed_kmh:.1f} km/h | Steer: {steering:.2f} | "
        f"CAMERA INPUT: {camera_input_label}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    pad_x = 14
    pad_y = 14
    box_x1, box_y1 = 10, 10
    box_x2 = min(w - 10, box_x1 + tw + pad_x * 2)
    box_y2 = box_y1 + th + pad_y * 2
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    cv2.putText(
        vis_img,
        info_text,
        (20, 50),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    source_text = "unknown" if source_frame_id is None else str(source_frame_id)
    age_text = "unknown" if source_age_s is None else f"{source_age_s:.2f}s"
    requested = requested_control or {}
    applied = applied_control or {}
    layer_lines = (
        (
            f"ALPAMAYO PROPOSAL | source frame {source_text} | age {age_text} | "
            f"admission={plan_admission_status or 'unknown'} | "
            f"road={near_term_road_status or 'unknown'}/"
            f"{full_path_road_status or 'unknown'}",
            (255, 255, 0),
        ),
        (
            "CONTROLLER EXECUTION | "
            f"{controller_state} | request "
            f"S/T/B={requested.get('steering', 0.0):.2f}/"
            f"{requested.get('throttle', 0.0):.2f}/"
            f"{requested.get('brake', 0.0):.2f} | "
            f"road cap="
            f"{'none' if road_speed_cap_mps is None else f'{road_speed_cap_mps:.2f}m/s'}",
            (80, 255, 120),
        ),
        (
            "SAFETY OVERRIDE | "
            f"{'ACTIVE' if safety_override_applied else 'INACTIVE'} | "
            f"source={applied_control_source} | "
            f"applied S/T/B={applied.get('steering', 0.0):.2f}/"
            f"{applied.get('throttle', 0.0):.2f}/"
            f"{applied.get('brake', 0.0):.2f} | "
            f"reason={safety_override_reason or 'none'}",
            (0, 80, 255) if safety_override_applied else (180, 180, 180),
        ),
    )
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, 72), (w - 10, 182), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.65, vis_img, 0.35, 0)
    for line_index, (line, color) in enumerate(layer_lines):
        cv2.putText(
            vis_img,
            line,
            (20, 100 + line_index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )

    status = "PAUSED" if paused else "RUNNING"
    nav_display = navigation_text or "(no navigation text)"
    nav_display = nav_display[:160] + "..." if len(nav_display) > 160 else nav_display
    cv2.putText(
        vis_img,
        f"{status} | Nav: {nav_display} | Weight: {navigation_weight:.2f}",
        (20, h - 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cot_display = cot_text[:200] + "..." if len(cot_text) > 200 else cot_text
    lines = textwrap.wrap(f"CoT: {cot_display}", width=120)
    y_offset = h - 115
    for line in lines[:3]:
        cv2.putText(
            vis_img,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def create_open_loop_visualization_frame(
    cam_img,
    pred_xyz,
    frame_count,
    total_frames,
    inference_time,
    cot_text,
):
    """Create one open-loop visualization frame with trajectory and text overlays."""

    vis_img = project_trajectory_to_image(cam_img, pred_xyz, selected_idx=0)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    height, width = vis_img.shape[:2]

    header = f"Frame: {frame_count}/{total_frames} | Inference: {inference_time:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(header, font, font_scale, thickness)
    pad_x = 14
    pad_y = 14
    box_x1 = 10
    box_y1 = 10
    box_x2 = min(width - 10, box_x1 + text_width + pad_x * 2)
    box_y2 = box_y1 + text_height + pad_y * 2
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.65, vis_img, 0.35, 0)
    cv2.putText(
        vis_img,
        header,
        (box_x1 + pad_x, box_y1 + pad_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, height - 170), (width - 10, height - 10), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    cot_display = str(cot_text or "").strip()
    cot_display = cot_display[:240] + "..." if len(cot_display) > 240 else cot_display
    lines = textwrap.wrap(f"Chain-of-Causation: {cot_display}", width=120)
    y_offset = height - 125
    for line in lines[:4]:
        cv2.putText(
            vis_img,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def save_open_loop_video(
    predictions,
    camera_images,
    cot_texts,
    inference_times,
    output_path,
    fps=5,
):
    """Render and save the open-loop inference summary video."""

    total_frames = len(predictions)
    recorder = VideoRecorder(output_path, fps=fps)
    for frame_index, (pred_xyz, cam_img, cot_text, inference_time) in enumerate(
        zip(predictions, camera_images, cot_texts, inference_times, strict=True),
        start=1,
    ):
        recorder.add_frame(
            create_open_loop_visualization_frame(
                cam_img=cam_img,
                pred_xyz=pred_xyz,
                frame_count=frame_index,
                total_frames=total_frames,
                inference_time=inference_time,
                cot_text=cot_text,
            )
        )
        if frame_index == 1 or frame_index % 10 == 0 or frame_index == total_frames:
            print(f"  Rendering frame {frame_index}/{total_frames}...")

    recorder.save()


def transcode_video_for_browser_compat(source_path, output_path):
    """Transcode OpenCV output to H.264/yuv420p for VS Code/browser players."""
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        "-crf",
        "18",
        output_path,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg failed"
    return True, "H.264/yuv420p"


class VideoRecorder:
    """Stream RGB frames to disk and optionally publish a live JPEG preview."""

    def __init__(self, output_path, fps=10, preview_path=None, preview_interval_frames=1):
        self.output_path = os.fspath(output_path)
        self.fps = fps
        self.preview_path = os.fspath(preview_path) if preview_path is not None else None
        self.preview_interval_frames = max(1, int(preview_interval_frames))
        self.frame_count = 0
        self.width = None
        self.height = None
        self._writer = None
        self._selected_codec = None
        self._temp_path = None

    def add_frame(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape HxWx3, got {frame.shape}")

        height, width = frame.shape[:2]
        if self._writer is None:
            self._initialize_writer(width, height)
        elif (width, height) != (self.width, self.height):
            raise ValueError(
                f"Video frame size changed from {self.width}x{self.height} "
                f"to {width}x{height}"
            )

        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

        if (
            self.preview_path is not None
            and (self.frame_count - 1) % self.preview_interval_frames == 0
        ):
            self._publish_preview(frame)

    def _create_writer(self, width, height, output_path):
        for codec in ("mp4v", "avc1", "H264"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None

    def _initialize_writer(self, width, height):
        output_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(output_dir, exist_ok=True)
        self._temp_path = os.path.join(
            output_dir,
            f".{os.path.basename(self.output_path)}.opencv-tmp.mp4",
        )
        writer, selected_codec = self._create_writer(width, height, self._temp_path)
        if writer is None:
            raise RuntimeError("Failed to initialize video writer.")
        if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

        self.width = width
        self.height = height
        self._writer = writer
        self._selected_codec = selected_codec

    def _publish_preview(self, frame):
        preview_dir = os.path.dirname(os.path.abspath(self.preview_path)) or "."
        os.makedirs(preview_dir, exist_ok=True)
        preview_name = os.path.basename(self.preview_path)
        preview_temp = os.path.join(preview_dir, f".{preview_name}.tmp.jpg")
        if not cv2.imwrite(preview_temp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write live preview image: {self.preview_path}")
        os.replace(preview_temp, self.preview_path)

    def save(self):
        if self.frame_count == 0:
            print("No frames to save.")
            return

        print(f"\nFinalizing video with {self.frame_count} frames...")
        self._writer.release()
        self._writer = None

        transcoded, transcode_msg = transcode_video_for_browser_compat(
            self._temp_path,
            self.output_path,
        )
        if not transcoded:
            shutil.move(self._temp_path, self.output_path)
            print(
                f"Warning: H.264 transcode skipped ({transcode_msg}); "
                f"saved OpenCV {self._selected_codec} output."
            )
        else:
            os.remove(self._temp_path)

        print(f"Video saved: {self.output_path}")
        print(
            f"  Codec: {transcode_msg if transcoded else self._selected_codec}, "
            f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, "
            f"Frames: {self.frame_count}"
        )

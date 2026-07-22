"""Configuration for CARLA closed-loop Alpamayo pipeline."""

import os

# User Config (Edit for your local CARLA version/layout)
# Used only when CARLA_ROOT/CARLA_HOME env vars are not set.
CARLA_AGENT_ROOT = os.path.expanduser("~/carla")

# Alpamayo Configuration
# Keep the camera role, CARLA calibration, order, and Alpamayo identity in one
# place.  Alpamayo was trained with camera IDs [0, 1, 2, 6] and a 120-degree
# front-wide camera.  The old prototype silently used a 95-degree wide camera
# and omitted the IDs from the model prompt.
CAMERA_SPECS = (
    {
        "name": "cam_front_left",
        "alpamayo_id": 0,
        "x": 1.0,
        "y": -0.5,
        "z": 2.4,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -60.0,
        "fov": 120.0,
    },
    {
        "name": "cam_front_wide",
        "alpamayo_id": 1,
        "x": 1.5,
        "y": 0.0,
        "z": 2.4,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": 120.0,
    },
    {
        "name": "cam_front_right",
        "alpamayo_id": 2,
        "x": 1.0,
        "y": 0.5,
        "z": 2.4,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 60.0,
        "fov": 120.0,
    },
    {
        "name": "cam_front_tele",
        "alpamayo_id": 6,
        "x": 1.5,
        "y": 0.0,
        "z": 2.4,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": 30.0,
    },
)
NUM_CAMERAS = len(CAMERA_SPECS)
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
IMG_CHANNELS = 3
# Offscreen Epic rendering can flicker heavily with CARLA camera postprocess bloom/exposure.
CAMERA_ENABLE_POSTPROCESS_EFFECTS = True
NUM_HISTORY = 16
NUM_FRAMES = 4
NUM_TRAJ_SAMPLES = 1

# Video Configuration
SAVE_VIDEO = True
OUTPUT_VIDEO = os.environ.get(
    "CARLAMAYO_OUTPUT_VIDEO",
    "carla_alpamayo_closed_loop_result.mp4",
)
LIVE_PREVIEW_IMAGE = os.environ.get(
    "CARLAMAYO_LIVE_PREVIEW_IMAGE",
    "carla_alpamayo_closed_loop_latest.jpg",
)
VIDEO_FPS = 10
PYGAME_WINDOW_WIDTH = 1280
PYGAME_WINDOW_HEIGHT = 900

# CARLA Configuration
CARLA_MAP = "Town03"  # Urban-style map
NPC_VEHICLE_COUNT = 50
NPC_WALKER_COUNT = 50
NPC_EXCLUDED_VEHICLE_KEYWORDS = (
    "ambulance",
    "carlacola",
    "cybertruck",
    "firetruck",
    "fusorosa",
    "sprinter",
)

# Control config
CONTROL_DT = 0.1
THROTTLE_MAX = 0.6
BRAKE_MAX = 1.0
CONTROL_SMOOTH_ALPHA = 0.25

# Timestamped trajectory validity.  Alpamayo predicts 64 future waypoints at
# 10 Hz.  A plan must retain enough of that horizon to be useful when it reaches
# the controller; older results fail closed instead of being re-anchored to the
# current ego pose.
TRAJECTORY_NUM_POINTS = 64
TRAJECTORY_WAYPOINT_DT = 0.1
TRAJECTORY_MAX_PLAN_AGE_S = 4.4
TRAJECTORY_MIN_REMAINING_HORIZON_S = 2.0
TRAJECTORY_STOP_MAX_DISPLACEMENT_M = 0.75
TRAJECTORY_MIN_FORWARD_PROGRESS_M = 1.0
TRAJECTORY_MAX_BACKWARD_M = 0.75
TRAJECTORY_MAX_LATERAL_M = 12.0
TRAJECTORY_MAX_STEP_M = 5.0
TRAJECTORY_MAX_SPEED_MPS = 35.0 / 3.6
TRAJECTORY_TIME_EPSILON_S = 1e-6
TRAJECTORY_MAX_TRACKING_ERROR_M = 2.5
TRAJECTORY_MAX_HEADING_ERROR_DEG = 45.0
# A stop is represented geometrically: at least this many terminal samples stay
# inside a small cluster and have near-zero spacing.  This catches a moving path
# followed by repeated endpoint samples as well as an all-stationary proposal.
TRAJECTORY_STOP_TAIL_MIN_POINTS = 6
TRAJECTORY_STOP_MAX_STEP_M = 0.05
TRAJECTORY_STOP_CLUSTER_RADIUS_M = 0.25

# Conservative safety shield.  Plan points must lie on a CARLA driving lane and
# nearby vehicles/pedestrians in the ego corridor trigger immediate braking.
SAFETY_PATH_SAMPLE_SPACING_M = 0.5
SAFETY_LATERAL_CLEARANCE_M = 0.25
SAFETY_LONGITUDINAL_CLEARANCE_M = 0.25
SAFETY_REACTION_TIME_S = 0.5
SAFETY_ASSUMED_DECELERATION_MPS2 = 4.0
SAFETY_STOP_BUFFER_M = 2.0
SAFETY_HARD_GAP_M = 1.0
SAFETY_TTC_THRESHOLD_S = 2.0
SAFETY_MINIMUM_CLOSING_SPEED_MPS = 0.1
SAFETY_PREDICTION_HORIZON_S = 3.0
SAFETY_PREDICTION_TIME_STEP_S = 0.1
SAFETY_EMERGENCY_HOLD_TICKS = 5
SAFETY_CLEAR_TICKS_TO_RELEASE = 3

# Auto-respawn after collision.
RESPAWN_COLLISION_COOLDOWN_FRAMES = 10

# Keep Alpamayo's original Qwen-VL image-token budget fixed from config.
VLM_IMAGE_PIXELS = 196608

# Official PID follower config
PID_LOOKAHEAD_MIN_M = 4.0
PID_LOOKAHEAD_MAX_M = 12.0
PID_LOOKAHEAD_SPEED_GAIN = 0.4
# Retained for callers of OfficialPIDFollower.compute_control().  The fixed-world
# controller derives speed from waypoint timestamps and intentionally has no
# positive minimum speed.
PID_TARGET_SPEED_MIN_KMH = 10.0
PID_TARGET_SPEED_MAX_KMH = 35.0
PID_TARGET_SPEED_EXTENT_GAIN = 0.5
PID_STOP_SPEED_THRESHOLD_MPS = 0.3
PID_STOP_POSITION_TOLERANCE_M = 0.5
PID_COMFORTABLE_DECEL_MPS2 = 2.5
PID_DECELERATION_STATE_DELTA_MPS = 0.25
PID_LAT_KP = 1.1
PID_LAT_KI = 0.02
PID_LAT_KD = 0.15
PID_LON_KP = 0.9
PID_LON_KI = 0.05
PID_LON_KD = 0.0

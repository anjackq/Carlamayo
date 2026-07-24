# Camera Geometry Alignment

CarlaMayo can run an opt-in camera input ablation that separates camera pose
and projection mismatch from controller and safety behavior.

## Modes

| Mode | Sensor pose | Model-facing projection |
| --- | --- | --- |
| `baseline` | Existing CARLA rig | Existing pinhole pixels |
| `pose-only` | PhysicalAI profile pose | Nominal pinhole pixels |
| `projection-only` | Existing CARLA rig | Profile F-theta pixels |
| `pose-projection` | PhysicalAI profile pose | Profile F-theta pixels |

`baseline` remains the default until both frozen-input and closed-loop promotion
gates pass. A non-baseline mode fails at startup unless a valid local profile is
provided. There is no silent fallback.

## Build a private profile

The generator uses the existing Hugging Face login and reads only
`sensor_extrinsics`, `camera_intrinsics`, and `vehicle_dimensions`. It never
requests camera video.

```bash
python scripts/build_camera_profile.py \
  --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --output ~/.cache/carlamayo/camera-profiles/hyperion8-example.json
```

The output is written with permission `0600`. Profiles must remain outside the
Git repository. Runtime logs contain only profile ID, dataset revision,
platform class, and profile SHA-256; they do not contain transforms,
principal points, or polynomial coefficients.

## Run one aligned simulation

```bash
python carlamayo_closed_loop.py \
  --empty-road \
  --mode navigation \
  --scenario-seed 0 \
  --camera-alignment pose-projection \
  --camera-profile ~/.cache/carlamayo/camera-profiles/hyperion8-example.json \
  --max-episode-seconds 20
```

`CARLAMAYO_CAMERA_PROFILE` may provide the profile path. An explicit
`--camera-profile` takes precedence.

For each F-theta camera, startup computes the minimum CARLA pinhole overscan FOV
with a two-degree margin. Startup fails if the target needs rear-facing rays,
more than 160 degrees, or predicts more than 0.5% invalid output pixels. The
precomputed maps are applied once per exact CARLA frame with `cv2.remap`;
conditioned frames are cached before entering the temporal buffer.

The observation contract distinguishes:

- `camera_source_intrinsics`: the CARLA overscan pinhole matrices;
- `camera_output_models`: the actual pinhole or F-theta projection associated
  with model-facing pixels;
- `camera_extrinsics`: the actual sensor-to-ego transforms.

The compatibility property `camera_intrinsics` refers only to source pinhole
intrinsics and must not be used to project onto a warped image.

## Capture frozen inputs

Capture holds the ego at full brake and waits for four consecutive camera
frames plus 16 real, contiguous ego-history ticks. Startup padding is not
accepted.

```bash
python carlamayo_closed_loop.py \
  --empty-road \
  --mode navigation \
  --scenario-seed 0 \
  --camera-alignment pose-projection \
  --camera-profile ~/.cache/carlamayo/camera-profiles/hyperion8-example.json \
  --capture-inference-fixture \
    ~/.cache/carlamayo/camera-fixtures/combined-seed0.npz \
  --capture-only
```

Fixtures are permission `0600`, must be outside the repository, and contain
only synthetic CARLA pixels, ego history, frame/timestamp identity, navigation
text, scene metadata, and profile/fixture hashes. They do not contain gated
PhysicalAI frames or calibration coefficients.

## Build a private visual comparison

The comparison generator reads authorized PhysicalAI camera frames and the
three CARLA fixtures needed to separate pose from projection changes. The
output contains gated images and exact mounting translations, so the script
rejects output directories inside the repository and writes owner-only files:

```bash
python scripts/build_private_camera_comparison.py \
  --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --camera-profile \
    ~/.cache/carlamayo/camera-profiles/hyperion8-example.json \
  --physical-t0-us 5100000 10100000 15100000 \
  --fixture \
    baseline=~/.cache/carlamayo/camera-fixtures/baseline-seed0.npz \
  --fixture \
    projection-only=~/.cache/carlamayo/camera-fixtures/projection-only-seed0.npz \
  --fixture \
    pose-projection=~/.cache/carlamayo/camera-fixtures/pose-projection-seed0.npz \
  --output-dir ~/.cache/carlamayo/camera-comparisons/example
```

The output includes one high-resolution comparison for each camera ID, an
all-position contact sheet, real and synthetic temporal sheets, a mounting
position/heading diagram, and private comparison metadata. PhysicalAI and
CARLA show different scenes, so these artifacts support qualitative checks of
FOV, horizon, vehicle occlusion, camera overlap, and mounting geometry—not
pixel-level correspondence or image-quality metrics.

Capture each mode independently, then run:

```bash
python scripts/run_frozen_camera_ablation.py \
  --fixture baseline=~/.cache/carlamayo/camera-fixtures/baseline.npz \
  --fixture pose=~/.cache/carlamayo/camera-fixtures/pose.npz \
  --fixture projection=~/.cache/carlamayo/camera-fixtures/projection.npz \
  --fixture combined=~/.cache/carlamayo/camera-fixtures/combined.npz \
  --repeats 16 \
  --num-traj-samples 3 \
  --diffusion-temperature 1.0
```

The model is loaded once. Every group reuses the seed sequence `0..15` and
records all CoCs, trajectories, stop intents, pairwise ADE, latency, and
candidate hashes in a private JSONL report under
`~/.cache/carlamayo/ablations/` by default.

## Promotion rule

Do not tune PID, safety, ego-history padding, navigation routing, candidate
selection, or model weights while evaluating this patch. Promote
`pose-projection` to the default only if it passes the documented geometry,
preprocessing, frozen-input, and three-seed closed-loop gates. Otherwise keep it
as an opt-in research mode and report the failed gate.

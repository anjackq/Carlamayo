<div align="center">

# CarlaMayo

### NVIDIA Alpamayo 1.5 + CARLA Simulator

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

[![CI](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml/badge.svg)](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml)

</div>

> **📖 Please read the [Hugging Face Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first.**
> The model card contains model architecture, inputs/outputs, licensing, and tested hardware details. This repository focuses on CARLA setup, data collection, and open/closed-loop inference scripts.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | NVIDIA GPU with ≥24 GB VRAM for Alpamayo, ≥6 GB VRAM for CARLA |
| **OS** | Linux tested; other platforms unverified |
| **CARLA** | 0.9.16 |

> ⚠️ GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors for full-precision Alpamayo inference. The 4-bit quantization path can reduce memory usage.

## Installation

Environment setup by following document:

- [Environment Setup](docs/environment-setup.md)

## Running Inference

Data collection, open-loop inference, and closed-loop inference by following document:

- [Data Collection and Inference](docs/inference-workflows.md)

### Closed-Loop UI Modes

The closed-loop runner supports `normal`, `navigation`, and `vqa` modes through
`--mode`. See the mode-specific usage guides:

- [Navigation Mode](docs/navigation-mode.md)
- [VQA Mode](docs/vqa-mode.md)

### Current Safe Closed-Loop Baseline

The current prototype implements a conservative, auditable closed-loop baseline:

- Each accepted observation contains one exact CARLA snapshot and four images
  with the same frame ID and simulation timestamp. The canonical Alpamayo camera
  order is front-left, front-wide, front-right, and front-tele, with camera IDs
  `[0, 1, 2, 6]` and nominal FOVs `[120, 120, 120, 30]` degrees.
- A selected 64-point Alpamayo trajectory is validated and transformed from the
  source ego pose into fixed CARLA world coordinates once. Its waypoints retain
  the source-simulation timestamps `+0.1` through `+6.4` seconds; the path is not
  re-anchored as the ego vehicle moves.
- The controller follows monotonic progress along that fixed-world path. Speed
  comes from timestamped waypoint spacing, and a stationary or repeated terminal
  tail can produce a real deceleration and full stop. Missing, malformed,
  misaligned, stale, or exhausted plans command braking instead of throttle.
- A stop-only safety shield checks the path and nearby vehicles/pedestrians using
  the exact CARLA snapshot. An unsafe result, unknown input, or adapter failure
  applies a latched emergency brake and is recorded separately from the
  controller request.
- With `--telemetry-jsonl PATH`, every proposal is auditable through an
  `alpamayo_proposal` event containing the full, untruncated CoC text, its SHA-256
  digest, source identity, and candidate trajectory. The same JSONL stream links
  that proposal ID to validation events and to tick events containing the
  controller request, final applied control, and safety-override reasons.

The runner defaults to synchronous inference; `--async` is opt-in. In normal and
navigation modes, inference is scheduled no more often than once per 1.0 second
of CARLA simulation time after the initial four-frame warm-up. Synchronous model
inference blocks the next world tick, so wall-clock time may be much longer while
simulation-time source age remains stable.

A bounded, telemetry-enabled baseline run is:

```bash
python carlamayo_closed_loop.py \
  --telemetry-jsonl runs/baseline/runtime.jsonl \
  --max-episode-seconds 60
```

#### Closed-Loop Artifacts

| Artifact | Default location |
|----------|------------------|
| Recorded front-wide view (H.264 when `ffmpeg` is available) | `./carla_alpamayo_closed_loop_result.mp4` |
| Live JPEG preview | `./carla_alpamayo_closed_loop_latest.jpg` |
| Recorded Pygame window, with `--pygame-ui` | `./carla_alpamayo_closed_loop_result_pygame_ui.mp4` |
| Runtime and full-CoC audit log | The path passed to `--telemetry-jsonl` |

Set `CARLAMAYO_OUTPUT_VIDEO` and `CARLAMAYO_LIVE_PREVIEW_IMAGE` to move the video
and preview. The provided Slurm script writes the video, preview, and
`runtime.jsonl` under `/home/aqiu/carlamayo-runs/<SLURM_JOB_ID>/`; its CARLA server
log remains in the repository root as `carla-server-<SLURM_JOB_ID>.log`.

#### Important Limitations

This remains a research integration prototype, not a production autonomous-
driving stack. In particular:

- there is no destination-aware route planner or automatic route-to-prompt
  generation;
- traffic lights, stop signs, right-of-way, and other traffic rules are not
  handled as driving policy;
- `NUM_TRAJ_SAMPLES` is currently `1`, so multi-candidate safety ranking and
  rerouting around hazards are not implemented;
- the safety shield is a privileged CARLA-ground-truth integration layer, not an
  onboard perception system or evidence that Alpamayo itself made a safe choice;
- the current video/Pygame overlay labels `ALPAMAYO PROPOSAL`,
  `CONTROLLER EXECUTION`, and `SAFETY OVERRIDE`, but the planned four-camera,
  CoC, BEV, and telemetry dashboard is not yet complete; and
- the deterministic fixed-route CARLA release gate in the implementation plan
  has not yet been run and passed.

## Project Structure

```
<repo-root>/
├── data_collect.py              # Collect synchronized CARLA camera/LiDAR/trajectory data.
├── carlamayo_open_loop.py       # Run Alpamayo 1.5 inference on recorded CARLA data.
├── carlamayo_closed_loop.py     # Run closed-loop CARLA control modes.
├── module/                      # Shared CARLA, inference, control, UI, and visualization helpers.
├── tests/                       # Simulator-free unit tests for lightweight helpers.
├── docs/                        # Environment setup and workflow guides.
├── assets/                      # README images and demo media.
├── third_party/alpamayo1.5/     # NVIDIA Alpamayo 1.5 git submodule.
├── third_party/oom-free-alpamayo/ # OOM-free demand-layering git submodule (optional --oom-free).
├── .github/workflows/ci.yml     # Lightweight GitHub Actions test workflow.
├── pyproject.toml               # Python project metadata and Ruff configuration.
├── uv.lock                      # Locked uv dependency graph for reproducible installs.
├── requirements-alpamayo.txt    # Additional Alpamayo runtime packages.
└── requirements-carla.txt       # CARLA 0.9.16 data-collection/runtime packages.
```

Generated data, the standard video/preview patterns, and `runs/` are ignored by
git. Put telemetry and other large runtime artifacts under `runs/` (or outside
the repository) to keep them out of commits.

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues, use PyTorch's scaled dot-product attention instead in the Alpamayo config:

```python
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Use **OOM-free mode** (`--oom-free`). See [OOM-Free Mode](docs/oom-free-mode.md).
2. Try 4-bit quantization with `--quantization`.
3. Ensure you have a GPU with enough VRAM for the selected precision and trajectory count.
4. Keep `num_traj_samples` low on smaller GPUs.
5. Close other GPU-intensive applications.

## License and Third-Party Licenses

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This repository does not vendor NVIDIA Alpamayo 1.5 source code directly. Alpamayo is linked as a git submodule under `third_party/alpamayo1.5` and is licensed separately under Apache License 2.0. See `third_party/alpamayo1.5/LICENSE`.

NVIDIA Alpamayo 1.5 model weights are not redistributed by this repository and are not covered by this repository's Apache License 2.0. Review the [Hugging Face model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for the model license and usage restrictions, including non-commercial restrictions where applicable.

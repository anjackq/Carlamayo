# Motion Stability Stabilization Report

## Configuration

- Code: `7c792b7`
- Map/spawn: `Town03`, centered spawn `0`
- Scenario: empty road
- Camera input: `projection-only`
- Alpamayo: `K=3`, diffusion temperature `1.0`
- Navigation text: `At the roundabout in 20m, take the first exit to the right.`
- Episode: 20 simulation seconds at 10 Hz

Runtime JSONL, videos, camera frames, model weights, and calibration data remain
outside the repository.

## Live results

| Seed | Slurm job | Distance (m) | Mean / peak speed (m/s) | Fallback ticks | Direct / latch-only | Collision | Ego-road UNSAFE | Motion+reserve p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `22864930` | 71.12 | 3.57 / 9.98 | 3 | 0 / 0 | 0 | 0 | 1.42 |
| 1 | `22864971` | 58.29 | 2.94 / 6.69 | 3 | 0 / 0 | 0 | 0 | 1.40 |
| 2 | `22864972` | 49.86 | 2.50 / 8.78 | 3 | 0 / 0 | 0 | 0 | 1.40 |

Median integrated distance was **58.29 m**, above the 50.2 m gate. All fallback
ticks were the initial ticks 1–3 before the first valid plan. There were no
cap-saturation-only emergencies and no activation of a worse non-stop motion
class while the active plan had more than 3.0 s remaining.

## Standby handoff result

The retained-plan liveness path was exercised live:

- Seed 0 activated retained standby plans twice at the retention deadline.
- Seed 1 activated one retained standby plan at the retention deadline.
- Seed 2 activated retained standby plans twice at the retention deadline.

In seed 0, a candidate retained at frame 24 survived later invalid proposals and
was freshly revalidated, road-assessed, and activated at frame 48. The episode
did not reproduce the bridge-expiry `WAITING_FOR_PLAN` gap from job `22863527`.

## Gate decision

The stabilization gate passes. Patch 5 may proceed with the serial/manual
defaults unchanged. The process road backend remains opt-in until its exact
parity and performance gates pass.

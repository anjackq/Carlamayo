# Exact serial road-batch rollout report

Date: 2026-07-24  
Code: `9a3ed9a` (`Batch exact serial road assessments`)

## Configuration

- Town03, centered spawn 0, empty road
- projection-only camera alignment
- Alpamayo K=3, diffusion temperature 1.0
- manual navigation prompt unchanged from the stabilization baseline
- 20 seconds of synchronous simulation time
- exact serial CARLA road-assessment backend

The three Slurm jobs were `22865308`, `22865309`, and `22865310` for
scenario seeds 0, 1, and 2 respectively. Runtime JSONL, videos, camera
frames, calibration, and model data remain outside Git.

## Results

| Seed / job | Distance | Mean / peak speed | Fallback ticks | Direct / latch-only | Ego-road UNSAFE | Collision |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 22865308 | 66.278 m | 3.318 / 9.987 m/s | 18 | 1 / 5 | 1 | 0 |
| 1 / 22865309 | 35.639 m | 1.784 / 7.511 m/s | 3 | 0 / 0 | 0 | 0 |
| 2 / 22865310 | 78.064 m | 3.909 / 8.587 m/s | 3 | 1 / 5 | 0 | 0 |

Median distance was 66.278 m. The seed-0 fallback excess was one
`plan_source_age_exceeded` gap at ticks 119–133 after consecutive unusable
proposals. Its direct road trigger at tick 189 was a genuine
`center_off_driving_lane` event. Seed 2's direct trigger was a genuine
exhausted stopping envelope: ego speed 2.849 m/s exceeded the raw physical
cap of 2.711 m/s with 4.274 m to the first bad point. No run contained a
cap-clipping-only emergency.

Scenario/model randomness is seeded, but the CUDA inference path is not
configured for bitwise determinism. Therefore, candidate trajectories from
these reruns are not assumed to match the earlier stabilization rollouts
bit for bit.

## Exact serial latency

| Job | Road batch p50 / p95 / max | Selection p95 | Backend errors |
|---|---:|---:|---:|
| 22865308 | 331.74 / 545.57 / 553.12 ms | 560.53 ms | 0 |
| 22865309 | 205.82 / 539.17 / 556.33 ms | 554.54 ms | 0 |
| 22865310 | 301.57 / 511.11 / 526.53 ms | 527.92 ms | 0 |

For job 22865308, exact candidate batches performed 24,265 map queries.
Map-query wall time dominated the profile; ranking remained below 0.1 ms.
The serial backend is semantically useful as the exact reference and
fallback, but it does not meet the 100 ms control/result-processing budget.

## Gate decision

- PASS: collisions were zero.
- PASS: median integrated distance exceeded 50.2 m.
- PASS: direct override counts `1/0/1` did not exceed K=1 limits `3/0/3`.
- PASS: cap-saturation-only emergency count was zero.
- FAIL: seed 0 had 18 fallback ticks instead of at most 3.
- FAIL: seed 0 had one current-ego road-UNSAFE tick instead of zero.
- FAIL: road-batch and selection latency exceeded their performance gates.

The exact serial backend remains the default reference. Process mode may
continue as an opt-in implementation within Patch 5, but it cannot be
promoted until serial/process parity and live performance gates pass.
Route and CoC work may be implemented behind opt-in/diagnostic contracts,
but the failed stabilization criteria prevent default promotion.

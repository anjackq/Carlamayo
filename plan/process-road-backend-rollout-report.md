# Exact process road backend rollout

Baseline commit: `256c631`  
Jobs: `22915684`, `22915685`, `22915686`  
Configuration: Town03, spawn 0, empty road, projection-only camera input,
K=3, temperature 1.0, 20 simulation seconds, six process workers.

## Outcome

The commissioning/steady deadline split fixed the operational failure.  Every
seed completed its first exact serial/process shadow comparison with
`process_shadow_pass`; every later non-empty batch remained on the exact
`process` backend.  There were no process timeouts, approximate fallbacks, or
parity failures.

| Seed/job | Distance | Fallback ticks | Direct/latch-only | Ego road UNSAFE | Collision |
|---|---:|---:|---:|---:|---:|
| 0 / 22915684 | 76.834 m | 3 | 2 / 4 | 0 | 0 |
| 1 / 22915685 | 54.292 m | 3 | 0 / 0 | 0 | 0 |
| 2 / 22915686 | 50.162 m | 3 | 0 / 0 | 0 | 0 |

Median distance was 54.292 m.  The stabilization safety gates passed:
collisions were zero, fallback ticks were at most three, current ego road
UNSAFE ticks were zero, and direct overrides did not exceed the K=1 seed
baselines `{3, 0, 3}`.

## Performance

The numbers below exclude the one commissioning shadow batch and batches with
no valid plan to assess.

| Job | Process batches | road batch p50/p95/max | map query p50/p95/max | selection p50/p95/max |
|---|---:|---:|---:|---:|
| 22915684 | 12 | 61.15 / 163.73 / 166.99 ms | 35.64 / 90.25 / 91.19 ms | 72.87 / 180.69 / 183.45 ms |
| 22915685 | 15 | 75.95 / 153.48 / 164.58 ms | 36.27 / 88.80 / 90.93 ms | 92.16 / 169.07 / 182.03 ms |
| 22915686 | 17 | 91.56 / 161.80 / 167.06 ms | 37.41 / 89.70 / 92.54 ms | 108.73 / 179.40 / 183.92 ms |

The exact worker map-query portion is near the 90 ms p95 target and is much
faster than the prior serial rollout, whose road-selection p95 was roughly
512–546 ms.  End-to-end batching still misses the promotion gates:

- `road_batch_wall_ms` p95 must be below 90 ms and max below 150 ms;
- `selection_compute_ms` p95 must be below 100 ms.

The remaining overhead is main-process validation, DTO preparation,
aggregation, and scheduling around the worker query.  Consequently the
process backend remains opt-in; defaults stay `serial`, `manual`, and K=1.

## Decision

The 500 ms commissioning and 250 ms steady operational deadlines are retained.
They are failover deadlines, not promotion thresholds.  No safety-semantic
change is needed.  A later performance-only patch may reduce main-process
batch overhead, but route grounding work can proceed independently.

# Synchronous Policy-Isolation Experiment Report

Date: 2026-07-28

## Decision

These experiments use only CARLA synchronous mode. Alpamayo inference blocks
the next CARLA tick, so model wall latency does not age the active plan in
simulation time. Async execution is deliberately excluded.

Two changes are promoted:

- bounded CARLA junction lane canonicalization;
- collision-free Slurm CARLA port reservation.

Far-horizon lateral safe-prefix execution is not promoted. It remains available
only through the explicit `--route-lateral-safe-prefix` research flag. Strict
lateral validation remains the default.

## Synchronous clock contract

Each inference cycle is:

1. CARLA produces one synchronized camera/history observation.
2. CARLA simulation remains paused while Alpamayo performs inference.
3. All `K=3` candidates are validated and road/route assessed.
4. The admitted candidate goes through active-plan handoff.
5. CARLA resumes at 10 Hz and executes the selected trajectory.

For job `22928385`, 45.0 seconds of CARLA time required about 185 seconds of
wall time. The dual-clock timeline is stored outside Git at:

```text
/home/aqiu/carlamayo-runs/22928385/pipeline_timeline.csv
/home/aqiu/carlamayo-runs/22928385/pipeline_timeline.png
```

The wall-clock panel contains the inference-duration blocks. The CARLA-clock
panel contains only source/result events at the same paused simulation time.
Consequently, the experiment measures Alpamayo trajectory quality without
confounding it with async inference delay.

## Experiment A — Oracle low-speed control

The model was bypassed. A direct route-derived trajectory requested a constant
speed while the normal PID, smoothing, CARLA drivetrain, road gate, and route
gate remained active.

| Target | Job | Distance to capture | Mean / peak speed | Stop-go cycles | Road/route result |
|---:|---:|---:|---:|---:|---|
| 0.5 m/s | `22927130` | 9.96 m | 0.286 / 2.391 m/s | 43 | all SAFE / MATCH |
| 1.0 m/s | `22927131` | 10.12 m | 0.575 / 2.599 m/s | 11 | all SAFE / MATCH |
| 2.0 m/s | `22927146` | 11.01 m | 1.067 / 2.879 m/s | 4 | all SAFE / MATCH |
| 5.0 m/s | `22927133` | 10.21 m | 2.544 / 5.130 m/s | 0 | all SAFE / MATCH |

There were no collisions or safety overrides. This isolates a controller and
vehicle-physics problem below roughly 2 m/s: smoothing, PID output, brake
release, and drivetrain response create oscillatory stop-go even for a perfect
route trajectory. Alpamayo is not the only source of low-speed instability.

## Experiment B — Oracle route authorization

The first full-route oracle run (`22927135`) stopped at 64.37/97.08 m route
progress. It recorded 269 near-route `DEVIATE` ticks despite executing the
route-derived trajectory itself. CARLA exposes overlapping junction road and
lane identities (`1577`, `1624`, `1608`, including alternate sections/lanes),
so an exact identity-only check rejected a geometrically correct route.

The bounded canonicalization patch authorizes only:

- geometrically overlapping junction lane identities within 1.5 m;
- route road IDs present in a bounded 4.5 m lookback / 18 m lookahead;
- same-road section/lane aliases inside that bounded junction window.

It does not authorize a different spatial branch or an adjacent non-route lane.

| Run | Job | Result | Route progress | Distance | Near-route status | Collision / override |
|---|---:|---|---:|---:|---|---:|
| Before | `22927135` | episode limit | 64.37 m | 64.08 m | 181 MATCH / 269 DEVIATE | 0 / 0 |
| After | `22927192` | destination arrived | 95.62 m | 95.06 m | 279 MATCH / 0 DEVIATE | 0 / 0 |

The patched run stopped at the destination with final speed approximately
0 m/s. This is a clean route-policy self-consistency pass.

## Experiment C — Matched CARLA pixels and ego histories

Private frozen fixtures were captured in CARLA at the same route progress.
Every group used 16 requests, `K=3`, temperature 1.0, the same model, and
source-matched four-camera/four-frame tensors plus 16 real ego-history poses.

| Source condition | Moving candidates | Moving coverage at K | Explicit stops |
|---|---:|---:|---:|
| stationary | 1/48 | 6.25% | 16 |
| 0.5 m/s target capture | 47/48 | 100% | 1 |
| 1.0 m/s target capture | 46/48 | 100% | 1 |
| 2.0 m/s target capture | 18/48 | 75% | 27 |
| 5.0 m/s target capture | 44/48 | 100% | 1 |

The 2 m/s fixture contains an accelerating history rather than a settled
2 m/s history. Results therefore depend on the temporal motion pattern, not a
single scalar current speed.

A second factorial reused the stationary history while swapping only the
camera bundle. Moving candidates changed from `1/48` to `38/48`, `42/48`,
`1/48`, and `9/48` for the stationary, 0.5, 1.0, 2.0, and 5.0 pixel bundles.
The images have correct frame spacing, no black-frame contamination, and
normal per-camera statistics. Both image content and ego history materially
condition the trajectory distribution.

## Experiment D — Junction frozen inputs

Six matched junction fixtures were captured at approach, entry, and active-turn
positions at nominal 2 and 4 m/s. Each group again used `16 x K=3`.

With strict validation:

| Fixture | Valid candidates | Moving coverage at K | Full route-branch coverage at K |
|---|---:|---:|---:|
| approach, 2 m/s | 16/48 | 68.75% | 18.75% |
| approach, 4 m/s | 23/48 | 87.50% | 75.00% |
| entry, 2 m/s | 32/48 | 100% | 0% |
| entry, 4 m/s | 23/48 | 81.25% | 12.50% |
| active, 2 m/s | 7/48 | 37.50% | 0% |
| active, 4 m/s | 5/48 | 31.25% | 12.50% |

Nearly all generic failures are `excessive_lateral_displacement`. Every first
lateral violation is after 3.3 seconds, beyond the 1.5-second execution
horizon. The navigation-direction text generally agrees with the requested
right turn, but the trajectory geometry does not follow the authorized branch.
In active-turn fixtures, predicted terminal displacement is roughly 39–46 m
forward and 19–24 m lateral, much larger than the matching oracle motion.

Allowing only far-horizon lateral safe prefixes makes all 48 candidates pass
generic validation, but it does not repair route geometry:

| Fixture | Near-route MATCH coverage at K | Full route-branch coverage at K |
|---|---:|---:|
| approach, 2 m/s | 100% | 18.75% |
| approach, 4 m/s | 100% | 81.25% |
| entry, 2 m/s | 100% | 0% |
| entry, 4 m/s | 100% | 12.50% |
| active, 2 m/s | 100% | 0% |
| active, 4 m/s | 25% | 25% |

The relaxation is useful as a diagnostic of receding-horizon execution. It is
not evidence that the full prediction follows the route.

## Experiment E — Closed-loop lateral safe prefix

Configuration: Town03 fixed route, empty road, projection-only aligned cameras,
`K=3`, temperature 1.0, exact serial road assessment, 45 seconds, synchronous
inference.

| Seed / job | Route progress | Distance | Longest stationary streak | Fallback | Road/route result |
|---|---:|---:|---:|---:|---|
| 0 / `22927450` | 58.67 m (60.44%) | 58.53 m | 152 ticks | 158 | 6 ego-road UNSAFE and 6 direct overrides |
| 1 / `22928385` | 63.29 m (65.19%) | 62.26 m | 121 ticks | 33 | ego road SAFE, 0 direct overrides |
| 2 / `22928898` | 64.35 m (66.29%) | 64.03 m | 142 ticks | 13 | ego road SAFE, 0 direct overrides |

Seed 0 followed an admitted Alpamayo prefix until its footprint left the
Driving surface near the junction, then entered a fallback-stop suffix. Seed 1
remained road-safe, but selected an `EXPLICIT_STOP` at tick 338, followed by
mostly `DELAYED_START` proposals, and selected another `EXPLICIT_STOP` at tick
438. Its final full brake came from the active Alpamayo trajectory, not the
road shield or route emergency policy.

Seed 2 reproduced the same low-motion distribution. It remained stationary for
142 consecutive ticks and briefly restarted near the episode end. That restart
prevents a literal terminal absorbing-stop classification, but does not make
the 14.2-second interruption acceptable autonomous-driving behavior.

The corresponding strict-validator runs reached about 68.21% and 65.19% route
completion for seeds 1 and 2. The relaxation therefore has not demonstrated a
closed-loop improvement and already violates the zero ego-road-UNSAFE gate.

## CoC finding

The source-synchronized audit remains diagnostic-only. Across the junction
fixtures and live rollout, CoC often describes the correct right-turn intent
while the paired trajectory stops, starts late, or follows the wrong geometry.
This is a `CoC / trajectory mismatch`, not evidence that CoC should receive
control authority.

## Gate decisions

### Promote

- Junction topology canonicalization: oracle trajectory changed from 269
  false `DEVIATE` ticks to zero and completed the route.
- Slurm port reservation: node-local locks plus live four-port probing prevent
  the CARLA RPC collision that invalidated job `22927220`.
- Synchronous policy-isolation harnesses and dual-clock timeline.

### Keep opt-in

- `K=3`.
- `--route-lateral-safe-prefix`.
- Frozen policy audit and oracle route runner.

### Do not promote

- Automatic far-lateral relaxation: seed 0 introduced real road-surface
  violations, and seed 1 still entered an Alpamayo-driven absorbing stop.
- Any async execution claim: async mode was not tested in this experiment.

## Next synchronous patches

1. Calibrate low-speed longitudinal control against the oracle suite at
   0.5/1/2/5 m/s. Promotion requires monotonic speed tracking without changing
   road or route semantics.
2. Add a frozen junction metric that compares predicted arc length and terminal
   displacement against the source-speed reachable envelope. Keep it
   diagnostic first; do not spatially snap Alpamayo output to the route.
3. Test whether selector ranking can reject `EXPLICIT_STOP` and
   `DELAYED_START` only when a route-matching moving candidate exists. It must
   never invent motion when every candidate stops or when scene truth requires
   stopping.
4. If Alpamayo continues to provide no route-matching candidate at active
   junction positions, evaluate a separate architecture in which CARLA route
   geometry owns lateral control and Alpamayo supplies longitudinal/behavioral
   intent. Report this as a change in Alpamayo's control role, not as a
   trajectory-following fix.

Runtime JSONL, MP4 files, camera frames, gated calibration, frozen fixtures,
model weights, and generated profiles remain outside Git.

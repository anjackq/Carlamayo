# Multi-candidate road-aware selector ablation

## Decision

The road-aware `K=3` selector is a useful opt-in research mode, but it must not
replace the default `K=1` configuration yet.

It materially improves distance, removes the seed-1 fallback-stop lock, and
reduces selected full-path road violations. It does not pass every closed-loop
promotion gate: seed 2 has more direct road-safety trigger ticks than its
single-sample baseline, and per-request candidate road evaluation exceeds the
100 ms control budget.

## Scope

This patch changes only proposal selection and its telemetry. It does not
change Alpamayo weights, camera alignment, PID gains, the safety shield,
trajectory validation thresholds, or the navigation prompt.

For every Alpamayo request, the runtime now:

1. extracts every CoC and trajectory sample;
2. builds and validates a fixed-world `TrajectoryPlan` for each sample;
3. evaluates each valid plan with the CARLA road execution envelope;
4. ranks candidates deterministically with safety admission first;
5. hands off only the selected admitted plan, or retains the executable active
   plan when no new candidate is admissible.

The ranking order is:

1. admitted plan, active-plan retention, fallback, then generic invalid;
2. empty-road stop penalty;
3. full-safe, safe-prefix, then recovery-prefix admission;
4. navigation turn-direction consistency;
5. larger full-path road margin;
6. continuity with the previous selected trajectory;
7. forward progress;
8. stable candidate index.

Normal traffic does not apply the empty-road stop penalty. CoC remains
diagnostic text and is not used as a safety signal.

## Telemetry

Each request adds:

- one `candidate_evaluation` event per sample, containing validation outcome,
  admission, road envelope, progress, lateral displacement, continuity, and
  transparent ranking terms;
- one `candidate_selection` event containing the continuity preselection,
  final selected index, all ranked evaluations, selection reason, and selector
  latency;
- one `alpamayo_proposal` event for the final selected CoC and trajectory;
- one `plan_admission` event when the selected candidate reached road
  admission.

If all new samples fail generic validation, the active plan and its previous
admission status are retained. An invalid proposal cannot erase the active
plan's admission telemetry.

## Experiment

Both groups used:

- Town03, spawn 0, empty road;
- seeds 0, 1, and 2;
- 20 seconds, 200 simulation ticks;
- navigation mode with the same roundabout-right instruction;
- projection-only aligned cameras using the same local PhysicalAI profile;
- diffusion temperature 1.0;
- no NPCs.

The control group used one trajectory sample. The patch group used three
trajectory samples.

| Seed | Control job | K=3 job |
|---:|---:|---:|
| 0 | 22858757 | 22859856 |
| 1 | 22858758 | 22859868 |
| 2 | 22858759 | 22859869 |

Runtime JSONL, videos, camera frames, generated profiles, and gated data remain
outside Git.

## Closed-loop results

| Seed | Distance K=1 → K=3 | Fallback ticks | Direct override ticks | Selected full-path UNSAFE | Longest near-zero-speed streak |
|---:|---:|---:|---:|---:|---:|
| 0 | 45.1 → 50.2 m | 3 → 3 | 3 → 1 | 8/15 → 3/17 | 43 → 17 |
| 1 | 29.7 → 50.1 m | 65 → 3 | 0 → 1 | 10/17 → 9/17 | 59 → 21 |
| 2 | 38.6 → 51.3 m | 3 → 3 | 3 → 5 | 10/16 → 10/18 | 109 → 37 |
| Median | 38.6 → 50.2 m | 3 → 3 | 3 → 1 | — | 59 → 21 |

Aggregate observations:

- median integrated distance improved by 30.2%;
- all six runs had zero collision and zero current-ego-road UNSAFE ticks;
- pooled selected full-path UNSAFE proposals fell from 28/48 (58.3%) to
  22/52 (42.3%), a 27.4% relative reduction;
- seed 1's six `REJECT_FALLBACK_STOP` admissions and 65-tick terminal waiting
  state disappeared;
- the selector changed the old continuity preselection on 23/60 requests;
- 180/180 generated candidates received a recorded evaluation;
- K=3 selector latency was 189.8 ms p50, 335.0 ms p95, and 478.3 ms maximum.

The selector therefore fixed a real selection-layer failure. It did not weaken
the fail-closed shield: no collision or ego-surface violation was introduced.

## Remaining stop-go and override causes

The remaining stop-go behavior is not an absorbing fallback state. Most
near-zero ticks are `TRACKING` or `DECELERATING`, with a fresh or retained
active plan.

Alpamayo frequently proposes very low-progress speed profiles near the
roundabout. In seed 0, one selected non-stop, full-safe candidate advances only
about 1.02 m across its full horizon. The final selected proposal begins with
millimetre-scale waypoint motion. Its CoC repeatedly says to adapt or slow for
the roundabout. The controller is following the proposal's low speed rather
than the safety shield inventing a stop.

K=3 model inference remains slow. Per-seed inference-latency p50 was 4.69,
5.05, and 5.71 seconds. Invalid proposal batches can leave the retained plan
several simulation seconds old, which contributes to braking near the end of
its executable speed profile.

Road-only emergency braking also remains:

- seed 0: one direct trigger tick;
- seed 1: one direct trigger tick;
- seed 2: five direct trigger ticks.

These happen when a selected safe-prefix plan is executed near its first bad
point and the stopping envelope becomes exhausted. Seed 2 therefore fails the
per-seed requirement that direct road overrides must not increase.

The current lexical CoC audit reported no positive actor hallucinations.
Qualitative mismatches still exist outside that audit: one seed-0 CoC speculates
about an upcoming yellow traffic control, and one seed-2 CoC recommends a right
lane change despite the strict-lane baseline. Candidate road safety can reject
unsafe geometry, but it cannot make the reasoning or route instruction
semantically correct.

## Promotion status

Passed:

- collisions remain zero;
- current ego road UNSAFE ticks remain zero;
- median distance does not regress;
- median fallback does not increase;
- no new absorbing stop state;
- pooled full-path road-UNSAFE rate improves by more than 20% relative.

Not passed:

- direct road override ticks increase on seed 2;
- candidate selection p50/p95 exceed the 100 ms control budget;
- slow/near-stop Alpamayo proposals still cause visible stop-go;
- CoC/navigation semantics are not grounded against route and traffic-control
  truth.

Keep `--num-traj-samples 3` opt-in. Do not change the default from one sample.

## Recommended next patch order

1. Add a trajectory motion-quality profile to each candidate: initial speed,
   horizon progress, deceleration, stop intent, and time-to-effective-stop.
   Penalize unexplained near-stop candidates on verified empty road without
   overriding road admission or explicit stop behavior.
2. Add stopping-robustness ranking for safe-prefix plans. Prefer candidates with
   larger braking reserve, and include controller tracking/acceleration margin
   so a plan admitted at the current speed cannot immediately exhaust its
   envelope after one tick.
3. Cache or batch common CARLA map queries across the three candidates and
   measure selector p50/p95 again.
4. Replace the static natural-language navigation prompt with route-derived
   current-lane and junction-exit semantics. Keep strict lane-keep authority
   separate from free-form CoC.
5. Extend the CoC audit to traffic lights, signs, lane-change intent, and route
   action; compare claims with CARLA ground truth and trajectory direction.
6. Repeat the same three-seed gate. Promote K=3 only if every seed has no
   increase in direct overrides and selection meets the timing budget.

# CarlaMayo Autonomous-Agent Implementation Plan

## Status

CarlaMayo now has an implemented safe closed-loop baseline on the prototype
branch:

- Exact-frame CARLA observations bind the ego state and canonical left, wide,
  right, and tele cameras to one simulation snapshot.
- Alpamayo receives camera IDs `[0, 1, 2, 6]` and produces auditable
  Chain-of-Causation reasoning plus a 64-point trajectory.
- A validated proposal is anchored once at its capture pose, timestamped in
  simulation time, and followed as an immutable world path by a stopping-capable
  CARLA PID controller.
- A privileged CARLA-ground-truth, stop-only safety shield can replace nominal
  control with a latched emergency brake. Unsafe and unknown assessments both
  fail closed.
- Structured telemetry distinguishes the Alpamayo proposal, controller request,
  post-processed nominal control, final applied control, fallback, and safety
  override. Proposal events retain the complete, untruncated CoC text and an
  integrity hash.
- Synchronous inference is the default. Normal and navigation submissions use a
  one-second CARLA simulation-time cadence after observation warm-up; `--async`
  remains an explicit opt-in mode.

This is more than proof of integration, but it has not passed the deterministic
CARLA release gate below. The next milestone is:

> Validate geometrically and temporally correct lane following and stopping on a
> deterministic CARLA route, then use the evidence to tune the baseline without
> weakening its fail-closed behavior.

Destination-aware routing, traffic-rule policy, multi-trajectory selection, the
complete four-panel UI, fine-tuning, V2X, and world-model integration remain
deferred until the basic-driving gate passes.

### Implementation progress

- PR 1, runtime contracts and baseline telemetry: implemented.
- PR 2, synchronized and correctly labelled observations: implemented.
- PR 3, coordinate-system and camera-calibration correctness: implemented for the
  canonical CARLA rig and current calibrated front-wide projection. Exact
  reproduction of Alpamayo's undisclosed training-rig extrinsics is not claimed.
- PR 4, timestamped fixed-world trajectory lifecycle: implemented.
- PR 5, fixed-world, stopping-capable controller: implemented.
- Safe-baseline hardening, including the CARLA-ground-truth adapter, fail-closed
  stop-only shield, and full CoC proposal audit: implemented.
- PR 6, traceable four-panel UI: partially implemented. The current view exposes
  persistent proposal/controller/override labels and requested versus applied
  control, but the four thumbnails, dedicated CoC panel, BEV, and full telemetry
  panel remain pending.
- PR 7, deterministic CARLA validation harness and the release-gate runs: pending.

Simulator-free tests cover the implemented contracts. No release claim should be
made until the fixed-route CARLA criteria below have also passed.

### Current runtime defaults and artifacts

- Closed-loop execution defaults to synchronous inference. After the initial
  four-frame observation warm-up, normal/navigation inference is submitted no
  more often than once per 1.0 second of CARLA simulation time. `--async` is
  optional and uses the same minimum simulation-time submission interval.
- The default recorded view is
  `./carla_alpamayo_closed_loop_result.mp4`; the live preview is
  `./carla_alpamayo_closed_loop_latest.jpg`.
- `--pygame-ui` additionally records
  `./carla_alpamayo_closed_loop_result_pygame_ui.mp4`.
- Runtime JSONL is opt-in at the exact path supplied to `--telemetry-jsonl`. It is
  also the complete CoC audit artifact; proposal text is stored in
  `alpamayo_proposal.coc_text_full` with SHA-256 and length metadata.
- `CARLAMAYO_OUTPUT_VIDEO` and `CARLAMAYO_LIVE_PREVIEW_IMAGE` override the two
  default media paths. The supplied Slurm runner places its video, preview, and
  `runtime.jsonl` in `/home/aqiu/carlamayo-runs/<SLURM_JOB_ID>/` and writes the
  CARLA server log to `./carla-server-<SLURM_JOB_ID>.log`.

## Review of the Proposed Roadmap

The original roadmap is directionally correct. Its diagnosis—proof of integration
rather than a reliable autonomous-driving agent—described the original prototype.
The current baseline addresses the defects below, but still requires the planned
simulator evaluation before it can be treated as a reliable agent.

### Critical defects addressed in the baseline

1. Model input now includes the canonical Alpamayo camera indices `[0, 1, 2, 6]`
   in left/wide/right/tele order.
2. The exact-frame collector accepts a camera bundle only when all four images
   match the requested CARLA frame; snapshot and sensor timestamps are checked,
   while missing, old, duplicate, and future packets are handled explicitly.
3. A selected ego-frame trajectory is transformed at observation capture once and
   remains fixed in CARLA world coordinates throughout its lifetime.
4. Target speed is derived from waypoint distance over waypoint time. The legacy
   10 km/h floor is not used by the fixed-world controller, and geometric terminal
   stop tails produce deceleration and full braking.
5. Plan source age starts at the exact source observation's CARLA simulation time
   and therefore includes all simulation-time delay.
6. The canonical front-wide FOV is 120 degrees and the current path overlay uses
   the captured camera pose and intrinsic matrix.
7. Plans have explicit source-age, remaining-horizon, prompt/respawn revision, and
   tracking-alignment validity. An invalid proposal is never activated; an older
   plan may continue only while independently valid, otherwise control brakes.

### Corrections to the proposed timing policy

- Plan validity now uses CARLA simulation time, not `time.time()`.
- The one-second inference interval is a minimum submission interval measured in
  CARLA simulation time. Synchronous mode blocks world ticks while inference runs;
  asynchronous mode permits ticking but still allows only one pending request, so
  its effective frequency can be inference-limited.
- The recorded asynchronous prototype run (`22843540`) had approximately 6.62
  seconds median inference latency and 10.19 seconds p95 latency. A fixed
  two-second source-age limit would reject most current results.
- Alpamayo waypoints represent future times `[0.1, 0.2, ..., 6.4]` seconds.
  Expired points are selected through these explicit timestamps to avoid an
  off-by-one error.
- The initial policy requires a two-second minimum remaining horizon and caps
  source age at 4.4 seconds. These limits must be recalibrated from measured
  latency and drift rather than treated as universal constants.

### Sequencing correction

Evaluation cannot be postponed until the final gate. Runtime telemetry is now in
place and every implemented slice produces comparable metrics. Full simulator
evaluation follows the completed geometry, timing, control, and shield work.

## Target Runtime Architecture

```text
CARLA world tick
      |
      v
SynchronizedObservation
frame ID + simulation time + capture pose + ego state + calibrated images
      |
      v
InferenceRequest
source observation + ego history + camera IDs + prompt/respawn revisions
      |
      v
Alpamayo 1.5
CoC + ego-frame trajectory proposal (current K=1; multi-candidate K>1 deferred)
      |
      v
TrajectoryPlan
capture-pose anchoring + fixed world points + explicit waypoint times
      |
      v
Plan validator/selector
shape + finite values + age + remaining horizon + drift + revision checks
      |
      v
World-trajectory controller
monotonic progress + lateral lookahead + stopping-capable speed profile
      |
      v
Safety state machine
WAITING / TRACKING / DECELERATING / STOPPED / FALLBACK / EMERGENCY_BRAKE
      |
      v
CARLA VehicleControl
```

## Runtime Data Contracts

Introduce immutable records in `module/runtime_types.py`.

### `SynchronizedObservation`

Required fields:

- `frame_id`
- `simulation_time_s`
- `ego_pose_world`
- `ego_velocity_world`
- ordered camera images
- camera IDs
- camera intrinsics and extrinsics
- ego-history samples with frame IDs and simulation timestamps

### `InferenceRequest`

Required fields:

- source observation identity
- image and ego-history tensors
- capture pose
- navigation prompt and weight
- prompt revision
- respawn revision
- submission wall time for performance telemetry only

### `TrajectoryPlan`

Required fields:

- source frame and source simulation time
- capture pose
- selected ego-frame points
- fixed world-frame points
- waypoint times `[0.1, ..., 6.4]`
- CoC text and candidate metadata
- inference wall latency
- prompt and respawn revisions

### `PlanValidation`

Required fields:

- valid/invalid status
- rejection reason
- simulation-time source age
- remaining horizon
- lateral and heading drift
- first usable waypoint index

### `ControlDecision`

Required fields:

- controller state
- target point and target speed
- controller-requested steering, throttle, and brake
- final applied steering, throttle, and brake
- fallback state and reason
- whether a safety override was applied
- safety-override type and reason
- source plan identity

The controller request and final applied control must remain separate even when
they are numerically identical. This prevents a successful emergency intervention
from being attributed to Alpamayo or to the nominal trajectory controller.

### `VisualizationSnapshot`

Required fields:

- current CARLA frame and simulation time
- source observation frame and simulation time
- synchronized source camera bundle and current display camera
- all Alpamayo candidate trajectories and the selected proposal
- controller reference trajectory, target point, and requested control
- actual ego trail and final applied control
- safety zones, conflict zones, and override decision
- CoC text and navigation prompt
- inference backend (`local` or `remote`), state, and latency
- plan source age, remaining horizon, fallback state, and rejection reason

The renderer consumes an immutable snapshot. It must never read partially updated
global state or block the simulation/control loop.

## Implementation Sequence

### PR 1: Runtime contracts and baseline telemetry — implemented

#### Files

- Add `module/runtime_types.py`.
- Add `module/runtime_metrics.py`.
- Update `carlamayo_closed_loop.py` to emit structured events.
- Add unit tests for serialization and metric aggregation.

#### Work

1. Add the immutable runtime records described above without changing control
   behavior.
2. Write per-tick and per-inference JSONL telemetry.
3. Record source/arrival frames, wall latency, simulation-time age, remaining
   horizon, controller state, requested control, applied control, and
   rejection/override reasons.
4. Preserve episode-wide collision counts even if the ego vehicle respawns.
5. Add a bounded episode duration so a run can produce a final summary.

#### Acceptance

- Existing simulator-free tests continue to pass.
- Every applied control can be traced to a source observation and plan.
- p50, p95, and p99 latency and plan-age metrics are reported.
- No PID or inference behavior changes in this PR.

### PR 2: Synchronized and correctly labelled observations — implemented

#### Files

- Update `module/config.py`.
- Update `module/carla_interface.py`.
- Update `module/inference.py`.
- Update `carlamayo_closed_loop.py`.
- Extend `tests/test_carla_interface.py` and `tests/test_inference_utils.py`.

#### Work

1. Centralize camera specifications instead of maintaining order, FOV, and
   Alpamayo identity in separate places:

   ```text
   front-left  -> Alpamayo ID 0, 120 degrees
   front-wide  -> Alpamayo ID 1, 120 degrees
   front-right -> Alpamayo ID 2, 120 degrees
   front-tele  -> Alpamayo ID 6, 30 degrees
   ```

2. Make camera callbacks enqueue `(frame_id, camera_name, image)`.
3. Make `CARLAInterface.tick()` return the CARLA frame and matching world
   snapshot.
4. Adapt the exact-frame collector in `module/data_collection.py` for the
   closed-loop camera bundle.
5. Discard and count old frames. Report missing frames instead of mixing them.
6. Obtain ego pose and velocity from the same world snapshot.
7. Update ego history only after a complete synchronized bundle is accepted.
8. Include `torch.tensor([0, 1, 2, 6], dtype=torch.long)` in model data and
   validate the configured camera order.

#### Acceptance

- All accepted camera images have the expected identical frame ID.
- Accepted frame mismatch rate is zero.
- Missing/dropped bundles are counted in telemetry.
- Temporal observation spacing is 0.1 seconds except for reported drops.
- Generated Alpamayo messages contain the four correct camera names and frame
  numbers.

### PR 3: Coordinate-system and camera-calibration correctness — implemented

#### Files

- Add `module/geometry.py`.
- Add `tests/test_geometry.py`.
- Update `module/carla_interface.py`.
- Update `module/pid_controller.py`.
- Update `module/visualization.py`.

#### Work

1. Replace scattered yaw negation and lateral sign changes with matrix-based
   conversions:
   - CARLA world to/from CARLA ego;
   - CARLA ego `(x forward, y right, z up)` to/from model ego
     `(x forward, y left, z up)`;
   - relative rotation basis conversion;
   - model ego at capture time to CARLA world;
   - CARLA world to current camera coordinates.
2. Apply the same conventions to ego-history positions and rotations.
3. Represent camera intrinsics and extrinsics explicitly.
4. Replace the approximate overlay projection with calibrated projection.
5. Project fixed world points through the camera transform associated with the
   current image.
6. Document that matching nominal FOV and role does not reproduce unknown
   training-rig extrinsics exactly.

#### Acceptance

- Float64 world/local/world position and rotation round trips have error below
  `1e-6`.
- Positive model `y` produces a left-world path and negative model `y` produces
  a right-world path.
- A straight world trajectory remains fixed when the ego vehicle moves or turns.
- Known camera-axis points project within two pixels of expected locations.
- Visualization and control consume the same world trajectory.

### PR 4: Timestamped, fixed-world trajectory lifecycle — implemented

#### Files

- Add `module/trajectory_runtime.py`.
- Add `tests/test_trajectory_runtime.py`.
- Update `carlamayo_closed_loop.py`.
- Update `module/config.py` with validity settings.

#### Work

1. Attach the source frame, source simulation time, and capture pose to every
   request and result.
2. Validate output rank, point count, finite values, displacement, and curvature
   before accepting a result.
3. Anchor model points to the capture pose exactly once.
4. Assign explicit future waypoint timestamps.
5. On every control tick, compute:

   ```text
   source_age_s = current_simulation_time_s - source_simulation_time_s
   ```

6. Remove or interpolate past points using waypoint timestamps.
7. Reject old prompt/respawn revisions, insufficient remaining horizon, excessive
   lateral or heading drift, and implausible geometry.
8. Schedule inference using simulation frames/time. Keep wall time only for
   performance measurement.
9. If a new result is invalid, continue an existing valid plan only until its own
   validity expires; otherwise enter fallback immediately.

#### Initial configurable policy

- `minimum_remaining_horizon_s = 2.0`
- `maximum_plan_age_s <= 4.4`
- lateral and heading drift limits calibrated from deterministic runs
- no fixed two-second source-age rejection rule

#### Acceptance

- No waypoint at or before the current simulation time reaches the controller.
- No expired, prompt-stale, or respawn-stale plan is applied.
- A plan's world coordinates do not change between control ticks.
- Invalid output enters fallback within one 0.1-second control tick.
- Source/arrival identities, source age, remaining horizon, and wall latency are
  logged separately.

### PR 5: World-trajectory and stopping-capable controller — implemented

#### Files

- Refactor `module/pid_controller.py` or introduce
  `module/trajectory_controller.py`.
- Update `module/config.py`.
- Update `carlamayo_closed_loop.py`.
- Extend `tests/test_pid_controller.py`.

#### Work

1. Change the controller API to consume a fixed world trajectory and waypoint
   timestamps.
2. Project current ego position onto the world polyline.
3. Maintain monotonic progress and choose lateral lookahead from current progress,
   not waypoint zero.
4. Include the capture origin at `t=0` when deriving the speed profile.
5. Compute desired speed from waypoint spacing and timestamp differences.
6. Smooth the speed profile and apply acceleration, deceleration, curvature, and
   global speed limits.
7. Remove `PID_TARGET_SPEED_MIN_KMH = 10`.
8. Detect terminal near-zero waypoint spacing and command a full stop.
9. Introduce explicit states:

   ```text
   WAITING -> TRACKING -> DECELERATING -> STOPPED
                         \-> FALLBACK / EMERGENCY_BRAKE
   ```

10. Allow smoothing during normal tracking, but bypass it for emergency braking.

#### Acceptance

- Constant 0.5 m spacing at 0.1-second intervals yields 5 m/s.
- Repeated terminal points yield zero desired speed and active braking.
- Straight, left, and right paths produce the expected steering direction.
- Progress along a path is monotonic.
- Moving the ego transform does not move the planned world trajectory.
- Missing or expired plans command braking and never throttle.

### Safe-baseline hardening: Fail-closed ground-truth shield — implemented

This slice is an integration guard around the nominal controller. It is not a
claim that Alpamayo perceives or resolves these hazards itself, and it is not a
substitute for an onboard perception stack.

#### Files

- Add `module/safety_shield.py`.
- Add `module/carla_safety_adapter.py`.
- Add `module/proposal_audit.py`.
- Update `carlamayo_closed_loop.py`, `module/config.py`, and
  `module/visualization.py`.
- Add simulator-free tests for the shield, CARLA adapter, proposal audit, and
  closed-loop arbitration.

#### Implemented behavior

1. Evaluate the selected fixed-world path at no more than 0.5-metre spacing,
   including the ego footprint center and corners, against CARLA driving-lane
   queries. Junction transitions remain valid only while the footprint is on a
   drivable lane.
2. Read vehicle and pedestrian poses, velocities, and bounding boxes from the
   same exact CARLA snapshot as the control tick.
3. Evaluate oriented-box overlap, predicted path conflict, stopping distance,
   hard-gap, and time-to-collision conditions with auditable reason codes.
4. Treat every unsafe or unknown road/obstacle assessment, missing controller
   request, invalid nominal control, or adapter exception as an emergency-brake
   trigger.
5. Latch emergency braking for a minimum hold interval and require consecutive
   clear ticks before releasing it.
6. Keep controller-requested, nominal post-processed, and final applied controls
   separate. Record `CONTROLLER_EXECUTION` or `SAFETY_OVERRIDE` as the applied
   control source.
7. Emit one `alpamayo_proposal` event per extracted proposal with full
   `coc_text_full`, `coc_sha256`, length metadata, source identity, candidate
   geometry, and a stable proposal ID. Link separate validation and control-tick
   events to that ID; UI truncation never truncates the audit log.

#### Acceptance status

- Simulator-free road-containment, obstacle, hysteresis, adapter, arbitration,
  and proposal-audit tests pass.
- Unknown safety state demonstrably produces full braking rather than allowing
  nominal throttle.
- CARLA fixed-route hazard trials remain part of PR 7 and have not yet passed a
  release gate.

### PR 6: Traceable four-panel visualization UI — partially implemented

The current calibrated primary view already shows persistent `ALPAMAYO PROPOSAL`,
`CONTROLLER EXECUTION`, and `SAFETY OVERRIDE` layers, including source age,
requested control, applied control, and override reason. The composed four-panel
dashboard described below is still pending and can be developed in parallel with
the validation harness. Semantic traceability is required; decorative polish is
not a prerequisite for the basic-driving release gate.

#### Files

- Refactor `module/visualization.py` into reusable panel renderers.
- Update `module/pygame_ui.py` to render a composed dashboard.
- Update `carlamayo_closed_loop.py` to publish immutable
  `VisualizationSnapshot` records.
- Add `tests/test_visualization_dashboard.py`.

#### Layout

Use one primary view and four information panels:

```text
+----------------------------------------------------------+
| CARLA spectator or calibrated front-wide view            |
| selected trajectory + safety/conflict zones              |
+----------------------+-----------------------------------+
| four camera          | Chain-of-Causation                |
| thumbnails           | source frame/time + prompt        |
| left/wide/right/tele |                                   |
+----------------------+-----------------------------------+
| BEV                  | system telemetry                  |
| proposal/reference   | inference latency and source age  |
| actual ego trail     | local/remote + fallback/override  |
| NPCs/conflict zones  | requested versus applied control |
+----------------------+-----------------------------------+
```

The primary view may switch between the CARLA spectator and front-wide camera.
Trajectory projection is permitted only when the selected view has explicit
intrinsics and extrinsics. A spectator view must therefore use its actual CARLA
camera transform rather than the front-wide approximation.

#### Mandatory semantic layers

Every dashboard frame must explicitly distinguish:

1. `ALPAMAYO PROPOSAL`
   - raw candidate trajectories;
   - model-selected trajectory;
   - source observation frame and age.
2. `CONTROLLER EXECUTION`
   - fixed-world reference consumed by the controller;
   - current target point and requested control;
   - actual ego path and final applied control.
3. `SAFETY OVERRIDE`
   - inactive/active badge;
   - intervention type, such as reject, rerank, speed limit, brake, or emergency
     stop;
   - triggering object/zone and machine-readable reason.

Do not rely on color alone. Use persistent text badges plus different line styles
and colors. Suggested defaults are dashed cyan for Alpamayo proposals, solid green
for controller execution, and solid orange/red for safety interventions.

#### Panel contents

Primary view:

- current front-wide or spectator image;
- calibrated projection of the selected fixed-world path;
- controller target and safety/conflict zones;
- visible proposal/controller/override legend.

Camera panel:

- front-left, front-wide, front-right, and front-tele thumbnails;
- camera ID, source frame, and simulation timestamp on every thumbnail;
- a visible stale/mismatch indicator rather than silently mixing frames.

CoC panel:

- complete or scrollable Chain-of-Causation text;
- navigation prompt and CFG weight;
- source frame/time and plan revision;
- explicit `NO VALID PLAN` or `INFERENCE ERROR` state when applicable.

BEV panel:

- all Alpamayo candidates and the selected proposal;
- controller reference trajectory and target;
- actual ego trail;
- ego/NPC footprints, lane or drivable-area boundary, and conflict zones;
- safety-rejected path segments when an override occurs.

Telemetry panel:

- current frame, simulation time, speed, steering, throttle, and brake;
- inference backend (`local` or `remote`), pending/ready/error state, wall latency,
  source age, and remaining horizon;
- controller state and plan validation state;
- requested control beside final applied control;
- fallback, safety-override, collision, and synchronized-bundle counters.

#### Runtime rules

1. Render from a bounded snapshot queue or latest-value buffer so UI/video work
   cannot delay control.
2. Clearly label current-view time separately from proposal-source time. Showing a
   current camera image beside an older CoC/trajectory without both timestamps is
   prohibited.
3. The MP4 recorder consumes the same composed frame shown by Pygame.
4. Headless operation and `--no-ui` must preserve identical control behavior and
   telemetry.
5. Safety geometry may be sparse until the full dashboard is implemented, but
   `SAFETY OVERRIDE: INACTIVE` or `SAFETY OVERRIDE: ACTIVE` must always be
   explicit.

#### Acceptance

- A recorded frame is traceable to current and source CARLA frame IDs.
- Proposal, controller request, and applied control are simultaneously visible.
- An injected emergency-brake decision changes the override badge, reason, applied
  control, and BEV overlay within one control tick.
- A deliberately stale or mismatched camera bundle is visibly marked and is not
  presented as synchronized input.
- Disabling the dashboard produces the same serialized `ControlDecision` sequence
  in a replay test.
- Dashboard composition and video recording pass headless simulator-free tests.

### PR 7: Deterministic CARLA validation harness — pending

#### Files

- Add `module/scenario.py`.
- Add `module/evaluation.py`.
- Add `scripts/run_validation.py`.
- Add fixed scenario definitions under `scenarios/`.
- Add `tests/test_scenario.py` and `tests/test_evaluation.py`.

#### Scenario configuration

- fixed map, spawn, and destination;
- fixed weather;
- fixed Python, NumPy, Torch, Traffic Manager, and CARLA seeds;
- fixed 0.1-second simulation step;
- explicit episode duration;
- configurable NPC and walker counts;
- automatic respawn disabled during evaluation;
- route used as reference truth, not hidden controller guidance;
- isolated output directory per episode.

#### Trajectory providers

1. `scripted`: verifies geometry and controller independently of Alpamayo.
2. `replay`: replays recorded plans for deterministic regression tests.
3. `alpamayo`: runs the complete model integration.

#### Initial scenarios

1. Scripted straight, left, and right sign checks.
2. Scripted cruising followed by a planned stop.
3. Alpamayo empty-road straight lane following.
4. Injected inference timeout or invalid output requiring safe stopping.

#### Metrics

Driving:

- route progress and completion;
- collisions by actor type;
- lane invasions and off-road frames;
- red-light and stop-sign violations when those scenarios are introduced.

Control:

- cross-track and heading error;
- target and actual speed;
- stopping error;
- acceleration, jerk, and steering rate.

Runtime:

- inference and queue latency;
- source and execution age;
- remaining horizon;
- invalid-output and timeout rate;
- fallback, emergency-brake, and safety-override counts;
- synchronized-bundle drops and mismatches.

## Basic-Driving Release Gate

The following are initial engineering criteria, not an official Alpamayo
benchmark.

### Simulator-free and scripted criteria

- All unit and simulator-free tests pass.
- Scripted geometry and control scenarios pass 10 out of 10 runs.
- Accepted synchronized-frame mismatch count is zero.
- Applied expired/stale trajectory count is zero.
- Injected inference timeout safely stops without a collision.

### Alpamayo fixed-route criteria

Complete 10 recorded episodes on the fixed basic route with:

- zero collisions;
- zero off-road events;
- zero lane invasions;
- route completion at least 95 percent;
- cross-track error p95 no greater than 0.75 m and maximum no greater than 1.5 m;
- heading error p95 no greater than 8 degrees;
- final stopping error no greater than 1.5 m;
- final speed no greater than 0.2 m/s.

Comfort thresholds remain provisional until a stable scripted-controller baseline
has been measured.

## Deferred Work

The following begins only after the basic-driving release gate passes.

### Route and traffic capability

- CARLA `GlobalRoutePlanner` destination and route progress;
- route-progress to Alpamayo navigation-text generation;
- traffic-light and stop-sign state handling;
- pedestrians, lane changes, cut-ins, and obstacle preview;
- preserve the existing separation of Alpamayo proposals, controller execution,
  safety overrides, and fallback decisions as route/rule policy is added.

### Multi-trajectory selection

- increase `NUM_TRAJ_SAMPLES` only after measuring VRAM and latency;
- evaluate collision, lane-boundary, route-progress, curvature, comfort, and
  traffic-rule costs;
- log both the Alpamayo proposals and the selected/overridden result.

### Research extensions

- CARLA-domain or failure-focused fine-tuning;
- remote accelerator inference;
- V2X integration;
- AlpaSim or world-model rendering integration;
- CoC/trajectory consistency evaluation.

## Explicit Non-Goals for the Next Milestone

- Hiding Alpamayo path errors by snapping its trajectory to CARLA map waypoints.
- Treating a successful safety override as an Alpamayo success.
- Using one demonstration video as a reliability benchmark.
- Fine-tuning before the observation, geometry, timing, and controller contracts
  are validated.
- Running traffic-heavy scenarios before deterministic empty-road driving passes.

## Dependency Order

```text
Runtime contracts and baseline metrics
        |
        v
Synchronized and labelled observations
        |
        v
Coordinate and camera geometry
        |
        v
Capture-pose anchoring and simulation-time alignment
        |
        v
Stopping-capable world-trajectory controller
        |
        +------> Traceable four-panel UI
        |
        +------> Deterministic CARLA evaluation
                         |
                         v
Route, traffic, multi-candidate safety ranking, and research extensions
```

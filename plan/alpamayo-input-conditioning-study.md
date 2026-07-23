# Alpamayo Input and Conditioning Study

## Purpose

Separate four sources of CarlaMayo proposal instability:

1. camera projection/extrinsic domain mismatch;
2. ego-history contract mismatch;
3. stochastic VLM and diffusion sampling;
4. navigation-conditioning mismatch.

This study treats CoC as diagnostic output, not a safety signal.

## Dataset access

The official Alpamayo 1.5 example loader reads:

- `nvidia/PhysicalAI-Autonomous-Vehicles`

The official AlpaSim/NuRec workflow additionally uses:

- `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`

Both repositories use automatic gated access. The current Hugging Face login is
valid and now has file access to both repositories. The official example clip's
camera calibration, vehicle dimensions, and egomotion metadata were read
successfully with the existing token; no replacement token is required.

Only metadata, one official example clip's camera calibration, and the four
short image streams needed for a fixed-input fixture should be downloaded.
Dataset-scale video or NuRec assets are not required for this study.

The gated raw calibration and image data must remain in the local Hugging Face
cache and must not be committed to this repository. The four camera feature
archives for the example chunk total several GiB, so the fixed-input experiment
should stream only the selected clip members rather than download whole chunk
archives.

## Contract findings

CarlaMayo currently matches the public contract for:

- camera IDs and order `[0, 1, 2, 6]`;
- four RGB frames on a contiguous 10 Hz grid;
- 16 ego poses on a contiguous 10 Hz grid;
- exact camera/snapshot frame and timestamp synchronization;
- CARLA `y-right` to model `y-left` conversion.

The first two empty-road requests do not have a complete 1.5-second ego history:
they repeat the earliest available pose. Inference should eventually wait for
all 16 history samples, but this does not explain later-run hallucinations.

The high-risk mismatch is camera geometry. Calibration from the official
example clip confirms that:

- the PhysicalAI-AV cameras are calibrated F-theta cameras, while CARLA renders
  the current rig as ideal rectilinear pinhole cameras;
- the current CARLA cameras are all mounted at 2.4 m, approximately one metre
  above the official front-wide/tele cameras and substantially above the cross
  cameras;
- the official cross cameras have a longer forward and wider lateral baseline,
  with optical headings farther outward than the current simple `+/-60 degree`
  placement;
- the front-wide pinhole/F-theta radial projection differs strongly through the
  middle of the image even though both have a nominal 120-degree horizontal
  field of view.

Alpamayo receives camera identity labels but no numeric intrinsics or
extrinsics, so it cannot compensate for these differences at inference time.
The appropriate ablation is therefore `pose only`, `F-theta warp only`, and
`pose + F-theta warp`; changing the controller would confound this experiment.

## Multi-sample experiment

The default remains one sample. Use:

```bash
python carlamayo_closed_loop.py \
  --empty-road \
  --mode navigation \
  --scenario-seed 0 \
  --num-traj-samples 3 \
  --diffusion-temperature 1.0 \
  --max-episode-seconds 20
```

The runtime log records every candidate trajectory, complete candidate CoC, and
CoC hash. Candidate selection is still continuity-based:

- without an active predecessor, select candidate 0;
- otherwise select the candidate with lowest mean XY distance to the previous
  selected model-frame trajectory.

This is an observability experiment, not completed multi-candidate safety
ranking. Road admission currently evaluates only the selected candidate.

Two full-precision runs isolated diffusion temperature:

| Job | Samples | Diffusion temperature | Scene |
| --- | ---: | ---: | --- |
| `22852968` | 3 | 1.0 | Town03, spawn 0, empty road, seed 0 |
| `22852969` | 3 | 0.6 | Town03, spawn 0, empty road, seed 0 |

Both jobs completed without collisions:

| Metric | temperature 1.0 | temperature 0.6 |
| --- | ---: | ---: |
| Integrated distance | 64.9 m | 16.9 m |
| Mean / peak speed | 3.26 / 7.50 m/s | 0.85 / 7.68 m/s |
| Candidate stop trajectories | 2 / 60 | 26 / 60 |
| Selected stop trajectories | 0 / 20 | 13 / 20 |
| Stopped ticks (`speed < 0.1 m/s`) | 43 / 200 | 164 / 200 |
| Pairwise ADE median / mean / max | 2.75 / 3.67 / 21.63 m | 0.54 / 0.92 / 4.52 m |

Lowering diffusion temperature to `0.6` did reduce geometric diversity, but it
collapsed frequently into stationary trajectories and made closed-loop progress
far worse. Temperature `1.0` should remain the baseline. More samples expose
useful alternatives, but temperature reduction is not a remedy for CoC/path
mismatch or for the continuity selector's stop-state bias.

A separate 4-bit smoke run, job `22852979`, completed on two RTX A4000
GPUs. It is not directly comparable to the full-precision baseline, but it
validates the three-candidate pipeline:

- 20 requests and 60 candidate CoC/trajectory pairs were recorded;
- integrated distance was 38.5 m, with no collision;
- median/mean/maximum within-request pairwise trajectory ADE was
  0.95/1.57/8.38 m;
- 8 requests produced three distinct CoCs, 5 produced two, and 7 produced one;
- 17 of 60 candidates were short stop trajectories;
- the selected plan requested stop in 9 of 19 validated proposals;
- requests 15--20 selected a stop trajectory despite at least one moving
  alternative on every request.

The last point exposes a continuity-selection absorbing state. Once the active
trajectory is a stop, minimum distance to the previous selected trajectory
systematically prefers another stop. Increasing the sample count therefore
creates useful alternatives but does not improve execution until every
candidate receives generic validation, road admission, route score, and
progress/stop-intent scoring before selection.

Compare:

- per-request pairwise trajectory ADE and final-point spread;
- CoC exact-hash diversity and normalized action/entity diversity;
- selected candidate index and continuity scores;
- candidate generic/road validity when replayed offline;
- distance, moving time, stop proposals, admissions and overrides;
- hallucinated dynamic actors in the zero-NPC scene.

## Navigation and map conditioning

The released pretrained interface has a native text navigation channel:

```text
<|route_start|>Turn right in 30m<|route_end|>
```

It does not expose a structured OpenDRIVE map, lane graph, route polyline, or
BEV raster input. Dumping map data into the route text is outside the tested
contract.

The compatible integration is:

```text
destination
  -> CARLA GlobalRoutePlanner.trace_route()
  -> sequence of (Waypoint, RoadOption)
  -> next meaningful maneuver and along-route distance
  -> concise route text
  -> Alpamayo trajectory candidates
```

The route must remain an explicit proposal input and evaluation reference. It
must not silently steer the low-level controller around Alpamayo, or the
experiment can no longer attribute behavior to the model.

Direct structured-map conditioning would require one of:

- an adapter/fine-tune that embeds a route polyline or lane graph;
- an additional calibrated BEV/map image followed by post-training;
- a model architecture with explicit map tokens.

These are research extensions, not prompt-only changes.

## Prompt design

Use one maneuver, one distance, and optional road/street identity:

- `Turn right in 30m.`
- `Turn left onto Main Street in 40m.`
- `Continue straight for 50m.`
- `At the roundabout in 20m, take the first exit to the right.`

Do not combine route guidance with long behavioral rules such as “never change
lanes, remain centered, obey every boundary, and stop if uncertain.” Alpamayo's
navigation channel is soft conditioning, not a hard policy language. Lane
authorization, collision avoidance, and uncertainty handling belong in
candidate ranking, controller constraints, and the safety shield.

## Fixed-input follow-up experiment

1. Keep the downloaded official calibration in the private local cache.
2. Record the exact four-camera CARLA input and 16-pose history for one request.
3. Repeat each frozen input at least 16 times without advancing CARLA.
4. Compare diffusion temperature `1.0` and `0.6`.
5. Compare current CARLA pinhole input, official-rig camera poses, an F-theta
   warp, and the combined pose-plus-warp input.
6. Change only ego history, then only navigation text, to measure causal
   sensitivity.
7. Do not tune the controller during this open-loop experiment.

The frozen-input design distinguishes stochastic instability from changing
closed-loop observations and camera-domain mismatch.

## Visual correction from job 22852979

The final front-wide image confirms that Town03 spawn 0 approaches a real
roundabout. Consequently, CoCs mentioning a roundabout or yield are generally
scene-consistent, not hallucinations. The first redesigned prompt described the
topology only as a right curve, which was under-specified. The fixed empty-road
prompt now names the roundabout and requested exit explicitly.

This correction strengthens the case for route-to-text generation: a manually
written prompt can mislabel the same topology that the model recognizes
correctly. The zero-NPC run still contains no evidence for CoCs that name a
specific interacting vehicle, pedestrian, or cyclist.

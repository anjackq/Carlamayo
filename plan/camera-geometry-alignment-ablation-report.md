# Camera Geometry Alignment Ablation Report

## Decision

Keep `--camera-alignment baseline` as the closed-loop default.

The opt-in camera patch is technically valid and materially improves some
trajectory metrics, but `pose-projection` does not pass the frozen-input or
closed-loop promotion gates. A subsequent `projection-only` three-seed run
passes the aggregate distance, road-quality, collision, and override gates,
but still produces one terminal fallback lock and more positive actor
hallucinations than baseline. It is therefore the preferred camera research
mode, not the closed-loop default. No controller, safety, history, selector,
prompt, or model-weight parameter was changed to conceal the failed gates.

Implementation checkpoints:

- `812147b` — sampling/navigation research baseline;
- `3e094f0` — opt-in camera geometry alignment and frozen ablation harness.

## Private profile and geometry validation

The authorized PhysicalAI-AV example clip generated a local
`hyperion_8-030c760c` profile. The profile and all fixtures remain outside the
repository with permission `0600`. Git contains no calibration coefficients,
official frames, generated profiles, fixtures, videos, or runtime JSONL.

The local profile validation produced:

- vehicle relative dimension error: length `2.70%`, width `2.24%`, height
  `0.79%`;
- F-theta pixel/ray/pixel median and p99 error below `1e-8 px` for all four
  cameras;
- rotation determinant `+1` and maximum orthogonality error below `3e-16`;
- required source FOV below `160 degrees` for all cameras;
- predicted remap valid ratio `100%` for all cameras.

The complete unit suite passed: `258 passed`.

## Capture and preprocessing

Capture-only job `22854005` produced a four-camera/four-frame fixture after 16
real contiguous ego-history ticks:

- camera bundle accepted: `16/16`;
- missing bundle: `0`;
- frame mismatch: `0`;
- timestamp mismatch: `0`;
- collision: `0`;
- fixture tensor: `(4, 4, 3, 1080, 1920)`, `uint8`;
- camera IDs: `[0, 1, 2, 6]`.

After avoiding a full invalid-mask scan for profiles with 100% valid pixels,
job `22854013` measured the four-camera F-theta remap:

| Metric | Result | Gate |
| --- | ---: | ---: |
| remap p50 | 8.77 ms | < 10 ms |
| remap p95 | 12.30 ms | < 20 ms |
| remap maximum | 17.24 ms | < 50 ms |
| total decode + remap maximum | 71.13 ms | < 100 ms |

The three closed-loop combined runs independently measured remap p50 between
`8.56` and `8.69 ms`, with no missing or timestamp-mismatched bundles.
Consequently, the native CARLA wide-angle fallback evaluation was not
triggered.

## Private image-space comparison

The authorized example clip was sampled at three times across all four model
camera IDs and compared with the frozen CARLA `baseline`, `projection-only`,
and `pose-projection` inputs. Gated frames, exact mounting translations, and
the generated contact sheets remain owner-readable under
`~/.cache/carlamayo/camera-comparisons/` and are not committed.

The qualitative comparison separates several effects:

- `projection-only` preserves the CARLA scene composition while applying the
  expected F-theta compression, so it is the cleanest projection ablation;
- `pose-projection` lowers the front-wide view and introduces ego-hood
  visibility closer to the real rig, but the Tesla hood shape and black
  self-occlusion differ substantially from the PhysicalAI collection vehicle;
- cross-left and cross-right show the largest pose/composition shift because
  the aligned rig uses a wider, lower lateral mounting arrangement and
  different optical headings than the original roof-level CARLA cameras;
- front-tele changes least geometrically, although CARLA rendering,
  photometry, traffic density, and scene realism remain far from the real
  sample.

The images therefore support treating lens projection and mounting pose as
separate research variables. They do not support pixel correspondence or a
quantitative image-similarity claim because the real and synthetic scenes are
different. In particular, official pose alignment should not be promoted
until vehicle-body occlusion and rig/vehicle compatibility are addressed.

## Frozen-input ablation

Job `22854057` ran four groups with one model load, seeds `0..15`, 16 requests
per group, three candidates per request, and diffusion temperature `1.0`.
Every group used the baseline fixture's exact ego history and navigation text;
only model-facing camera pixels differed.

| Input | Stop candidates | Pairwise ADE median | Positive nonexistent-actor hallucinations |
| --- | ---: | ---: | ---: |
| baseline | 21/48 (43.75%) | 0.752 m | 0/48 |
| pose-only | 33/48 (68.75%) | 0.206 m | 0/48 |
| projection-only | 15/48 (31.25%) | 2.432 m | 0/48 |
| pose-projection | 25/48 (52.08%) | 0.529 m | 1/48 |

Failed frozen gates:

- combined stop-candidate rate is baseline `+8.33 percentage points`, exceeding
  the allowed `+5 points`;
- combined introduces one positive hallucination of a vehicle in the synthetic
  zero-NPC scene;
- combined is worse than projection-only on stop-candidate rate, so it is not
  better than or equal to the better single-variable ablation.

The result suggests that projection correction is useful, while changing the
pose and projection together shifts the model toward stationary modes.

## Three-seed closed-loop comparison

Town03, spawn 0, empty road, 20 seconds, three trajectory samples, temperature
`1.0`, and the same navigation prompt:

| Seed | Input | Distance | Ego UNSAFE ticks | Fallback ticks | Direct overrides | Full-path UNSAFE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | baseline | 59.9 m | 0 | 3 | 3 | 13/17 (76.5%) |
| 0 | pose-projection | 45.4 m | 5 | 20 | 17 | 9/18 (50.0%) |
| 1 | baseline | 7.0 m | 0 | 13 | 11 | 14/17 (82.4%) |
| 1 | pose-projection | 23.7 m | 0 | 110 | 6 | 13/18 (72.2%) |
| 2 | baseline | 13.9 m | 0 | 50 | 4 | 17/18 (94.4%) |
| 2 | pose-projection | 11.2 m | 0 | 3 | 0 | 0/20 (0.0%) |
| median | baseline | 13.9 m | 0 | 13 | 4 | 82.4% |
| median | pose-projection | 23.7 m | 0 | 20 | 6 | 50.0% |

Both groups had zero collisions. Combined improved median distance by `70.6%`
and reduced median full-path UNSAFE rate by `39.3%` relative. Those improvements
are real but insufficient for promotion.

Failed closed-loop gates:

- seed 0 produced five current-ego road-UNSAFE ticks;
- median fallback ticks increased from `13` to `20`;
- median direct road overrides increased from `4` to `6`;
- seed 1 entered a 107-tick consecutive `WAITING_FOR_PLAN` sequence;
- seed 2 entered a 156-tick consecutive model-selected `STOPPED` sequence;
- selected CoCs with positive nonexistent-actor claims increased from zero in
  all baseline runs to `3/2/1` in combined seeds `0/1/2`.

The UI inspection confirmed that the F-theta trajectory overlay follows the
warped front-wide image and clearly distinguishes `ALPAMAYO PROPOSAL`,
`CONTROLLER EXECUTION`, and `SAFETY OVERRIDE`.

## Projection-only closed-loop follow-up

Jobs `22858757/22858758/22858759` repeated the same Town03, spawn 0,
empty-road, 20-second, three-sample experiment with only the F-theta
projection enabled:

| Seed | Distance | Ego UNSAFE ticks | Fallback ticks | Direct overrides | Full-path UNSAFE | Longest WAITING / STOPPED |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 45.1 m | 0 | 3 | 3 | 8/15 (53.3%) | 3 / 37 |
| 1 | 29.7 m | 0 | 65 | 0 | 10/17 (58.8%) | 62 / 0 |
| 2 | 38.6 m | 0 | 3 | 3 | 10/16 (62.5%) | 3 / 20 |
| median | 38.6 m | 0 | 3 | 3 | 58.8% | 3 / 20 |

All three runs had zero collisions. Against the baseline medians,
`projection-only`:

- increased integrated distance from `13.9 m` to `38.6 m`;
- reduced fallback ticks from `13` to `3`;
- reduced direct road overrides from `4` to `3`;
- reduced full-path UNSAFE rate by `28.6%` relative, from `82.4%` to
  `58.8%`;
- kept current-ego road UNSAFE ticks at zero.

Remap performance also remained within the production gate across these runs:
p50 `8.67 ms`, p95 `9.81 ms`, worst observed remap `24.69 ms`, and worst
decode-plus-remap `93.85 ms`.

It nevertheless fails reliability promotion:

- seed 1 ended with 62 consecutive `WAITING_FOR_PLAN` ticks; the physical ego
  surface remained SAFE, but the buffered clearance was `-0.182 m` and the
  near-term candidate began outside the clearance envelope, so proposals
  15–20 were all `REJECT_FALLBACK_STOP`;
- seed 0 ended with 37 model-selected `STOPPED` ticks even though later plans
  were fully road-safe, exposing the remaining CoC/trajectory stop-mode
  mismatch;
- selected CoCs made positive nonexistent-actor claims in seeds 0/1/2 at
  `1/2/0`, while all three baseline runs had zero;
- median selected stop plans increased from `2` in baseline to `4`.

The projection correction is therefore beneficial and should be retained as
the next research baseline. Default promotion should wait until candidate
selection, CoC/trajectory consistency, and bounded recovery from marginal
clearance violations are addressed and the same three-seed gate is repeated.

## Interpretation and follow-up

Camera mismatch was a real input-domain problem, but it was not the only cause
of unstable CarlaMayo driving. The experiment isolates three remaining
problems:

1. stochastic stop-mode and continuity-selector bias;
2. route/navigation-to-trajectory disagreement at the roundabout;
3. junction containment and fallback recovery.

The next camera-specific experiment should evaluate `projection-only` in
closed loop because it had the lowest frozen stop rate. That experiment must
remain separate from candidate-selector, prompt, controller, and safety
changes. The current `pose-projection` implementation remains useful as an
opt-in research ablation, but must not become the production default.

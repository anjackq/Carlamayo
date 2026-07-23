# Closed-Loop Navigation Mode

Navigation mode runs Alpamayo closed-loop trajectory generation with a runtime text instruction. Use it when you want the ego vehicle to follow a natural-language driving command such as turning, lane choice, or route preference.

## Start CARLA

Start CARLA before launching the integration script:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen
```

> Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

Set the CARLA PythonAPI root if it is not already configured:

```bash
export CARLA_ROOT=~/carla
```

## Run Navigation Mode

From the repository root:

```bash
source a1_5_carla_venv/bin/activate
python carlamayo_closed_loop.py --mode navigation --pygame-ui
```

Closed-loop loading defaults to full precision. On lower-VRAM machines, add `--quantization`:

```bash
python carlamayo_closed_loop.py --mode navigation --pygame-ui --quantization
```

The pygame UI starts paused automatically so you can enter the first navigation prompt
before the CARLA loop begins driving.

## Enter a Navigation Prompt

When the pygame UI opens, type a command in this format:

```text
Turn right in 30m | 1.0
```

Then press `Enter`.

- Text before `|` becomes the navigation instruction.
- The number after `|` becomes the navigation guidance weight.
- Weight `1.0` uses normal navigation conditioning.
- Weights other than `1.0` use Alpamayo classifier-free guidance navigation and may require more VRAM.

Examples:

```text
Turn right in 30m | 1.0
Turn left onto Main Street in 40m | 1.0
Continue straight for 50m | 1.0
At the roundabout in 20m, take the first exit to the right | 1.0
```

Prefer a concise route maneuver plus distance. Alpamayo 1.5 was demonstrated
with instructions such as `Turn right in 30m`; long behavioral-policy prompts
such as `never change lanes and obey every lane boundary` are not hard
constraints and should be enforced by route, controller, and safety layers.

The pretrained interface accepts navigation text between Alpamayo's route
tokens. It does not accept a CARLA/OpenDRIVE map tensor or waypoint polyline
directly. The intended CarlaMayo integration is therefore:

```text
CARLA GlobalRoutePlanner / OpenDRIVE route
                  -> next maneuver + distance
                  -> concise Alpamayo navigation text
                  -> trajectory candidates
```

Structured map conditioning or a BEV map image would require a separately
trained adapter/fine-tune; simply appending a long map dump to the navigation
text is outside the released model's tested input contract.

## UI Controls

- `Ctrl+P`: pause or resume the synchronous CARLA loop.
- `Enter`: apply the text in the input box.
- `Esc`: quit.
- Plain spaces and `p` characters are accepted in the text input.

## Useful Options

```bash
# Non-blocking inference worker.
python carlamayo_closed_loop.py --mode navigation --pygame-ui --async

# Lower VRAM model loading.
python carlamayo_closed_loop.py --mode navigation --pygame-ui --quantization

# Exact returned-logits baseline for debugging memory changes.
python carlamayo_closed_loop.py --mode navigation --pygame-ui --keep-generate-logits

# Generate and audit three CoC/trajectory candidates per inference.
python carlamayo_closed_loop.py --mode navigation --num-traj-samples 3

# Lower-diversity diffusion diagnostic used by the Alpamayo navigation notebook.
python carlamayo_closed_loop.py --mode navigation \
  --num-traj-samples 3 --diffusion-temperature 0.6

```

## Output

If video recording is enabled in `module/config.py`, the script writes:

- `carla_alpamayo_closed_loop_result.mp4`

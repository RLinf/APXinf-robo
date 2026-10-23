# Examples

Run commands from the APXinf-robo repository root after
[installation](../README.md#build-apxinf-robo). Start with
`robot_policy_infer.py` for inference using a robot preset.

| Example | Layer | Shows | Needs |
|---|---|---|---|
| [`register_preset.py`](register_preset.py) | — | Register your own `Embodiment x Convention` from your own package. | — |
| [`preflight_check.py`](preflight_check.py) | — | `check_checkpoint` — will this checkpoint drive this robot correctly? | a checkpoint dir |
| [`bare_model_infer.py`](bare_model_infer.py) | L1 | `load_bare_model` and `infer_rgb`'s exact contract, for callers with their own transforms. | GPU + checkpoint |
| [`policy_infer.py`](policy_infer.py) | L2 | `load_policy` with no preset — you state the wire contract argument by argument. | GPU + checkpoint |
| [`robot_policy_infer.py`](robot_policy_infer.py) | L2 + preset | `build_robot_policy` — Run inference with a robot preset. | GPU + checkpoint |
| [`serve_websocket.py`](serve_websocket.py) | L3 | `websocket_server`, plus a real client reading the served contract off the wire. | GPU + checkpoint + `[serve]` |
| [`lerobot_loop.py`](lerobot_loop.py) | L2 + preset | Drop `ApxInfPolicy` into lerobot's `record_loop`, leaving the robot side untouched. | lerobot + GPU |
| [`g1_adapter_smoke.py`](g1_adapter_smoke.py) | L2 + preset | Verify G1 processing and action shapes. | GPU + 3-view checkpoint |
| [PI0-FAST on LIBERO](#pi0-fast-on-libero) | L2 + preset | Keep both finger joints and use checkpoint-owned normalization. | GPU + PI0-FAST checkpoint/tokenizers |
| [GR00T N1.7 on LIBERO](#gr00t-n17-on-libero) | L2 + environment conversion | Pass named state and convert the decoded gripper for robosuite. | GPU + prepared GR00T checkpoint/processor |
| [Qwen-Drive planning](#qwen-drive-planning) | L2 | Run a driving scene through Robo without a robot preset. | GPU + Qwen-Drive checkpoint and scene inputs |

To verify L1/L2 numerical parity with a compatible checkpoint:

```sh
APXINF_PARITY_CHECKPOINT=/path/to/checkpoint pytest tests/test_parity.py
```

The test compares model outputs bitwise for the same captured inputs.
For engine-only APIs, see the bundled engine's
[Python examples](../apxinf/python/apxinf/examples/README.md).

## Dependencies

- Everything except `register_preset.py` and `preflight_check.py` needs the
  **`apxinf_py` CUDA binding** (see [Build APXinf-robo](../README.md#build-apxinf-robo))
  and a checkpoint directory.
- `register_preset.py` and `preflight_check.py` load no weights, import no
  torch, and touch no network — they run on a laptop.
- `serve_websocket.py` needs `pip install -e ".[serve]"` (msgpack / websockets),
  plus `openpi_client` for its client half.
- `lerobot_loop.py` needs `pip install -e ".[lerobot]"` plus lerobot itself.
- `g1_adapter_smoke.py` needs a checkpoint with **three real camera views**. The
  published LIBERO checkpoints are two-view (their third slot is `empty_cameras`
  padding) and are rejected before inference.

## Quick start

```sh
# No checkpoint, no GPU — the extension point and the wire contract
python examples/register_preset.py

# Check the checkpoint against the target robot preset
python examples/preflight_check.py --robot franka_libero --model-dir /path/to/checkpoint

# L1: the raw forward pass, you own every transform
python examples/bare_model_infer.py --model-dir /path/to/checkpoint

# L2: the engine's preprocessing, wire contract stated by hand
python examples/policy_infer.py --model-dir /path/to/checkpoint

# L2 + preset: the same, with a named robot filling those arguments in
python examples/robot_policy_infer.py --model-dir /path/to/checkpoint

# L3: serve it, and call it from a real client
python examples/serve_websocket.py --model-dir /path/to/checkpoint

# Does a whole body's adapter chain run? (G1: needs a three-view checkpoint)
python examples/g1_adapter_smoke.py --model-dir /path/to/checkpoint
```

`preflight_check.py` exits `0` on a clean check, `1` on WARN, and `2` on FAIL.
Pass `--robot` to check your target preset; omitting it checks all presets.
This checks checkpoint/robot compatibility. Verify GPU execution by running
`robot_policy_infer.py`; `g1_adapter_smoke.py` additionally asserts G1 action shapes.
For PI0-FAST and GR00T, preflight checks local required files and reports WARN
because their tokenizer and processor resources are fully validated at policy
load time.

## The checkpoint

`--model-dir` is a checkpoint directory (`model.safetensors`, `config.json`, and
that model's tokenizer/normalizer assets); none ships with this package. For the
published LIBERO numbers see
[Get the checkpoint](../README.md#get-the-checkpoint) — that checkpoint needs its
`norm_stats.json` downloaded separately and passed with `--norm-stats`.

Every example that unnormalizes accepts `--norm-stats`. `bare_model_infer.py`
returns normalized actions and does not accept `--norm-stats`.

## PI0-FAST on LIBERO

Use Robo's `franka_libero` preset for the two camera keys and action width.
PI0-FAST still needs **both** mirrored finger joints in its eight-value state;
the seven-value PI0.5 evaluation default would change its prompt. The policy
reads normalization statistics from its own LeRobot checkpoint, so do not pass
`--norm-stats` or flow-step options. Its text and FAST tokenizers must be present
at the checkpoint-declared local paths or in the local Hugging Face cache. Set
these paths when the tokenizers are stored elsewhere:

```sh
APXINF_PALIGEMMA_TOKENIZER=/models/paligemma-tokenizer \
APXINF_FAST_TOKENIZER=/models/fast-tokenizer \
python examples/robot_policy_infer.py \
  --robot franka_libero --model-dir /models/pi0fast-libero-v044 \
  --precision bf16
```

That command uses synthetic observations to check the loading and inference
contract. For frames from a LIBERO simulator, feed the same policy through
Robo's environment conversion:

```python
from apxinf_robo import build_robot_policy
from apxinf_robo.envs.libero import libero_images, libero_state

policy = build_robot_policy("franka_libero", model_dir, precision="bf16")
try:
    keys = policy.metadata["image_keys"]
    frames = libero_images(raw["agentview_image"], raw["robot0_eye_in_hand_image"])
    observation = {
        keys[0]: frames[0],
        keys[1]: frames[1],
        policy.metadata["state_key"]: libero_state(raw, finger_joints=2),
        policy.metadata["prompt_key"]: task_description,
    }
    actions = policy.infer(observation)["actions"]
finally:
    policy.close()
```

Here `raw` is one LIBERO simulator observation and `task_description` is its
instruction. `actions` is the detokenized, unnormalized action chunk; the
policy also returns `action_tokens` for inspection. Replace the synthetic
command with real observations before measuring task success.

The same checkpoint can be served or evaluated through Robo's commands. Supply
the tokenizer environment variables above to either process:

```sh
apxinf-robo serve --robot franka_libero --model-dir /models/pi0fast-libero-v044 --precision bf16
apxinf-robo eval-libero --backend in-process --model-dir /models/pi0fast-libero-v044 \
  --precision bf16 --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl pi0fast-results.jsonl --summary-json pi0fast-summary.json
```

## GR00T N1.7 on LIBERO

First [prepare the checkpoint's local Cosmos processor resources](../apxinf/doc/gr00t-n1.7.md#loading)
and install the compatible Isaac-GR00T and Transformers environment. GR00T's
official processor owns its model-specific state and action decode, so use
Robo's L2 `load_policy` entry point with explicit LIBERO wire keys. Robo's
`franka_libero` preset does not apply GR00T's decoded-gripper conversion;
`libero_gr00t_action` supplies that conversion before sending actions to
robosuite.

```python
from apxinf_robo import load_policy
from apxinf_robo.envs.libero import (
    libero_gr00t_action,
    libero_gr00t_state,
    libero_images,
)

image_keys = ("observation/image", "observation/wrist_image")
policy = load_policy(
    "/models/GR00T-N1.7-LIBERO/libero_10",
    precision="bf16",
    image_keys=image_keys,
    state_key="observation/state",
    prompt_key="prompt",
)
try:
    frames = libero_images(raw["agentview_image"], raw["robot0_eye_in_hand_image"])
    observation = {
        image_keys[0]: frames[0],
        image_keys[1]: frames[1],
        "observation/state": libero_gr00t_state(raw),
        "prompt": task_description,
    }
    actions = libero_gr00t_action(policy.infer(observation)["actions"])
finally:
    policy.close()
```

GR00T consumes named XYZ, axis-angle and **two** finger joint values. The action
conversion maps its decoded `0=closed, 1=open` gripper to robosuite's
`+1=closed, -1=open` convention. Here `raw` and `task_description` are the
simulator observation and instruction described above. The generic
`robot_policy_infer.py` builds a flat synthetic state and does not perform this
conversion; use the code above
for simulator integration. FP8 additionally needs a matching `calibration=`
profile, as described in the engine's GR00T guide.

The prepared checkpoint also works with Robo's service and LIBERO evaluator:

```sh
apxinf-robo serve --robot franka_libero \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16
apxinf-robo eval-libero --backend in-process \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 \
  --precision bf16 --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl gr00t-results.jsonl --summary-json gr00t-summary.json
```

## Qwen-Drive planning

Qwen-Drive produces a driving trajectory, not LIBERO robot actions. Use Robo's
model-agnostic `load_policy` entry point with the released checkpoint and its
planner. Run the constructed-input benchmark without scene files:

```sh
python scripts/bench_qwen_drive.py --model-dir /models/Qwen-Drive-1.0-4B
```

For a real scene with a prepared input directory, use the existing example:

```sh
python examples/qwen_drive_infer.py \
  --model-dir /models/Qwen-Drive-1.0-4B \
  --inputs /path/to/public-inputs
```

See [the Qwen-Drive guide](../doc/qwen-drive.md#accuracy-evaluation) for the
full NAVSIM evaluator and its scoring command.

To serve the same planning policy over Robo's OpenPI-compatible WebSocket
transport, omit the robot preset and supply its planner:

```sh
apxinf-robo serve --robot none --model-dir /models/Qwen-Drive-1.0-4B \
  --precision bf16 \
  --policy-options '{"planner":"/models/Qwen-Drive-1.0-4B/planner-sft","mode":"direct_planning"}'
```

Clients send the scene's `views` and prompt fields; the response contains a
`(50, 3)` trajectory in `actions`. `eval-libero` applies only to LIBERO robot
policies and rejects Qwen-Drive.

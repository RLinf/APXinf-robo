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

To verify L1/L2 numerical parity with a compatible checkpoint:

```sh
APXINF_PARITY_CHECKPOINT=/path/to/checkpoint pytest tests/test_parity.py
```

The test compares model outputs bitwise for the same captured inputs.
For engine-only APIs, see the bundled engine's
[Python examples](https://github.com/infinigence/ApxInf/tree/ba968f63c9820f7db368bea0ce17bb890aa90781/python/apxinf/examples).

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

## The checkpoint

`--model-dir` is a checkpoint directory (`model.safetensors`, `config.json`, and
that model's tokenizer/normalizer assets); none ships with this package. For the
published LIBERO numbers see
[Get the checkpoint](../README.md#get-the-checkpoint) — that checkpoint needs its
`norm_stats.json` downloaded separately and passed with `--norm-stats`.

Every example that unnormalizes accepts `--norm-stats`. `bare_model_infer.py`
returns normalized actions and does not accept `--norm-stats`.

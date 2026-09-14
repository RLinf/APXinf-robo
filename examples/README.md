# apxinf-robo examples

Minimal samples that **teach the interface**. Evaluation harnesses (LIBERO
rollouts, resumable ledgers, benchmarks) are the CLI's job — `apxinf-robo
eval-libero`, `apxinf-robo serve`, `apxinf-robo inspect` — and the simulator
bridges live in [`scripts/`](../scripts). Those are ops tooling, not API demos.

Each example runs straight from a source checkout: they share a `sys.path` shim
and a synthetic-observation builder in `_common.py`, so no dataset or simulator
is needed to see the API work. Where a checkpoint is needed the actions are
meaningless — the call shape is the point.

## Three layers, three examples

The same checkpoint is reachable at three layers, and which one you want is the
first decision to make. Each one hands you more and asks for less:

| Layer | Entry point | You supply | You get back |
|---|---|---|---|
| **L1** | `load_bare_model` | preprocessed tensors, your own tokenizer | normalized, full-width actions |
| **L2** | `load_policy` | an observation dict of raw frames, and the wire keys | deployable actions |
| **L3** | `websocket_server` | a port | the same, over the wire |

`build_robot_policy` is L2 with the wire keys filled in from a named robot
preset. It is the call most deployments want, which is why it gets its own
example rather than being a footnote to L2's.

| Example | Layer | Shows | Needs |
|---|---|---|---|
| [`register_preset.py`](register_preset.py) | — | Register your own `Embodiment x Convention` from your own package. | — |
| [`preflight_check.py`](preflight_check.py) | — | `check_checkpoint` — will this checkpoint drive this robot correctly? | a checkpoint dir |
| [`bare_model_infer.py`](bare_model_infer.py) | L1 | `load_bare_model` and `infer_rgb`'s exact contract, for callers with their own transforms. | GPU + checkpoint |
| [`policy_infer.py`](policy_infer.py) | L2 | `load_policy` with no preset — you state the wire contract argument by argument. | GPU + checkpoint |
| [`robot_policy_infer.py`](robot_policy_infer.py) | L2 + preset | `build_robot_policy` — the headline call: preset + checkpoint → deployable actions. | GPU + checkpoint |
| [`serve_websocket.py`](serve_websocket.py) | L3 | `websocket_server`, plus a real client reading the served contract off the wire. | GPU + checkpoint + `[serve]` |
| [`lerobot_loop.py`](lerobot_loop.py) | L2 + preset | Drop `ApxInfPolicy` into lerobot's `record_loop`, leaving the robot side untouched. | lerobot + GPU |
| [`g1_adapter_smoke.py`](g1_adapter_smoke.py) | L2 + preset | Prove a body's whole adapter chain runs, asserting on shape rather than printing. | GPU + 3-view checkpoint |

`g1_adapter_smoke.py` is the one that checks rather than tours. It is the
runnable last step of [`doc/adding-an-embodiment.md`](../doc/adding-an-embodiment.md):
the G1 is a worked instance you copy for your own body. It stays here rather than
in `tests/` because it needs a GPU and a three-view checkpoint that no CI has —
`tests/test_parity.py` can be skipped when its checkpoint is absent, but nothing
would ever run this one.

L1 and L2 must return the same numbers for the same inputs, and that is not
automatic — it is checked by [`tests/test_parity.py`](../tests/test_parity.py),
not by an example, because it is an invariant rather than a usage pattern:

```sh
APXINF_PARITY_CHECKPOINT=/path/to/checkpoint pytest tests/test_parity.py
```

That test records what L2 hands the model, replays it through a bare handle, and
requires the two to agree **bitwise**. It also pins the failure that motivated
it: a bare handle loaded without the tuned GEMM tactics diverges from the policy
path on byte-identical inputs, which is what `--no-tactics` in
`bare_model_infer.py` reproduces by hand.

The engine has its own examples for the layer below —
[`python/apxinf/examples/`](https://github.com/infinigence/ApxInf/tree/main/python/apxinf/examples)
covers `AutoPolicy`, `Pi05Policy`, and the raw websocket server, none of which
know what a robot is. Reach for those when the question is about a *checkpoint*;
reach for these when it is about a *robot*.

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
  padding) and are rejected before inference, which is the correct answer rather
  than a limitation of the example.

## Quick start

```sh
# No checkpoint, no GPU — the extension point and the wire contract
python examples/register_preset.py

# Vet a checkpoint against every registered preset before serving it
python examples/preflight_check.py --model-dir /path/to/checkpoint

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

`preflight_check.py` exits `2` on a FAIL and `1` on a WARN, the same contract as
`apxinf-robo inspect`, so it is usable as a deployment gate. With no `--robot` it
checks *every* registered preset, so a single-robot checkpoint exits `2` by
design — a LIBERO checkpoint's 7-wide statistics are a FAIL against the 16-DoF
G1. Pass the preset you actually intend to serve to gate on one.

Note what it does *not* check: your GPU, your driver, your memory, your
precision, whether a tuned kernel database exists for this device. Preflight
answers "does this checkpoint match this **robot**", not "will it run on this
**machine**" — the machine failures are loud, and you find out by loading.

## The checkpoint

`--model-dir` is a checkpoint directory (`model.safetensors`, `config.json`, and
that model's tokenizer/normalizer assets); none ships with this package. For the
published LIBERO numbers see
[Get the checkpoint](../README.md#get-the-checkpoint) — that checkpoint needs its
`norm_stats.json` downloaded separately and passed with `--norm-stats`.

Every example that unnormalizes accepts `--norm-stats`. `bare_model_infer.py`
does not, and that absence is the point: L1 returns normalized-domain actions
and never reads the statistics at all.

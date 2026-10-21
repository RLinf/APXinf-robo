# Adding an Embodiment

A guide for serving a robot through the OpenPI-compatible websocket API.

The companion doc for the *model* side lives in the engine repository
([`doc/adding-a-new-model.md`](https://github.com/infinigence/ApxInf/blob/main/doc/adding-a-new-model.md)).
That one is about weights and kernels. This one is about the **wire contract**:
which keys the client sends, how state is routed, and what the action vector
means when it comes back.

The split between the two repositories follows exactly that line. If your change
is caused by a new *checkpoint*, it belongs in ApxInf. If it is caused by a new
*body*, *dataset dialect*, or *simulator*, it belongs here.

## Wire contracts and presets

A pi05 checkpoint does not carry its wire contract. The weights fix the view
count and the action width; they say nothing about whether the base camera
arrives as `observation/image` or as `obs["images"]["cam_high"]`, whether the
state vector is discretized into the prompt or dropped, or whether the 32-wide
model output should be truncated to 7 or to 16.

The model layer does not define dataset wire keys — that is the whole reason this
package exists. A robot preset selects the wire keys, state routing, action
width, and robot processing steps as one deployable contract:

```
openpi:      serve_policy.py --policy.config pi05_UnitreeG1_groundwire
apxinf-robo: apxinf-robo serve --robot unitree_g1
```

## Contract namespaces

| | what it is | where it lives | on the wire? |
|---|---|---|---|
| **view slots** | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` | `VIEW_SLOTS` in ApxInf's `policies/base.py` | never — the *order* is baked into the weights |
| **wire keys** | `observation/image`, `images/cam_high`, … | a `Convention` in `conventions/` | yes — this is what the client sends |
| **training feature names** | LeRobot `config.json` `input_features` | the checkpoint | no |

View slots are *model* vocabulary and stay in the engine; this package imports
the tuple and never redefines it. Wire keys are a *dataset's* and live here.

A preset pairs each view slot with the wire key that fills it. `image_keys` is
order-significant: entry *i* is stacked into model view slot *i*. A tuple written
in the wrong order still stacks, still has the right shape, and silently feeds
the wrong camera to each slot — pairing every key with its slot makes that
reviewable instead of positional.

A policy without explicit camera keys uses its view slot names, and a request
with different keys raises a `KeyError` that reports the expected and received
contracts.

## Launching a server for an existing embodiment

`--robot` is the only flag that has to be right. It selects the wire keys, the
state routing, the action width, and the robot pre/post steps together.

```bash
# Franka Panda under LIBERO's key convention (the default)
apxinf-robo serve \
  --model-dir "$APXINF_MODEL_DIR" \
  --robot franka_libero \
  --precision bf16 --host 0.0.0.0 --port 8000

# Unitree G1: 3 cameras, nested keys, state discretized into the prompt
apxinf-robo serve \
  --model-dir "$G1_MODEL_DIR" \
  --robot unitree_g1 \
  --precision bf16 --host 0.0.0.0 --port 8000
```

`--robot` defaults to `franka_libero`, so an existing LIBERO launch command needs
no change beyond the executable name. `apxinf-robo serve --help` prints every
registered preset with its slot→key mapping, and does so without importing CUDA.

### Confirm the contract before trusting the numbers

The server logs what it ended up serving and publishes the same thing in its
connect-time metadata. Read one of them **before** concluding anything about
accuracy:

```
INFO serving robot=unitree_g1 robot_steps=True H=50 x D=32
     image_keys=['images/cam_high', 'images/cam_left_wrist', 'images/cam_right_wrist']
     state=state discrete_state=False
```

From a client:

```python
from openpi_client import websocket_client_policy as wcp
c = wcp.WebsocketClientPolicy(host="127.0.0.1", port=8000)
meta = c.get_server_metadata()
assert meta["robot"] == "unitree_g1"
assert meta["robot_steps"] is True      # this robot's arithmetic is actually wired
assert meta["image_keys"] == [...]      # the keys you are actually sending
assert meta["discrete_state"] is True   # a joint-space robot needs this on
```

Published keys: `robot`, `robot_steps`, `robot_slots`, `state_dim`, `model_type`,
`image_keys`, `state_key`, `prompt_key`, `discrete_state`, `state_normalized`,
`action_horizon`, `action_dim`, `model_action_dim`, `num_views`, `image_size`,
`input_pipeline`, `output_pipeline`. A key mismatch is invisible on the wire but
obvious here. `robot_steps` is `False` on a server that serves a preset's keys
without running its builder — see `--random-weights` below.

### Per-field overrides

A deployed client may already speak a fixed dialect that does not match any
preset. Override the individual field rather than editing the preset:

```bash
apxinf-robo serve \
  --model-dir "$APXINF_MODEL_DIR" --robot franka_libero \
  --image-keys 'observation/exterior_image_1_left,observation/wrist_image_left' \
  --state-key 'observation/joint_position'
```

Nested layouts are written as a slash path: `--image-keys
'images/cam_high,images/cam_left_wrist'`. `--discrete-state` /
`--no-discrete-state` and `--action-dim` override the remaining fields.

Register a preset for a contract used by multiple deployments.

### Serving fewer cameras than the checkpoint declares

```bash
apxinf-robo serve \
  --model-dir "$APXINF_MODEL_DIR" --robot franka_libero \
  --image-keys 'observation/image' --num-views 1 \
  --precision bf16 --port 8000
```

This drops the trailing view slots at **load** time. It is numerically equivalent
to what OpenPI does by zero-padding the absent views and masking them out —
masked tokens consume no RoPE position (`positions = cumsum(input_mask) - 1`),
are excluded from attention (`valid_mask = mask[:, None, :] * mask[:, :, None]`),
and `embed_prefix` runs the same `PaliGemma.img` encoder per view with no
per-slot learned embedding. `num_views` only sizes the prefix; nothing
weight-shaped depends on it. ApxInf skips the ~256 patch tokens per absent view
(measured 1.38× on a 2-view LIBERO checkpoint served with one camera, on Orin).

`--num-views` must be specified when loading fewer views. A shorter
`--image-keys` list alone is invalid. This is a server-side loading option and
does not change the OpenPI wire protocol.

## Adding a new embodiment

### Step 0 — write down the contract

Before any code, get these seven facts from the OpenPI `TrainConfig` the
checkpoint was fine-tuned under (its `data_transforms` pair is the source of
truth):

1. camera wire keys, **in view-slot order**, and whether they are flat or nested;
2. the state wire key;
3. whether state is discretized into the prompt or dropped;
4. state width and layout;
5. model action width vs. deployable action width;
6. whether actions are deltas needing the current state to resolve;
7. any robot-space convention (joint sign flips, gripper mapping).

Facts 1–5 are just a table row. Facts 6–7 are the only ones that need code.

`python scripts/from_openpi.py --model-dir "$CKPT" --robot <preset>` reads facts 1–5
straight out of a checkpoint's own `metadata.pt` and diffs them against a preset,
which is faster than reading the training config by hand and cannot misremember.

### Step 1 — decide whether you need robot steps at all

| the checkpoint… | what to write |
|---|---|
| emits absolute actions in robot space, at the deployable width | **nothing** — a preset row is the whole port |
| needs a fixed truncation only | a preset row with `action_dim=N` |
| emits deltas, or needs a sign/gripper convention | processing steps + a builder (Steps 2–3) |

`franka_libero` requires only a preset row.

### Step 2 — robot processing steps

`src/apxinf_robo/embodiments/<robot>/steps.py`. Each step is an
`apxinf.processors.ProcessorStep`: `dict -> dict` over the shared data dict, with
a name. These must be **model-agnostic** — they reference no policy symbols and
vary only with the robot body. Importing `ProcessorStep` is fine: it is shared
type information, not an engine instance.

The data-dict contract:

* input chain reads `observation` / `prompt`, writes `rgb` (uint8 NHWC),
  `token_ids` (uint32), `noise`;
* output chain reads `normalized_actions`, writes `trimmed` then `actions`
  (unnormalized float32). `observation` is threaded through so an output step can
  read state.

Resolve every wire key through `lookup_key` / `has_key` / `set_key` from
`apxinf.processors.transforms` rather than indexing the observation directly.
They resolve **flat first, then as a nested path**, so one flat key string
addresses either layout: `"observation/image"` hits `data["observation/image"]`
if that key literally exists, and `"images/cam_high"` walks
`data["images"]["cam_high"]` when it does not. `set_key` returns a copy — the
client's dict is never mutated.

Write each step as a no-op when its input is absent. `UnitreeG1DecodeState`
returns `data` unchanged when there is no state, so state-off serving degrades to
"this step does nothing" instead of raising deep in the pipeline.

### Step 3 — the builder

`src/apxinf_robo/embodiments/<robot>/builder.py`. One factory that loads the
checkpoint through `apxinf_robo.engine.load_policy` — so `config.json` decides
which model it is, not this file — and then wraps the robot steps *around*
whatever chain that policy has:

```python
from apxinf.policies.base import ComposablePolicy

from ...engine import load_policy

# state_key and image_keys come from the selected convention.
def build_<robot>_policy(model_dir, *, state_key, image_keys, **load_kwargs):
    base = load_policy(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,      # keep full model width; the encode step truncates
        state_key=state_key,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        raise TypeError(f"{type(base).__name__} has no with_adapter(); ...")
    return base.with_adapter(
        before=[("<robot>_decode_state", <Robot>DecodeState(state_key))],
        after=[("<robot>_absolute", <Robot>AbsoluteActions(state_key)),
               ("<robot>_encode",   <Robot>EncodeActions())],
        action_dim=ROBOT_DIM,             # the width the encode step leaves behind
        metadata={"robot": "<robot>"},
    )
```

`load_policy` is imported from `engine.py` rather than reaching for
`apxinf.AutoPolicy` directly. That is the rule for this package: **one module
loads and runs the engine**, so a submodule bump is one diff to read and a
compatibility shim has one place to live. `tests/test_robot_presets.py`
monkeypatches that name, which is only possible because it is a module-level
import here rather than an attribute lookup on `apxinf`.

The builder depends on the `ComposablePolicy` interface, not on a concrete model
class or the names of model-specific processing steps. `with_adapter` prepends
robot input steps and appends robot output steps around the model pipeline —
strict onion nesting, not a bisection of the model's own chain.

The builder preserves these orderings:

* the decode-state step goes **before** the model's whole input chain, so
  discretized state (when on) and the delta→absolute output step both see the
  decoded state;
* the model pipeline resolves the checkpoint's deployable action width before
  robot output steps run. Pass `action_dim=None` when the appended encode step
  owns the final width.

Declare `action_dim=` only for the width a step you appended actually produces.
Without the truncating step there is nothing to claim — inherit the model's own
width instead of advertising one nothing emits.

### Step 4 — register the convention and the preset

A preset pairs an `Embodiment` with a `Convention` and declares whether the
checkpoint discretizes state into the prompt. The embodiment defines the robot's
camera count, action widths, and processing builder. The convention defines
dataset wire keys. They are separate because they vary independently: LIBERO and
DROID are the same arm with different dialects. State encoding belongs to the
preset because it follows the checkpoint's training config, not the wire keys.

```python
# src/apxinf_robo/conventions/mydataset.py — the dialect, independent of any arm
from .base import Convention, register_convention

MY_DATASET_KEYS = register_convention(Convention(
    name="mydataset",
    image_keys=("observation/image", "observation/wrist_image"),  # in view-slot order
    state_key="observation/state",
))

# src/apxinf_robo/embodiments/myarm.py — the body
from .base import Embodiment

MY_ARM = Embodiment(
    name="myarm",
    num_cameras=2,
    action_dim=7,               # None keeps full width when an encode step truncates
    state_dim=8,                # how wide norm_stats["state"] has to be
    action_width=7,             # how wide norm_stats["actions"] has to be
    builder=build_my_robot_policy,   # omit for the stock policy
    builder_kwargs={},          # constants the builder always receives
)

# src/apxinf_robo/presets.py — the pairing
MY_ROBOT = register_robot_preset(RobotPreset(
    name="myarm_mydataset",
    embodiment=MY_ARM,
    convention=conventions.MY_DATASET_KEYS,
    summary="MyArm, MyDataset keys: 2 cameras, 7-dim action",
    discrete_state=False,       # False *drops* state entirely — not "keeps it raw"
))
```

To serve the same embodiment with another supported dataset contract, register
another convention and preset pairing. A client that routes its cameras to view
slots itself, such as RPent's APXInf server, needs no dialect: pair the preset
with `conventions.CANONICAL`, which uses the engine's slot names.

For a robot defined in another package, call `register_convention(...)` and
`register_robot_preset(..., aliases=(...))` at your module scope; importing that
module before the server starts is enough to make it `--robot <name>`.
Re-registering an existing name needs an explicit `replace=True`, and an alias
may never shadow a canonical preset name — a silent overwrite would change what a
launch command already in production resolves to.

Only registered preset pairings are deployable through `--robot`.

Validation runs when the module loads. `Convention.__post_init__` rejects
duplicate wire keys and more keys than there are view slots;
`Embodiment.__post_init__` rejects a camera count outside `1..len(VIEW_SLOTS)`;
`RobotPreset.__post_init__` rejects a convention whose camera count disagrees
with the body's. Slot names are *derived* from `image_keys` in order, so "base +
right wrist" is not expressible at all — a checkpoint fills view slots from 0 up,
and the only way to spell "wrist camera only" is to put it in slot 0.

Set `state_dim` and `action_width` so preflight can validate normalization
metadata. `action_dim` controls model-output trimming during loading;
`action_width` records the robot action width. If a robot output step performs
the truncation, set `action_dim=None` and declare the result in `action_width`.

Set `norm_dtype="float64"` in `builder_kwargs` when matching an OpenPI
checkpoint's normalization and state-discretization numerics. Otherwise the
normalizers follow the input dtype.

Pass `aliases=(...)` to `register_robot_preset` for supported alternate launch
names.

### Naming: `<arm>_<key convention>`

The arm alone does not determine the contract. LIBERO and DROID are both Franka
Panda, yet LIBERO sends `observation/image` with a 7-dim EEF-delta action while
DROID sends `observation/exterior_image_1_left` with a different action space. So
a preset is named for the arm **and** the dataset convention whose keys it
implements: `franka_libero`, not `libero` (a benchmark, not a robot) and not
`franka` (ambiguous). A single-embodiment robot that owns its convention needs no
suffix — `unitree_g1`.

### Step 5 — tests

Add to `tests/test_robot_presets.py`. Five things are worth asserting, and they
run on CPU with a mock model — no GPU, no checkpoint:

1. **the preset matches the openpi transform it mirrors** — keys, order, widths,
   `discrete_state`, spelled out literally rather than derived;
2. **an unmodified openpi client round-trips** — `UnitreeG1ServingTest` starts
   the real `WebsocketPolicyServer` on a mock model and sends the observation the
   integrator's client actually sends;
3. **the wrong dialect fails loudly** — a LIBERO-shaped observation against your
   server must raise, naming the served `image_keys`;
4. **slot order is load-bearing** — reordering `image_keys` reorders the stacked
   views (`ImageSlotOrderTest` asserts this against the tensor, not the metadata);
5. **the contract you publish is the one you serve** — `BuildRobotPolicyTest`
   asserts the resolved keys and `robot_steps` reach the builder and the metadata,
   and `SyntheticContractTest` asserts a checkpoint-free server names what it
   cannot honour instead of passing for the real embodiment.

```bash
pytest tests/test_robot_presets.py          # no GPU needed
```

### Step 6 — check that preflight refuses the wrong checkpoint

Run preflight before serving a checkpoint:

```bash
apxinf-robo inspect --model-dir "$CKPT" --robot <preset>
```

It validates normalization metadata from OpenPI `norm_stats.json` or LeRobot
processor sidecars. Missing optional statistics are reported as identity
passthrough; declared but unreadable or incompatible statistics are fatal. When
the checkpoint carries OpenPI `metadata.pt`, `scripts/from_openpi.py` also
checks the preset against its serialized `TrainConfig` (camera keys, state
routing, action mode, horizon, and token length). The server runs the same checks
before loading weights and refuses fatal mismatches.

Preflight is split across the two repositories along the same line as everything
else: `apxinf.checkpoints.inspect_checkpoint` answers "what does this checkpoint
say about itself", and `apxinf_robo.preflight.check_checkpoint` answers "does
that match this preset". They share one `Finding` vocabulary so a report reads as
one list.

Add cases to `tests/test_preflight.py` for embodiment-specific width rules. A
valid checkpoint and preset pairing must produce no fatal findings.

## Verification recipe

Offline first, then one GPU pass. In order, because each step localizes a
different class of bug:

```bash
# 1. contract + plumbing, CPU only, mock model
pytest tests/test_robot_presets.py tests/test_preflight.py

# 2. does the checkpoint agree with the preset? (seconds; no weights loaded)
python scripts/from_openpi.py --model-dir "$CKPT" --robot <preset>

# 3. real checkpoint, native contract
apxinf-robo serve --model-dir "$CKPT" --robot <preset> --precision bf16 --port 8000
#    -> read the "serving robot=..." line; assert every field

# 4. real transport, unmodified openpi client
#    -> assert actions.shape == (meta["action_horizon"], meta["action_dim"])
#    -> assert np.isfinite(actions).all()

# 5. wrong-dialect rejection: send another preset's keys; it must raise

# 6. the two engine depths still agree on this checkpoint
APXINF_PARITY_CHECKPOINT="$CKPT" pytest tests/test_parity.py
```

Step 4's shape assertion should read the shape **off the metadata**, not off a
constant. A hard-coded expected shape tests your memory of the checkpoint rather
than the server.

For a transport-only check with no weights on disk, `apxinf-robo serve
--random-weights --robot <preset>` serves the preset's key layout and view count
on synthetic weights. That is all it serves: the actions are numerically
meaningless, and the preset's `builder` never runs, so none of its robot pre/post
steps is wired. It publishes `robot_steps=false` and warns once at startup naming
every gap — `discrete_state` (the synthetic tokenizer never reads state) and, for
a robot-step preset, the skipped factory and the action width that came out of
the model instead of out of the encode step.

### If you are serving FP8

An FP8 profile is bound to a *dialect* as well as to a checkpoint: it is
calibrated on activations produced by whatever keys the client sent. Serving a
preset under a profile calibrated for another one is silent — the scales are the
wrong size and the actions merely get worse.

`apxinf-robo capture-libero --robot <preset>` writes an NPZ directory whose
field names come from that preset's convention, and ApxInf's
`scripts/calibrate_pi05.py --input-dir` turns it into the profile. The engine
defaults its wire keys to the model's own slot names rather than to any preset,
so the calibrator has to be told the dialect — `capture-libero` prints the
`--image-key` / `--state-key` / `--prompt-key` line for the preset it just
captured under, so run that rather than retyping it. Calibrate through the
preset you serve through. For an embodiment with no simulator here, write the
same NPZ layout from your own capture — one file per observation, wire keys as
array names, prompt as a 0-d unicode array — or export a JSONL manifest instead.

## Operational constraints

1. **`discrete_state=False` drops state; it does not pass it through raw.**
   There is no third state. A joint-space robot served with `discrete_state=False`
   loses its proprioception silently, which also makes any delta→absolute step a
   no-op (a delta cannot be resolved without current joint positions). This is
   also why `state_key` is **required whenever it is read**: a wrong camera key
   raises on the first inference, but a missing state key is silent, so the
   policy refuses to be built with `discrete_state=True` and no key.
2. **`image_keys` order is the view slot order.** Wrong order → right shape,
   wrong cameras, no error. This is why presets pair keys with slot names.
3. **Truncate after unnormalize, not before.** Unnormalize at full model width so
   the delta→absolute step sees the whole action; let the robot encode step do
   the 32→16.
4. **`action_dim` in a preset is the *deployable* width, not the model's.**
   `None` is correct when a robot output step does the truncation itself —
   metadata still reports both (`action_dim` and `model_action_dim`).
5. **Nested keys need `lookup_key`, not `obs[k]`.** Both wire shapes exist in the
   wild and arrive as one string. Indexing directly works for LIBERO and raises
   for G1.
6. **A preset's `num_views` must equal the checkpoint's**, unless you pass
   `--num-views` to load fewer. `Pi05Policy.default_pipelines` rejects a mismatch
   and names the fix in the message.
7. **`--random-weights` previews the wire layer only.** Its synthetic tokenizer
   ignores state, so the published metadata says `discrete_state=False`, and it
   never runs `preset.builder`, so no robot pre/post step is wired and the action
   width is the model's rather than the encode step's. Truthful about that server,
   misleading as a preview — hence `robot_steps=false` in the metadata and a
   startup warning per gap. Use a checkpoint for a real contract check.
8. **Metadata is the contract; assert it from the client.** Every silent failure
   in this area is visible in the connect-time metadata one line before it
   becomes an "accuracy problem."

## Rules

1. **The embodiment is explicit, never inferred.** No preset is guessed from the
   checkpoint, and `--num-views` is not inferred from `len(image_keys)`. An
   omission should fail at startup.
2. **Robot steps are model-agnostic; builders are where they meet a policy.**
   `embodiments/<robot>/steps.py` imports no policy symbols;
   `embodiments/<robot>/builder.py` does the assembly. Neither holds a wire key:
   those are a dataset's, and live in `conventions/`.
3. **Only `engine.py` loads or runs the engine.** Everything else may import
   ApxInf for *vocabulary* — `Policy`, `ProcessorStep`, `Finding`, `VIEW_SLOTS`,
   `Unnormalizer` — because shared type information belongs where it is used.
   Nothing else constructs a model, allocates a device, or serves one.
4. **Fail loudly with the served contract in the message.** Every error in this
   layer names what the server is actually serving, because the person reading it
   is comparing two dialects.
5. **A preset row is the cheapest port; prefer it.** Only write steps for a
   convention the model output genuinely does not satisfy.
6. **Overrides exist for deployed clients, not for new robots.** If the same
   `--image-keys` shows up twice, add a preset.

## Concrete example

The Unitree G1 port is the reference implementation — it exercises every step,
including the ones `franka_libero` skips:

- `src/apxinf_robo/presets.py` — the registry and both presets
- `src/apxinf_robo/embodiments/base.py` — `Embodiment` / `RobotPreset` machinery
- `src/apxinf_robo/conventions/` — the LIBERO, G1, and canonical wire keys
- `src/apxinf_robo/embodiments/unitree_g1/steps.py` — decode-state,
  delta→absolute, 32→16 encode
- `src/apxinf_robo/embodiments/unitree_g1/builder.py` — the builder that splices
  them in
- `apxinf.processors.transforms` (engine side) — `lookup_key`/`has_key`/`set_key`
- `tests/test_robot_presets.py` — contract, serving, and slot-order tests
- `examples/g1_adapter_smoke.py --model-dir "$G1_CKPT"` — the one end-to-end
  check that needs a GPU; copy it for your own body

<div align="center">
  <img src="https://media.githubusercontent.com/media/apxinf/apxinf.brand/refs/heads/main/logo.png" alt="apxinf-logo" width="512"/>
</div>

# APXinf-robo

ApxInf is a reimagined edge inference engine born of the agentic coding era,
combining high performance, reliability, and energy efficiency across devices
with an evolving agentic workflow that radically simplifies custom model development.

- implemented with system language Rust with no other externel dependencies
- embodied AI is highest priority, VLA/WAM models on Jetson/DriveOS Thor/Orin
- Agentically optimized CUDA Kernels

The first version of ApxInf ships with highly optimized PI-0.5 VLA model on Jetson Thor &
Orin devices, and supports BF16, FP8 and INT8 precisions.

APXinf-robo carries ApxInf as a git submodule at `apxinf/`.

## Quick start

Make sure you have ApxInf built and installed, see [Build APXinf-robo](#build-apxinf-robo) for instructions.

### Quickly benchmarking PI-0.5

Benchmarking PI-0.5 with randomly generated weights.

```bash
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 \
  --warmup 10 --samples 100 --autotune
```

```bash
python scripts/bench_pi05.py --random-weights --precision fp8 --layer l1 \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 --autotune
```

Reported latency is P50 over 30 samples after 10 warm-up iterations
(`--warmup` / `--samples`).

### Run a policy through Python API

```python
import numpy as np
from apxinf_robo import build_robot_policy

policy = build_robot_policy("franka_libero", "<path-to-model>", precision="bf16")

observation = {
    key: np.zeros((256, 256, 3), np.uint8)
    for key in policy.metadata["image_keys"]
}
observation[policy.metadata["prompt_key"]] = "put both moka pots on the stove"
if state_key := policy.metadata.get("state_key"):
    observation[state_key] = np.zeros(policy.metadata["state_dim"], np.float32)

result = policy.infer(observation)

result["actions"]   # (H, policy.action_dim) float32, unnormalized
result["timing"]    # model_ms / total_ms
policy.close()
```

`<path-to-model>` is a checkpoint directory (`model.safetensors`, `config.json`,
and that model's tokenizer/normalizer assets); none ships with this package.
Resize, tokenization, normalization, and the flow sampler all run inside `infer`
— pass raw frames.

### Serve it with OpenPI compatible websocket server

```bash
apxinf-robo serve --robot franka_libero \
  --model-dir <path-to-model> --precision bf16 --port 8000
```

An unmodified `openpi-client` connects to it:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
actions = client.infer(observation)["actions"]
```

## Performance

Two views, 224x224 NHWC `uint8`, 10 flow steps, `H=10`, batch 1. Latency is
steady-state CUDA Graph replay P50.

| Hardware | Precision | Latency | Throughput |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 72.45 ms | 13.8 Hz |
| Jetson AGX Thor | FP8 | **41.16 ms** | **24.3 Hz** |
| Jetson AGX Orin | BF16 | 165.67 ms | 6.0 Hz |
| RTX 4090 | BF16 | 31.38 ms | 31.9 Hz |
| RTX 4090 | INT8 | 25.99 ms | 38.5 Hz |

LIBERO-10, 10 tasks x 50 episodes, `H=10`, `replan=5`, seed 7. PI0.5 reference
is 92.4%.

| Hardware | Precision | Trials | Success | Rate |
|---|---|---:|---:|---:|
| Jetson AGX Thor | BF16 | 500 | 464 | 92.8% |
| Jetson AGX Thor | FP8 | 500 | 470 | 94.0% |
| Jetson AGX Orin | BF16 | 500 | 460 | 92.0% |


## Port a new model with an agent

`apxinf/skills/model-port-workflow` drives the whole sequence.

Install it once, from the repository root:

```bash
# Claude Code
mkdir -p .claude/skills && ln -s ../../apxinf/skills/model-port-workflow .claude/skills/

# Codex
mkdir -p ~/.agents/skills && ln -s "$(pwd)/apxinf/skills/model-port-workflow" ~/.agents/skills/
```

Then invoke it with the model, the target, and the acceptance bar:

```
/model-port-workflow port GR00T N1.7 from <path-to-reference-implementation> to
ApxInf, Jetson Thor, BF16, parity against the reference within 1e-2
```

It works from the same guides a human would follow. These live in the engine, so
they are linked upstream rather than by path: GitHub does not serve a
submodule's files under this repository's tree, though a recursive clone has
them all under `apxinf/doc/`.
- [porting workflow](https://github.com/infinigence/ApxInf/blob/main/doc/porting-workflow.md),
- [adding a new model](https://github.com/infinigence/ApxInf/blob/main/doc/adding-a-new-model.md),
- [model-layer architecture](https://github.com/infinigence/ApxInf/blob/main/doc/model-layer-architecture.md),
- [adding new kernels](https://github.com/infinigence/ApxInf/blob/main/doc/adding-new-kernels.md).

## Build APXinf-robo

```bash
git clone --recursive <repo-url> && cd APXinf-robo
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
pip install -e "apxinf/python/apxinf[serving]"
pip install -e ".[libero,serve]"
```

Activate a venv or conda env before installing. One extra covers both model
families: tokenization is native — the binding carries SentencePiece for PI0.5
and the HF tokenizer for WallOSS — so `serving` only adds the
msgpack/websockets transport. Drop it for in-process use.

`--features cuda` is a Cargo feature, not a CUDA installation: it compiles the
CUDA backend into the binding, and it is required — the PI0.5 runtime is only
registered on CUDA devices, as is WallOSS.

The build queries the visible GPU for its compute capability and compiles the
kernels for exactly that architecture, so build on the machine you deploy to;
cross-compiling fails unless `APXINF_CUDA_ARCH` names the target (`sm_87` Orin,
`sm_101` Thor-U, `sm_110` Thor).

Confirm the binding imports and reaches the GPU:

```bash
python -c 'import apxinf_py; print(apxinf_py.__version__)'
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 --samples 5
```

Needs a Linux host with an NVIDIA driver and a CUDA toolkit, a stable Rust
toolchain, and `cmake` — the binding links SentencePiece statically, which
builds its C++ library from source. If `nvcc --version`, `cargo --version` or
`cmake --version` fails, set them up first:

- [NVIDIA build environment](#nvidia-build-environment)
- [Rust toolchain](#rust-toolchain)
- `cmake` (`apt install cmake`)


## Using APXinf-robo from Python

Three public layers, each wrapping the one before. Pick the outermost one that
still leaves you the control you need.

All three reach the engine through `apxinf_robo.engine`, the one module in this
package that imports `apxinf` — so pinning a new submodule SHA is one diff to
read and a compatibility shim has one place to live. The engine's own names
(`apxinf.Model`, `apxinf.AutoPolicy`, `apxinf.serving.WebsocketPolicyServer`)
still work, but they are not this package's seam; see
[Adding an embodiment](doc/adding-an-embodiment.md).

### L1 — bare model

You own resize, tokenization, noise, and unnormalization; the model takes
already-resized frames and returns a **normalized-domain** chunk.

```python
from apxinf_robo import load_bare_model

model = load_bare_model("<path-to-model>", precision="bf16")

# rgb: uint8 [views, H, W, 3] at model.image_size; tokens: uint32; noise: float32
actions = model.infer_rgb(rgb, "nhwc", token_ids, noise)   # (H, action_dim)
model.action_horizon, model.num_views, model.image_size    # what it was loaded for
```

### L2 — policy

Adds the pre/post pipelines and reads the checkpoint's tokenizer and
`norm_stats`, so it takes a raw observation dict and returns deployable actions.
`build_robot_policy` is the call to reach for: it pairs the checkpoint with a
robot preset, so the camera keys, state routing, and deployable action width
come from the robot rather than from the caller. That pairing is what this
package adds over the engine.

```python
from apxinf_robo import build_robot_policy

policy = build_robot_policy(
    "franka_libero", "<path-to-model>",
    precision="bf16",
)

policy.metadata             # robot, robot_slots, image_keys, state_key, action_horizon, ...

# Pipelines are ordered named steps — image_stack -> tokenize in, trim -> unnormalize
# out — and every mutation returns a new one. A custom step is any
# apxinf.processors.ProcessorStep subclass; a bare callable is rejected.
policy.input_pipeline = policy.input_pipeline.replace("tokenize", MyTokenizeStep())
policy.output_pipeline = policy.output_pipeline.insert_after(
    "unnormalize", ("clip", MyClip())
)

result = policy.infer(observation)
result["normalized_actions"]  # what L1 returned, before trim + unnormalize
```

Each preset field can be overridden per call (`image_keys`, `state_key`,
`prompt_key`, `action_dim`, `discrete_state`) for a client that already speaks a
fixed dialect. When no robot applies — vetting a checkpoint, reproducing the
engine's own numbers — `load_policy` takes the same arguments and gives you the
checkpoint's own contract with no body attached:

```python
from apxinf_robo import load_policy

policy = load_policy(
    "<path-to-model>",
    precision="bf16",
    action_dim=None,        # default: infer the model's full vector from checkpoint weights
)
```

### L3 — websocket server

Wraps an L2 policy in the OpenPI wire protocol. See
[OpenPI-compatible serving](#openpi-compatible-serving).

## OpenPI-compatible serving

The server speaks OpenPI's websocket protocol, so an existing `openpi-client`
robot stack connects without a code change — swap the endpoint and keep the
observation dict you already send.

```python
from apxinf_robo import build_robot_policy, websocket_server

policy = build_robot_policy("unitree_g1", "<path-to-model>", precision="bf16")
websocket_server(policy, "0.0.0.0", 8000).serve_forever()
```

`--robot` selects the wire contract — camera keys, state routing, deployable
action width — the way OpenPI selects a `TrainConfig`. It is not negotiated at
connect time, so a checkpoint fine-tuned for another robot must name its preset;
a mismatch produces wrong actions, not an error.

| Preset | Cameras | State | Action |
|---|---|---|---|
| `franka_libero` | `observation/image`, `observation/wrist_image` | `observation/state`; encoding is checkpoint-specific | 7-dim EEF delta |
| `unitree_g1` | 3 views | 16-dim, discretized into the prompt | delta joints, 32→16 encode |

`--help` lists every preset. If an installed client uses different keys, pass the
concrete policy fields through `--image-keys` / `--state-key` or register a named
preset; do not edit the generic server. See
[Adding an embodiment](doc/adding-an-embodiment.md).

The server keeps the checkpoint's native action width unless the user supplies
`--action-dim` or selects a preset with a deployable width. It publishes the
resolved wire contract in connect-time metadata so the client can assert it
rather than guess.


## Precisions

### BF16

The default, on every supported device. Runs on the checkpoint alone; no
calibration.

```bash
apxinf-robo serve --robot franka_libero \
  --model-dir <path-to-model> --precision bf16 \
  --port 8000
```

```python
policy = build_robot_policy("franka_libero", "<path-to-model>", precision="bf16")
```

### FP8

Thor only, where it is the fastest path. Orin has no FP8 Tensor Cores and is not
supported.

FP8 needs per-tensor activation scales. Pass the calibration generated for the
deployment data explicitly; when omitted, ApxInf falls back to
`<path-to-model>/calibration.json`:

```bash
apxinf-robo serve --robot franka_libero \
  --model-dir <path-to-model> --precision fp8 \
  --calibration <path-to-calibration.json> \
  --port 8000
```

```python
policy = build_robot_policy(
    "franka_libero", "<path-to-model>",
    precision="fp8",
    calibration="<path-to-calibration.json>",
)
```

If the checkpoint does not contain `calibration.json`, generate one from
representative Observations:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --manifest <path-to-observations.jsonl>
```

For the PI0.5 LIBERO checkpoint, capture task-balanced observations directly
from the native LIBERO10 simulator:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --libero-suite libero_10
```

That drives MuJoCo inside the calibrator, under the *engine's* default wire keys.
To calibrate for the dialect a robot preset actually serves — or to keep MuJoCo
out of the machine that holds the checkpoint — capture the frames first and hand
the calibrator the directory:

```bash
apxinf-robo capture-libero --suite libero_10 --output-dir /tmp/libero-calib
```

`capture-libero` prints the matching `calibrate_pi05.py --input-dir` line filled
in with `--robot`'s keys; run that rather than transcribing them. The captured
NPZ files are a reviewable, re-runnable artifact, which the in-process path is
not.

See [PI0.5 FP8 calibration](https://github.com/infinigence/ApxInf/blob/main/doc/pi05-fp8-calibration.md) for
the Observation format, native LIBERO sampling, and output options. Both native
paths use the same LIBERO/MuJoCo dependencies as
[LIBERO evaluation](#libero-evaluation).

### INT8

W8A8, optimized for Orin (SM87) and Ada (SM89). Needs nothing beyond the
checkpoint.

```bash
apxinf-robo serve --robot franka_libero \
  --model-dir <path-to-model> --precision int8 --port 8000
```

```python
policy = build_robot_policy("franka_libero", "<path-to-model>", precision="int8")
```


## LIBERO evaluation

### Get the checkpoint

The published accuracy is `pi05_libero_base`, π0.5 fine-tuned on LIBERO — an
arbitrary π0.5 checkpoint might not reproduce it.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download lerobot/pi05_libero_base --local-dir <path-to-model>
curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  -o <path-to-model>/norm_stats.json
```

The `lerobot/pi05_libero_base` checkpoint lost its normalization statistics
during repository updates. To reproduce the officially reported performance,
download OpenPI's LIBERO `norm_stats.json` separately as shown above and pass it
explicitly with `--norm-stats`.

### Run

The rollout needs LIBERO and MuJoCo:

```bash
python -c 'from libero.libero import benchmark'
```

If that fails, install LIBERO from source:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git <path-to-libero>
pip install -r <path-to-libero>/requirements.txt   # robosuite brings MuJoCo; pins numpy==1.22.4
pip install -e <path-to-libero>
export MUJOCO_GL=egl                               # headless; osmesa if the machine has no EGL
```

That numpy pin is the one thing to watch: it predates Python 3.11, so on a newer
interpreter it has no wheel and builds from source. `--no-deps` skips it.

`--backend websocket` additionally needs `openpi-client`, from an openpi
checkout:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git <path-to-openpi>
pip install -e <path-to-openpi>/packages/openpi-client
```

`apxinf-robo eval-libero` builds the policy in-process — no server involved:

```bash
apxinf-robo eval-libero --backend in-process --model-dir <path-to-model> \
  --norm-stats <path-to-model>/norm_stats.json \
  --precision bf16 --action-horizon 10 \
  --suite libero_10 --tasks all --trials-per-task 50 \
  --results-jsonl <out-dir>/results.jsonl --summary-json <out-dir>/summary.json
```

That is the published protocol: all 10 LIBERO-10 tasks x 50 episodes at seed 7
(the default), 500 episodes in total.

### Options

- `--suite` picks the task suite, `--tasks` a comma list within it, and
  `--trials-per-task` the episode count; a smoke run is
  `--tasks 0 --trials-per-task 1`.
- The model flags — `--model-type`, `--norm-stats`, `--action-horizon`, `--action-dim`,
  `--discrete-state`, FP8 `--calibration` — belong to `--backend in-process`
  alone.
- `--backend websocket --host <h> --port <p>` evaluates a running
  [server](#openpi-compatible-serving) instead, on this machine or another. The
  model flags belong to the server there, and `--precision` only asserts what
  the server reports, so a mismatch fails at connect instead of skewing a run.
  Pass `--norm-stats <path-to-model>/norm_stats.json` to `apxinf-robo serve`
  when serving this checkpoint.
- Runs are resumable: completed task/trial rows in the JSONL ledger are skipped,
  and the summary reports success rate alongside per-segment latency.


## Benchmark

`scripts/bench_pi05.py` times the concentric serving shells so a
regression can be attributed to the engine, the processors, or the transport.

```bash
python scripts/bench_pi05.py --model-dir <path-to-model> --precision bf16 --layer l1,l2
```

- `--layer` selects any subset of `l1` (bare model), `l2` (engine policy), `l3`
  (websocket round trip). L3 attaches to a running server and needs no local
  weights.
- `--model-dir` runs a real checkpoint at its native horizon; `--random-weights`
  runs the engine with no checkpoint on disk, and the shape knobs (`--views`,
  `--image-size`, `--action-horizon`, `--num-flow-steps`, `--token-count`)
  select the synthetic workload.
- `--calibration` is FP8-only and synthetic-only; a checkpoint reads
  `calibration.json` from its own directory.
- `--action-horizon` also applies to a checkpoint — the horizon is a sequence
  length, not a weight dimension — which is what makes a real checkpoint
  comparable to a synthetic run.
- `--warmup` / `--samples` set the sampling protocol (default 10 and 30);
  `--out` writes the report as JSON.

Any registered model type works: `AutoPolicy` dispatches on the checkpoint's
`config.json`, so the same command benchmarks the next model without a flag
change.

`l2` is the engine policy, not `build_robot_policy`, so a preset's own pre/post
steps are outside the measurement. `franka_libero` adds none — it only supplies
constructor arguments — but `unitree_g1` wires real arithmetic (state
discretization into the prompt, 32→16 action encode) that `l2` does not time.
Benchmark that through `l3` against a server started under the preset, or through
`eval-libero`'s per-segment latency, until this script grows a `--robot` path.

For one-step evaluation, please refer to [run warmstart with onestep](https://github.com/infinigence/ApxInf/blob/main/doc/run_warmstart_with_onestep.md)


## NVIDIA build environment

A complete CUDA toolkit is required: `nvcc`, CUDA headers and runtime, cuBLAS
and cuBLASLt development libraries, and NVTX (`libnvToolsExt` on Jetson,
`libnvtx3interop` on desktop CUDA). Also a C/C++ compiler, linker, `ar`, Git,
`pkg-config`, and Python 3. The CUDA kernels, CUTLASS, and FlashAttention
sources are vendored — no external checkout needed.

Install the driver and toolkit through the JetPack, DRIVE OS, or CUDA
distribution for the machine, then check `nvcc --version`. If CUDA does not live
at `/usr/local/cuda`, point `CUDA_PATH` at it.

| Device | Architecture | Validated toolkit |
|---|---:|---:|
| Jetson AGX Thor | `sm_110` | CUDA 13.0 |
| Thor-U | `sm_101` | CUDA 12.8 |
| Jetson AGX Orin | `sm_87` | CUDA 12.6, 13.2 |
| RTX 4090 | `sm_89` | CUDA 12.8 |


## Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
```

Built with Rust 1.95 and 1.96; no minimum supported version is declared.


## The engine submodule

`apxinf/` is the ApxInf inference engine, pinned by SHA.

- A fresh clone does not carry it. Initialize it explicitly with
  `git submodule update --init --recursive`, then install it per
  [Build APXinf-robo](#build-apxinf-robo).
- The pin is a SHA, never a branch. Bumping it is a reviewed change and must
  keep `tests/test_parity.py` green.
- To co-develop against an unpushed engine change, point the submodule at a
  local checkout for the duration and put it back before committing:

  ```bash
  git -C apxinf remote add local /path/to/ApxInf && git -C apxinf fetch local
  ```


## License

Apache 2.0. Vendored third-party components retain their own licenses.


## Community

Scan the QR Code to join our Wechat Group

<div align="left">
  <img src="https://media.githubusercontent.com/media/apxinf/apxinf.brand/refs/heads/main/wechat.jpg" alt="wechat-group" width="256"/>
</div>

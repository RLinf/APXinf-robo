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

These commands report P50 after 10 warm-up iterations. `--samples` selects the
measurement count (100 in the BF16 example, 30 by default).
Synthetic inputs check runtime and latency; use [LIBERO evaluation](#libero-evaluation)
to measure task success.

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

The workflow follows these guides to implement and validate the model port:

- [Port a model](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/porting-workflow.md)
- [Add a model implementation](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/adding-a-new-model.md)
- [Model API reference](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/model-layer-architecture.md)
- [Add a CUDA kernel](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/adding-new-kernels.md)

## Build APXinf-robo

```bash
git clone --recursive https://github.com/RLinf/APXinf-robo.git
cd APXinf-robo
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
pip install -e "./apxinf/python/apxinf[serving]" --config-settings editable_mode=strict
pip install -e ".[libero,serve]"
```

Install on Linux with an NVIDIA driver, CUDA toolkit, Rust, and `cmake`.
Use `--features cuda` when building the binding. The `serving` extra installs
msgpack and websockets; omit it for in-process use. Strict editable installation
picks up edits to existing engine Python files without reinstalling. Reinstall
after adding modules or changing dependencies, and keep its generated
`apxinf/python/apxinf/build/__editable__.*` directory.

The build queries the visible GPU for its compute capability and compiles the
kernels for exactly that architecture, so build on the machine you deploy to;
cross-compiling fails unless `APXINF_CUDA_ARCH` names the target (`sm_87` Orin,
`sm_101` Thor-U, `sm_110` Thor).

Check the compiled extension, the Robo package, and GPU inference:

```bash
python -c 'import apxinf_py; print(apxinf_py.__version__)'
python -c 'import apxinf_robo; print(apxinf_robo.__version__)'
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 --samples 5
```

Both imports must succeed and the benchmark must complete with latency results.
If `nvcc --version`, `cargo --version`, or `cmake --version` fails, install:

- [NVIDIA build environment](#nvidia-build-environment)
- [Rust toolchain](#rust-toolchain)
- `cmake` (`apt install cmake`)


## Using APXinf-robo from Python

Use `build_robot_policy` for a named robot. Choose another entry point when your
application supplies its own processing or needs a network service.

| Entry point | Input | Output |
|---|---|---|
| `build_robot_policy` | Robot preset, checkpoint, raw observations | Robot actions |
| `load_policy` | Checkpoint, input field names, raw observations | Policy actions |
| `load_bare_model` | Preprocessed images, tokens, noise | Normalized model actions |
| `websocket_server` | A policy and listening address | OpenPI-compatible service |

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

Pass raw images, state, and a prompt to `build_robot_policy`. The selected
preset supplies input field names, state processing, and output action width.

```python
from apxinf_robo import build_robot_policy

policy = build_robot_policy(
    "franka_libero", "<path-to-model>",
    precision="bf16",
)

policy.metadata             # robot, robot_slots, image_keys, state_key, action_horizon, ...

result = policy.infer(observation)
result["normalized_actions"]  # what L1 returned, before trim + unnormalize
```

Each preset field can be overridden per call (`image_keys`, `state_key`,
`prompt_key`, `action_dim`, `discrete_state`) to match your client. To run a checkpoint without a robot preset, use
`load_policy`:

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

Select the preset matching your checkpoint and client with `--robot`. Match the
client's camera fields, state encoding, and action layout to that preset.

| Preset | Cameras | State | Action |
|---|---|---|---|
| `franka_libero` | `observation/image`, `observation/wrist_image` | `observation/state`; encoding is checkpoint-specific | 7-dim EEF delta |
| `unitree_g1` | 3 views | 16-dim, discretized into the prompt | delta joints, 32→16 encode |

`--help` lists every preset. If an installed client uses different keys, pass the
input fields through `--image-keys` / `--state-key`, or register a preset. See
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

For LIBERO, first capture observations using the robot preset. This step needs
[LIBERO and MuJoCo](#run), but no model checkpoint:

```bash
apxinf-robo capture-libero --robot franka_libero --suite libero_10 \
  --output-dir devlocal/fp8-calibration/observations
```

The output directory contains one NPZ file per observation. On the calibration
machine, read those files with the same input fields:

```bash
python scripts/calibrate_pi05.py --model-dir <path-to-model> \
  --image-key observation/image --image-key observation/wrist_image \
  --state-key observation/state --prompt-key prompt \
  --input-dir devlocal/fp8-calibration/observations \
  --output <path-to-model>/calibration.json
```

For another preset, use the field arguments printed by `capture-libero`.
Keep the NPZ directory to reuse the same inputs in later calibration runs.
Successful calibration prints the sample count and output profile path. Pass
that profile to `serve --precision fp8 --calibration <path-to-model>/calibration.json`.

Alternatively, run LIBERO directly on the calibration machine:

```bash
python scripts/calibrate_pi05.py --model-dir <path-to-model> \
  --libero-suite libero_10 --output <path-to-model>/calibration.json
```

This uses the calibrator's default input fields. Override them with `--image-key`,
`--state-key`, and `--prompt-key` when needed. For NPZ and manifest formats, see
[PI0.5 FP8 calibration](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/pi05-fp8-calibration.md).

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
pip install -U huggingface_hub
hf download lerobot/pi05_libero_base --local-dir <path-to-model>
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

LIBERO pins NumPy 1.22.4, which is incompatible with Python 3.12. Use a
compatible simulator environment; `--no-deps` does not skip packages explicitly
listed in `requirements.txt`.

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
  model flags belong to the server. At connection, the evaluator checks
  `--precision` and any published image, state, and prompt field names against
  `franka_libero`. A mismatch stops the run; unpublished fields are not checked.
  Pass `--norm-stats <path-to-model>/norm_stats.json` to `apxinf-robo serve`
  when serving this checkpoint.
- Runs are resumable: completed task/trial rows in the JSONL ledger are skipped,
  and the summary reports success rate alongside per-segment latency.


## Benchmark

Measure latency with a checkpoint:

```bash
python scripts/bench_pi05.py --model-dir <path-to-model> --precision bf16 \
  --layer l1,l2 --warmup 10 --samples 30 --out devlocal/benchmark/results.json
```

| Layer | Measurement |
|---|---|
| `l1` | Model inference from resized RGB, including the Python/Rust call |
| `l2` | Engine policy, including its preprocessing and postprocessing |
| `l3` | Client-to-server round trip, including the served policy and transport |

L2 excludes robot-specific processing added by `build_robot_policy`. To include
that processing, start a server with the required robot preset and measure L3:

```bash
python scripts/bench_pi05.py --layer l3 --precision bf16 \
  --host 127.0.0.1 --port 8000 --out devlocal/benchmark/server.json
```

- `--random-weights` runs without a checkpoint. Use `--views`, `--image-size`,
  `--action-horizon`, `--num-flow-steps`, and `--token-count` to set its workload.
  Synthetic L2 uses a fixed-length tokenizer and identity normalization.
- `--action-horizon` also overrides a checkpoint's native horizon.
- `--calibration` applies to synthetic FP8 runs; checkpoint runs read
  `calibration.json` from the checkpoint directory.
- `--warmup` and `--samples` default to 10 and 30. The command prints latency
  statistics; `--out` also saves JSON. Compare runs with matching layers,
  precision, input shapes, and sampling settings.

For LIBERO task success and per-segment latency, use [LIBERO evaluation](#libero-evaluation).

For one-step evaluation, see [Run warm-start with one-step](https://github.com/infinigence/ApxInf/blob/ba968f63c9820f7db368bea0ce17bb890aa90781/doc/run_warmstart_with_onestep.md).

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

Initialize or restore the engine version recorded by this checkout:

```bash
git submodule update --init --recursive
```

Record the Robo and engine revisions alongside benchmark or evaluation results:

```bash
git rev-parse HEAD
git -C apxinf rev-parse HEAD
git submodule status apxinf
```

If the output starts with `+`, the engine is checked out at a different commit
from the one recorded by this Robo checkout.
Use `git submodule update --remote` only for a deliberate engine upgrade; it
follows `main`, as configured in `.gitmodules`. Before submitting
an upgrade, review the gitlink diff and run `tests/test_parity.py` with
`APXINF_PARITY_CHECKPOINT` set to a compatible checkpoint.

## License

Apache 2.0. Vendored third-party components retain their own licenses.


## Community

Scan the QR Code to join our Wechat Group

<div align="left">
  <img src="https://media.githubusercontent.com/media/apxinf/apxinf.brand/refs/heads/main/wechat.jpg" alt="wechat-group" width="256"/>
</div>

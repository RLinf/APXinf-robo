# Qwen-Drive planning

Install APXinf-robo with its `drive` dependencies and the pinned ApxInf CUDA
binding. Keep `planner-sft` under the released model directory. The optional
real-scene example accepts a scene fixture directory containing `scenes.json`,
referenced frame arrays, and `initial-noise.npy`. Robo loads the policy through
`load_policy` and returns a `(50, 3)` trajectory.
For the Thor performance configuration, build the binding with the SM110 AOT
operator bundle as described in the [ApxInf build instructions](../apxinf/doc/qwen-drive-benchmark.md#build-and-load).

## Performance

Direct planning, batch 1, ten flow steps, twelve input frames. Request P50
includes decoded-image preprocessing and the host trajectory.

| Hardware | Precision | Latency | NAVSIM PDM (242 scenes) |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 482.60 ms | 85.6786 |

The shared benchmark constructs three cameras with four frames each, 16 history
states, and deterministic noise. It measures resident decoded RGB through the
policy and the returned `[50, 3]` host trajectory. No recorded input files are
required. The historical benchmark image sizes are explicit in the workload;
these are not an accuracy dataset or the official default-resolution protocol.

```sh
python scripts/bench_qwen_drive.py \
  --model-dir /models/Qwen-Drive-1.0-4B --precision bf16 \
  --warmup 10 --samples 30 --seed 0 \
  --out devlocal/qwen-drive-eval/latency.json
```

The same command works in ApxInf. Robo delegates input construction and timing
to the pinned engine while loading the policy through `apxinf_robo.load_policy`.
Without `--tactics`, the engine selects a compatible hardware/toolkit database
under `configs/tuning` when available; otherwise it uses provider defaults.
For controlled comparisons, supply `--tactics` and retain the database identity
and hash in the run evidence.
Lock CPU/GPU/EMC clocks and fan, exclude other compute jobs, and repeat the run
with a second output path. Report the median of all retained samples.

The table retains the previously published recorded-input result. Use the
command above for new measurements.

## Accuracy evaluation

Use real NAVSIM scene records with observed 10 Hz history and the official
metric caches. Never score the benchmark's constructed observations.
The scene JSONL uses Qwen-Drive's `messages`, `trajectory`, `meta_info` schema;
`hist_traj_10hz`, `hist_vel_10hz`, and `hist_acc_10hz` must each contain 16 states.
This evaluator does not interpolate missing histories. Scene provenance must
establish that the supplied states were observed, not interpolated upstream.

```sh
python scripts/eval_qwen_drive.py \
  --model-dir /models/Qwen-Drive-1.0-4B \
  --scenes /data/navsim/navtest-observed-history.jsonl \
  --image-root /data/navsim/images --seed 42 \
  --metric-cache /data/navsim/metric-cache/metadata/cache.csv --maps /data/nuplan/maps \
  --results-jsonl devlocal/qwen-drive-eval/predictions.jsonl \
  --summary-json devlocal/qwen-drive-eval/summary.json
```

If some scene images live in a separate directory with the same relative
paths, pass `--image-overlay /path/to/additional/images`. The evaluator checks
the primary image root first and then the overlay.

Prediction and scoring may use separate environments. Omit `--metric-cache`
from the prediction command, then score its existing output in the NAVSIM
Python environment without loading CUDA or model weights:

```sh
python scripts/eval_qwen_drive.py --score-only \
  --scenes /data/navsim/navtest-observed-history.jsonl \
  --results-jsonl devlocal/qwen-drive-eval/predictions.jsonl \
  --metric-cache /data/navsim/metric-cache/metadata/cache.csv --maps /data/nuplan/maps \
  --summary-json devlocal/qwen-drive-eval/summary.json
```

Install the official NAVSIM/nuPlan evaluator and configure its maps before
scoring. CUDA PyTorch supplies the same seeded initial noise as the reference.
The checkpoint, scene/image profile, raw-history construction and metric-cache
versions must be pinned together. Official Qwen scene-generation details are
not fully published, so this command alone does not establish byte-identical
reproduction of their published PDM score. The prior 242-scene interpolated
history score of 85.6786 is historical, not an acceptance target for corrected
observed-history inputs.

For serving, see [the service example](../examples/README.md#qwen-drive-planning).

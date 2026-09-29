# Qwen-Drive planning

Install APXinf-robo with its `drive` dependencies and the pinned ApxInf CUDA
binding. Keep `planner-sft` under the released model directory. Prediction
also needs CUDA PyTorch in the selected Python environment for reference noise
generation, and Pillow 12.3.0 for the validated preprocessing contract. Torch
runs in a child process so its bundled CUDA/cuBLAS libraries do not preload
into native inference. Scoring can use a separate CPU environment.

## Performance

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
No explicit tactics are required for the accepted operator-default path. If
supplying `--tactics`, keep the database identity and hash in the run evidence.
Lock CPU/GPU/EMC clocks and fan, exclude other compute jobs, and repeat the run
with a second output path. Report the median of all retained samples.

Constructed-input Thor BF16 latency measured on 2026-09-29 is **487.60 ms P50 /
492.93 ms P95**, pooled over two runs of 30 requests. Both runs returned
identical trajectories. CPU/GPU/EMC were locked to 2.601/1.575/4.266 GHz and
fan PWM 255. The previous recorded-input latency was 482.60 ms.

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

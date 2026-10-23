# GR00T N1.7 on LIBERO

Install APXinf-robo with its LIBERO dependencies, the LIBERO simulator, the
ApxInf CUDA binding, and the matching Isaac-GR00T/Transformers processor
environment. Download the LIBERO-10 checkpoint and the Cosmos processor
resources (Cosmos weights are not needed):

```sh
pip install -U huggingface_hub
hf download nvidia/GR00T-N1.7-LIBERO \
  --include 'libero_10/*.json' --include 'libero_10/*.safetensors' \
  --local-dir /models/GR00T-N1.7-LIBERO
hf download nvidia/Cosmos-Reason2-2B --exclude '*.safetensors' \
  --local-dir /models/nvidia/Cosmos-Reason2-2B
python - <<'PY'
from apxinf import Gr00tPolicy

Gr00tPolicy.prepare_assets(
    "/models/GR00T-N1.7-LIBERO/libero_10",
    "/models/nvidia/Cosmos-Reason2-2B",
)
PY
```

Cosmos-Reason2-2B is gated on Hugging Face; accept its license and authenticate
before downloading. The asset preparation copies the processor resources into
the checkpoint; later benchmark and inference commands need only `--model-dir`.
See the [ApxInf loading guide](../apxinf/doc/gr00t-n1.7.md#loading) for the
required files and processor environment.

## Performance

Batch 1, best recorded model-core P50 from fixed processor tensors to returned
actions, following the [GR00T benchmark procedure](../apxinf/doc/gr00t-n1.7.md#fixed-input-benchmark):

| Hardware | Precision | 1-view P50 | 2-view P50 |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 51.834 ms | 54.216 ms |
| Jetson AGX Thor | FP8 | 32.557 ms | 35.436 ms |
| Jetson AGX Orin | BF16 | 75.778 ms | 84.864 ms |
| Jetson AGX Orin | W8A8 | 56.711 ms | 64.924 ms |

The benchmark constructs deterministic tensors in memory in the pinned engine.
It needs the prepared model directory only; processor assets are discovered
under `assets/cosmos`. It does not read images, saved tensors or a dataset.
The model-core timing boundary and 90/156-token one/two-view shapes match the
engine benchmark. Synthetic outputs are for latency, not task accuracy.

Build once and run each view count for the selected hardware/precision:

```sh
cargo build --manifest-path apxinf/Cargo.toml --release \
  -p apxinf-model --features cuda --example gr00t_bench
for views in 1 2; do
  python scripts/bench_gr00t.py \
    --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16 \
    --views "$views" --warmup 30 --samples 200 \
    --binary apxinf/target/release/examples/gr00t_bench \
    --tactics "devlocal/gr00t-eval/thor-bf16-${views}v-tactics.json" --autotune \
    --out "devlocal/gr00t-eval/thor-bf16-${views}v.json"
done
```

| Table row | Precision argument | Additional argument |
|---|---|---|
| Thor BF16 | `--precision bf16` | None |
| Thor FP8 | `--precision fp8` | `--calibration /path/to/matching-calibration.json` |
| Orin BF16 | `--precision bf16` | None |
| Orin W8A8 | `--precision int8` | None |

Use separate paths for each hardware, precision and view count. The first run
creates the tactic database; repeat with `--tactics` and omit `--autotune` to
verify reuse. Lock clocks/fan and exclude other GPU jobs. CUDA/cuBLAS and kernel
identities must match the database. Reports must say `cuda-graph`; retain the
raw samples and artifact hashes. See the engine's
[benchmark contract](../apxinf/doc/gr00t-n1.7.md#fixed-input-benchmark).

The table retains the previously published results. Use the command above for
new measurements. See the engine document for timing boundaries.

## Accuracy evaluation

LIBERO-10, two views, ten episodes per task:

| Hardware | Precision | Episodes | Successes | Success rate |
|---|---|---:|---:|---:|
| Jetson AGX Thor | BF16 | 100 | 94 | 94.0% |
| Jetson AGX Thor | FP8 | 100 | 92 | 92.0% |
| Jetson AGX Orin | BF16 | 100 | 93 | 93.0% |
| Jetson AGX Orin | W8A8 | 100 | 93 | 93.0% |

Run the full suite with Robo's GR00T state and action conversion:

```sh
apxinf-robo eval-libero --backend in-process \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16 \
  --suite libero_10 --trials-per-task 10 --seed 7 \
  --max-steps 720 --replan-steps 8 \
  --results-jsonl devlocal/gr00t-eval/full-results.jsonl \
  --summary-json devlocal/gr00t-eval/full-summary.json
```

For a service deployment, use `apxinf-robo serve --robot franka_libero
--model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16`, then select
`--backend websocket` in the evaluator. See the [observation and gripper
contract](../examples/README.md#gr00t-n17-on-libero).

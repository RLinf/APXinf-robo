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

Run the same pinned model-core CUDA Graph benchmark from the Robo checkout.
The script generates a deterministic two-view LIBERO observation, processes it
once with the checkpoint's official NVIDIA processor, and keeps preprocessing
outside the timed region. No input file or separate backbone path is needed:

```sh
python scripts/bench_gr00t.py \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16 \
  --warmup 10 --iterations 50 \
  --output devlocal/gr00t-eval/latency.json
```

The report measures preprocessed host tensors through model-core action D2H,
including steady-state CUDA Graph replay. The generated observation is synthetic,
so its output is for latency testing, not task accuracy or numerical parity with
the historical fixed-input run. On Thor, its two-view image grid, 156-token
prompt, state shape, and output shape matched the original two-view input; the
new and original inputs measured 58.69 ms and 58.38 ms P50 respectively with
the same binary and 10 warmups plus 50 samples. The published table used a
different recorded build and conditions, and also includes a one-view input. If
`apxinf/target/release/examples/gr00t_bench` already exists, the script uses it
without rebuilding; `--binary /path/to/gr00t_bench` selects another existing
build. Otherwise Cargo builds the pinned runner. FP8 also needs a matching
`--calibration` profile.

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

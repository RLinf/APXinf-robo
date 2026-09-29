# GR00T N1.7 on LIBERO

Install APXinf-robo with its LIBERO dependencies, the LIBERO simulator, the
ApxInf CUDA binding, and
the matching Isaac-GR00T/Transformers processor environment. Prepare the
checkpoint's local Cosmos processor resources as described in the
[ApxInf loading guide](../apxinf/doc/gr00t-n1.7.md#loading).

## Performance

Batch 1, best recorded model-core P50 from fixed processor tensors to returned
actions, following the [GR00T benchmark procedure](../apxinf/doc/gr00t-n1.7.md#fixed-input-benchmark):

| Hardware | Precision | 1-view P50 | 2-view P50 |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 51.834 ms | 54.216 ms |
| Jetson AGX Thor | FP8 | 32.557 ms | 35.436 ms |
| Jetson AGX Orin | BF16 | 75.778 ms | 84.864 ms |
| Jetson AGX Orin | W8A8 | 56.711 ms | 64.924 ms |

Use the official processor fixture and the pinned model-core CUDA Graph runner
from the Robo checkout. Choose the one- or two-view fixture for the matching
table column:

```sh
python scripts/bench_gr00t.py \
  --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/GR00T-N1.7-LIBERO/libero_10/assets/cosmos \
  --fixture /data/gr00t/libero-two-view --precision bf16 \
  --warmup 10 --iterations 50 \
  --output devlocal/gr00t-eval/latency.json
```

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

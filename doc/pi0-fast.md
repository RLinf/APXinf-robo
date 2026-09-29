# PI0-FAST on LIBERO

Install APXinf-robo with its LIBERO dependencies, the LIBERO simulator, and the
ApxInf CUDA binding.
Download the PI0-FAST LIBERO checkpoint, including its normalization assets:

```sh
pip install -U huggingface_hub
hf download lerobot/pi0fast-libero-v044 --local-dir /models/pi0fast-libero-v044
```

Make both text and action tokenizers available at the checkpoint paths, in the local cache, or
through these environment variables:

```sh
export APXINF_PALIGEMMA_TOKENIZER=/models/paligemma-tokenizer
export APXINF_FAST_TOKENIZER=/models/fast-tokenizer
```

## Performance

Two views, 224×224 RGB, batch 1. Prefix and per-token latency is fitted from
fixed frames with different action-token counts, as in the [PI0-FAST benchmark](../apxinf/scripts/bench_pi0_fast.py):

| Hardware | Precision | Prefix | Per Token |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 33.1 ms | 17.16 ms |
| Jetson AGX Thor | FP8 | 31.0 ms | 9.68 ms |
| Jetson AGX Orin | BF16 | 117.3 ms | 25.34 ms |
| RTX 4090 | BF16 | 20.9 ms | 5.24 ms |

Capture LIBERO frames, then run the same L1 token-count fit from the Robo
checkout. The report gives `fixed_ms`, `per_step_ms`, and fit quality:

```sh
python scripts/collect_pi0_fast_frames.py \
  --suite libero_10 --trials-per-task 2 \
  --out devlocal/pi0fast-eval/libero-frames.npz
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --frames devlocal/pi0fast-eval/libero-frames.npz \
  --out devlocal/pi0fast-eval/latency.json
```

## Accuracy evaluation

Run all ten LIBERO-10 tasks with ten trials each. The evaluator builds the
PI0-FAST policy through Robo, preserves both finger joints in the observation,
and writes per-task success rates to the summary.

```sh
apxinf-robo eval-libero --backend in-process \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --suite libero_10 --trials-per-task 10 --seed 7 \
  --results-jsonl devlocal/pi0fast-eval/full-results.jsonl \
  --summary-json devlocal/pi0fast-eval/full-summary.json
```

For server evaluation, run `apxinf-robo serve --robot franka_libero --model-dir
/models/pi0fast-libero-v044 --precision bf16` and change the evaluator backend
to `websocket`, omitting `--model-dir`. See the [working observation contract](../examples/README.md#pi0-fast-on-libero).

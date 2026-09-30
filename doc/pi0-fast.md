# PI0-FAST on LIBERO

Install APXinf-robo with its LIBERO dependencies, the LIBERO simulator, and the
ApxInf CUDA binding.
Download the PI0-FAST LIBERO checkpoint, including its normalization assets:

```sh
pip install -U huggingface_hub
hf download lerobot/pi0fast-libero-v044 --local-dir /models/pi0fast-libero-v044
```

Keep both tokenizers inside the prepared checkpoint so the command needs only
`--model-dir`:

```sh
hf download google/paligemma-3b-pt-224 --include '*token*' 'special_tokens_map.json' \
  --local-dir /models/pi0fast-libero-v044/assets/paligemma-tokenizer
hf download jadechoghari/fast-libero-tokenizer-mean-std \
  --local-dir /models/pi0fast-libero-v044/assets/fast-tokenizer
```

Record downloaded revisions and calibration identity. See the engine's
[benchmark and evaluation contract](../apxinf/doc/pi0-fast-benchmark.md).

## Performance

Two views, 224×224 RGB, batch 1. Prefix and per-token latency is fitted from
the same constructed input with verified token stopping points, as in the [PI0-FAST benchmark](../apxinf/scripts/bench_pi0_fast.py):

| Hardware | Precision | Prefix | Per Token |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 33.1 ms | 17.16 ms |
| Jetson AGX Thor | FP8 | 31.0 ms | 9.68 ms |
| Jetson AGX Orin | BF16 | 117.3 ms | 25.34 ms |
| RTX 4090 | BF16 | 20.9 ms | 5.24 ms |

The table retains the previously published results. Use the command below for
new measurements on each hardware and precision.

The benchmark uses deterministic constructed camera images and state. It calls
the pinned engine benchmark through Robo's policy loader and requires no frame
archive. Both repositories accept the same command arguments:

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --state-key observation/state --layer l1 --mode all \
  --frames-count 1 --repeats 5 --warmup 10 --samples 30 \
  --tactics devlocal/pi0fast-eval/thor-bf16-tactics.json --autotune \
  --out devlocal/pi0fast-eval/latency.json
```

This covers Thor, Orin and RTX 4090 BF16. On Thor, repeat with `--precision fp8`
and `--calibration /path/to/matching-calibration.json` for the FP8 row. Use a
separate output and tactic path for each device and precision. Generate the
native GEMV/GEMM database with `--autotune` once; omit that flag and reuse
`--tactics` for subsequent measurements. FP8 calibration is a separate artifact.
The database's toolkit/device/kernel identity must match the tested binary.


The report retains latency samples and token counts. Prefix/per-token fitting
uses the same constructed observation at verified token stopping points; it
fails if fewer than two distinct decode lengths are available. Do not report
full-request latency as prefix latency. Lock clocks/fan, exclude other GPU work
and record the exact engine binary and calibration identity. Historical values
require new measurements before they are claimed for constructed inputs.

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

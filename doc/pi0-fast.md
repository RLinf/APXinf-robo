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
[benchmark contract](../apxinf/doc/pi0-fast-benchmark.md).

## Performance

Two views, 224×224 RGB, batch 1. The table retains the previously published
Prefix and Per Token values. The [ApxInf benchmark](../apxinf/scripts/bench_pi0_fast.py)
reports these fields from a fit over frames with distinct generated token counts:

| Hardware | Precision | Prefix | Per Token |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 33.1 ms | 17.16 ms |
| Jetson AGX Thor | FP8 | 31.0 ms | 9.68 ms |
| Jetson AGX Orin | BF16 | 117.3 ms | 25.34 ms |
| RTX 4090 | BF16 | 20.9 ms | 5.24 ms |

The Robo entry point calls the pinned ApxInf benchmark through Robo's policy
loader. By default, ApxInf constructs deterministic camera images and state, so
the following complete-request latency check requires no frame archive. Both
repositories accept the same arguments. PI0-FAST times its autoregressive native
call; its runtime does not implement captured CUDA Graph replay:

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --state-key observation/state --layer l1 --mode latency \
  --frames-count 10 --warmup 10 --samples 30 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json --autotune \
  --out devlocal/model-bench-inputs/pi0fast/latency.json
```

Run the same latency check on Thor, Orin and RTX 4090 BF16. On Thor, repeat with
`--precision fp8` and `--calibration /path/to/matching-calibration.json` for
FP8. Use a separate output and tactic path for each device and precision.
`--autotune` creates the native GEMV/GEMM database; omit that flag and reuse
`--tactics` for subsequent measurements. FP8 calibration is a separate
artifact. The database's toolkit/device/kernel identity must match the tested
binary. To check the original script's default tactic path, omit both tactic
flags. That baseline can be substantially slower and can generate different
tokens, so keep its results separate from the tuned run.

To obtain a Prefix / Per Token fit using the original measurement method,
supply recorded LIBERO frames with varying decode lengths:

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --state-key observation/state --layer l1 --mode ar \
  --frames /path/to/libero_frames.npz --frame-variant raw \
  --survey 20 --repeats 5 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json \
  --out devlocal/model-bench-inputs/pi0fast/ar.json
```

The optional `.npz` must contain camera, state and task arrays in the format
described by the [engine contract](../apxinf/doc/pi0-fast-benchmark.md). Robo
resolves `--frames` relative to the caller's directory before invoking ApxInf.
The fit is present only if at least two token lengths occur. The input-free
latency command does not reproduce the table's Prefix / Per Token values.
Lock clocks/fan, exclude other GPU work and record the exact engine binary,
tactics, calibration and input identities when comparing historical results.

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

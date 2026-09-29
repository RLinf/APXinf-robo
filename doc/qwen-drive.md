# Qwen-Drive planning

Install APXinf-robo with its `drive` dependency and the ApxInf CUDA binding.
Use the released Qwen-Drive checkpoint, its `planner-sft` directory, and the
scene fixture format (`scenes.json`, referenced frame arrays, and
`initial-noise.npy`) from the [ApxInf benchmark guide](../apxinf/doc/qwen-drive-benchmark.md).
Robo loads the policy through `load_policy` and returns a `(50, 3)` trajectory.

## Performance

Direct planning, batch 1, ten flow steps, twelve input frames. Request P50
includes decoded-image preprocessing and the host trajectory.

| Hardware | Precision | Latency | NAVSIM PDM (242 scenes) |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 482.60 ms | 85.6786 |

Measure request latency through Robo's example; its JSON reports P50 and P95:

```sh
python examples/qwen_drive_infer.py \
  --model-dir /models/Qwen-Drive-1.0-4B \
  --inputs /data/qwen-drive/public-inputs \
  --warmup 10 --samples 30 \
  --out devlocal/qwen-drive-eval/latency.json
```

For the pooled 60-request result, repeat with a second output path and take
the median of both reports' `samples_ms` arrays.

## Accuracy evaluation

The fixed NAVSIM subset has 242 scenes and a PDM score of 85.6786. Its
trajectories match the accepted padded implementation for 242/242 scenes.
Compare a scene's Robo trajectory with the reference array:

```sh
python examples/qwen_drive_infer.py \
  --model-dir /models/Qwen-Drive-1.0-4B \
  --inputs /data/qwen-drive/public-inputs \
  --scene-index 0 \
  --reference /data/qwen-drive/reference-direct/scene-0-repeat-0.npy \
  --save-actions devlocal/qwen-drive-eval/scene-0-actions.npy \
  --out devlocal/qwen-drive-eval/scene-0.json
```

The output includes `reference_max_abs` and `reference_relative_l2`. Use the
same command for each available scene and score the resulting trajectories
with NAVSIM to reproduce the aggregate PDM result. For the WebSocket entry
point, see the [service example](../examples/README.md#qwen-drive-planning).

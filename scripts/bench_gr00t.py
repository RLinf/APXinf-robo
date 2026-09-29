#!/usr/bin/env python3
"""Run the pinned GR00T CUDA Graph benchmark with generated processor input.

The public command needs only a prepared checkpoint. Raw synthetic LIBERO
observations are processed once before timing; ApxInf's model-core runner keeps
the existing host-tensor-to-action-D2H timing boundary and Graph check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_KEYS = ("observation/image", "observation/wrist_image")
PROMPT = "put the white mug on the left plate and put the yellow and white mug on the right plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--binary", type=Path, help="existing gr00t_bench binary")
    parser.add_argument("--output", type=Path, default=Path("devlocal/gr00t-eval/latency.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    return args


def _write_tensor(root: Path, name: str, values: np.ndarray, dtype: str) -> dict:
    values = np.ascontiguousarray(values)
    if dtype == "bfloat16":
        import torch

        values = (
            torch.from_numpy(values.astype(np.float32, copy=False))
            .to(torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .astype("<u2", copy=False)
        )
    elif dtype == "uint32":
        values = values.astype("<u4", copy=False)
    else:
        values = values.astype(np.uint8, copy=False)
    filename = f"{name}.bin"
    values.tofile(root / filename)
    return {"file": filename, "dtype": dtype, "shape": list(values.shape)}


def _generate_input(root: Path, model_dir: Path) -> Path:
    from apxinf.policies.impls._gr00t_assets import resolve_assets
    from apxinf.policies.impls.gr00t import _NvidiaProcessorAdapter

    backbone = resolve_assets(model_dir)
    adapter = _NvidiaProcessorAdapter.load(
        model_dir,
        backbone=backbone,
        embodiment="libero_sim",
        image_keys=IMAGE_KEYS,
        state_key="observation/state",
        prompt_key="prompt",
        action_key=None,
    )
    rng = np.random.default_rng(0)
    observation = {
        key: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for key in IMAGE_KEYS
    }
    observation["observation/state"] = np.zeros(adapter.state_dim, dtype=np.float32)
    observation["prompt"] = PROMPT
    encoded = adapter.encode(observation)

    config = json.loads((model_dir / "config.json").read_text())
    horizon = int(config.get("action_horizon", 40))
    width = int(config.get("max_action_dim", 132))
    noise = np.zeros((1, horizon, width), dtype=np.float32)
    tensors = {
        name: _write_tensor(root, name, values, dtype)
        for name, values, dtype in (
            ("pixel_values", encoded["pixel_values"], "bfloat16"),
            ("image_grid_thw", encoded["image_grid_thw"], "uint32"),
            ("token_ids", encoded["token_ids"], "uint32"),
            ("attention_mask", encoded["attention_mask"], "uint8"),
            ("state", encoded["state"], "bfloat16"),
            ("noise", noise, "bfloat16"),
        )
    }
    manifest = {
        "schema": "apxinf.gr00t-n1.7.preprocessed-fixture.v1",
        "fixture": "robo-synthetic-libero-two-view-zero-noise-v1",
        "embodiment_id": encoded["embodiment_id"],
        "tensors": tensors,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return backbone


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "apxinf/scripts/bench_gr00t.py"
    if not runner.is_file():
        raise SystemExit("ApxInf submodule is missing; run git submodule update --init")
    binary = args.binary or ROOT / "apxinf/target/release/examples/gr00t_bench"
    with TemporaryDirectory(prefix="gr00t-input-", dir=output.parent) as directory:
        input_dir = Path(directory)
        backbone = _generate_input(input_dir, model_dir)
        command = [
            sys.executable, str(runner),
            "--checkpoint", str(model_dir),
            "--backbone", str(backbone),
            "--fixture", str(input_dir),
            "--precision", args.precision,
            "--device", str(args.device),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--output", str(output),
        ]
        for flag, value in (("--calibration", args.calibration), ("--tactics", args.tactics)):
            if value is not None:
                command.extend((flag, str(value.resolve())))
        if args.autotune:
            command.append("--autotune")
        if binary.is_file():
            command.extend(("--binary", str(binary.resolve())))
        elif args.binary is not None:
            raise SystemExit(f"benchmark binary does not exist: {binary}")
        completed = subprocess.run(
            command, cwd=ROOT / "apxinf", capture_output=True, text=True
        )
        if completed.returncode:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            completed.check_returncode()
    report = json.loads(output.read_text())
    print(json.dumps({
        "execution": report["execution"],
        "precision": report["precision"],
        "input": report["input"],
        "p50_ms": report["latency_ms"]["p50"],
        "p95_ms": report["latency_ms"]["p95"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one Qwen-Drive scene through Robo's model-agnostic L2 loader.

The input directory follows ApxInf's Qwen-Drive benchmark fixture format:
``scenes.json``, referenced image ``.npy`` files, and ``initial-noise.npy``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _common  # noqa: F401 - adds checkout source directories to sys.path
from apxinf_robo import load_policy


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--planner", type=Path)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--reference", type=Path, help="reference trajectory .npy for output comparison")
    parser.add_argument("--save-actions", type=Path, help="write the predicted trajectory .npy")
    parser.add_argument("--out", type=Path, help="write measured latency and comparison JSON")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    return args


def main() -> None:
    args = parse_args()

    scene = json.loads((args.inputs / "scenes.json").read_text())[args.scene_index]
    observation = dict(scene)
    observation["views"] = {
        camera: [
            {"image": np.load(args.inputs / frame["image"]),
             "target_size": frame["target_size"]}
            for frame in frames
        ]
        for camera, frames in scene["views"].items()
    }
    noise = np.load(args.inputs / "initial-noise.npy")
    policy = load_policy(
        args.model_dir,
        precision="bf16",
        mode="direct_planning",
        planner=args.planner or args.model_dir / "planner-sft",
    )
    try:
        for _ in range(args.warmup):
            policy.infer(observation, noise=noise)
        durations = []
        first_actions = None
        for _ in range(args.samples):
            start = time.perf_counter()
            result = policy.infer(observation, noise=noise)
            durations.append((time.perf_counter() - start) * 1000)
            current_actions = np.asarray(result["actions"])
            if first_actions is None:
                first_actions = current_actions.copy()
            elif not np.array_equal(first_actions, current_actions):
                raise ValueError("fixed Qwen-Drive input produced different trajectories")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (50, 3) or not np.isfinite(actions).all():
            raise ValueError(f"expected finite trajectory (50, 3), got {actions.shape}")
        report = {
            "scene_index": args.scene_index,
            "trajectory_shape": list(actions.shape),
            "request_p50_ms": float(np.median(durations)),
            "request_p95_ms": float(np.percentile(durations, 95)),
            "samples_ms": durations,
            "samples": args.samples,
            "warmup": args.warmup,
        }
        if args.reference is not None:
            reference = np.load(args.reference)
            if reference.shape == (1, *actions.shape):
                reference = reference[0]
            if reference.shape != actions.shape or not np.isfinite(reference).all():
                raise ValueError(f"reference must be finite (50, 3), got {reference.shape}")
            difference = actions.astype(np.float64) - reference
            report["reference_max_abs"] = float(np.max(np.abs(difference)))
            report["reference_relative_l2"] = float(
                np.linalg.norm(difference) / np.linalg.norm(reference)
            )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        if args.save_actions is not None:
            args.save_actions.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_actions, actions)
        print(json.dumps(report))
    finally:
        policy.close()


if __name__ == "__main__":
    main()

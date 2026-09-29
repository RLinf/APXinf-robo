#!/usr/bin/env python3
"""Benchmark fixed two-view LIBERO policy requests through Robo's L2 loader.

Capture one simulator observation before timing; no simulator work is timed.
These are full policy requests, not engine prefix/token or core-only timings.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

from apxinf_robo import load_policy
from apxinf_robo.envs.libero import (
    libero_gr00t_state,
    libero_images,
    libero_state,
    make_env,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="bf16")
    parser.add_argument("--calibration", type=pathlib.Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    from apxinf.policies.auto import _read_model_type

    model_type = _read_model_type(args.model_dir)
    state_converter = {
        "pi0_fast": lambda raw: libero_state(raw, finger_joints=2),
        "pi0fast": lambda raw: libero_state(raw, finger_joints=2),
        "gr00t": libero_gr00t_state,
        "gr00tn1d7": libero_gr00t_state,
        "Gr00tN1d7": libero_gr00t_state,
    }.get(model_type)
    if state_converter is None:
        raise ValueError(f"expected a PI0-FAST or GR00T checkpoint, got {model_type!r}")

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task {args.task_id} outside {args.suite} (0..{suite.n_tasks - 1})")
    task = suite.get_task(args.task_id)
    env = make_env(task, args.seed)
    try:
        env.reset()
        raw = env.set_init_state(suite.get_task_init_states(args.task_id)[0])
        frames = libero_images(raw["agentview_image"], raw["robot0_eye_in_hand_image"])
    finally:
        env.close()

    image_keys = ("observation/image", "observation/wrist_image")
    observation = {
        key: frame for key, frame in zip(image_keys, frames)
    }
    observation["observation/state"] = state_converter(raw)
    observation["prompt"] = str(task.language)

    options = {
        "precision": args.precision,
        "image_keys": image_keys,
        "state_key": "observation/state",
        "prompt_key": "prompt",
    }
    if args.calibration is not None:
        options["calibration"] = args.calibration
    policy = load_policy(args.model_dir, **options)
    try:
        def infer():
            actions = np.asarray(policy.infer(observation)["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise ValueError(f"expected finite LIBERO actions (H, 7), got {actions.shape}")
            return actions

        for _ in range(args.warmup):
            infer()
        durations = []
        for _ in range(args.samples):
            start = time.perf_counter()
            actions = infer()
            durations.append((time.perf_counter() - start) * 1000.0)
        report = {
            "model_type": model_type,
            "precision": args.precision,
            "suite": args.suite,
            "task_id": args.task_id,
            "action_shape": list(actions.shape),
            "request_p50_ms": float(statistics.median(durations)),
            "request_p95_ms": float(np.percentile(durations, 95)),
            "warmup": args.warmup,
            "samples": args.samples,
        }
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        policy.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture LIBERO frames in the pinned PI0-FAST benchmark's input format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apxinf_robo.envs.libero import libero_images, libero_state, make_env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--trials-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.trials_per_task < 1:
        parser.error("--trials-per-task must be positive")
    return args


def main() -> None:
    args = parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    images, wrists, states, prompts = [], [], [], []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        try:
            for trial_id in range(min(args.trials_per_task, len(initial_states))):
                env.reset()
                raw = env.set_init_state(initial_states[trial_id])
                for _ in range(10):
                    raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
                base, wrist = libero_images(
                    raw["agentview_image"], raw["robot0_eye_in_hand_image"]
                )
                images.append(base)
                wrists.append(wrist)
                states.append(libero_state(raw, finger_joints=2))
                prompts.append(str(task.language))
        finally:
            env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        base_flipped=np.stack(images),
        wrist_flipped=np.stack(wrists),
        state=np.stack(states),
        task=np.asarray(prompts),
    )
    print(f"wrote {len(images)} LIBERO frames to {args.out}")


if __name__ == "__main__":
    main()

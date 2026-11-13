#!/usr/bin/env python3
"""Save one native LIBERO observation in the S600 Pi05 demo's local format.

Example:
    MUJOCO_GL=egl /home/wwxq/miniconda3/envs/libero_sim/bin/python \
        scripts/drobotics_pi05_libero_fixture.py \
        --output-dir devlocal/s600-pi05-stage3/fixture

The demo reads two RGB JPEGs, an fp64 state.bin, and prompt.txt. The original
simulator fields are also saved in observation.npz for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f"task-id must be in 0..{suite.n_tasks - 1}")
    task = suite.get_task(args.task_id)
    states_path = (Path(get_libero_path("init_states")) /
                   task.problem_folder / task.init_states_file)
    initial_states = torch.load(states_path, weights_only=False)
    if not 0 <= args.initial_state_id < len(initial_states):
        parser.error(f"initial-state-id must be in 0..{len(initial_states) - 1}")
    bddl_path = (Path(get_libero_path("bddl_files")) /
                 task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(initial_states[args.initial_state_id])
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        raw = {
            key: np.asarray(observation[key]).copy()
            for key in (
                "agentview_image", "robot0_eye_in_hand_image",
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            )
        }
    finally:
        env.close()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = str(task.language)
    state = np.concatenate((
        np.asarray(raw["robot0_eef_pos"], dtype=np.float64).reshape(-1),
        axis_angle(raw["robot0_eef_quat"]),
        np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64).reshape(-1),
    ))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite state[8], got {state}")
    state.astype("<f8").tofile(output_dir / "state.bin")
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    np.savez_compressed(output_dir / "observation.npz", **raw)

    images = []
    for index, key in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
        source = raw[key]
        if source.shape != (256, 256, 3) or source.dtype != np.uint8:
            raise ValueError(f"unexpected {key}: {source.shape} {source.dtype}")
        # Match the existing LIBERO policy client's camera orientation.
        rgb = np.ascontiguousarray(source[::-1, ::-1])
        image_path = output_dir / f"image_{index}.jpg"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"failed to save {image_path}")
        images.append({
            "source_key": key, "source_shape": list(source.shape),
            "source_dtype": str(source.dtype), "saved_shape": list(rgb.shape),
            "saved_dtype": str(rgb.dtype), "orientation": "flip height and width",
            "file": image_path.name,
        })

    files = [output_dir / name for name in
             ("image_0.jpg", "image_1.jpg", "state.bin", "prompt.txt", "observation.npz")]
    manifest = {
        "suite": args.suite, "task_id": args.task_id,
        "initial_state_id": args.initial_state_id, "seed": args.seed,
        "settle_steps": args.settle_steps, "task_language": prompt,
        "bddl_file": str(bddl_path), "initial_states_file": str(states_path),
        "images": images, "state_shape": [8], "state_dtype": "float64 little-endian",
        "state_components": "eef_pos_xyz, eef_quat_xyzw_to_axis_angle_xyz, gripper_qpos_2",
        "state_finite": bool(np.isfinite(state).all()),
        "state_values": state.tolist(),
        "file_sha256": {path.name: sha256(path) for path in files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

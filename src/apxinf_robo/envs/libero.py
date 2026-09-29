"""Native LIBERO simulator observations, translated for an ApxInf policy.

This is the **in-process** path: the caller holds a live ``Policy`` and feeds it
frames directly. For the out-of-process path -- a simulator host with no ApxInf
installed, talking to a WebSocket server -- copy ``scripts/connect_libero.py``
instead; it is deliberately standalone and shares no code with this module.

``libero_state`` keeps the historical seven-value state by default. Checkpoints
trained with both finger joints (including PI0-FAST) request eight values with
``finger_joints=2``. GR00T uses a named state and a separate decoded-gripper
conversion, matching its official LIBERO processor contract.
"""

from __future__ import annotations

import math
import pathlib
from typing import Tuple

import numpy as np

__all__ = [
    "quat_to_axis_angle",
    "libero_images",
    "libero_state",
    "libero_gr00t_state",
    "libero_gr00t_action",
    "make_env",
    "to_apxinf_observation",
]


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def libero_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Orient raw LIBERO frames; the selected policy owns model-specific resize."""
    return np.stack(
        [np.ascontiguousarray(base[::-1, ::-1]), np.ascontiguousarray(wrist[::-1, ::-1])]
    )


def libero_state(observation, *, finger_joints: int = 1) -> np.ndarray:
    """Return a seven- or eight-value state for the selected checkpoint."""
    gripper = np.asarray(observation["robot0_gripper_qpos"]).reshape(-1)
    if gripper.size != 2:
        raise ValueError(f"robot0_gripper_qpos must have 2 values, got {gripper.size}")
    if finger_joints not in (1, 2):
        raise ValueError(f"finger_joints must be 1 or 2, got {finger_joints}")
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            gripper[:finger_joints],
        )
    ).astype(np.float32, copy=False)


def libero_gr00t_state(observation) -> dict[str, np.ndarray]:
    """Preserve GR00T's named XYZ, axis-angle and two-finger state."""
    state = libero_state(observation, finger_joints=2)
    if state.size != 8:
        raise ValueError(f"GR00T LIBERO state must have 8 values, got {state.size}")
    return {
        "x": np.ascontiguousarray(state[0:1]),
        "y": np.ascontiguousarray(state[1:2]),
        "z": np.ascontiguousarray(state[2:3]),
        "roll": np.ascontiguousarray(state[3:4]),
        "pitch": np.ascontiguousarray(state[4:5]),
        "yaw": np.ascontiguousarray(state[5:6]),
        "gripper": np.ascontiguousarray(state[6:8]),
    }


def libero_gr00t_action(actions: np.ndarray) -> np.ndarray:
    """Map GR00T's decoded gripper (0 closed, 1 open) to robosuite (+1/-1)."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim not in (1, 2) or array.shape[-1] != 7:
        raise ValueError(f"GR00T LIBERO actions must end in 7 values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("GR00T LIBERO actions must contain only finite values")
    converted = np.ascontiguousarray(array.copy())
    converted[..., -1] = -np.sign(2.0 * converted[..., -1] - 1.0)
    return converted


def make_env(task, seed: int):
    """Build the same off-screen LIBERO environment used by evaluation."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ImportError(
            "native LIBERO observations require the LIBERO and MuJoCo evaluation "
            "dependencies; install them with `pip install apxinf-robo[libero]` plus "
            "LIBERO itself, as described in README.md"
        ) from error

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def to_apxinf_observation(
    observation,
    *,
    prompt: str,
    image_keys: Tuple[str, str],
    prompt_key: str,
    state_key: str,
) -> dict:
    """Convert one raw simulator frame using the evaluation-time convention.

    The keys are passed in rather than read from a preset because a server may
    have been started with overrides; take them from the policy's published
    ``metadata`` so what is sent matches what is served.
    """
    images = libero_images(
        observation["agentview_image"],
        observation["robot0_eye_in_hand_image"],
    )
    state = libero_state(observation)
    return {
        image_keys[0]: images[0],
        image_keys[1]: images[1],
        state_key: state,
        prompt_key: prompt,
    }

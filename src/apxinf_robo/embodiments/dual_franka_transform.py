"""RLinf dual-Franka TCP rot6d state and body-frame action transforms, with no engine behind them.

Adapted from RLinf's dual_franka_tcp_rot6d_policy, rigid_body_delta, and utils.rot6d. The
controller consumes two [xyz, rot6d, gripper] poses; wire observations put both grippers
first. Rotation columns follow RLinf's float32 rounding between Gram-Schmidt and float64
SE(3) composition.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..wirekeys import lookup_key, set_key

__all__ = [
    "DUAL_FRANKA_ACTION_DIM",
    "DualFrankaTransform",
    "dual_franka_absolute_actions",
    "reorder_state",
]

DUAL_FRANKA_ACTION_DIM = 2 * (3 + 6 + 1)  # Two [xyz, rot6d, gripper] poses.


def _rotation(rot6d: np.ndarray) -> np.ndarray:
    vectors = np.asarray(rot6d, dtype=np.float64)
    first, second = vectors[..., :3], vectors[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate dual-Franka rotation: first column is zero")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate dual-Franka rotation: columns are collinear")
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=-1).astype(np.float32)


def reorder_state(state: Any) -> np.ndarray:
    """Reorder RPent/RLinf wire state into the checkpoint's per-arm pose layout."""
    state = np.asarray(state)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("Dual Franka requires a finite 20-value state")
    return np.concatenate([state[2:11], state[:1], state[11:20], state[1:2]])


def dual_franka_absolute_actions(actions: Any, state: Any) -> np.ndarray:
    """Compose each body-frame delta with the observed (reordered) pose, preserving grips."""
    state = np.asarray(state)
    raw = np.asarray(actions)
    if state.ndim != 1 or state.size < 20 or not np.isfinite(state).all():
        raise ValueError("Dual Franka requires at least 20 finite reordered state values")
    if raw.ndim != 2 or raw.shape[1] < DUAL_FRANKA_ACTION_DIM or not np.isfinite(raw).all():
        raise ValueError("Dual Franka requires a finite [horizon, >=20] action chunk")
    out = raw[:, :DUAL_FRANKA_ACTION_DIM].copy()
    for start in (0, 10):
        xyz = slice(start, start + 3)
        rotation = slice(start + 3, start + 9)
        current = _rotation(state[rotation]).astype(np.float64)
        delta = _rotation(out[:, rotation]).astype(np.float64)
        absolute = current @ delta
        out[:, xyz] = (current @ out[:, xyz].astype(np.float64).T).T + state[xyz]
        out[:, rotation] = np.concatenate(
            [absolute[..., :, 0], absolute[..., :, 1]], axis=-1
        ).astype(np.float32)
    return out


class DualFrankaTransform:
    """Dual-Franka's state reorder on the way in, body-frame composition on the way out."""

    min_input_dim = DUAL_FRANKA_ACTION_DIM

    def __init__(self, state_key: str) -> None:
        self.state_key = state_key

    def before(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return set_key(observation, self.state_key, reorder_state(lookup_key(observation, self.state_key)))

    def after(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        return dual_franka_absolute_actions(actions, lookup_key(observation, self.state_key))

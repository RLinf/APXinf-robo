# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RLinf dual-Franka TCP rot6d state and body-frame action transforms.

Adapted from RLinf's dual_franka_tcp_rot6d_policy, rigid_body_delta, and
utils.rot6d. The controller consumes two [xyz, rot6d, gripper] poses; wire
observations put both grippers first. Rotation columns follow RLinf's float32
rounding between Gram-Schmidt and float64 SE(3) composition.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Sequence

import numpy as np

from apxinf.policies.base import Policy
from apxinf.processors.base import ProcessorStep
from apxinf.processors.transforms import ACTIONS, OBSERVATION, lookup_key, set_key

from .base import Embodiment

__all__ = ["DUAL_FRANKA_BODY", "build_dual_franka_policy"]


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
    return np.stack([first, second, np.cross(first, second)], axis=-1).astype(
        np.float32
    )


class DualFrankaState(ProcessorStep):
    """Reorder RPent/RLinf state into the checkpoint's per-arm pose layout."""

    def __init__(self, state_key: str):
        self.state_key = state_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data[OBSERVATION]
        state = np.asarray(lookup_key(observation, self.state_key))
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("Dual Franka requires a finite 20-value state")
        ordered = np.concatenate([state[2:11], state[:1], state[11:20], state[1:2]])
        data[OBSERVATION] = set_key(observation, self.state_key, ordered)
        return data


class DualFrankaAbsoluteActions(ProcessorStep):
    """Compose each body-frame delta with the observed pose, preserving grips."""

    def __init__(self, state_key: str):
        self.state_key = state_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        state = np.asarray(lookup_key(data[OBSERVATION], self.state_key))
        raw = np.asarray(data[ACTIONS])
        if state.ndim != 1 or state.size < 20 or not np.isfinite(state).all():
            raise ValueError(
                "Dual Franka requires at least 20 finite reordered state values"
            )
        if raw.ndim != 2 or raw.shape[1] < 20 or not np.isfinite(raw).all():
            raise ValueError(
                "Dual Franka requires a finite [horizon, >=20] action chunk"
            )
        actions = raw[:, :20].copy()
        for start in (0, 10):
            xyz = slice(start, start + 3)
            rotation = slice(start + 3, start + 9)
            current = _rotation(state[rotation]).astype(np.float64)
            delta = _rotation(actions[:, rotation]).astype(np.float64)
            absolute = current @ delta
            actions[:, xyz] = (
                current @ actions[:, xyz].astype(np.float64).T
            ).T + state[xyz]
            actions[:, rotation] = np.concatenate(
                [absolute[..., :, 0], absolute[..., :, 1]], axis=-1
            ).astype(np.float32)
        data[ACTIONS] = actions
        return data


def build_dual_franka_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    prompt_key: str = "prompt",
    action_dim: int = 20,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Wrap a body-delta checkpoint with RLinf's state and action transforms."""
    from apxinf.policies.base import ComposablePolicy

    from ..engine import load_policy

    if action_dim != 20:
        raise ValueError("Dual Franka TCP rot6d actions must have 20 values")
    base = load_policy(
        model_dir,
        state_key=state_key,
        image_keys=tuple(image_keys),
        prompt_key=prompt_key,
        action_dim=20,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        base.close()
        raise TypeError("Dual Franka requires a ComposablePolicy for pose transforms")
    return base.with_adapter(
        before=[("dual_franka_state", DualFrankaState(state_key))],
        after=[("dual_franka_absolute", DualFrankaAbsoluteActions(state_key))],
        action_dim=20,
        metadata={"robot": "dual_franka_tcp_rot6d", **(metadata or {})},
    )


DUAL_FRANKA_BODY = Embodiment(
    name="dual_franka",
    num_cameras=3,
    action_dim=20,
    state_dim=20,
    # RLinf computes stats after DualFrankaTcpRot6dInputs pads actions to the
    # model width (32), so checking for 20-wide statistics would reject them.
    action_width=None,
    builder=build_dual_franka_policy,
    builder_kwargs={
        # Match OpenPI's float64 unnormalization before SE(3) composition.
        "norm_dtype": "float64",
    },
)

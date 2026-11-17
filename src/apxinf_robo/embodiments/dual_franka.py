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
from .dual_franka_transform import (  # noqa: F401
    DUAL_FRANKA_ACTION_DIM,
    _rotation,
    dual_franka_absolute_actions,
    reorder_state,
)

__all__ = ["DUAL_FRANKA_BODY", "build_dual_franka_policy"]


class DualFrankaState(ProcessorStep):
    """ApxInf shell over :func:`~apxinf_robo.embodiments.dual_franka_transform.reorder_state`."""

    def __init__(self, state_key: str):
        self.state_key = state_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data[OBSERVATION]
        ordered = reorder_state(lookup_key(observation, self.state_key))
        data[OBSERVATION] = set_key(observation, self.state_key, ordered)
        return data


class DualFrankaAbsoluteActions(ProcessorStep):
    """ApxInf shell over :func:`~apxinf_robo.embodiments.dual_franka_transform.dual_franka_absolute_actions`."""

    def __init__(self, state_key: str):
        self.state_key = state_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        state = lookup_key(data[OBSERVATION], self.state_key)
        data[ACTIONS] = dual_franka_absolute_actions(data[ACTIONS], state)
        return data


def build_dual_franka_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    prompt_key: str = "prompt",
    action_dim: int = DUAL_FRANKA_ACTION_DIM,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Wrap a body-delta checkpoint with RLinf's state and action transforms."""
    from apxinf.policies.base import ComposablePolicy

    from ..engine import load_policy

    if action_dim != DUAL_FRANKA_ACTION_DIM:
        raise ValueError(
            f"Dual Franka TCP rot6d actions must have {DUAL_FRANKA_ACTION_DIM} values"
        )
    base = load_policy(
        model_dir,
        state_key=state_key,
        image_keys=tuple(image_keys),
        prompt_key=prompt_key,
        action_dim=DUAL_FRANKA_ACTION_DIM,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        base.close()
        raise TypeError("Dual Franka requires a ComposablePolicy for pose transforms")
    return base.with_adapter(
        before=[("dual_franka_state", DualFrankaState(state_key))],
        after=[("dual_franka_absolute", DualFrankaAbsoluteActions(state_key))],
        action_dim=DUAL_FRANKA_ACTION_DIM,
        metadata={"robot": "dual_franka_tcp_rot6d", **(metadata or {})},
    )


DUAL_FRANKA_BODY = Embodiment(
    name="dual_franka",
    num_cameras=3,
    action_dim=DUAL_FRANKA_ACTION_DIM,
    state_dim=20,
    # Only the effective pose dimensions are consumed. Statistics may include
    # checkpoint padding; preflight checks coverage rather than exact width.
    action_width=None,
    builder=build_dual_franka_policy,
    builder_kwargs={
        # Match OpenPI's float64 unnormalization before SE(3) composition.
        "norm_dtype": "float64",
    },
)

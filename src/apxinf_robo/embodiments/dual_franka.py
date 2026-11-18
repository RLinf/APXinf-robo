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
"""Dual-Franka body (RLinf TCP rot6d): geometry and its transform.

The arithmetic is :mod:`~apxinf_robo.embodiments.dual_franka_transform`; nothing here
imports an inference engine except :func:`build_dual_franka_policy`, which loads ApxInf
when called.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..engines import load_engine_policy
from ..protocol import Policy, apply_transform
from .base import Embodiment
from .dual_franka_transform import (  # noqa: F401
    DUAL_FRANKA_ACTION_DIM,
    DualFrankaTransform,
    dual_franka_absolute_actions,
    reorder_state,
)

__all__ = ["DUAL_FRANKA_BODY", "build_dual_franka_policy"]


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
    """Load a body-delta checkpoint on ApxInf and wrap RLinf's state and action transforms."""
    if action_dim != DUAL_FRANKA_ACTION_DIM:
        raise ValueError(
            f"Dual Franka TCP rot6d actions must have {DUAL_FRANKA_ACTION_DIM} values"
        )
    base = load_engine_policy(
        "apxinf",
        model_dir,
        state_key=state_key,
        image_keys=tuple(image_keys),
        prompt_key=prompt_key,
        action_dim=DUAL_FRANKA_ACTION_DIM,
        **load_kwargs,
    )
    return apply_transform(
        base,
        DualFrankaTransform(state_key),
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
    transform=DualFrankaTransform,
    engine_options={
        # Match OpenPI's float64 unnormalization before SE(3) composition.
        "apxinf": {"norm_dtype": "float64"},
    },
)

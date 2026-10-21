"""Dual-YAM joint policy transforms for RLinf's pi05_yam_joint config.

Reference: RLinf 196d40b6dec6f7a4063edee31cfeaa6ca0ee435a (#1702),
``rlinf/models/embodiment/openpi/dataconfig/yam_dataconfig.py``.
Both arms use six joint positions followed by an absolute gripper value.
The training transform subtracts state from joints only; inference must
restore it after unnormalization.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Sequence

import numpy as np

from apxinf.policies.base import Policy
from apxinf.processors.base import ProcessorStep
from apxinf.processors.transforms import ACTIONS, OBSERVATION, lookup_key

from .base import Embodiment

__all__ = ["YAM_BODY", "build_yam_policy"]

YAM_DIM = 14
YAM_DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


class YamAbsoluteActions(ProcessorStep):
    """Restore absolute joints and strip model padding, preserving grippers.

    Each prediction is relative to the observation, not the previous timestep
    in the chunk. Missing state is an error: sending unresolved deltas to the
    absolute-position controller would change the meaning of the action.
    """

    def __init__(self, state_key: str):
        self.state_key = state_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        state = np.asarray(lookup_key(data[OBSERVATION], self.state_key))
        if state.shape != (YAM_DIM,) or not np.isfinite(state).all():
            raise ValueError("YAM requires a finite 14-value joint state")
        raw = np.asarray(data[ACTIONS])
        if raw.ndim != 2 or raw.shape[1] < YAM_DIM or not np.isfinite(raw).all():
            raise ValueError("YAM requires a finite [horizon, >=14] action chunk")
        actions = raw[:, :YAM_DIM].copy()
        offset = np.where(YAM_DELTA_MASK, state, 0)
        actions[...] = actions + offset
        data[ACTIONS] = actions
        return data


def build_yam_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    prompt_key: str = "prompt",
    action_dim: int = YAM_DIM,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Load the RLinf delta-joint YAM policy and expose absolute qpos14.

    Checkpoint selection and model dimensions remain the engine's concern.
    The ``pi05_yam_joint`` training contract uses deltas relative to the
    observation for the twelve joints and absolute values for both grippers.
    """
    from apxinf.policies.base import ComposablePolicy

    from ..engine import load_policy

    if action_dim != YAM_DIM:
        raise ValueError("YAM controller actions must have 14 values")
    base = load_policy(
        model_dir,
        state_key=state_key,
        image_keys=tuple(image_keys),
        prompt_key=prompt_key,
        action_dim=YAM_DIM,
        **load_kwargs,
    )
    if not isinstance(base, ComposablePolicy):
        base.close()
        raise TypeError("YAM requires a ComposablePolicy to restore absolute joints")
    return base.with_adapter(
        after=[("yam_absolute", YamAbsoluteActions(state_key))],
        action_dim=YAM_DIM,
        metadata={"robot": "yam_joint", **(metadata or {})},
    )


YAM_BODY = Embodiment(
    name="yam",
    num_cameras=3,
    action_dim=YAM_DIM,
    state_dim=YAM_DIM,
    action_width=YAM_DIM,
    builder=build_yam_policy,
    builder_kwargs={
        # Match OpenPI's float64 normalization. State values near a discretization
        # bin edge can otherwise produce different token ids.
        "norm_dtype": "float64",
    },
)

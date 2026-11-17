"""Dual-YAM joint transform for RLinf's ``pi05_yam_joint`` config, with no engine behind it.

Reference: RLinf 196d40b6dec6f7a4063edee31cfeaa6ca0ee435a (#1702),
``rlinf/models/embodiment/openpi/dataconfig/yam_dataconfig.py``. Both arms use six joint
positions followed by an absolute gripper value. The training transform subtracts state
from joints only; inference must restore it after unnormalization.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..wirekeys import lookup_key

__all__ = ["YAM_DELTA_MASK", "YAM_DIM", "YamTransform", "yam_absolute_actions"]

YAM_DIM = 14
YAM_DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def yam_absolute_actions(actions: Any, state: Any) -> np.ndarray:
    """Restore absolute joints and strip model padding, preserving grippers.

    Each prediction is relative to the observation, not the previous timestep in the
    chunk. A state or chunk of the wrong shape, or a non-finite one, is an error: sending
    unresolved deltas to the absolute-position controller would change the action's meaning.
    """
    state = np.asarray(state)
    if state.shape != (YAM_DIM,) or not np.isfinite(state).all():
        raise ValueError("YAM requires a finite 14-value joint state")
    raw = np.asarray(actions)
    if raw.ndim != 2 or raw.shape[1] < YAM_DIM or not np.isfinite(raw).all():
        raise ValueError("YAM requires a finite [horizon, >=14] action chunk")
    out = raw[:, :YAM_DIM].copy()
    offset = np.where(YAM_DELTA_MASK, state, 0)
    out[...] = out + offset
    return out


class YamTransform:
    """YAM's output side: delta joints to absolute qpos14. The observation passes unchanged."""

    min_input_dim = YAM_DIM

    def __init__(self, state_key: str) -> None:
        self.state_key = state_key

    def before(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return observation

    def after(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        # A missing state raises KeyError rather than shipping unresolved deltas.
        return yam_absolute_actions(actions, lookup_key(observation, self.state_key))

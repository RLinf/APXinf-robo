"""The Unitree G1's arithmetic as an engine-neutral :class:`~apxinf_robo.protocol.RobotTransform`.

16 DoF laid out ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, trained as delta joint
actions with absolute grippers. Ports an integrator's ``pi05_UnitreeG1`` OpenPI transforms
(``_decode_state``, ``AbsoluteActions``, ``unitreeG1Outputs`` + ``_encode_actions``)
without importing their framework, and without importing an inference engine either: the
functions below are plain numpy, and :class:`G1Transform` composes them as

    decode state  ->  [ whatever runs the model ]  ->  delta -> absolute  ->  32 -> 16 encode

Wire keys come from a :class:`~apxinf_robo.conventions.base.Convention` and reach the
transform as ``state_key``; what is left here changes only if the hardware does.

.. note::
   The joint-flip mask and gripper maps in the integrator's file are near-identity
   placeholders (the flip is all ``+1``, the gripper maps are ``clip(0, 1)``). They are
   ported faithfully as the hooks where real G1 calibration would live.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...wirekeys import lookup_key, set_key

__all__ = [
    "G1_DELTA_MASK",
    "G1_ROBOT_DIM",
    "G1Transform",
    "absolute_actions",
    "decode_state",
    "encode_actions",
]

#: G1 state/action layout: [L-arm 7, L-gripper 1, R-arm 7, R-gripper 1].
G1_ROBOT_DIM = 16
_GRIPPER_INDICES = (7, 15)


def _joint_flip_mask() -> np.ndarray:
    """Per-joint ``+1/-1`` sign convention between G1 and pi runtime.

    Integer dtype on purpose, matching openpi's ``np.array([1, 1, ...])``: these are exact
    signs, and the ``int64 * float32 -> float64`` promotion is what puts openpi's whole G1
    chain in float64. Reproducing it keeps our discretized prompt bit-identical to theirs.
    """
    return np.ones(G1_ROBOT_DIM, dtype=np.int64)


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    """G1 gripper -> pi angular space; currently ``clip(value, 0, 1)``."""
    return np.clip(value, 0.0, 1.0)


def _gripper_from_angular(value: np.ndarray) -> np.ndarray:
    """pi angular -> G1 gripper space; inverse of :func:`_gripper_to_angular`."""
    return np.clip(value, 0.0, 1.0)


def _make_bool_mask(*dims: int) -> np.ndarray:
    """openpi ``make_bool_mask``: ``+n`` -> ``n`` True, ``-n`` -> ``n`` False."""
    return np.concatenate([np.full(abs(n), n > 0, dtype=bool) for n in dims])


#: Delta-action mask: arm joints are trained as deltas, grippers as absolute.
G1_DELTA_MASK = _make_bool_mask(7, -1, 7, -1)


def decode_state(raw: Any) -> np.ndarray:
    """Raw G1 state -> pi runtime space: joint flip, then gripper-to-angular.

    Promotes to at least float32 before the flip so an integer state cannot be truncated
    by the gripper clip (openpi has no such guard); the int64 mask then promotes the
    product to float64, matching openpi.
    """
    raw = np.asarray(raw)
    state = raw.astype(np.result_type(raw.dtype, np.float32), copy=False) * _joint_flip_mask()
    idx = list(_GRIPPER_INDICES)
    state[idx] = _gripper_to_angular(state[idx])
    return state


def absolute_actions(actions: Any, state: Any) -> np.ndarray:
    """Add the decoded state back onto the masked (arm-joint) dims of delta ``actions``.

    Both operands keep their own dtype: openpi's output chain is float64 end to end, so
    coercing either side to float32 would reintroduce the ~2e-7 divergence the rest of
    the G1 path was aligned to remove.
    """
    actions = np.array(actions, copy=True)
    state = np.asarray(state)[:G1_ROBOT_DIM]
    offset = np.where(G1_DELTA_MASK, state, state.dtype.type(0))
    actions[:, :G1_ROBOT_DIM] = actions[:, :G1_ROBOT_DIM] + offset
    return actions


def encode_actions(actions: Any) -> np.ndarray:
    """Absolute pi actions -> G1 robot space: truncate to 16, flip, gripper-from-angular.

    No dtype coercion in either direction: the int64 mask promotes float32 input to
    float64, which is what openpi's server puts on the wire.
    """
    out = np.asarray(actions)[:, :G1_ROBOT_DIM] * _joint_flip_mask()
    idx = list(_GRIPPER_INDICES)
    out[:, idx] = _gripper_from_angular(out[:, idx])
    return np.ascontiguousarray(out)


class G1Transform:
    """The G1's pre/post processing, for any policy of the :class:`~apxinf_robo.protocol.Policy` shape.

    ``adapt_to_pi=False`` drops the decode and encode conventions (raw robot space);
    ``use_delta_joint_actions=False`` drops delta -> absolute (the model already emits
    absolute actions). Without a state in the observation there is nothing to resolve a
    delta against, so ``before`` and the absolute step are no-ops, as in the serving
    path that discards state.
    """

    min_input_dim = G1_ROBOT_DIM

    def __init__(
        self,
        state_key: str,
        *,
        use_delta_joint_actions: bool = True,
        adapt_to_pi: bool = True,
    ) -> None:
        self.state_key = state_key
        self.use_delta_joint_actions = use_delta_joint_actions
        self.adapt_to_pi = adapt_to_pi
        #: ``encode_actions`` truncates to 16; without it the model's width is left alone.
        self.output_dim = G1_ROBOT_DIM if adapt_to_pi else None

    def before(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.adapt_to_pi:
            return observation
        raw = lookup_key(observation, self.state_key, None)
        if raw is None:
            return observation
        return set_key(observation, self.state_key, decode_state(raw))

    def after(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        if self.use_delta_joint_actions:
            state = lookup_key(observation, self.state_key, None)
            if state is not None:
                actions = absolute_actions(actions, state)
        if self.adapt_to_pi:
            actions = encode_actions(actions)
        return actions

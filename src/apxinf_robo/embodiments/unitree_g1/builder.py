"""Compatibility entry point: load a checkpoint on ApxInf and wrap the G1 transform around it.

The G1's arithmetic is :class:`~.transform.G1Transform`, which knows no engine. This
function is kept for callers that want the ApxInf policy with G1 processing in one call;
it is the same thing :func:`~apxinf_robo.presets.build_robot_policy` does for the
``unitree_g1`` preset, and it imports ApxInf only when called.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ...engines import load_engine_policy
from ...protocol import Policy, apply_transform
from .transform import G1_ROBOT_DIM, G1Transform

__all__ = ["build_unitree_g1_policy"]


def build_unitree_g1_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    prompt_key: str = "prompt",
    use_delta_joint_actions: bool = True,
    adapt_to_pi: bool = True,
    discrete_state: bool = True,
    action_dim: Optional[int] = None,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Load ``model_dir`` on ApxInf and wrap the Unitree G1 transform around it.

    The checkpoint's own chain is untouched: the engine is loaded at its full width
    (``action_dim=None``), and the transform decodes the state going in and does
    delta -> absolute and the 32 -> 16 encode coming out. ``adapt_to_pi=False`` drops the
    decode/encode conventions; ``use_delta_joint_actions=False`` drops delta -> absolute.
    ``action_dim`` is the width the policy *reports*, defaulting to
    :data:`~.transform.G1_ROBOT_DIM` when the encode step truncates.

    ``state_key`` and ``image_keys`` are required: they are a recording convention
    (:data:`apxinf_robo.conventions.UNITREE_G1`), not a fact about this body.
    """
    base = load_engine_policy(
        "apxinf",
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,  # the transform owns the truncation
        state_key=state_key,
        prompt_key=prompt_key,
        discrete_state=discrete_state,
        **load_kwargs,
    )
    transform = G1Transform(
        state_key, use_delta_joint_actions=use_delta_joint_actions, adapt_to_pi=adapt_to_pi
    )
    return apply_transform(
        base,
        transform,
        action_dim=action_dim,
        metadata={"robot": "unitree_g1", **(metadata or {})},
    )

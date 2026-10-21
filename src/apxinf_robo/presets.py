"""Named robot presets: the deployable pairings, and how to build one.

A preset is ``Embodiment x Convention``, cross-checked. This module holds the
table of built-in pairings and :func:`build_robot_policy`, the one call a server
or an evaluation harness needs.

The machinery itself (``Embodiment`` / ``RobotPreset`` / ``register_robot_preset``)
lives in :mod:`apxinf_robo.embodiments.base`; external packages register their
own pairings from their own module scope and never edit this file.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from apxinf.policies.base import Policy

from . import conventions
from .embodiments.base import (
    ROBOT_ALIASES,
    ROBOT_PRESETS,
    VIEW_SLOTS,
    RobotPreset,
    available_robots,
    get_robot_preset,
    register_robot_preset,
)
from .embodiments.dual_franka import DUAL_FRANKA_BODY
from .embodiments.franka import FRANKA
from .embodiments.unitree_g1 import UNITREE_G1_BODY
from .embodiments.yam import YAM_BODY

__all__ = [
    "FRANKA_LIBERO",
    "UNITREE_G1",
    "DUAL_FRANKA",
    "YAM",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "build_robot_policy",
]

#: --- deployable pairings -----------------------------------------------------

#: Franka Panda under LIBERO's keys: 2 cameras, 7-dim action.
FRANKA_LIBERO = register_robot_preset(
    RobotPreset(
        name="franka_libero",
        embodiment=FRANKA,
        convention=conventions.LIBERO,
        summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
    ),
    aliases=("libero",),
)

#: Unitree G1 under its native wire convention.
UNITREE_G1 = register_robot_preset(
    RobotPreset(
        name="unitree_g1",
        embodiment=UNITREE_G1_BODY,
        convention=conventions.UNITREE_G1,
        summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
    )
)


DUAL_FRANKA = register_robot_preset(
    RobotPreset(
        name="dual_franka_tcp_rot6d",
        embodiment=DUAL_FRANKA_BODY,
        convention=conventions.DUAL_FRANKA,
        summary="Dual Franka: body-frame delta checkpoint, absolute TCP rot6d output",
    ),
)
YAM = register_robot_preset(
    RobotPreset(
        name="yam_joint",
        embodiment=YAM_BODY,
        convention=conventions.YAM,
        summary="Dual YAM: delta-joint checkpoint, absolute qpos14 output",
    ),
)


def build_robot_policy(
    robot: str,
    model_dir,
    *,
    image_keys: Optional[Sequence[str]] = None,
    state_key: Optional[str] = None,
    prompt_key: Optional[str] = None,
    action_dim: Optional[int] = None,
    discrete_state: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Policy:
    """Load ``model_dir`` under the named preset, with per-argument overrides.

    Each override defaults to the preset's value; passing one replaces just that
    field. ``image_keys``, ``state_key``, and ``prompt_key`` exist because a
    deployed client may already speak a fixed dialect — they let a server match
    an installed robot stack without editing the preset or touching the client.

    The resulting wire contract is published in the policy ``metadata``
    (``robot`` / ``robot_steps`` / ``image_keys`` / ``state_key`` /
    ``discrete_state``), which the server pushes on connect, so a client can
    assert it rather than guess. ``robot_steps`` says whether this robot's
    pre/post steps are actually wired, so a server that serves the preset's keys
    without its arithmetic cannot pass for the real thing.
    """
    preset = get_robot_preset(robot)
    keys = tuple(image_keys) if image_keys is not None else preset.image_keys
    state = state_key if state_key is not None else preset.state_key
    prompt = prompt_key if prompt_key is not None else preset.prompt_key
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    width = preset.action_dim if action_dim is None else action_dim
    policy_kwargs = {
        "image_keys": keys,
        "state_key": state,
        "prompt_key": prompt,
        "action_dim": width,
        "metadata": {
            "robot": preset.name,
            "robot_steps": preset.has_robot_steps,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            "state_dim": preset.state_dim,
            **(dict(metadata) if metadata else {}),
        },
        **dict(preset.builder_kwargs),
        **kwargs,
    }
    if discrete is not None:
        policy_kwargs["discrete_state"] = discrete
    return preset.builder(model_dir, **policy_kwargs)

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
from .engines import ENGINES, load_engine_policy
from .protocol import Policy, apply_transform

__all__ = [
    "ENGINES",
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

#: Franka Panda under LIBERO's keys: 2 cameras, 7-dim action. State encoding is
#: checkpoint-specific: PI0.5 drops it by default, while WallOSS discretizes it.
FRANKA_LIBERO = register_robot_preset(
    RobotPreset(
        name="franka_libero",
        embodiment=FRANKA,
        convention=conventions.LIBERO,
        summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
        discrete_state=None,
    ),
    aliases=("libero",),
)


#: Unitree G1 under its native wire convention, with state discretized into the
#: prompt.
UNITREE_G1 = register_robot_preset(
    RobotPreset(
        name="unitree_g1",
        embodiment=UNITREE_G1_BODY,
        convention=conventions.UNITREE_G1,
        summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
        discrete_state=True,
    )
)

#: RLinf ``pi05_dualfranka_tcp_rot6d``: slots hold base, left wrist, right wrist.
#: The checkpoint omits state tokens; state is still needed to restore absolute
#: body-frame actions.
DUAL_FRANKA = register_robot_preset(
    RobotPreset(
        name="dual_franka_tcp_rot6d",
        embodiment=DUAL_FRANKA_BODY,
        convention=conventions.CANONICAL,
        summary="Dual Franka: body-frame delta checkpoint, absolute TCP rot6d output",
        discrete_state=False,
    ),
)
#: RLinf ``pi05_yam_joint``: slots hold top, left, right.
YAM = register_robot_preset(
    RobotPreset(
        name="yam_joint",
        embodiment=YAM_BODY,
        convention=conventions.CANONICAL,
        summary="Dual YAM: delta-joint checkpoint, absolute qpos14 output",
        discrete_state=True,
    ),
)


def build_robot_policy(
    robot: str,
    model_dir,
    *,
    engine: str = "apxinf",
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

    ``engine`` selects what runs the model: ``apxinf``, or ``drobotics`` for the D-Robotics
    board, where ``model_dir`` is the SDK's ``oellm_runtime_s600`` directory and may be
    ``None``. Both results satisfy :class:`~apxinf_robo.protocol.Policy` and take
    observations under the same keys, but they are not interchangeable in every
    respect. The drobotics policy has no ``noise=`` (the runtime samples its own, so
    passing one raises) and does not return ``normalized_actions``. Callers that need
    those must use ``engine="apxinf"``.

    A preset's robot arithmetic is an engine-neutral transform that wraps whichever
    policy the engine produced (``before -> engine -> after``), so it runs on either engine.
    What decides whether a pairing is servable is the engine's own deployment contract:
    camera count, state width and action width are checked there, and a transform that needs
    wider actions than the engine produces is refused. ``transform_options`` reaches the
    transform's constructor (G1's ``use_delta_joint_actions`` / ``adapt_to_pi``).
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; known: {list(ENGINES)}")
    transform_options = dict(kwargs.pop("transform_options", None) or {})
    preset = get_robot_preset(robot)
    keys = tuple(image_keys) if image_keys is not None else preset.image_keys
    state = state_key if state_key is not None else preset.state_key
    prompt = prompt_key if prompt_key is not None else preset.prompt_key
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    transform = preset.transform(state, **transform_options) if preset.has_transform else None
    if transform is None:
        width = preset.action_dim if action_dim is None else action_dim
    else:
        # The body owns the deployable width: the engine runs at the preset's width and the
        # transform produces the rest. A caller can confirm it but not change it.
        if (
            action_dim is not None
            and transform.output_dim is not None
            and action_dim != transform.output_dim
        ):
            raise ValueError(
                f"robot preset {preset.name!r} produces {transform.output_dim}-value actions; "
                f"action_dim={action_dim} cannot be served"
            )
        width = preset.action_dim
    policy_kwargs = {
        "image_keys": keys,
        "state_key": state,
        "prompt_key": prompt,
        "action_dim": width,
        "metadata": {
            "robot": preset.name,
            "robot_steps": preset.has_transform,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            "state_dim": preset.state_dim,
            **(dict(metadata) if metadata else {}),
        },
        **dict(preset.engine_options.get(engine, {})),
        **kwargs,
    }
    if engine == "drobotics":
        policy_kwargs["state_dim"] = preset.state_dim
        policy_kwargs["discrete_state"] = discrete
    elif discrete is not None:
        policy_kwargs["discrete_state"] = discrete
    metadata_out = policy_kwargs.pop("metadata")
    engine_policy = load_engine_policy(engine, model_dir, metadata=metadata_out, **policy_kwargs)
    if transform is None:
        return engine_policy
    return apply_transform(engine_policy, transform, metadata=metadata_out)

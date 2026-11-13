"""Robot bodies: DoF geometry, body arithmetic, and preset machinery.

A body is what changes when you swap hardware. It is deliberately separate from
a :mod:`~apxinf_robo.conventions` dialect (what changes when you swap the
recorder) and from the engine (what changes when you swap the checkpoint).

The G1, YAM and dual-Franka bodies add processor steps to ApxInf's chain, so they
exist only where ApxInf does. Franka needs no steps and is available everywhere.
"""

from __future__ import annotations

from ..engine import apxinf_available, require_apxinf
from .base import (
    ROBOT_ALIASES,
    ROBOT_PRESETS,
    VIEW_SLOTS,
    Embodiment,
    RobotPreset,
    available_robots,
    get_robot_preset,
    register_robot_preset,
)
from .franka import FRANKA

_APXINF_BODIES = (
    "DUAL_FRANKA_BODY",
    "YAM_BODY",
    "build_dual_franka_policy",
    "build_yam_policy",
    "UNITREE_G1_BODY",
    "build_unitree_g1_policy",
)

if apxinf_available():
    from .dual_franka import DUAL_FRANKA_BODY, build_dual_franka_policy
    from .unitree_g1 import UNITREE_G1_BODY, build_unitree_g1_policy
    from .yam import YAM_BODY, build_yam_policy
else:

    def __getattr__(name: str):
        if name in _APXINF_BODIES:
            require_apxinf()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VIEW_SLOTS",
    "Embodiment",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "FRANKA",
    *(_APXINF_BODIES if apxinf_available() else ()),
]

"""Robot bodies: DoF geometry, body arithmetic, and preset machinery.

A body is what changes when you swap hardware. It is deliberately separate from
a :mod:`~apxinf_robo.conventions` dialect (what changes when you swap the
recorder) and from the engine (what changes when you swap the checkpoint).
"""

from __future__ import annotations

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
from .dual_franka import DUAL_FRANKA_BODY, build_dual_franka_policy
from .franka import FRANKA
from .unitree_g1 import UNITREE_G1_BODY, build_unitree_g1_policy
from .yam import YAM_BODY, build_yam_policy

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
    "DUAL_FRANKA_BODY",
    "YAM_BODY",
    "build_dual_franka_policy",
    "build_yam_policy",
    "UNITREE_G1_BODY",
    "build_unitree_g1_policy",
]

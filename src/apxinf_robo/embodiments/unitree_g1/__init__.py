"""Unitree G1: body arithmetic (:mod:`.transform`) and geometry (:mod:`.body`).

Nothing here imports an inference engine. :func:`build_unitree_g1_policy` is the one
compatibility call that loads ApxInf, and it does so only when called.
"""

from __future__ import annotations

from .body import UNITREE_G1_BODY
from .builder import build_unitree_g1_policy
from .transform import G1_DELTA_MASK, G1_ROBOT_DIM, G1Transform

__all__ = [
    "UNITREE_G1_BODY",
    "build_unitree_g1_policy",
    "G1Transform",
    "G1_ROBOT_DIM",
    "G1_DELTA_MASK",
]

"""Wire-key dialects: what a client calls its cameras, state, and prompt.

Importing this package registers the built-in conventions. Third-party dialects
call :func:`register_convention` from their own module scope.
"""

from __future__ import annotations

from .base import (
    CONVENTIONS,
    Convention,
    available_conventions,
    get_convention,
    register_convention,
)
from .libero import LIBERO
from .rlinf import DUAL_FRANKA, YAM
from .unitree_g1 import UNITREE_G1

__all__ = [
    "Convention",
    "CONVENTIONS",
    "LIBERO",
    "DUAL_FRANKA",
    "YAM",
    "UNITREE_G1",
    "available_conventions",
    "get_convention",
    "register_convention",
]

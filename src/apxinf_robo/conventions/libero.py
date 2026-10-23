"""LIBERO's recording dialect."""

from __future__ import annotations

from .base import Convention, register_convention

__all__ = ["LIBERO"]

#: LIBERO (the 7-DoF sim benchmark), mirroring openpi's ``LiberoInputs``: flat
#: ``observation/...`` keys.
LIBERO = register_convention(
    Convention(
        name="libero",
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state",
    )
)

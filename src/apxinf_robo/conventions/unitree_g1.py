"""The ``pi05_UnitreeG1`` fine-tune's recording dialect."""

from __future__ import annotations

from .base import Convention, register_convention

__all__ = ["UNITREE_G1"]

#: From its ``unitreeG1Inputs``/``Outputs``: cameras nested one level under
#: ``images`` (spelled as a path for
#: :func:`~apxinf.processors.transforms.lookup_key`) and a flat top-level
#: ``state`` rather than LIBERO's ``observation/state``.
UNITREE_G1 = register_convention(
    Convention(
        name="unitree_g1",
        image_keys=("images/cam_high", "images/cam_left_wrist", "images/cam_right_wrist"),
        state_key="state",
    )
)

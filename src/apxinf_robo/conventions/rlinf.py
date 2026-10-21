"""Split-view serving conventions for RLinf real-robot policies.

The bridge unpacks RLinf's stacked extra views into these keys. Order here is
model slot order (base, left wrist, right wrist), independent of wire order.
"""

from .base import Convention, register_convention

__all__ = ["DUAL_FRANKA", "YAM"]

DUAL_FRANKA = register_convention(
    Convention(
        name="rlinf_dual_franka",
        image_keys=("images/base", "images/left_wrist", "images/right_wrist"),
        state_key="state",
        # pi05_dualfranka_tcp_rot6d omits state tokens; state is still needed
        # to restore absolute body-frame actions.
        discrete_state=False,
    )
)
YAM = register_convention(
    Convention(
        name="rlinf_yam",
        image_keys=("images/top", "images/left", "images/right"),
        state_key="state",
        discrete_state=True,
    )
)

"""LIBERO conversion helpers shared with apxinf_robo.envs.libero.

These exports preserve imports used by the Robo calibration scripts.
"""

from __future__ import annotations

try:
    from apxinf_robo.envs.libero import (  # noqa: F401
        libero_images,
        libero_state,
        make_env,
        quat_to_axis_angle,
        to_apxinf_observation,
    )
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError(
        "the vendored calibration scripts read LIBERO observations through "
        "apxinf_robo; install this package first (`pip install -e .`), as "
        "described in README.md"
    ) from error

__all__ = [
    "quat_to_axis_angle",
    "libero_images",
    "libero_state",
    "make_env",
    "to_apxinf_observation",
]

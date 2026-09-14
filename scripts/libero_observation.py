"""Native LIBERO observation conversion, re-exported for the vendored scripts.

The conversions themselves live in :mod:`apxinf_robo.envs.libero` and are *not*
restated here. Evaluation (``apxinf-robo eval-libero``), capture
(``apxinf-robo capture-libero``), and calibration (``scripts/calibrate_pi05.py``)
must agree on camera orientation and robot-state layout down to the byte: a
divergence there changes success rates and FP8 scales with no error anywhere, and
the goldens in ``tests/test_libero_observation.py`` only pin the package copy.

This module exists because the engine's ``scripts/pi05_calibration_data.py`` --
which this repository vendors -- imports it by that name. It is a shim, not a
mirror.
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

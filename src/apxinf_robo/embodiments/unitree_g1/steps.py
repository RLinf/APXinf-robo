"""Unitree G1 (humanoid, dual-arm + 2 dexterous hands) processing steps.

Ports the robot-specific numpy transforms of an integrator's Unitree G1
``DataTransformFn`` pipeline (a ``pi05_UnitreeG1`` OpenPI finetune: 3 cameras,
16-DoF state ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, ``action_dim=32``,
``action_horizon=50``, delta joint actions) into ApxInf's
:class:`~apxinf.processors.base.ProcessorStep` contract **without importing their
framework**:

===========================  ==============================================
integrator transform          apxinf equivalent
===========================  ==============================================
camera rename + CHW->HWC       ``image_keys`` config + ``ParseImage`` (already
  + float->uint8                handles CHW->HWC and float->uint8)
``_decode_state``              :class:`UnitreeG1DecodeState` (input step)
32-wide unnormalize            existing ``Unnormalize`` (full model width)
``AbsoluteActions`` (delta)    :class:`UnitreeG1AbsoluteActions` (output step)
``unitreeG1Outputs`` 32->16    :class:`UnitreeG1EncodeActions` (output step)
  + joint flip + gripper
===========================  ==============================================

These steps are model-agnostic — they reference no policy symbols and vary only
with the G1 body. They are also **dataset-agnostic**: the G1 client's wire keys
(``images/cam_high``, a flat ``state``) are a recording convention, so they live
in :mod:`apxinf_robo.conventions` and reach these steps as arguments. What is
left here is arithmetic that changes only if the hardware does — the 16-DoF
layout, the delta mask, the gripper and joint-sign conventions.

The :func:`~apxinf_robo.embodiments.unitree_g1.builder.build_unitree_g1_policy`
adapter wraps them around whichever policy the checkpoint turns out to be, using
:meth:`~apxinf.policies.base.ComposablePolicy.with_adapter`; it names no model
class either.

.. note::
   The joint-flip mask and gripper transforms in the integrator's file are
   near-identity placeholders ("需要根据实际参数更新"): the flip mask is all
   ``1``, the gripper maps are ``clip(0,1)`` round-trips. They are ported
   faithfully (as the hooks where real G1 calibration would live) but currently
   pass their input through unchanged.
"""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from apxinf.processors.base import ProcessorStep
from apxinf.processors.transforms import ACTIONS, OBSERVATION, lookup_key, set_key

__all__ = [
    "UnitreeG1DecodeState",
    "UnitreeG1AbsoluteActions",
    "UnitreeG1EncodeActions",
    "G1_ROBOT_DIM",
    "G1_DELTA_MASK",
]

from .transform import (  # noqa: E402  (arithmetic lives in the engine-neutral module)
    G1_DELTA_MASK,
    G1_ROBOT_DIM,
    absolute_actions,
    decode_state,
    encode_actions,
)


class UnitreeG1DecodeState(ProcessorStep):
    """Decode raw G1 state into pi runtime space before it enters the model.

    Ports ``_decode_state``: joint-flip then gripper-to-angular on the two
    gripper dims. Runs *before* ``tokenize`` so discretized state (when enabled)
    and the delta->absolute output step both see the decoded state. Operates on a
    shallow copy of the observation so the caller's dict is left untouched. A
    no-op when no state is present (state-off serving).

    ``state_key`` is required and has no default: which key carries state is a
    property of the *convention* the data was recorded under
    (:mod:`apxinf_robo.conventions`), not of this body, and a default here would
    be one dataset's dialect built into a robot's arithmetic.
    """

    def __init__(self, state_key: str, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key)
        if observation is None:
            return data
        raw = lookup_key(observation, self.state_key, None)
        if raw is None:
            return data
        data[self.observation_key] = set_key(observation, self.state_key, decode_state(raw))
        return data


class UnitreeG1AbsoluteActions(ProcessorStep):
    """Convert the model's delta arm-joint actions back to absolute using state.

    Ports openpi ``AbsoluteActions``: on masked dims (arm joints), add the current
    decoded state; gripper dims are already absolute and pass through. Reads the
    unnormalized ``actions`` and the decoded ``observation`` state threaded in by
    ``Pi05Policy.infer``. A no-op when state is absent (delta cannot be resolved).
    ``state_key`` is required for the same reason as in
    :class:`UnitreeG1DecodeState`.
    """

    def __init__(self, state_key: str, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key) or {}
        state = lookup_key(observation, self.state_key, None)
        if state is None:
            return data
        data[ACTIONS] = absolute_actions(data[ACTIONS], state)
        return data


class UnitreeG1EncodeActions(ProcessorStep):
    """Map absolute pi actions back to G1 robot space: 32->16, flip, gripper.

    Ports ``unitreeG1Outputs`` + ``_encode_actions``: truncate to the 16 robot
    dims, apply the joint flip, and gripper-from-angular on the two gripper dims.
    """

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        data[ACTIONS] = encode_actions(data[ACTIONS])
        return data

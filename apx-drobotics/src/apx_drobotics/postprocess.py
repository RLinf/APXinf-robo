"""Apx-drobotics-postprocess: the runtime's reply, turned into the Policy result.

The SDK has already unnormalized the actions, and nothing else is applied here: the
runtime's delta actions go to the controller unchanged. What is left is checking the
chunk, because a NaN or a wrong horizon must never reach a robot, and giving the
timing the common names.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import numpy as np

from . import contract as contract_lib
from .errors import RuntimeFailure

__all__ = ["Postprocess"]

logger = logging.getLogger(__name__)

#: The runtime's profile entries that are time spent in the HBMs.
_MODEL_PARTS = ("visual", "lm", "action")


class Postprocess:
    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = contract
        self._warned_parts = False

    def __call__(self, reply: Mapping[str, Any], payload: bytes) -> Tuple[np.ndarray, dict]:
        """Return ``(actions, timing)``; actions are ``float32 [horizon, dim]``."""
        action = self.contract["action"]
        expected = [1, action["horizon"], action["dim"]]
        shape = reply.get("action_shape")
        if shape != expected:
            raise RuntimeFailure(f"runtime returned action shape {shape}, contract says {expected}")
        values = np.frombuffer(payload, dtype="<f8")
        if values.size != action["horizon"] * action["dim"]:
            raise RuntimeFailure(f"runtime returned {values.size} action values for shape {shape}")
        actions = values.reshape(shape[1:])
        problems = contract_lib.check_actions(self.contract, actions)
        if problems:
            raise RuntimeFailure("; ".join(problems))
        profile = reply.get("timing_ms", {})
        missing = [part for part in _MODEL_PARTS if part not in profile]
        if missing and not self._warned_parts:
            # model_ms would silently under-report, and latency is what this feeds.
            self._warned_parts = True
            logger.warning("runtime profile has no %s entries; model_ms is incomplete", missing)
        timing = {
            "model_ms": float(sum(profile.get(part, 0.0) for part in _MODEL_PARTS)),
            **{f"sdk_{name}_ms": float(value) for name, value in profile.items()},
        }
        return actions.astype(np.dtype(action["output_dtype"])), timing

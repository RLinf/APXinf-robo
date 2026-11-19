"""The vocabulary every engine shares, owned here so importing robo needs no engine.

ApxInf and apx-drobotics stand side by side under this package, and neither is available
everywhere: ApxInf does not run on the S600 board, and apx-drobotics runs nowhere else. So
what the layers above them are written against lives here.

There is nothing to adapt between the engines at run time. An engine's policy is any
object of the :class:`Policy` shape, and robo gives each one the same thing, an
observation dict under the preset's wire keys. Everything specific to a model, such as
resizing or tokenizing, happens inside the policy. A new engine is therefore one
``engine_<name>.py`` next to :mod:`apxinf_robo.engine`.

These values restate ApxInf's own (``apxinf.policies.base``). They are the defaults robo
owns; ApxInf does not import them back. A model's camera slots and their order are the
checkpoint's or the deployment contract's to say, not these constants'; a drift test
(``tests/test_protocol.py``) holds the two copies together where ApxInf is installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CANONICAL_IMAGE_KEYS",
    "CANONICAL_PROMPT_KEY",
    "CANONICAL_STATE_KEY",
    "Policy",
    "RobotTransform",
    "apply_transform",
    "TransformedPolicy",
    "VIEW_SLOTS",
]

#: Camera view slots, in the order a checkpoint's weights consume them.
VIEW_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

#: The keys of a client that fills view slots directly and speaks no dataset's dialect.
CANONICAL_IMAGE_KEYS = VIEW_SLOTS
CANONICAL_STATE_KEY = "state"
CANONICAL_PROMPT_KEY = "prompt"


@runtime_checkable
class Policy(Protocol):
    """``observation dict -> result dict``, which every engine's policy satisfies.

    ``infer`` guarantees two result keys: ``actions``, deployable ``[horizon, dim]``,
    and ``timing``, with at least ``model_ms`` and ``total_ms``. ``metadata`` is the
    static description a server publishes on connect.
    """

    metadata: Mapping[str, Any]

    @property
    def action_dim(self) -> int: ...

    @property
    def action_horizon(self) -> int: ...

    def infer(
        self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None
    ) -> Dict[str, Any]: ...

    def close(self) -> None: ...


class RobotTransform(Protocol):
    """A body's arithmetic around a model, independent of what runs the model.

    ``before`` rewrites the client's observation on the way in (G1 decodes its state,
    dual-Franka reorders it) and ``after`` turns the model's deployable actions into the
    body's (delta to absolute, then truncate or re-encode). Both are pure: ``before``
    returns a new mapping and never touches its argument, and ``after`` is handed
    ``before``'s output, so a delta resolves against the same state the model saw.

    ``after`` receives actions in the engine's deployable domain: unnormalized
    ``[horizon, width]``, not clipped or re-encoded for any body. ``min_input_dim`` is the
    narrowest such chunk it can work on.
    """

    #: Narrowest engine action width ``after`` accepts.
    min_input_dim: int
    #: Width of the chunk ``after`` returns, or ``None`` when it leaves the width alone.
    output_dim: Optional[int]

    def before(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def after(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray: ...


class TransformedPolicy:
    """``transform.before -> inner.infer -> transform.after``, as one :class:`Policy`.

    This is how a body reaches either engine: the wrapper needs only ``inner`` to satisfy
    :class:`Policy`. Everything else in the result (``normalized_actions``, ``token_ids``,
    whatever the engine adds) passes through untouched, except ``metadata``: if the engine
    returns one it is replaced by a copy of this policy's, so it describes the actions
    actually returned (an engine that returns none still returns none). ``timing`` keeps the engine's
    entries with the transform's own time added to ``total_ms`` and reported as
    ``robot_ms``.

    The caller owns ``inner`` until construction succeeds: if this raises, close ``inner``.
    Afterwards ``close()`` closes it.
    """

    def __init__(
        self,
        inner: Policy,
        transform: RobotTransform,
        *,
        action_dim: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if inner.action_dim < transform.min_input_dim:
            raise ValueError(
                f"the robot transform needs actions at least {transform.min_input_dim} wide, "
                f"but the engine's policy produces {inner.action_dim}"
            )
        self._inner = inner
        self._transform = transform
        if action_dim is None:
            action_dim = transform.output_dim if transform.output_dim is not None else inner.action_dim
        self._action_dim = int(action_dim)
        self.metadata: Dict[str, Any] = {
            **dict(inner.metadata),
            **(dict(metadata) if metadata else {}),
            "action_dim": self._action_dim,
        }

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._inner.action_horizon

    def infer(
        self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        prepared = self._transform.before(observation)
        robot_ms = (time.perf_counter() - started) * 1000.0
        # ``noise`` is forwarded only when given: an engine that owns its noise rejects it.
        result = dict(self._inner.infer(prepared, **({} if noise is None else {"noise": noise})))
        started = time.perf_counter()
        result["actions"] = self._transform.after(result["actions"], prepared)
        if "metadata" in result:
            # The engine describes its own actions; these are the wrapped ones. A copy, so a
            # caller editing the result cannot change this policy's metadata.
            result["metadata"] = dict(self.metadata)
        robot_ms += (time.perf_counter() - started) * 1000.0
        timing = dict(result.get("timing", {}))
        timing["robot_ms"] = robot_ms
        timing["total_ms"] = timing.get("total_ms", 0.0) + robot_ms
        result["timing"] = timing
        return result

    def close(self) -> None:
        self._inner.close()


def apply_transform(
    inner: Policy,
    transform: RobotTransform,
    *,
    action_dim: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TransformedPolicy:
    """Wrap ``inner`` in ``transform``; if that fails, close ``inner`` before raising.

    A loaded engine policy can hold a model (an ApxInf handle, an S600 worker process).
    Whoever loaded it and then fails to wrap it must release it, and this is that step,
    so a bad configuration cannot leave a worker occupying the accelerator.
    """
    try:
        return TransformedPolicy(inner, transform, action_dim=action_dim, metadata=metadata)
    except BaseException:
        inner.close()
        raise

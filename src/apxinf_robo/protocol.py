"""The vocabulary every engine shares, owned here so importing robo needs no engine.

ApxInf and apx-drobotics stand side by side under this package, and neither is available
everywhere: ApxInf does not run on the S600 board, and apx-drobotics runs nowhere else. So
what the layers above them are written against lives here.

There is nothing to adapt between the engines at run time. An engine's policy is any
object of the :class:`Policy` shape, and robo gives each one the same thing, an
observation dict under the preset's wire keys. Everything specific to a model, such as
resizing or tokenizing, happens inside the policy. A new engine is therefore one
``engine_<name>.py`` next to :mod:`apxinf_robo.engine`.

These values restate ApxInf's own (``apxinf.policies.base``).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CANONICAL_IMAGE_KEYS",
    "CANONICAL_PROMPT_KEY",
    "CANONICAL_STATE_KEY",
    "Policy",
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

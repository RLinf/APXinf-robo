"""Engine-canonical keys for clients that fill model view slots directly.

Some bridges, such as RPent's APXInf server, route their cameras to model view
slots themselves and write each frame under the policy's ``image_keys`` in slot
order. They need no dataset dialect, so this convention uses the engine's
canonical spelling and leaves camera routing to the bridge.
"""

from __future__ import annotations

from ..engine import require_apxinf
from .base import Convention, register_convention

__all__ = ["CANONICAL"]

_ENGINE_KEYS = require_apxinf().policies.base

CANONICAL = register_convention(
    Convention(
        name="canonical",
        image_keys=_ENGINE_KEYS.CANONICAL_IMAGE_KEYS,
        state_key=_ENGINE_KEYS.CANONICAL_STATE_KEY,
        prompt_key=_ENGINE_KEYS.CANONICAL_PROMPT_KEY,
    )
)

"""Engine-canonical keys for clients that fill model view slots directly.

Some bridges, such as RPent's APXInf server, route their cameras to model view
slots themselves and write each frame under the policy's ``image_keys`` in slot
order. They need no dataset dialect, so this convention uses the engine's
canonical spelling and leaves camera routing to the bridge.
"""

from __future__ import annotations

from ..protocol import CANONICAL_IMAGE_KEYS, CANONICAL_PROMPT_KEY, CANONICAL_STATE_KEY
from .base import Convention, register_convention

__all__ = ["CANONICAL"]

CANONICAL = register_convention(
    Convention(
        name="canonical",
        image_keys=CANONICAL_IMAGE_KEYS,
        state_key=CANONICAL_STATE_KEY,
        prompt_key=CANONICAL_PROMPT_KEY,
    )
)

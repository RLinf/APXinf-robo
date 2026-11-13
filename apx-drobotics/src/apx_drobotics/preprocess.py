"""Apx-drobotics-preprocess: the raw observation, turned into what the SDK should receive.

The SDK has its own preprocessing, which stays on and which this package treats as a
black box. Two things cannot be left to it, both found by tracing tensors through the
runtime (stage 4):

* Below 224 pixels it would resize with ``cv2.INTER_LINEAR``, which does not antialias
  and differs from OpenPI's preprocessing by up to about 34/255 on one pixel. At
  exactly 224 it uses the image unchanged, so the letterbox happens here.
* It encodes the prompt as given, so an underscore reaches the model as its own token.
  ApxInf's tokenizer cleans the prompt; on this path nothing else would.

What an observation must contain is checked strictly, because none of these mistakes
fail on their own: the runtime accepts a seven-value state or an uncleaned prompt, and
only the success rate changes. What it may contain is as open as with ApxInf: keys that
are not inputs are ignored, so one client works against either engine. The state is
required here although the SDK's preprocessing ignores discretized state, because the
runtime itself is handed it with every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from . import contract as contract_lib
from .errors import RequestError

__all__ = ["ModelRequest", "Preprocess", "clean_prompt", "resize_with_pad"]


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox one HWC uint8 image, as ``openpi_client.image_tools.resize_with_pad``."""
    if image.shape[:2] == (height, width):
        return image
    source = Image.fromarray(image)
    ratio = max(source.width / width, source.height / height)
    resized_height = int(source.height / ratio)
    resized_width = int(source.width / ratio)
    resized = source.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(
        resized,
        (max(0, int((width - resized_width) / 2)), max(0, int((height - resized_height) / 2))),
    )
    return np.asarray(canvas)


def clean_prompt(text: str) -> str:
    """The same cleaning as ApxInf's tokenizer."""
    return text.strip().replace("_", " ").replace("\n", " ")


@dataclass(frozen=True)
class ModelRequest:
    """One preprocessed request, in view-slot order."""

    images: Tuple[np.ndarray, ...]
    state: np.ndarray
    prompt: str


class Preprocess:
    """Validate an observation and convert it to a :class:`ModelRequest`.

    Keys are the caller's, as with ApxInf: ``image_keys[i]`` fills view slot ``i``.
    """

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        image_keys: Sequence[str],
        state_key: str,
        prompt_key: str,
    ) -> None:
        self.contract = contract
        self.image_keys = tuple(image_keys)
        if state_key in self.image_keys or prompt_key in self.image_keys or state_key == prompt_key:
            raise ValueError(
                f"image keys {list(self.image_keys)}, state key {state_key!r} and prompt key "
                f"{prompt_key!r} must all be different"
            )
        self.state_key = state_key
        self.prompt_key = prompt_key

    def _image_problems(self, key: str, image: Any) -> list[str]:
        want = self.contract["camera"]["input"]
        if not isinstance(image, np.ndarray):
            return [f"{key}: expected an array, got {type(image).__name__}"]
        problems = []
        if image.ndim != 3 or image.shape[-1] != want["channels"]:
            problems.append(f"{key}: shape {list(image.shape)}, expected HWC with {want['channels']} channels")
        if image.dtype.name != want["dtype"]:
            problems.append(f"{key}: dtype {image.dtype.name}, contract says {want['dtype']}")
        return problems

    def _state_problems(self, state: Any) -> list[str]:
        want = self.contract["state"]
        if not isinstance(state, np.ndarray):
            return [f"{self.state_key}: expected an array, got {type(state).__name__}"]
        problems = []
        if state.shape != (want["dim"],):
            problems.append(f"{self.state_key}: shape {list(state.shape)}, contract says [{want['dim']}]")
        if state.dtype.name != want["input_dtype"]:
            problems.append(f"{self.state_key}: dtype {state.dtype.name}, contract says {want['input_dtype']}")
        elif not np.isfinite(state).all():
            problems.append(f"{self.state_key}: contains non-finite values")
        return problems

    def __call__(self, observation: Mapping[str, Any]) -> ModelRequest:
        if not isinstance(observation, Mapping):
            raise RequestError([f"expected a map of observation keys, got {type(observation).__name__}"])
        keys = (*self.image_keys, self.state_key, self.prompt_key)
        problems = [f"key {key!r} is missing" for key in keys if key not in observation]
        for key in self.image_keys:
            if key in observation:
                problems += self._image_problems(key, observation[key])
        if self.state_key in observation:
            problems += self._state_problems(observation[self.state_key])
        prompt = observation.get(self.prompt_key)
        if self.prompt_key in observation:
            if not isinstance(prompt, str):
                problems.append(f"{self.prompt_key}: must be a string, got {type(prompt).__name__}")
            elif not clean_prompt(prompt).strip():
                problems.append(f"{self.prompt_key}: must not be empty")
        if problems:
            raise RequestError(problems)

        height, width, _ = self.contract["camera"]["model_shape"]
        request = ModelRequest(
            images=tuple(
                np.ascontiguousarray(resize_with_pad(observation[key], height, width))
                for key in self.image_keys
            ),
            state=observation[self.state_key],
            prompt=clean_prompt(prompt),
        )
        problems = contract_lib.check_model_input(
            self.contract, request.images, request.state, request.prompt
        )
        if problems:
            raise RequestError(problems)
        return request

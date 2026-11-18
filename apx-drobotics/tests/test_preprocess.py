"""Pin Apx-drobotics-preprocess: what it changes, and what it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from apx_drobotics import RequestError, load_contract
from apx_drobotics.preprocess import Preprocess, clean_prompt, resize_with_pad

from conftest import KEYS, observation


@pytest.fixture
def preprocess():
    return Preprocess(load_contract(), **KEYS)


@pytest.mark.parametrize("size", [(256, 256), (480, 640), (640, 480), (100, 300), (224, 225)])
def test_resize_matches_openpi_byte_for_byte(size):
    """The resize moved here from the client; its output must not change."""
    image_tools = pytest.importorskip("openpi_client.image_tools")
    image = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
    expected = image_tools.resize_with_pad(image, 224, 224)
    assert np.array_equal(resize_with_pad(image, 224, 224), expected)


def test_images_at_model_size_pass_through_unchanged():
    image = np.zeros((224, 224, 3), np.uint8)
    assert resize_with_pad(image, 224, 224) is image


def test_request_keeps_slot_order_and_model_shape(preprocess):
    raw = observation(256)
    request = preprocess(raw)
    assert [image.shape for image in request.images] == [(224, 224, 3)] * 2
    assert np.array_equal(request.images[0], resize_with_pad(raw["cam/base"], 224, 224))
    assert np.array_equal(request.images[1], resize_with_pad(raw["cam/wrist"], 224, 224))
    assert np.array_equal(request.state, raw["proprio"])


@pytest.mark.parametrize(
    "raw, cleaned",
    [
        ("pick up the soup", "pick up the soup"),
        ("  pick up the soup\n", "pick up the soup"),
        ("pick_up the\nsoup", "pick up the soup"),
    ],
)
def test_prompt_is_cleaned(preprocess, raw, cleaned):
    assert clean_prompt(raw) == cleaned
    assert preprocess({**observation(), "task": raw}).prompt == cleaned


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"cam/base": np.zeros((3, 224, 224), np.uint8)}, "cam/base: shape"),
        ({"cam/base": np.zeros((224, 224), np.uint8)}, "cam/base: shape"),
        ({"cam/wrist": np.zeros((224, 224, 3), np.float32)}, "cam/wrist: dtype"),
        ({"cam/wrist": b"jpeg bytes"}, "cam/wrist: expected an array"),
        ({"proprio": np.zeros(7, np.float32)}, "proprio: shape"),
        ({"proprio": np.zeros(8, np.float64)}, "proprio: dtype"),
        ({"proprio": np.full(8, np.nan, np.float32)}, "non-finite"),
        ({"proprio": [0.0] * 8}, "proprio: expected an array"),
        ({"task": ""}, "must not be empty"),
        ({"task": " _\n"}, "must not be empty"),
        ({"task": 7}, "must be a string"),
    ],
)
def test_bad_observation_is_rejected(preprocess, change, needle):
    with pytest.raises(RequestError) as error:
        preprocess({**observation(), **change})
    assert any(needle in problem for problem in error.value.problems), error.value.problems


def test_missing_key_is_rejected(preprocess):
    raw = observation()
    del raw["proprio"]
    with pytest.raises(RequestError, match="'proprio' is missing"):
        preprocess(raw)


def test_every_problem_is_reported_at_once(preprocess):
    raw = {**observation(), "proprio": np.zeros(7, np.float32), "task": 7}
    with pytest.raises(RequestError) as error:
        preprocess(raw)
    assert len(error.value.problems) == 2


def test_observation_must_be_a_mapping(preprocess):
    with pytest.raises(RequestError, match="expected a map"):
        preprocess([1, 2, 3])


def test_keys_that_are_not_inputs_are_ignored_as_with_apxinf(preprocess):
    """One client must work against either engine, and clients send extra keys."""
    raw = {**observation(), "observation/image": np.zeros((224, 224, 3), np.uint8), "reset": True}
    request = preprocess(raw)
    assert len(request.images) == 2


@pytest.mark.parametrize(
    "keys",
    [
        {"image_keys": ("cam/base", "cam/wrist"), "state_key": "cam/wrist", "prompt_key": "task"},
        {"image_keys": ("cam/base", "cam/wrist"), "state_key": "proprio", "prompt_key": "proprio"},
        {"image_keys": ("cam/base", "task"), "state_key": "proprio", "prompt_key": "task"},
    ],
)
def test_state_prompt_and_image_keys_must_differ(keys):
    with pytest.raises(ValueError, match="must all be different"):
        Preprocess(load_contract(), **keys)

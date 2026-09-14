"""Pin the connect-time dialect check between the evaluator and a remote server.

The websocket backend builds its observation from the preset table while the
server resolved its own wire contract, so nothing structural keeps the two in
step. A mismatch is silent on the wire -- the server accepts the dict, reads the
keys it expects as absent, and the run reports a low success rate that looks like
an accuracy regression rather than a misconfiguration. These tests pin the
comparison, including the part that must *not* fire: a server that publishes less
metadata is not a mismatch.
"""

import pytest

from apxinf_robo.cli import eval_libero


def served_metadata(**overrides):
    convention = eval_libero.libero_convention()
    metadata = {
        "precision": "bf16",
        "image_keys": list(convention.image_keys),
        "state_key": convention.state_key,
        "prompt_key": convention.prompt_key,
    }
    metadata.update(overrides)
    return metadata


def test_matching_dialect_passes():
    eval_libero._assert_server_speaks_the_same_dialect(served_metadata())


def test_tuple_image_keys_are_not_a_mismatch():
    """msgpack round-trips a list; an in-process server may hand back a tuple."""
    convention = eval_libero.libero_convention()
    eval_libero._assert_server_speaks_the_same_dialect(
        served_metadata(image_keys=tuple(convention.image_keys))
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("image_keys", ["observation/images/base", "observation/images/wrist"]),
        ("state_key", "observation/qpos"),
        ("prompt_key", "task"),
    ],
)
def test_mismatched_field_fails_at_connect(field, value):
    with pytest.raises(RuntimeError) as error:
        eval_libero._assert_server_speaks_the_same_dialect(
            served_metadata(**{field: value})
        )
    assert field in str(error.value)
    assert eval_libero.LIBERO_PRESET in str(error.value)


def test_unpublished_fields_are_not_asserted():
    """A non-ApxInf OpenPI server publishes less; that is not a known mismatch."""
    eval_libero._assert_server_speaks_the_same_dialect({"precision": "bf16"})

"""The robot transforms against ``fixtures/robot_golden.npz``.

The fixture was generated from the pre-refactor ApxInf ``ProcessorStep`` chain
(``fixtures/make_robot_golden.py``, run at commit 55760e2), so it is the oracle: values,
shapes and dtypes. These tests need no engine and run with ApxInf blocked.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

GOLDEN = np.load(pathlib.Path(__file__).with_name("fixtures") / "robot_golden.npz")
KEY = "state"


def test_the_fixture_is_present_and_complete():
    names = set(GOLDEN.files)
    for body in ("g1/f32", "g1/f64", "g1/i64", "g1/clip", "yam/f64", "yam/f32",
                 "dual_franka/f64", "dual_franka/f32", "g1_policy"):
        assert any(n.startswith(body + "/") for n in names), body


def _same(actual, expected):
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


# --- the engine-neutral transforms, which need no engine to import --------------------
#
# Everything below must run with ApxInf blocked (APXINF_ROBO_FORBID_IMPORT=apxinf).


def _golden_g1(case):
    return lambda tag: GOLDEN[f"g1/{case}/{tag}"]


def _same_up_to_matmul(actual, expected):
    """Exact dtype and shape; values to a few ulp of float64.

    Only the dual-Franka composition uses ``@``. Its float64 result is the same
    mathematics through BLAS, and the kernel (hence the last bit) differs between numpy
    builds: numpy 1.26 here gives bit-identical values to the fixture, a CI machine on
    numpy 2.5 differs by ~4e-16. The element-wise G1 and YAM paths stay bit-exact above.
    """
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["f32", "f64", "i64", "clip"])
def test_g1_transform_reproduces_golden(case):
    from apxinf_robo.embodiments.unitree_g1.transform import (
        G1Transform,
        absolute_actions,
        decode_state,
        encode_actions,
    )

    g = _golden_g1(case)
    state, raw = g("state"), g("raw")
    original = (state.copy(), raw.copy())
    _same(decode_state(state), g("decoded"))
    _same(absolute_actions(raw, g("decoded")), g("absolute"))
    _same(encode_actions(g("absolute")), g("encoded"))

    transform = G1Transform(KEY)
    observation = {KEY: state, "prompt": "p"}
    prepared = transform.before(observation)
    _same(prepared[KEY], g("decoded"))
    assert prepared["prompt"] == "p"
    _same(transform.after(raw, prepared), g("encoded"))
    # Neither the client's observation nor the engine's actions are modified.
    np.testing.assert_array_equal(state, original[0])
    np.testing.assert_array_equal(raw, original[1])
    assert observation[KEY] is state


def test_g1_transform_variants():
    from apxinf_robo.embodiments.unitree_g1.transform import G1Transform

    state, raw = GOLDEN["g1/f32/state"], GOLDEN["g1/f32/raw"]
    # Raw robot space: no decode, no encode.
    transform = G1Transform(KEY, adapt_to_pi=False, use_delta_joint_actions=False)
    obs = {KEY: state}
    assert transform.before(obs) is obs
    assert transform.after(raw, obs) is raw
    # Without state a delta cannot be resolved: the action is left as the model made it.
    stateless = G1Transform(KEY)
    assert stateless.before({}) == {}
    np.testing.assert_array_equal(
        stateless.after(raw, {}), G1Transform(KEY, use_delta_joint_actions=False).after(raw, {})
    )


def test_g1_transform_nested_state_key():
    from apxinf_robo.embodiments.unitree_g1.transform import G1Transform

    obs = {"robot": {"state": GOLDEN["g1/f32/state"]}}
    prepared = G1Transform("robot/state").before(obs)
    _same(prepared["robot"]["state"], GOLDEN["g1/f32/decoded"])


class _ConstantPolicy:
    """Stands in for a loaded engine policy: a constant action chunk, and a record."""

    metadata = {"protocol": "openpi.websocket_policy"}
    action_dim = 32
    action_horizon = 4

    def __init__(self, value):
        self.value, self.seen, self.closed = value, [], 0

    def infer(self, observation, *, noise=None):
        self.seen.append(observation)
        return {
            "actions": self.value.copy(),
            "normalized_actions": "passthrough",
            "timing": {"model_ms": 1.0, "total_ms": 2.0},
        }

    def close(self):
        self.closed += 1


def test_g1_wrapper_matches_the_with_adapter_chain():
    """The old route's end-to-end result (golden ``g1_policy``) from the new route."""
    from apxinf_robo.embodiments.unitree_g1.transform import G1_ROBOT_DIM, G1Transform
    from apxinf_robo.protocol import TransformedPolicy

    inner = _ConstantPolicy(GOLDEN["g1_policy/model_actions"])
    policy = TransformedPolicy(inner, G1Transform(KEY), action_dim=G1_ROBOT_DIM)
    state = GOLDEN["g1_policy/state"]
    result = policy.infer({KEY: state, "prompt": "p"})
    _same(result["actions"], GOLDEN["g1_policy/actions"])
    # The model saw the decoded state: what the old chain's tokenizer was handed.
    np.testing.assert_array_equal(inner.seen[0][KEY], GOLDEN["g1_policy/tokenizer_state"])
    assert result["normalized_actions"] == "passthrough"


@pytest.mark.parametrize("case", ["f64", "f32"])
def test_yam_transform_reproduces_golden(case):
    from apxinf_robo.embodiments.yam_transform import YamTransform

    state, raw = GOLDEN[f"yam/{case}/state"], GOLDEN[f"yam/{case}/raw"]
    kept = raw.copy()
    transform = YamTransform(KEY)
    obs = {KEY: state}
    assert transform.before(obs) is obs
    _same(transform.after(raw, obs), GOLDEN[f"yam/{case}/out"])
    np.testing.assert_array_equal(raw, kept)


@pytest.mark.parametrize("case", ["f64", "f32"])
def test_dual_franka_transform_reproduces_golden(case):
    from apxinf_robo.embodiments.dual_franka_transform import DualFrankaTransform

    state, raw = GOLDEN[f"dual_franka/{case}/state"], GOLDEN[f"dual_franka/{case}/raw"]
    kept = (state.copy(), raw.copy())
    transform = DualFrankaTransform(KEY)
    prepared = transform.before({KEY: state})
    _same(prepared[KEY], GOLDEN[f"dual_franka/{case}/reordered"])
    _same_up_to_matmul(transform.after(raw, prepared), GOLDEN[f"dual_franka/{case}/out"])
    np.testing.assert_array_equal(state, kept[0])
    np.testing.assert_array_equal(raw, kept[1])


def test_yam_and_dual_franka_wrappers_over_a_fake_engine():
    from apxinf_robo.embodiments.dual_franka_transform import DualFrankaTransform
    from apxinf_robo.embodiments.yam_transform import YamTransform
    from apxinf_robo.protocol import TransformedPolicy

    for body, transform, width in (("yam", YamTransform(KEY), 14),
                                   ("dual_franka", DualFrankaTransform(KEY), 20)):
        raw = GOLDEN[f"{body}/f64/raw"]
        inner = _ConstantPolicy(raw)
        policy = TransformedPolicy(inner, transform, action_dim=width)
        result = policy.infer({KEY: GOLDEN[f"{body}/f64/state"]})
        compare = _same_up_to_matmul if body == "dual_franka" else _same
        compare(result["actions"], GOLDEN[f"{body}/f64/out"])
        # The model is handed the transformed state: reordered for dual-Franka.
        seen = inner.seen[0][KEY]
        np.testing.assert_array_equal(
            seen, GOLDEN["dual_franka/f64/reordered"] if body == "dual_franka"
            else GOLDEN["yam/f64/state"])


def test_yam_and_dual_franka_keep_their_input_checks():
    from apxinf_robo.embodiments.dual_franka_transform import DualFrankaTransform
    from apxinf_robo.embodiments.yam_transform import YamTransform

    with pytest.raises(ValueError, match="14-value"):
        YamTransform(KEY).after(np.zeros((30, 32)), {KEY: np.zeros(13)})
    with pytest.raises(ValueError, match="action chunk"):
        YamTransform(KEY).after(np.zeros((30, 13)), {KEY: np.zeros(14)})
    with pytest.raises(KeyError):  # never ships unresolved deltas
        YamTransform(KEY).after(np.zeros((30, 32)), {})
    with pytest.raises(ValueError, match="20-value"):
        DualFrankaTransform(KEY).before({KEY: np.zeros(19)})
    with pytest.raises(ValueError, match="action chunk"):
        DualFrankaTransform(KEY).after(np.zeros((2, 19)), {KEY: np.ones(20)})

"""The robot transforms against ``fixtures/robot_golden.npz``.

The fixture was generated from the pre-refactor ApxInf ``ProcessorStep`` chain
(``fixtures/make_robot_golden.py``), so it is the oracle: values, shapes and dtypes.
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


@pytest.mark.parametrize("case", ["f32", "f64", "i64", "clip"])
def test_g1_steps_reproduce_golden(case):
    from apxinf.processors.transforms import ACTIONS, OBSERVATION

    from apxinf_robo.embodiments.unitree_g1.steps import (
        UnitreeG1AbsoluteActions,
        UnitreeG1DecodeState,
        UnitreeG1EncodeActions,
    )

    g = lambda tag: GOLDEN[f"g1/{case}/{tag}"]  # noqa: E731
    data = UnitreeG1DecodeState(KEY)({OBSERVATION: {KEY: g("state")}})
    _same(data[OBSERVATION][KEY], g("decoded"))
    absolute = UnitreeG1AbsoluteActions(KEY)({OBSERVATION: data[OBSERVATION], ACTIONS: g("raw")})
    _same(absolute[ACTIONS], g("absolute"))
    _same(UnitreeG1EncodeActions()({ACTIONS: absolute[ACTIONS]})[ACTIONS], g("encoded"))


# --- the engine-neutral transforms, which need no engine to import --------------------
#
# Everything below must run with ApxInf blocked (APXINF_ROBO_FORBID_IMPORT=apxinf).


def _golden_g1(case):
    return lambda tag: GOLDEN[f"g1/{case}/{tag}"]


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

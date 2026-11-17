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

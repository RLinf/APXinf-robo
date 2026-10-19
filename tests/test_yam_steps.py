"""Joint-space recovery at the YAM controller boundary."""

import numpy as np
import pytest

from apxinf_robo.embodiments.yam import YamAbsoluteActions


def test_yam_restores_each_arm_relative_to_observation_without_accumulating():
    state = np.arange(14, dtype=np.float64) / 10
    raw = np.full((30, 32), 0.25, dtype=np.float64)
    raw[:, 6] = 0.2
    raw[:, 13] = 0.8
    original = raw.copy()
    data = {"observation": {"state": state}, "actions": raw}
    result = YamAbsoluteActions("state")(data)["actions"]
    assert result.shape == (30, 14)
    np.testing.assert_allclose(result[:, :6], np.tile(state[:6] + 0.25, (30, 1)))
    np.testing.assert_allclose(result[:, 7:13], np.tile(state[7:13] + 0.25, (30, 1)))
    np.testing.assert_array_equal(result[:, 6], 0.2)
    np.testing.assert_array_equal(result[:, 13], 0.8)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize(
    "state", [np.zeros(13), np.zeros((1, 14)), np.full(14, np.nan)]
)
def test_yam_rejects_invalid_state(state):
    with pytest.raises(ValueError, match="14-value"):
        YamAbsoluteActions("state")(
            {"observation": {"state": state}, "actions": np.zeros((30, 32))}
        )


@pytest.mark.parametrize(
    "actions", [np.zeros((30, 13)), np.zeros(14), np.full((30, 32), np.inf)]
)
def test_yam_rejects_invalid_actions(actions):
    with pytest.raises(ValueError, match="action chunk"):
        YamAbsoluteActions("state")(
            {"observation": {"state": np.zeros(14)}, "actions": actions}
        )


def test_yam_missing_state_never_sends_unresolved_deltas():
    with pytest.raises(KeyError):
        YamAbsoluteActions("state")({"observation": {}, "actions": np.zeros((30, 32))})


def test_yam_nested_observation_keys():
    result = YamAbsoluteActions("robot/state")(
        {
            "observation": {"robot": {"state": np.ones(14)}},
            "actions": np.zeros((2, 14)),
        }
    )["actions"]
    np.testing.assert_array_equal(result[0], [1] * 6 + [0] + [1] * 6 + [0])

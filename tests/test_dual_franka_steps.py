"""Controller semantics for dual-Franka body-frame policy actions."""

import numpy as np
import pytest

from apxinf_robo.embodiments.dual_franka import (
    DualFrankaAbsoluteActions,
    DualFrankaState,
)

IDENTITY = [1, 0, 0, 0, 1, 0]
Z90 = [0, 1, 0, -1, 0, 0]


def test_rotated_body_translation_and_independent_arms():
    wire = np.array([0.1, 0.9, 1, 2, 3, *Z90, 4, 5, 6, *IDENTITY], dtype=float)
    action = np.array([1, 0, 0, *Z90, 0.3, 0, 2, 0, *IDENTITY, 0.7], dtype=float)
    data = {"observation": {"state": wire}, "actions": np.tile(action, (20, 1))}
    data = DualFrankaState("state")(data)
    assert data["observation"]["state"][9] == 0.1
    assert data["observation"]["state"][19] == 0.9
    result = DualFrankaAbsoluteActions("state")(data)["actions"]
    expected = [1, 3, 3, -1, 0, 0, 0, -1, 0, 0.3, 4, 7, 6, *IDENTITY, 0.7]
    np.testing.assert_allclose(result, np.tile(expected, (20, 1)), atol=1e-7)
    np.testing.assert_array_equal(wire[:2], [0.1, 0.9])


@pytest.mark.parametrize("rotation", [[0] * 6, [1, 0, 0, 2, 0, 0]])
def test_degenerate_prediction_is_rejected(rotation):
    state = np.array([0, 0, 0, *IDENTITY, 0, 0, 0, 0, *IDENTITY, 0], dtype=float)
    action = state.copy()
    action[3:9] = rotation
    with pytest.raises(ValueError, match="Degenerate"):
        DualFrankaAbsoluteActions("state")(
            {"observation": {"state": state}, "actions": action[None]}
        )

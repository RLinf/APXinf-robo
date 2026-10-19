"""Exercise robot builders around the engine's actual Pi05 pipelines on CPU."""

import numpy as np
import pytest

from apxinf import Pi05Policy
from apxinf.processors import Normalizer, Unnormalizer
from apxinf_robo import engine
from apxinf_robo.presets import build_robot_policy, get_robot_preset
from test_robot_presets import MockModel, RecordingTokenizer


@pytest.mark.parametrize(
    "robot,width", [("yam_joint", 14), ("dual_franka_tcp_rot6d", 20)]
)
def test_model_pipeline_preserves_state_for_absolute_action_recovery(
    monkeypatch, robot, width
):
    preset = get_robot_preset(robot)
    model = MockModel(num_views=3)
    predicted = np.zeros((model.action_horizon, model.action_dim), dtype=np.float32)
    if width == 20:
        predicted[:, 3:9] = [1, 0, 0, 0, 1, 0]
        predicted[:, 13:19] = [1, 0, 0, 0, 1, 0]
        state = np.array(
            [0.2, 0.8, 1, 2, 3, 1, 0, 0, 0, 1, 0, 4, 5, 6, 1, 0, 0, 0, 1, 0],
            dtype=np.float32,
        )
        expected = np.concatenate([state[2:11], [0], state[11:20], [0]])
    else:
        state = np.arange(14, dtype=np.float32) / 10
        expected = state.copy()
        expected[[6, 13]] = 0
    model.infer_rgb = lambda *args, **kwargs: predicted.copy()
    tokenizer = RecordingTokenizer()
    tokenizer.discrete_state = preset.discrete_state
    state_width = width
    # Non-identity state normalization makes reading the tokenized/normalized
    # state instead of the physical observation produce a visibly wrong pose.
    inputs, outputs = Pi05Policy.default_pipelines(
        model,
        tokenizer=tokenizer,
        unnormalizer=Unnormalizer(q01=[-1] * 32, q99=[1] * 32, eps=0),
        state_normalizer=Normalizer(
            q01=[-10] * state_width, q99=[10] * state_width, eps=0
        ),
        image_keys=preset.image_keys,
        state_key=preset.state_key,
    )
    base = Pi05Policy(
        model,
        input_pipeline=inputs,
        output_pipeline=outputs,
        image_keys=preset.image_keys,
        state_key=preset.state_key,
        metadata={"state_dim": state_width},
    )
    monkeypatch.setattr(engine, "load_policy", lambda *args, **kwargs: base)
    policy = build_robot_policy(robot, "unused")
    obs = {
        key: np.full((16, 16, 3), i * 40, dtype=np.uint8)
        for i, key in enumerate(preset.image_keys)
    }
    obs.update(state=state, prompt="pick the cup")
    result = policy.infer(obs)["actions"]
    np.testing.assert_allclose(
        result, np.tile(expected, (model.action_horizon, 1)), atol=1e-6
    )
    assert policy.action_dim == width
    assert (tokenizer.seen_states[-1] is None) == (robot == "dual_franka_tcp_rot6d")
    np.testing.assert_array_equal(obs["state"], state)

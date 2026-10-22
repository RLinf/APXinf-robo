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
    assert policy.metadata["state_dim"] == width
    assert (tokenizer.seen_states[-1] is None) == (robot == "dual_franka_tcp_rot6d")
    np.testing.assert_array_equal(obs["state"], state)


@pytest.mark.parametrize("model_width,stats_width", [(32, 20), (32, 24), (32, 32), (64, 64)])
def test_dual_franka_loads_matching_effective_statistics(
    monkeypatch, tmp_path, model_width, stats_width
):
    """Read real stats from disk; only model execution and tokenization are mocked."""
    import json
    from apxinf.policies.impls import pi05

    preset = get_robot_preset("dual_franka_tcp_rot6d")
    model = MockModel(num_views=3)
    model.action_dim = model_width
    predicted = np.zeros((model.action_horizon, model_width), dtype=np.float32)
    model.infer_rgb = lambda *args, **kwargs: predicted.copy()
    # Zero normalized actions decode to this non-identity physical delta.
    midpoint = np.zeros(stats_width)
    midpoint[:preset.action_dim] = [
        0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.4,
        0.5, 0.6, 0.7, 1, 0, 0, 0, 1, 0, 0.8,
    ]
    midpoint[preset.action_dim:] = 1234  # Padding must not reach the controller.
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": {
        "actions": {"q01": (midpoint - 1).tolist(), "q99": (midpoint + 1).tolist()}
    }}))
    monkeypatch.setattr(pi05, "_resolve_tokenizer", lambda *args: "unused")
    monkeypatch.setattr(pi05, "PromptTokenizer", lambda *args, **kwargs: RecordingTokenizer())
    monkeypatch.setattr(engine, "load_policy", lambda path, **kwargs:
                        Pi05Policy.from_pretrained(path, model_runner=model, **kwargs))
    policy = build_robot_policy(preset.name, tmp_path)
    state = np.array([0, 0, 1, 2, 3, 1, 0, 0, 0, 1, 0,
                      4, 5, 6, 1, 0, 0, 0, 1, 0], dtype=np.float64)
    observation = {key: np.zeros((16, 16, 3), dtype=np.uint8) for key in preset.image_keys}
    observation.update(state=state, prompt="move")
    result = policy.infer(observation)["actions"]
    expected = midpoint[:preset.action_dim].copy()
    expected[:3] += state[2:5]
    expected[10:13] += state[11:14]
    np.testing.assert_allclose(result, np.tile(expected, (model.action_horizon, 1)), atol=1e-5)
    assert result.shape == (model.action_horizon, preset.action_dim)

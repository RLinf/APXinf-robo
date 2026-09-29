"""Exercise the public serving and LIBERO seams with family-shaped inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf_robo.cli import eval_libero, serve
from apxinf_robo import check_checkpoint


@pytest.mark.parametrize("model_type", ["pi0_fast", "gr00t"])
def test_serve_uses_registered_policy_loader(monkeypatch, tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    captured = {}
    policy = SimpleNamespace(action_horizon=10, action_dim=7, close=lambda: None)
    monkeypatch.setattr(
        serve, "build_robot_policy",
        lambda robot, model_dir, **options: captured.update(
            robot=robot, model_dir=model_dir, options=options
        ) or policy,
    )
    monkeypatch.setattr(
        serve, "websocket_server",
        lambda *args: SimpleNamespace(serve_forever=lambda: captured.update(served=True)),
    )
    args = serve.build_parser().parse_args(
        ["--model-dir", str(tmp_path), "--robot", "franka_libero", "--precision", "bf16"]
    )
    serve.run(args)
    assert captured["served"]
    assert captured["options"]["model_type"] == model_type
    assert captured["options"]["precision"] == "bf16"
    assert captured["options"]["metadata"]["model_type"] == model_type
    assert "checkpoint_format" not in captured["options"]
    assert "norm_key" not in captured["options"]


def test_qwen_drive_serves_without_robot_preset(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen_drive"}))
    captured = {}
    policy = SimpleNamespace(action_horizon=50, action_dim=3, close=lambda: None)
    monkeypatch.setattr(
        serve, "load_policy",
        lambda model_dir, **options: captured.update(options=options) or policy,
    )
    monkeypatch.setattr(
        serve, "websocket_server",
        lambda *args: SimpleNamespace(serve_forever=lambda: captured.update(served=True)),
    )
    args = serve.build_parser().parse_args(
        ["--model-dir", str(tmp_path), "--robot", "none", "--policy-options",
         '{"planner": "/ckpt/planner", "mode": "direct_planning"}']
    )
    serve.run(args)
    assert captured["served"]
    assert captured["options"]["model_type"] == "qwen_drive"
    assert captured["options"]["planner"] == "/ckpt/planner"


def test_registered_checkpoint_preflight_uses_its_own_layout(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi0_fast"}))
    for name in (
        "model.safetensors",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (tmp_path / name).touch()
    findings = check_checkpoint(tmp_path, "franka_libero")
    assert not any(item.level == "FAIL" for item in findings)
    assert any(item.check == "policy resources" and item.level == "WARN" for item in findings)


@pytest.mark.parametrize("model_type", ["pi0_fast", "gr00t"])
def test_libero_rollout_sends_family_state_and_action(monkeypatch, model_type):
    monkeypatch.setattr(eval_libero, "WAIT_STEPS", 0)
    monkeypatch.setattr(eval_libero, "MAX_STEPS", 1)
    raw = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
    }

    class Env:
        action = None

        def reset(self):
            return raw

        def set_init_state(self, unused):
            return raw

        def step(self, action):
            self.action = action
            return raw, 0, True, {}

    class Backend:
        def __init__(self):
            self.model_type = model_type
            self.state = None

        def infer(self, base, wrist, state, prompt, noise=None):
            self.state = state
            actions = np.zeros((5, 7), dtype=np.float32)
            actions[:, -1] = 0.0
            return actions, actions, {}

    env, backend = Env(), Backend()
    result = eval_libero.run_episode(
        env, np.zeros(1), "libero_10", 0, 0, "pick up the block", backend,
        "in_process_api", 0, False, 0.5,
    )
    assert result["success"] and result["action_steps"] == 1
    if model_type == "pi0_fast":
        assert backend.state.shape == (8,)
        assert env.action[-1] == 0.0
    else:
        assert set(backend.state) == {"x", "y", "z", "roll", "pitch", "yaw", "gripper"}
        assert env.action[-1] == 1.0

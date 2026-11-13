"""Robo restates ApxInf's shared vocabulary so it can import without ApxInf.

These tests are what keeps the copy honest, and what shows that a policy of either
engine satisfies the shape the layers above are written against.
"""

from __future__ import annotations

import pytest

from apxinf_robo import build_robot_policy, engine_drobotics, protocol


def test_vocabulary_matches_apxinf():
    base = pytest.importorskip("apxinf.policies.base")
    assert protocol.VIEW_SLOTS == base.VIEW_SLOTS
    assert protocol.CANONICAL_IMAGE_KEYS == base.CANONICAL_IMAGE_KEYS
    assert protocol.CANONICAL_STATE_KEY == base.CANONICAL_STATE_KEY
    assert protocol.CANONICAL_PROMPT_KEY == base.CANONICAL_PROMPT_KEY


def test_policy_shape_matches_apxinf():
    base = pytest.importorskip("apxinf.policies.base")
    for name in ("action_dim", "action_horizon", "infer", "close"):
        assert hasattr(base.Policy, name) and hasattr(protocol.Policy, name)


def test_drobotics_policy_satisfies_the_shape(monkeypatch, tmp_path):
    testing = pytest.importorskip("apx_drobotics.testing")
    fake = testing.fake_deployment(tmp_path)
    load = engine_drobotics.load_policy
    monkeypatch.setattr(engine_drobotics, "load_policy",
                        lambda model_dir, **options: load(model_dir, **options, **fake.options))
    policy = build_robot_policy("franka_libero", fake.sdk_root, engine="drobotics")
    try:
        assert isinstance(policy, protocol.Policy)
        assert policy.metadata["robot"] == "franka_libero"
        assert policy.metadata["robot_steps"] is False
        assert policy.metadata["image_keys"] == ["observation/image", "observation/wrist_image"]
        assert policy.metadata["state_key"] == "observation/state"
        assert policy.metadata["state_dim"] == 8
    finally:
        policy.close()


def test_preset_with_robot_steps_cannot_use_the_drobotics():
    pytest.importorskip("apxinf.policies.base")  # unitree_g1 exists only with ApxInf
    with pytest.raises(ValueError, match="does not have"):
        build_robot_policy("unitree_g1", None, engine="drobotics")


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown engine"):
        build_robot_policy("franka_libero", None, engine="tpu")


def test_missing_package_gives_the_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "apx_drobotics":
            raise ImportError("no apx_drobotics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match="pip install -e"):
        engine_drobotics.require_apx_drobotics()

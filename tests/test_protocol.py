"""Robo restates ApxInf's shared vocabulary so it can import without ApxInf.

These tests are what keeps the copy honest, and what shows that a policy of either
engine satisfies the shape the layers above are written against.
"""

from __future__ import annotations

import pytest

from _require import needs_apxinf, needs_drobotics
from apxinf_robo import build_robot_policy, engine_drobotics, protocol


@needs_apxinf
def test_vocabulary_matches_apxinf():
    from apxinf.policies import base
    assert protocol.VIEW_SLOTS == base.VIEW_SLOTS
    assert protocol.CANONICAL_IMAGE_KEYS == base.CANONICAL_IMAGE_KEYS
    assert protocol.CANONICAL_STATE_KEY == base.CANONICAL_STATE_KEY
    assert protocol.CANONICAL_PROMPT_KEY == base.CANONICAL_PROMPT_KEY


@needs_apxinf
def test_policy_shape_matches_apxinf():
    from apxinf.policies import base
    for name in ("action_dim", "action_horizon", "infer", "close"):
        assert hasattr(base.Policy, name) and hasattr(protocol.Policy, name)


@needs_drobotics
def test_drobotics_policy_satisfies_the_shape(monkeypatch, tmp_path):
    from apx_drobotics import testing

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


def _fake_s600(monkeypatch, tmp_path):
    from apx_drobotics import testing

    fake = testing.fake_deployment(tmp_path)
    load = engine_drobotics.load_policy
    monkeypatch.setattr(engine_drobotics, "load_policy",
                        lambda model_dir, **options: load(model_dir, **options, **fake.options))
    return testing, fake


@needs_drobotics
def test_a_robot_preset_is_refused_by_the_contract_not_by_its_transform(monkeypatch, tmp_path):
    """G1 has 16 state values and three cameras; this deployment has 8 and two.

    The refusal comes from the deployment's own contract, and names what disagrees. It is
    not "robot steps are unsupported": the transform is engine-neutral.
    """
    _, fake = _fake_s600(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cannot serve this configuration on the S600") as caught:
        build_robot_policy("unitree_g1", fake.sdk_root, engine="drobotics")
    assert "state_dim=16" in str(caught.value)


@needs_drobotics
def test_a_compatible_robot_transform_runs_on_the_s600_engine(monkeypatch, tmp_path):
    """The same wrapper that serves ApxInf runs around the S600 policy, in order."""
    import numpy as np

    from apxinf_robo import conventions
    from apxinf_robo.embodiments.base import (
        ROBOT_PRESETS,
        Embodiment,
        RobotPreset,
    )

    testing, fake = _fake_s600(monkeypatch, tmp_path)
    seen = []

    class PlusOne:
        min_input_dim = 7
        output_dim = None

        def __init__(self, state_key):
            self.state_key = state_key

        def before(self, observation):
            seen.append(("before", observation[self.state_key].copy()))
            return {**observation, self.state_key: observation[self.state_key] + 5}

        def after(self, actions, observation):
            seen.append(("after", observation[self.state_key].copy()))  # what the model saw
            return actions + 1

    body = Embodiment(name="franka_plus", num_cameras=2, action_dim=7, state_dim=8,
                      action_width=7, transform=PlusOne)
    preset = RobotPreset(name="franka_plus_one", embodiment=body, convention=conventions.LIBERO,
                         discrete_state=None)
    ROBOT_PRESETS[preset.name] = preset
    try:
        policy = build_robot_policy("franka_plus_one", fake.sdk_root, engine="drobotics")
    finally:
        ROBOT_PRESETS.pop(preset.name)
    try:
        assert isinstance(policy, protocol.TransformedPolicy)
        assert policy.metadata["robot_steps"] is True
        assert policy.action_dim == 7
        image = np.zeros((256, 256, 3), np.uint8)
        state = np.arange(8, dtype=np.float32)
        result = policy.infer({"observation/image": image, "observation/wrist_image": image,
                               "observation/state": state, "prompt": "pick up the soup"})
        np.testing.assert_allclose(result["actions"], testing.EXPECTED_ACTIONS + 1)
        assert [name for name, _ in seen] == ["before", "after"]
        np.testing.assert_array_equal(seen[0][1], state)
        np.testing.assert_array_equal(seen[1][1], state + 5)  # after sees the transformed state
        assert "robot_ms" in result["timing"] and "model_ms" in result["timing"]
    finally:
        policy.close()


def _fake_workers():
    """Pids of running fake S600 workers, matched on argv so a shell that merely names one is not."""
    import os
    import time

    time.sleep(0.2)  # a closed worker needs a moment to be reaped
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if len(argv) > 1 and argv[1].endswith(b"fake_worker.py"):
            found.append(int(pid))
    return found


@needs_drobotics
def test_a_failed_wrap_stops_the_s600_worker(monkeypatch, tmp_path):
    """A transform that cannot take the engine's width must not leave a worker running."""
    from apxinf_robo import conventions
    from apxinf_robo.embodiments.base import ROBOT_PRESETS, Embodiment, RobotPreset

    _, fake = _fake_s600(monkeypatch, tmp_path)

    class Wide:
        min_input_dim = 9
        output_dim = None

        def __init__(self, state_key):
            pass

    body = Embodiment(name="franka_wide", num_cameras=2, action_dim=7, state_dim=8,
                      action_width=7, transform=Wide)
    preset = RobotPreset(name="franka_wide", embodiment=body, convention=conventions.LIBERO)
    ROBOT_PRESETS[preset.name] = preset
    assert _fake_workers() == []
    try:
        with pytest.raises(ValueError, match="at least 9 wide"):
            build_robot_policy("franka_wide", fake.sdk_root, engine="drobotics")
    finally:
        ROBOT_PRESETS.pop(preset.name)
    assert _fake_workers() == []


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


@needs_drobotics
def test_the_worker_check_can_see_a_live_worker(monkeypatch, tmp_path):
    """Control for the test above: the same check does find a worker that is running."""
    _, fake = _fake_s600(monkeypatch, tmp_path)
    policy = build_robot_policy("franka_libero", fake.sdk_root, engine="drobotics")
    try:
        assert len(_fake_workers()) == 1
    finally:
        policy.close()
    assert _fake_workers() == []

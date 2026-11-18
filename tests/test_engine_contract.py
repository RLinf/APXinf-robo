"""What every engine's policy must do, checked on both, bare and with a robot transform.

``isinstance(policy, Policy)`` only proves the attribute names exist. These tests drive
real policies: ApxInf's ``Pi05Policy`` over a mock model, and ``apx_drobotics``'s
``DroboticsPolicy`` over its fake worker. Each case body is shared, so a behaviour the
layers above rely on (the shape of ``actions``, the timing keys, the published metadata,
how an engine failure reaches the caller, that ``close()`` releases what the engine holds)
is asserted on both engines or on neither.

The two halves need different packages, and CI deliberately installs them in separate
jobs, so each half skips where its engine is absent. The ``engine-contract`` job installs
both, which is the only place the parametrization is whole.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from _require import installed, needs_apxinf, needs_drobotics

from apxinf_robo import conventions, protocol
from apxinf_robo.embodiments.base import ROBOT_PRESETS, Embodiment, RobotPreset
from apxinf_robo.presets import build_robot_policy

KEYS = ("observation/image", "observation/wrist_image")
STATE_KEY = "observation/state"


class PlusOne:
    """A robot transform with an effect on both sides, so wrapping is observable."""

    min_input_dim = 7
    output_dim = None

    def __init__(self, state_key):
        self.state_key = state_key

    def before(self, observation):
        return {**observation, self.state_key: np.asarray(observation[self.state_key]) + 1}

    def after(self, actions, observation):
        return np.asarray(actions) + 1


def _observation():
    image = np.full((256, 256, 3), 7, np.uint8)
    return {KEYS[0]: image, KEYS[1]: image, STATE_KEY: np.zeros(8, np.float32),
            "prompt": "pick up the soup"}


def _fake_workers() -> list:
    """Pids of running fake S600 workers, matched on argv."""
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


class Case:
    """One engine, with the few facts that differ and a way to build its policy."""

    def __init__(self, name, build, horizon, dim, base_actions, failure):
        self.name, self.build = name, build
        self.horizon, self.dim, self.base_actions, self.failure = horizon, dim, base_actions, failure
        self.closed = lambda: True


def _apxinf_case(tmp_path, monkeypatch) -> Case:
    from apxinf.policies import Pi05Policy
    from apxinf.processors import Unnormalizer
    from test_robot_presets import HORIZON, MODEL_DIM, MockModel, RecordingTokenizer

    class Model(MockModel):
        fail = False
        closes = 0

        def infer_rgb(self, rgb_u8, layout, token_ids, noise):
            if self.fail:
                raise RuntimeError("boom")
            return super().infer_rgb(rgb_u8, layout, token_ids, noise)

        def close(self):
            type(self).closes += 1

    model = Model(num_views=2, value=0.25)
    identity = Unnormalizer(q01=[-1.0] * MODEL_DIM, q99=[1.0] * MODEL_DIM, dims=MODEL_DIM, eps=0.0)
    tokenizer = RecordingTokenizer()
    tokenizer.discrete_state = False
    input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
        model, tokenizer=tokenizer, unnormalizer=identity, image_keys=KEYS, state_key=STATE_KEY)

    def fake_load(model_dir, **kwargs):
        # The loader returns a fresh policy over the shared mock model, as a real one would.
        return Pi05Policy(model, input_pipeline=input_pipeline, output_pipeline=output_pipeline,
                          image_keys=KEYS, state_key=STATE_KEY, metadata=kwargs.get("metadata"))

    from apxinf_robo import engine

    monkeypatch.setattr(engine, "load_policy", fake_load)

    def build(wrapped, *, fail=False):
        model.fail = fail
        return _build("apxinf", None, wrapped)

    case = Case("apxinf", build, HORIZON, MODEL_DIM,
                base_actions=np.full((HORIZON, MODEL_DIM), 0.25, np.float32),
                failure="boom")
    return case


def _drobotics_case(tmp_path, monkeypatch) -> Case:
    from apx_drobotics import testing

    from apxinf_robo import engine_drobotics

    state = {"mode": "ok", "n": 0}
    load = engine_drobotics.load_policy

    def fresh_loader(model_dir, **options):
        state["n"] += 1
        root = tmp_path / f"deployment-{state['n']}"
        root.mkdir()
        fake = testing.fake_deployment(root, state["mode"])
        return load(fake.sdk_root, **options, **fake.options)

    monkeypatch.setattr(engine_drobotics, "load_policy", fresh_loader)

    def build(wrapped, *, fail=False):
        state["mode"] = "error" if fail else "ok"
        return _build("drobotics", None, wrapped)

    case = Case("drobotics", build, 10, 7, base_actions=testing.EXPECTED_ACTIONS,
                failure="runtime error: boom")
    case.closed = lambda: _fake_workers() == []
    return case


def _build(engine, model_dir, wrapped):
    if not wrapped:
        return build_robot_policy("franka_libero", model_dir, engine=engine,
                                  **({"discrete_state": False} if engine == "apxinf" else {}))
    body = Embodiment(name="franka_plus", num_cameras=2, action_dim=7, state_dim=8,
                      action_width=7, transform=PlusOne)
    preset = RobotPreset(name="franka_plus_one", embodiment=body, convention=conventions.LIBERO)
    ROBOT_PRESETS[preset.name] = preset
    try:
        return build_robot_policy(preset.name, model_dir, engine=engine,
                                  **({"discrete_state": False} if engine == "apxinf" else {}))
    finally:
        ROBOT_PRESETS.pop(preset.name)


@pytest.fixture(
    params=[
        pytest.param("apxinf", marks=needs_apxinf),
        pytest.param("drobotics", marks=needs_drobotics),
    ]
)
def case(request, tmp_path, monkeypatch) -> Case:
    if request.param == "apxinf":
        return _apxinf_case(tmp_path, monkeypatch)
    return _drobotics_case(tmp_path, monkeypatch)


@pytest.fixture(params=[False, True], ids=["bare", "wrapped"])
def wrapped(request) -> bool:
    return request.param


def test_infer_result_shape_timing_and_metadata(case, wrapped):
    policy = case.build(wrapped)
    try:
        result = policy.infer(_observation())
        actions = np.asarray(result["actions"])
        assert actions.shape == (policy.action_horizon, policy.action_dim)
        assert (policy.action_horizon, policy.action_dim) == (case.horizon, case.dim)
        assert np.isfinite(actions).all()
        assert {"model_ms", "total_ms"} <= set(result["timing"])
        # Not ``total >= model``: the fake worker *reports* 6 ms without spending it.
        assert result["timing"]["total_ms"] >= 0 and result["timing"]["model_ms"] >= 0
        meta = policy.metadata
        assert list(meta["image_keys"]) == list(KEYS)
        assert meta["state_key"] == STATE_KEY
        assert meta["action_dim"] == policy.action_dim
        assert meta["action_horizon"] == policy.action_horizon
        assert meta["robot"] == ("franka_plus_one" if wrapped else "franka_libero")
        assert meta["robot_steps"] is wrapped
        assert isinstance(policy, protocol.Policy)
    finally:
        policy.close()


def test_a_transform_changes_exactly_what_it_says(case):
    """Wrapped output is the bare output plus one, on the same observation, on both engines."""
    bare, wrapped = case.build(False), case.build(True)
    try:
        observation = _observation()
        base = np.asarray(bare.infer(observation)["actions"])
        out = np.asarray(wrapped.infer(observation)["actions"])
        np.testing.assert_allclose(out, base + 1)
        assert "robot_ms" in wrapped.infer(observation)["timing"]
        # Only the first seven columns exist on the S600; ApxInf's chunk is the model's width.
        assert out.shape == base.shape
    finally:
        bare.close()
        wrapped.close()


def test_an_engine_failure_reaches_the_caller_and_close_still_works(case, wrapped):
    policy = case.build(wrapped, fail=True)
    try:
        with pytest.raises(Exception, match=case.failure):
            policy.infer(_observation())
    finally:
        policy.close()
    assert case.closed()


def test_close_releases_what_the_engine_holds(case, wrapped):
    policy = case.build(wrapped)
    policy.infer(_observation())
    policy.close()
    assert case.closed()


def test_the_s600_engine_rejects_noise_and_apxinf_accepts_it(case, wrapped):
    """The one documented difference in the call itself; ``build_robot_policy`` says so."""
    policy = case.build(wrapped)
    try:
        noise = np.zeros((case.horizon, case.dim), np.float32)
        if case.name == "drobotics":
            with pytest.raises(Exception, match="noise"):
                policy.infer(_observation(), noise=noise)
        else:
            policy.infer(_observation(), noise=noise)
    finally:
        policy.close()


def test_both_halves_are_installed_where_the_contract_is_whole():
    """Documents which environment makes the parametrization complete (``engine-contract``)."""
    if os.environ.get("APXINF_ROBO_REQUIRE_BOTH"):
        assert installed("apxinf") and installed("apx-drobotics")

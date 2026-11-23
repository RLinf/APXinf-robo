"""``TransformedPolicy``: the seam between a body's arithmetic and any engine's policy."""

from __future__ import annotations

import numpy as np
import pytest

from apxinf_robo.protocol import TransformedPolicy


class Recorder:
    """A transform that records what it was handed and what order it ran in."""

    min_input_dim = 4
    output_dim = None

    def __init__(self, log):
        self.log = log

    def before(self, observation):
        self.log.append(("before", dict(observation)))
        return {**observation, "tag": "prepared"}

    def after(self, actions, observation):
        self.log.append(("after", observation["tag"]))
        return actions[:, :2] + 100


class Inner:
    metadata = {"robot": "inner", "protocol": "x"}
    action_dim = 4
    action_horizon = 3

    def __init__(self, log, *, fail=False):
        self.log, self.fail, self.closed, self.noises = log, fail, 0, []

    def infer(self, observation, *, noise=None):
        self.log.append(("infer", observation["tag"]))
        self.noises.append(noise)
        if self.fail:
            raise RuntimeError("engine failed")
        return {"actions": np.ones((3, 4)), "extra": 7, "timing": {"model_ms": 5.0, "total_ms": 6.0}}

    def close(self):
        self.closed += 1


def test_runs_before_then_engine_then_after_and_passes_the_rest_through():
    log = []
    policy = TransformedPolicy(Inner(log), Recorder(log), action_dim=2, metadata={"robot": "body"})
    result = policy.infer({"state": 1})
    assert [entry[0] for entry in log] == ["before", "infer", "after"]
    assert log[1][1] == log[2][1] == "prepared"  # after sees what before produced
    np.testing.assert_array_equal(result["actions"], np.full((3, 2), 101.0))
    assert result["extra"] == 7
    assert result["timing"]["model_ms"] == 5.0
    assert result["timing"]["total_ms"] >= 6.0
    assert result["timing"]["robot_ms"] == pytest.approx(result["timing"]["total_ms"] - 6.0)
    assert policy.action_dim == 2 and policy.action_horizon == 3
    assert policy.metadata["robot"] == "body" and policy.metadata["protocol"] == "x"
    assert policy.metadata["action_dim"] == 2


def test_noise_is_forwarded_only_when_given():
    inner = Inner([])
    policy = TransformedPolicy(inner, Recorder([]))

    class NoNoise(Inner):
        def infer(self, observation, **kwargs):
            assert kwargs == {}, "an engine that owns its noise must not receive noise="
            return super().infer(observation)

    TransformedPolicy(NoNoise([]), Recorder([])).infer({})
    noise = np.zeros(3)
    policy.infer({}, noise=noise)
    assert inner.noises == [noise]


def test_engine_errors_propagate_and_close_closes_the_engine():
    inner = Inner([], fail=True)
    policy = TransformedPolicy(inner, Recorder([]))
    with pytest.raises(RuntimeError, match="engine failed"):
        policy.infer({})
    policy.close()
    assert inner.closed == 1


def test_a_too_narrow_engine_is_rejected_at_construction():
    inner = Inner([])
    inner.action_dim = 3
    with pytest.raises(ValueError, match="at least 4 wide"):
        TransformedPolicy(inner, Recorder([]))
    assert inner.closed == 0  # the caller owns the engine until construction succeeds


class WithMetadata(Inner):
    """An engine whose result carries its own ``metadata`` (ApxInf's does; S600's does not)."""

    def infer(self, observation, *, noise=None):
        result = super().infer(observation, noise=noise)
        result["metadata"] = {"action_dim": 4, "protocol": "x", "robot": "inner"}
        return result


def test_a_result_metadata_describes_the_wrapped_policy_not_the_engine():
    inner = WithMetadata([])
    policy = TransformedPolicy(inner, Recorder([]), action_dim=2, metadata={"robot": "body"})
    result = policy.infer({})
    assert result["metadata"] == policy.metadata
    assert result["metadata"]["action_dim"] == result["actions"].shape[-1] == 2
    assert result["metadata"]["robot"] == "body"
    assert result["extra"] == 7  # other engine fields still pass through
    # A copy: editing the result changes neither this policy nor the engine's own metadata.
    result["metadata"]["robot"] = "edited"
    assert policy.metadata["robot"] == "body"
    assert inner.metadata == {"robot": "inner", "protocol": "x"} and inner.action_dim == 4


def test_an_engine_that_returns_no_metadata_still_returns_none():
    result = TransformedPolicy(Inner([]), Recorder([])).infer({})
    assert "metadata" not in result

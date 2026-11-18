"""Shared fixtures: real policy code against a fake worker and SDK."""

from __future__ import annotations

import numpy as np
import pytest

from apx_drobotics import DroboticsPolicy
from apx_drobotics.testing import fake_deployment

#: Keys are the caller's. These match no dataset, to show nothing assumes one.
KEYS = {"image_keys": ("cam/base", "cam/wrist"), "state_key": "proprio", "prompt_key": "task"}


def observation(size: int = 224) -> dict:
    rng = np.random.default_rng(1)
    return {
        "cam/base": rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
        "cam/wrist": rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
        "proprio": np.linspace(-1, 1, 8).astype(np.float32),
        "task": "pick up the alphabet soup",
    }


@pytest.fixture
def deployment(tmp_path):
    return fake_deployment(tmp_path)


@pytest.fixture
def make_policy(tmp_path):
    """Build a policy on a fresh fake deployment; returns ``(policy, deployment)``."""
    policies = []

    def make(mode: str = "ok", **options):
        root = tmp_path / f"deployment-{len(policies)}"
        root.mkdir()
        fake = fake_deployment(root, mode)
        policy = DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options, **options)
        policies.append(policy)
        return policy, fake

    yield make
    for policy in policies:
        policy.close()

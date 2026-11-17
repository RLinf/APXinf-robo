"""Regenerate ``robot_golden.npz``: the robot transforms' outputs on fixed inputs.

Run this **only against the pre-refactor commit (55760e2)**, where the transforms are
still ApxInf ``ProcessorStep`` classes wired through ``with_adapter``. The file it writes
is the oracle the engine-neutral transforms are held to afterwards, so regenerating it
from the refactored code would make that comparison vacuous.

    python tests/fixtures/make_robot_golden.py     # needs ApxInf installed

Every array keeps the dtype the original chain produced; ``test_robot_golden`` asserts
dtype as well as values, because the G1 path is deliberately float64 end to end.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("robot_golden.npz")
sys.path.insert(0, str(HERE.parents[1]))  # tests/, for the G1 mock-policy helpers
for entry in ("", str(HERE.parents[2])):  # keep the checkout's apxinf/ namespace off the path
    while entry in sys.path:
        sys.path.remove(entry)

from apxinf.processors.transforms import ACTIONS, OBSERVATION  # noqa: E402

from apxinf_robo.embodiments.dual_franka import (  # noqa: E402
    DualFrankaAbsoluteActions,
    DualFrankaState,
)
from apxinf_robo.embodiments.unitree_g1.steps import (  # noqa: E402
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)
from apxinf_robo.embodiments.yam import YamAbsoluteActions  # noqa: E402

KEY = "state"


def _g1(rng, out):
    cases = {
        "f32": (rng.normal(size=16).astype(np.float32), rng.normal(size=(10, 32)).astype(np.float32)),
        "f64": (rng.normal(size=16), rng.normal(size=(10, 32))),
        # An integer state must not be truncated by the gripper clip.
        "i64": (rng.integers(-3, 4, size=16).astype(np.int64), rng.normal(size=(10, 32)).astype(np.float32)),
        # Grippers outside [0, 1], so the clip is exercised in both directions.
        "clip": (np.linspace(-1.5, 2.5, 16).astype(np.float32), np.linspace(-2, 3, 320).reshape(10, 32).astype(np.float32)),
    }
    for name, (state, raw) in cases.items():
        data = UnitreeG1DecodeState(KEY)({OBSERVATION: {KEY: state}})
        decoded = data[OBSERVATION][KEY]
        after = UnitreeG1AbsoluteActions(KEY)({OBSERVATION: data[OBSERVATION], ACTIONS: raw})
        absolute = after[ACTIONS]
        encoded = UnitreeG1EncodeActions()({ACTIONS: absolute})[ACTIONS]
        for tag, value in (("state", state), ("raw", raw), ("decoded", decoded),
                           ("absolute", absolute), ("encoded", encoded)):
            out[f"g1/{name}/{tag}"] = value


def _yam(rng, out):
    for name, dtype in (("f64", np.float64), ("f32", np.float32)):
        state = rng.normal(size=14).astype(dtype)
        raw = rng.normal(size=(30, 32)).astype(dtype)
        result = YamAbsoluteActions(KEY)({OBSERVATION: {KEY: state}, ACTIONS: raw})[ACTIONS]
        out[f"yam/{name}/state"], out[f"yam/{name}/raw"], out[f"yam/{name}/out"] = state, raw, result


def _dual_franka(rng, out):
    for name, dtype in (("f64", np.float64), ("f32", np.float32)):
        state = rng.normal(size=20).astype(dtype)
        raw = rng.normal(size=(10, 32)).astype(dtype)
        data = DualFrankaState(KEY)({OBSERVATION: {KEY: state}})
        reordered = data[OBSERVATION][KEY]
        result = DualFrankaAbsoluteActions(KEY)({OBSERVATION: data[OBSERVATION], ACTIONS: raw})[ACTIONS]
        out[f"dual_franka/{name}/state"], out[f"dual_franka/{name}/raw"] = state, raw
        out[f"dual_franka/{name}/reordered"], out[f"dual_franka/{name}/out"] = reordered, result


def _g1_policy(rng, out):
    """The whole ``with_adapter`` chain over a mock model: what a client really gets."""
    from test_robot_presets import (  # noqa: E402
        G1_STATE_KEY,
        build_g1_mock_policy,
        g1_observation,
    )

    policy = build_g1_mock_policy()
    policy.model_runner._value = 0.3
    state = rng.normal(size=16).astype(np.float32)
    result = policy.infer(g1_observation(state))
    assert G1_STATE_KEY == KEY
    out["g1_policy/state"] = state
    out["g1_policy/actions"] = result["actions"]
    out["g1_policy/normalized"] = result["normalized_actions"]
    out["g1_policy/tokenizer_state"] = policy.tokenizer.seen_states[0]


def main() -> None:
    rng = np.random.default_rng(20261117)
    out: dict = {}
    _g1(rng, out)
    _yam(rng, out)
    _dual_franka(rng, out)
    _g1_policy(rng, out)
    np.savez(OUT, **out)
    print(f"wrote {OUT} ({len(out)} arrays)")


if __name__ == "__main__":
    main()

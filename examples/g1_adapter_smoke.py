#!/usr/bin/env python3
"""Smoke-test a body's full adapter chain end to end, using the Unitree G1.

This is the last step of ``doc/adding-an-embodiment.md`` made runnable: once you
have written an embodiment, this is how you prove the chain you wired actually
runs. The G1 is the worked instance -- copy it and swap in your own body's
conventions.

A G1-style observation (3 cameras ``cam_high`` / ``cam_left_wrist`` /
``cam_right_wrist``, 16-DoF state, a prompt) is fed to a
:func:`~apxinf_robo.embodiments.unitree_g1.build_unitree_g1_policy` policy and
must come back as a ``[action_horizon, 16]`` action chunk -- exercising every
stage (decode state -> the engine's tokenize/model/unnormalize -> delta->absolute
-> 32->16 encode).

Unlike the other examples this one asserts rather than only printing, because
those steps are arithmetic that fails silently: a delta convention applied twice,
or a truncation at the wrong width, yields a plausible array of wrong numbers.

Since we hold no G1 checkpoint/norm_stats, this validates **plumbing and shape**
on a stand-in pi05 checkpoint: the action unnormalizer is a full-model-width
identity, so values are not G1-calibrated (that needs the integrator's gripper
limits + the G1 weights). It proves the config runs end-to-end through the
adapter.

The stand-in must carry three *real* camera views, because G1 sends three and a
checkpoint cannot serve more cameras than it was trained on. ``pi05_libero_base``
and its relatives are two-view (their third slot is ``empty_cameras`` padding),
so they are rejected here -- correctly, and before inference rather than after.

Requires the ``apxinf_py`` CUDA binding and a three-view checkpoint directory.

    python examples/g1_adapter_smoke.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import _common  # noqa: F401  (installs the source-checkout path shim)

from apxinf.processors import Unnormalizer

from apxinf_robo import load_bare_model
from apxinf_robo.conventions import UNITREE_G1 as G1_KEYS
from apxinf_robo.embodiments.unitree_g1 import build_unitree_g1_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument(
        "--chw", action="store_true", help="feed CHW images (ParseImage must transpose)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load the handle first so the identity unnormalizer can be built at the
    # model's own width: [50,32] then passes through unchanged, so delta->absolute
    # and the 32->16 robot truncation run on the raw model output.
    model = load_bare_model(
        args.model_dir / "model.safetensors",
        model="pi05",
        device=args.device,
        precision=args.precision,
    )
    width = model.action_dim
    identity = Unnormalizer(
        mean=np.zeros(width, np.float32), std=np.ones(width, np.float32), mode="mean_std"
    )

    policy = build_unitree_g1_policy(
        args.model_dir,
        model=model,  # reuse the already-loaded handle
        use_delta_joint_actions=True,
        adapt_to_pi=True,
        state_key=G1_KEYS.state_key,
        image_keys=G1_KEYS.image_keys,
        unnormalizer=identity,  # injected into the model's own unnormalize step
        precision=args.precision,
        device=args.device,
        metadata={"config": "pi05_UnitreeG1_groundwire", "note": "stand-in ckpt, shape-only"},
    )

    print("wrapped policy :", type(policy).__name__, "around", type(policy._inner).__name__)
    print(
        "metadata       :",
        {
            k: policy.metadata[k]
            for k in ("robot", "action_dim", "model_action_dim", "num_views", "action_horizon")
        },
    )

    S = model.image_size
    rng = np.random.default_rng(0)
    if args.chw:
        cam = lambda: rng.integers(0, 256, (3, S, S), dtype=np.uint8)  # noqa: E731
    else:
        cam = lambda: rng.integers(0, 256, (S, S, 3), dtype=np.uint8)  # noqa: E731

    # Built here rather than with _common.observation_for_policy because the
    # nesting is what is under test: an unmodified openpi G1 client sends its
    # cameras nested one level under "images" with the state flat. The policy's
    # image_keys are the paths ("images/cam_high"), which lookup_key walks into
    # this dict; _common builds the flat form instead.
    groups: dict = {}
    for key in G1_KEYS.image_keys:
        group, _, camera = key.partition("/")
        assert camera, f"expected a nested camera path, got {key!r}"
        groups.setdefault(group, {})[camera] = cam()
    observation = dict(groups)
    observation[G1_KEYS.state_key] = rng.standard_normal(16).astype(np.float32)
    observation[G1_KEYS.prompt_key] = args.prompt

    out = policy.infer(observation)
    actions = np.asarray(out["actions"])
    H = model.action_horizon

    assert actions.shape == (H, 16), f"expected ({H}, 16), got {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"

    print(f"\nOK  actions {actions.shape}  dtype={actions.dtype}")
    print(
        f"    timing: model={out['timing']['model_ms']:.1f}ms  "
        f"total={out['timing']['total_ms']:.1f}ms"
    )
    print(f"    action[0] = {np.array2string(actions[0], precision=3, max_line_width=120)}")
    print(
        f"    gripper dims [7,15] over horizon in [0,1]? "
        f"{bool((actions[:, [7, 15]] >= 0).all() and (actions[:, [7, 15]] <= 1).all())}"
    )
    print("\nG1 ADAPTER SMOKE TEST PASSED (shape/plumbing)")


if __name__ == "__main__":
    main()

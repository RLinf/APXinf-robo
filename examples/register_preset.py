#!/usr/bin/env python3
"""Register a robot preset from your own package -- no GPU, no weights.

A preset is an ``Embodiment x Convention`` pairing:

* a :class:`~apxinf_robo.conventions.base.Convention` is a fact about a *client*
  -- what a simulator, a dataset recorder, or a robot stack calls its cameras,
  its state, and its prompt. It survives a change of robot;
* an :class:`~apxinf_robo.embodiments.base.Embodiment` is a fact about a *body*
  -- how many cameras it carries, how wide its state and action vectors are, and
  (optionally) the pre/post steps that turn its numbers into the model's;
* a :class:`~apxinf_robo.embodiments.base.RobotPreset` pairs them, cross-checked.

Both registries are open. A deployment that speaks its own dialect registers it
from its own module scope and never edits this package -- which is the whole
reason ``register_convention`` / ``register_robot_preset`` are public. Doing it
at import time is what makes ``--robot mine`` work in a process that imported
your module.

See doc/adding-an-embodiment.md for the full story, including how to attach a
body's arithmetic with a custom builder.

    python examples/register_preset.py
    python examples/register_preset.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

from _common import observation_for_preset  # noqa: E402 (path shim in _common)

from apxinf_robo import (
    Convention,
    Embodiment,
    RobotPreset,
    available_robots,
    get_robot_preset,
    register_convention,
    register_robot_preset,
)

# --- 1. the dialect the client speaks ----------------------------------------
#
# image_keys are in model view-slot order: entry i fills slot i. Getting the
# order wrong feeds the wrong camera to each slot, silently.
MY_KEYS = register_convention(
    Convention(
        name="my_rig",
        image_keys=("cam/overhead", "cam/wrist"),
        state_key="robot/joint_positions",
        prompt_key="prompt",
    )
)

# --- 2. the body --------------------------------------------------------------
#
# action_dim is a *loading* knob (trim the model's vector down to this width);
# action_width and state_dim are facts about the hardware, and they are what the
# preflight compares norm_stats against. They need not match: a Franka under
# LIBERO has 8-dim state and a 7-dim EEF-delta action.
#
# No transform is given, so the engine's own output is what the robot takes -- no
# robot-specific pre/post arithmetic, which is why has_transform is False below.
MY_ARM = Embodiment(
    name="myarm",
    num_cameras=2,
    action_dim=7,
    state_dim=7,
    action_width=7,
)

# --- 3. the deployable pairing ------------------------------------------------
MY_ROBOT = register_robot_preset(
    RobotPreset(
        name="myarm_my_rig",
        embodiment=MY_ARM,
        convention=MY_KEYS,
        summary="MyArm on my rig: 2 cameras, 7-dim state, 7-dim action",
    ),
    aliases=("myarm",),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=None,
        help="optional: run the preflight against the new preset (still loads no weights)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("registered presets:", list(available_robots()))
    print("resolved by alias :", get_robot_preset("myarm").name)
    print("describe          :", MY_ROBOT.describe())
    print(f"num_views={MY_ROBOT.num_views} action_dim={MY_ROBOT.action_dim} "
          f"state_dim={MY_ROBOT.state_dim} has_transform={MY_ROBOT.has_transform}")

    observation = observation_for_preset(MY_ROBOT)
    print("an observation this preset accepts:",
          {key: getattr(value, "shape", value) for key, value in observation.items()})

    # The pairing is validated, not just stored. A convention recorded on a
    # two-camera rig cannot be attached to a three-camera body: the checkpoint
    # fills view slots from 0 up, so the mismatch would be a silently wrong
    # number of frames rather than an error at inference time.
    try:
        RobotPreset(
            name="mismatch",
            embodiment=Embodiment(name="three_cam", num_cameras=3),
            convention=MY_KEYS,
        )
    except ValueError as error:
        print("\nrejected pairing (expected):", error)

    # Re-registering a name is an error too, because a silent overwrite would
    # change what an already-written preset resolves to.
    try:
        register_robot_preset(MY_ROBOT)
    except ValueError as error:
        print("rejected re-registration (expected):", error)

    if args.model_dir is not None:
        from apxinf_robo import check_checkpoint, format_findings

        print(f"\npreflight against {MY_ROBOT.name}:")
        print(format_findings(check_checkpoint(args.model_dir, "myarm_my_rig")))

    print("\nfrom here, build_robot_policy('myarm_my_rig', <model-dir>) serves it,")
    print("and `apxinf-robo serve --robot myarm_my_rig` works in any process that")
    print("imported this module. See doc/adding-an-embodiment.md.")


if __name__ == "__main__":
    main()

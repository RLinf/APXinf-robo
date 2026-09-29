"""The examples are documentation that runs, so they are tested like code.

Only the parts that need neither a GPU nor a checkpoint are exercised here --
which, by design, is most of what the examples teach: the wire contract a preset
declares, the observation shape it accepts, the registration mechanism, and the
recording wrapper that makes the L1/L2 boundary observable. The inference calls
themselves are covered by ``tests/test_parity.py`` against a real checkpoint.

Loading ``register_preset`` mutates the process-wide registries, so the two
registry dicts are snapshotted and restored around it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from apxinf_robo.conventions.base import CONVENTIONS
from apxinf_robo.embodiments.base import ROBOT_ALIASES, ROBOT_PRESETS
from apxinf_robo.presets import get_robot_preset

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str):
    """Exec an example as a module, with the examples directory importable."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


class _IsolatedRegistries(unittest.TestCase):
    """Undo whatever an example registered at import time.

    ``register_preset`` registers at module scope, which is the behavior being
    demonstrated -- registration has to happen on import for ``--robot mine`` to
    work in a process that imported your module. That makes loading it a
    side effect on the process, so any test that loads it restores the
    registries afterwards; re-registering the same name is an error by design.
    """

    def setUp(self) -> None:
        super().setUp()
        saved = [
            (registry, dict(registry))
            for registry in (ROBOT_PRESETS, ROBOT_ALIASES, CONVENTIONS)
        ]

        def restore() -> None:
            for registry, snapshot in saved:
                registry.clear()
                registry.update(snapshot)

        self.addCleanup(restore)


class _PolicyContractOnly:
    """Exposes nothing but the public ``Policy`` metadata contract."""

    metadata = {
        "image_keys": ["cam/front", "cam/wrist"],
        "state_key": "observation/state",
        "state_dim": 8,
        "prompt_key": "prompt",
    }
    action_dim = 7


class CommonHelpersTest(unittest.TestCase):
    """``_common`` is shared by every example, so its contract is pinned here."""

    def setUp(self) -> None:
        self.common = _load_example("_common")

    def test_policy_options_accept_a_json_object(self) -> None:
        self.assertEqual(
            self.common.json_object('{"norm_key":"x2_normal","action_dim":7}'),
            {"norm_key": "x2_normal", "action_dim": 7},
        )

    def test_policy_options_reject_non_objects(self) -> None:
        for value in ("[]", '"walloss"', "not-json"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                self.common.json_object(value)

    def test_generic_flags_win_over_duplicate_json_keys(self) -> None:
        options = self.common.policy_kwargs(
            {"norm_key": "x2_normal", "device": "wrong", "metadata": {"deployment": "lab"}},
            device="cuda:1",
            precision="fp8",
            action_dim=7,
            metadata={"robot": "myarm"},
        )
        self.assertEqual(
            options,
            {
                "norm_key": "x2_normal",
                "device": "cuda:1",
                "precision": "fp8",
                "action_dim": 7,
                "metadata": {"deployment": "lab", "robot": "myarm"},
            },
        )

    def test_action_dim_zero_is_left_out_rather_than_sent_as_zero(self) -> None:
        # 0 means "keep the checkpoint's full vector", which is a *missing*
        # argument downstream, not a width of zero.
        options = self.common.policy_kwargs({}, device="cuda:0", precision="bf16")
        self.assertNotIn("action_dim", options)

    def test_an_observation_is_built_from_a_preset_contract(self) -> None:
        preset = get_robot_preset("franka_libero")
        observation = self.common.observation_for_preset(preset)

        self.assertEqual(
            set(observation),
            set(preset.image_keys) | {preset.state_key, preset.prompt_key},
        )
        first = observation[preset.image_keys[0]]
        self.assertEqual(first.shape, (256, 256, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(observation[preset.state_key].shape, (preset.state_dim,))

    def test_an_observation_is_built_from_the_public_policy_contract(self) -> None:
        observation = self.common.observation_for_policy(_PolicyContractOnly())

        self.assertEqual(
            set(observation), {"cam/front", "cam/wrist", "observation/state", "prompt"}
        )
        self.assertEqual(observation["observation/state"].shape, (8,))

    def test_a_policy_with_no_usable_image_contract_is_rejected(self) -> None:
        for image_keys in (None, [], ["front", ""]):
            with self.subTest(image_keys=image_keys):

                class BadPolicy:
                    metadata = {} if image_keys is None else {"image_keys": image_keys}

                with self.assertRaises(ValueError) as caught:
                    self.common.observation_for_policy(BadPolicy())
                self.assertIn("image_keys", str(caught.exception))

    def test_a_served_state_key_without_a_width_is_rejected(self) -> None:
        # Guessing a width here would produce an observation the policy accepts
        # and silently misreads.
        class BadPolicy:
            metadata = {"image_keys": ["front"], "state_key": "observation/state"}

        with self.assertRaises(ValueError) as caught:
            self.common.observation_for_policy(BadPolicy())
        self.assertIn("state_dim", str(caught.exception))


class RegisterPresetExampleTest(_IsolatedRegistries):
    """The extension point the example demonstrates actually extends."""

    def setUp(self) -> None:
        super().setUp()
        self.example = _load_example("register_preset")

    def test_the_example_registers_a_usable_preset(self) -> None:
        self.assertIn("myarm_my_rig", ROBOT_PRESETS)
        self.assertIs(get_robot_preset("myarm"), self.example.MY_ROBOT)
        self.assertIn("my_rig", CONVENTIONS)

    def test_the_registered_preset_declares_a_complete_wire_contract(self) -> None:
        preset = self.example.MY_ROBOT
        self.assertEqual(preset.image_keys, ("cam/overhead", "cam/wrist"))
        self.assertEqual(preset.state_key, "robot/joint_positions")
        self.assertEqual(preset.num_views, 2)
        self.assertEqual(preset.action_dim, 7)
        # No builder was supplied, so the generic one applies: the preset's keys
        # are served without any robot-specific arithmetic.
        self.assertFalse(preset.has_robot_steps)

    def test_main_runs_without_a_checkpoint(self) -> None:
        # The example's whole point is that it needs no weights and no GPU.
        self.example.parse_args = lambda: argparse.Namespace(model_dir=None)
        self.example.main()


class ExampleSurfaceTest(_IsolatedRegistries):
    """Every example stays runnable and self-describing."""

    NAMES = (
        "bare_model_infer",
        "g1_adapter_smoke",
        "policy_infer",
        "preflight_check",
        "register_preset",
        "robot_policy_infer",
        "serve_websocket",
    )

    # ``lerobot_loop`` raises SystemExit at import time when lerobot is absent --
    # deliberately, so the failure reads as "install lerobot" rather than as an
    # ImportError from one of its moved helper modules. That makes it unloadable
    # here.
    UNLOADABLE = {"lerobot_loop": "imports lerobot at module scope"}

    def test_every_example_is_either_covered_or_excused(self) -> None:
        # Otherwise a new example is silently untested: this list is hand-written
        # because loading an example is the test.
        on_disk = {
            path.stem for path in EXAMPLES.glob("*.py") if not path.name.startswith("_")
        }
        self.assertEqual(on_disk, set(self.NAMES) | set(self.UNLOADABLE))

    def test_each_example_documents_itself_and_has_an_entry_point(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                module = _load_example(name)
                self.assertTrue((module.__doc__ or "").strip(), f"{name} has no docstring")
                self.assertTrue(callable(getattr(module, "main", None)))
                self.assertTrue(callable(getattr(module, "parse_args", None)))

    def test_the_checkpoint_examples_require_a_model_dir(self) -> None:
        # Defaulting to some path would turn a missing argument into a confusing
        # load failure several seconds later.
        needs_weights = (
            "bare_model_infer",
            "g1_adapter_smoke",
            "policy_infer",
            "preflight_check",
            "robot_policy_infer",
            "serve_websocket",
        )
        for name in needs_weights:
            with self.subTest(name=name):
                module = _load_example(name)
                argv, sys.argv = sys.argv, [name]
                try:
                    with self.assertRaises(SystemExit):
                        module.parse_args()
                finally:
                    sys.argv = argv

    def test_the_readme_lists_every_example(self) -> None:
        readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
        for path in sorted(EXAMPLES.glob("*.py")):
            if path.name.startswith("_"):
                continue
            with self.subTest(example=path.name):
                self.assertIn(path.name, readme)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

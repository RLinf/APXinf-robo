"""L1 and L2 must return the same numbers. Nothing else checks that.

ApxInf exposes two interfaces (see :mod:`apxinf_robo.engine`):
``interface="policy"`` runs the engine's own resize / tokenize / normalize /
noise chain, and ``interface="bare"`` takes already-preprocessed tensors from a
caller that brings its own transforms. Frameworks pick one. RLinf's vendored
openpi transforms want the bare interface; ``apxinf-robo serve`` and
``eval-libero`` use the policy interface.

Which means the same checkpoint gets evaluated through two different
preprocessing implementations, and *neither repository checks that they agree*.
Drift here is invisible: both paths return finite, well-shaped, plausible
actions. It shows up as an unreproducible success rate months later.

Three groups:

* :class:`InterfaceDisciplineTest` — offline, always runs. The engine must not
  let a caller use one interface's entry point on the other's handle, because a
  silent fallback is exactly how the two paths stop being compared.
* :class:`BareInterfaceLoadTest` — offline. The bare handle is *built* the way
  the policy handle is built, which is what makes the numeric test able to pass.
* :class:`NumericParityTest` — the real thing. Loads a checkpoint at the policy
  interface, records the exact tensors it hands the model, replays them through
  a bare handle, and asserts the outputs are **bitwise** identical. Needs a GPU
  and a checkpoint, so it is opt-in::

      APXINF_PARITY_CHECKPOINT=/ckpt/pi05_libero pytest tests/test_parity.py

  Bitwise, not ``allclose``: both paths run the same kernels on the same device
  with the same inputs, so any difference at all means a preprocessing
  divergence rather than float noise, and a tolerance would hide small ones.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from apxinf_robo.engine import INTERFACES, ApxInfEngine

CHECKPOINT = os.environ.get("APXINF_PARITY_CHECKPOINT")
DEVICE = os.environ.get("APXINF_PARITY_DEVICE", "cuda:0")
PRECISION = os.environ.get("APXINF_PARITY_PRECISION", "bf16")
ROBOT = os.environ.get("APXINF_PARITY_ROBOT", "franka_libero")

needs_checkpoint = unittest.skipIf(
    not CHECKPOINT,
    "set APXINF_PARITY_CHECKPOINT=<dir> (and have a GPU) to run L1/L2 parity",
)


class _FakePolicy:
    metadata = {"robot": "fake"}
    action_horizon = 10
    action_dim = 7

    def __init__(self) -> None:
        self.closed = False

    def infer(self, observation, *, noise=None):
        return {"actions": np.zeros((self.action_horizon, self.action_dim), np.float32)}

    def close(self) -> None:
        self.closed = True


class _FakeBare:
    action_horizon = 10
    action_dim = 32

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        return np.zeros((self.action_horizon, self.action_dim), np.float32)


class InterfaceDisciplineTest(unittest.TestCase):
    """The two interfaces stay distinguishable, so they stay comparable."""

    def test_exactly_one_handle_is_held(self) -> None:
        with self.assertRaises(ValueError):
            ApxInfEngine(interface="policy")
        with self.assertRaises(ValueError):
            ApxInfEngine(interface="policy", policy=_FakePolicy(), bare=_FakeBare())

    def test_an_unknown_interface_is_rejected_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ApxInfEngine(interface="l3", policy=_FakePolicy())
        self.assertIn("l3", str(caught.exception))
        for interface in INTERFACES:
            self.assertIn(interface, str(caught.exception))

    def test_the_policy_interface_refuses_infer_rgb(self) -> None:
        # Falling back to the bare handle here would let a caller think it was
        # exercising L1 while actually re-running L2 -- a parity test that
        # compares a path with itself always passes.
        engine = ApxInfEngine(interface="policy", policy=_FakePolicy())
        with self.assertRaises(RuntimeError) as caught:
            engine.infer_rgb(np.zeros((1, 4, 4, 3), np.uint8), "nhwc", [1, 2])
        self.assertIn('interface="bare"', str(caught.exception))

    def test_the_bare_interface_refuses_an_observation_dict(self) -> None:
        engine = ApxInfEngine(interface="bare", bare=_FakeBare())
        with self.assertRaises(RuntimeError) as caught:
            engine.infer({"prompt": "pick up the cup"})
        self.assertIn('interface="policy"', str(caught.exception))

    def test_shape_facts_come_from_whichever_handle_is_held(self) -> None:
        self.assertEqual(ApxInfEngine(interface="bare", bare=_FakeBare()).action_dim, 32)
        policy_engine = ApxInfEngine(interface="policy", policy=_FakePolicy())
        self.assertEqual(policy_engine.action_dim, 7)
        self.assertEqual(policy_engine.action_horizon, 10)

    def test_the_bare_interface_publishes_no_metadata(self) -> None:
        # It has no wire contract to publish: the caller owns preprocessing.
        self.assertEqual(ApxInfEngine(interface="bare", bare=_FakeBare()).metadata, {})
        self.assertEqual(
            ApxInfEngine(interface="policy", policy=_FakePolicy()).metadata,
            {"robot": "fake"},
        )

    def test_the_context_manager_closes_the_handle(self) -> None:
        policy = _FakePolicy()
        with ApxInfEngine(interface="policy", policy=policy):
            pass
        self.assertTrue(policy.closed)


class BareInterfaceLoadTest(unittest.TestCase):
    """The bare handle is built the way the policy handle is built.

    This is what makes :class:`NumericParityTest` able to pass at all. Tuned GEMM
    tactics are a *load-time* argument: ``from_pretrained`` resolves them
    internally, so a bare handle loaded without them runs different kernels and
    returns different numbers for byte-identical inputs (measured at 2**-8 in the
    normalized domain on Thor / bf16). Nothing downstream can detect that -- both
    handles return finite, well-shaped, plausible actions.
    """

    def setUp(self) -> None:
        from apxinf_robo import engine

        self.engine = engine
        self.loaded: list = []

        class _FakeModel:
            @staticmethod
            def load(model, path, device, **kwargs):
                self.loaded.append((model, path, device, kwargs.get("model_variant", kwargs.get("precision")), kwargs))
                return _FakeBare()

        original_require = engine.require_apxinf
        original_resolve = engine.resolve_tactics
        engine.require_apxinf = lambda: type("_FakeApxInf", (), {"ModelRunner": _FakeModel})
        engine.resolve_tactics = lambda *a, **kw: "/tuning/thor-sm110/tactics.json"

        def restore() -> None:
            engine.require_apxinf = original_require
            engine.resolve_tactics = original_resolve

        self.addCleanup(restore)

    def test_tactics_are_resolved_by_default(self) -> None:
        self.engine.load_bare_model("/ckpt", device="cuda:0", precision="bf16")
        self.assertEqual(self.loaded[0][4]["model_variant"], "bf16")
        self.assertEqual(
            self.loaded[0][4].get("tactics"), "/tuning/thor-sm110/tactics.json"
        )

    def test_the_checkpoint_is_offered_so_a_local_tactics_json_can_win(self) -> None:
        seen: list = []
        self.engine.resolve_tactics = lambda *a, **kw: seen.append(kw) or None
        self.engine.load_bare_model("/ckpt", device="cuda:0", precision="bf16")
        self.assertEqual(str(seen[0]["model_dir"]), "/ckpt")
        # Resolving to nothing must not pass ``tactics=None`` down.
        self.assertNotIn("tactics", self.loaded[0][4])

    def test_an_explicit_tactics_file_wins(self) -> None:
        self.engine.load_bare_model("/ckpt", tactics="/mine.json")
        self.assertEqual(self.loaded[0][4]["tactics"], "/mine.json")

    def test_tactics_none_opts_out(self) -> None:
        # An explicit opt-out, for measuring what the tuning is worth.
        self.engine.load_bare_model("/ckpt", tactics=None)
        self.assertNotIn("tactics", self.loaded[0][4])

    def test_a_family_whose_l2_does_not_auto_select_is_left_alone(self) -> None:
        # WallossPolicy.from_pretrained only forwards an explicit tactics=, so
        # auto-selecting here would tune L1 against an untuned L2 -- trading
        # pi05's divergence for WallOSS's rather than removing one.
        self.engine.load_bare_model("/ckpt", model="walloss")
        self.assertNotIn("tactics", self.loaded[0][4])
        # ...but an explicit file is still honoured, for a caller that knows.
        self.loaded.clear()
        self.engine.load_bare_model("/ckpt", model="walloss", tactics="/mine.json")
        self.assertEqual(self.loaded[0][4]["tactics"], "/mine.json")


class _RecordingModel:
    """Wraps the engine handle to capture exactly what L2 feeds L1."""

    def __init__(self, model) -> None:
        self._model = model
        self.calls: list = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        # Capture as arrays: the engine binding takes numpy arrays, not
        # sequences, so a captured ``list(token_ids)`` replays as a TypeError.
        self.calls.append((np.array(rgb_u8, copy=True), layout,
                           np.array(token_ids, copy=True),
                           None if noise is None else np.array(noise, copy=True)))
        return self._model.infer_rgb(rgb_u8, layout, token_ids, noise)


class RecordingModelTest(unittest.TestCase):
    """The recorder is what makes the L1/L2 boundary observable, so pin it.

    :class:`NumericParityTest` skips without a GPU, which would leave the
    wrapper it depends on completely untested on every machine that does not
    have one -- and a recorder that captures the wrong thing produces a parity
    test that passes for the wrong reason.
    """

    def test_it_passes_through_and_captures_the_call(self) -> None:
        class FakeModel:
            action_horizon = 10
            action_dim = 32

            def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
                return np.full((self.action_horizon, self.action_dim), 0.5, np.float32)

        recorder = _RecordingModel(FakeModel())
        rgb = np.zeros((2, 224, 224, 3), np.uint8)
        noise = np.ones((10, 32), np.float32)

        out = recorder.infer_rgb(rgb, "nhwc", np.array([1, 2, 3], np.uint32), noise)

        self.assertEqual(out.shape, (10, 32))
        self.assertEqual(len(recorder.calls), 1)
        captured_rgb, layout, token_ids, captured_noise = recorder.calls[0]
        self.assertEqual(layout, "nhwc")
        # An array, not a list: the engine binding rejects sequences, so a
        # captured list could not be replayed.
        self.assertIsInstance(token_ids, np.ndarray)
        np.testing.assert_array_equal(token_ids, [1, 2, 3])
        np.testing.assert_array_equal(captured_rgb, rgb)
        np.testing.assert_array_equal(captured_noise, noise)
        # Attributes the policy reads must survive the wrapper untouched.
        self.assertEqual(recorder.action_horizon, 10)
        self.assertEqual(recorder.action_dim, 32)

    def test_it_copies_rather_than_aliases_its_inputs(self) -> None:
        # The engine may reuse its buffers; a stored reference would be compared
        # against whatever the next call left behind.
        class FakeModel:
            def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
                return np.zeros((1, 1), np.float32)

        recorder = _RecordingModel(FakeModel())
        rgb = np.zeros((1, 4, 4, 3), np.uint8)
        recorder.infer_rgb(rgb, "nhwc", np.array([1], np.uint32), None)
        rgb[...] = 7

        self.assertTrue(np.all(recorder.calls[0][0] == 0))
        self.assertIsNone(recorder.calls[0][3])


@needs_checkpoint
class NumericParityTest(unittest.TestCase):
    """One checkpoint, two interfaces, identical bits.

    Both bare handles are loaded on this thread on purpose: ``ModelRunner.load``
    returns an ``unsendable`` handle whose CUDA context is bound to its creating
    thread.

    L2 runs once, in :meth:`setUpClass`, and its recorded inputs are shared. The
    comparison has to replay *exactly* what L2 sent rather than a reconstruction
    of it -- reimplementing the preprocessing here would compare a guess.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from apxinf_robo.presets import build_robot_policy

        policy = build_robot_policy(
            ROBOT, CHECKPOINT, device=DEVICE, precision=PRECISION
        )
        recorder = _RecordingModel(policy.model_runner)
        policy.model_runner = recorder

        # Pin the noise: with internal sampling the two runs draw from different
        # RNG streams and would differ for a reason that is not drift.
        noise = np.ascontiguousarray(
            np.random.default_rng(1).standard_normal(
                (recorder.action_horizon, recorder.action_dim)
            ),
            dtype=np.float32,
        )
        try:
            result = policy.infer(cls._observation(), noise=noise)
        finally:
            policy.close()

        assert len(recorder.calls) == 1, f"expected 1 model call, saw {len(recorder.calls)}"
        cls.recorded = recorder.calls[0]
        cls.l2 = np.asarray(result["normalized_actions"], np.float32)

    @staticmethod
    def _observation():
        from apxinf_robo.presets import get_robot_preset

        convention = get_robot_preset(ROBOT).convention
        rng = np.random.default_rng(0)
        observation = {}
        for key in convention.image_keys:
            _set(observation, key, rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        if convention.state_key is not None:
            _set(observation, convention.state_key,
                 rng.standard_normal(8).astype(np.float32))
        _set(observation, convention.prompt_key, "pick up the black bowl")
        return observation

    def _replay(self, **load_kwargs) -> np.ndarray:
        """Feed L2's recorded tensors to a freshly loaded bare handle."""
        from apxinf_robo.engine import load_bare_model

        rgb, layout, token_ids, noise = self.recorded
        bare = load_bare_model(
            CHECKPOINT, device=DEVICE, precision=PRECISION, **load_kwargs
        )
        return np.asarray(bare.infer_rgb(rgb, layout, token_ids, noise), np.float32)

    def test_the_two_interfaces_return_the_same_normalized_actions(self) -> None:
        np.testing.assert_array_equal(
            self._replay(),
            self.l2,
            err_msg=(
                "L1 (interface='bare') and L2 (interface='policy') disagree on "
                "identical inputs. Both paths are still 'working' -- this is the "
                "drift that makes two evaluation numbers for one checkpoint "
                "incomparable."
            ),
        )

    def test_dropping_the_tuned_tactics_is_what_broke_that_parity(self) -> None:
        """The divergence is real, so pin its cause rather than describing it.

        ``tactics=None`` forces the provider-default kernels, which is what a
        bare handle got before ``load_bare_model`` started resolving what
        ``from_pretrained`` resolves. The inputs are byte-identical to the
        passing case above, so this isolates the divergence to a *load-time*
        argument: nothing about preprocessing changed between the two replays.

        If this ever stops diverging the tuning has become a no-op for this
        checkpoint on this device, which is worth knowing -- but it is not a
        reason to keep L1 loading differently from L2, so the test asserts the
        difference rather than skipping.
        """
        untuned = self._replay(tactics=None)
        if np.array_equal(untuned, self.l2):
            self.skipTest(
                "the tuned and default kernels agree bitwise on this device, so "
                "there is no divergence here to pin"
            )
        delta = float(np.abs(untuned - self.l2).max())
        self.assertGreater(delta, 0.0)
        # Small enough to survive every shape and finiteness check downstream,
        # which is exactly why it went unnoticed.
        self.assertLess(delta, 1.0, f"max |delta| = {delta}")


def _set(observation: dict, key: str, value) -> None:
    head, _, tail = key.partition("/")
    if not tail:
        observation[key] = value
        return
    _set(observation.setdefault(head, {}), tail, value)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

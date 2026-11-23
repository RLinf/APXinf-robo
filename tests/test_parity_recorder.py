"""The L1/L2 parity recorder must reach the engine under a robot transform.

Split from ``test_parity.py`` because it needs ApxInf's Python front end (a real
``Pi05Policy`` over a mock model), and ``test_parity.py`` is kept importable without it.
No GPU and no checkpoint: this is the part of the parity setup that runs everywhere.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_parity import _engine_policy, install_recorder


try:
    from test_robot_presets import (
        G1_ROBOT_DIM,
        MockModel,
        RecordingTokenizer,
        build_g1_mock_policy,
        g1_observation,
    )
    from apxinf import Pi05Policy
except ImportError as exc:  # pragma: no cover - depends on the environment
    _ENGINE_IMPORT_ERROR: "str | None" = str(exc)
else:
    _ENGINE_IMPORT_ERROR = None


@unittest.skipIf(_ENGINE_IMPORT_ERROR is not None, f"needs ApxInf ({_ENGINE_IMPORT_ERROR})")
class RecorderOnWrappedPolicyTest(unittest.TestCase):
    """:func:`install_recorder` finds the engine under a robot transform. No GPU, no checkpoint.

    :class:`NumericParityTest` cannot run without hardware, so the part of its setup that broke
    when robots moved to ``TransformedPolicy`` is exercised here, on a real ``Pi05Policy`` over
    a mock model, with and without the wrapper.
    """

    def _g1(self):
        policy = build_g1_mock_policy()
        return policy, policy._inner, policy.model

    def _bare(self):
        from apxinf.processors import Unnormalizer

        from test_robot_presets import MODEL_DIM

        model = MockModel(num_views=2)
        tokenizer = RecordingTokenizer()
        tokenizer.discrete_state = False
        identity = Unnormalizer(
            q01=[-1.0] * MODEL_DIM, q99=[1.0] * MODEL_DIM, dims=MODEL_DIM, eps=0.0
        )
        inp, out = Pi05Policy.default_pipelines(
            model, tokenizer=tokenizer, unnormalizer=identity
        )
        policy = Pi05Policy(model, input_pipeline=inp, output_pipeline=out)
        return policy, policy, model

    def _bare_observation(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        return {"base_0_rgb": image, "left_wrist_0_rgb": image,
                "prompt": "pick up the bowl"}

    def test_the_wrapper_has_no_model_runner_and_the_helper_unwraps_it(self) -> None:
        policy, engine, _ = self._g1()
        self.assertFalse(hasattr(policy, "model_runner"))
        self.assertIs(_engine_policy(policy), engine)
        self.assertIs(_engine_policy(engine), engine)

    def test_the_recorder_sees_one_real_model_call_on_both_paths(self) -> None:
        for name, make, observation in (
            ("wrapped", self._g1, lambda: g1_observation(np.zeros(G1_ROBOT_DIM, np.float32))),
            ("bare", self._bare, self._bare_observation),
        ):
            with self.subTest(name):
                policy, engine, model = make()
                recorder = install_recorder(policy)
                self.assertIs(engine.model_runner, recorder)
                result = policy.infer(observation())  # the outer infer: full pre/engine/post
                self.assertEqual(len(recorder.calls), 1)
                rgb, layout, token_ids, _ = recorder.calls[0]
                # What was recorded is what the model received, not a reconstruction.
                self.assertEqual(len(model.images), 1)
                np.testing.assert_array_equal(rgb, model.images[0])
                self.assertEqual(layout, "nhwc")
                self.assertGreater(token_ids.size, 0)
                self.assertTrue(np.isfinite(np.asarray(result["actions"])).all())
                if name == "wrapped":  # the robot post-processing really ran
                    self.assertEqual(np.asarray(result["actions"]).shape[-1], G1_ROBOT_DIM)
                policy.close()

    def test_closing_the_outer_policy_releases_the_engine(self) -> None:
        policy, engine, _ = self._g1()
        install_recorder(policy)
        closed = []
        engine.close = lambda: closed.append(True)
        policy.close()
        self.assertEqual(closed, [True])

    def test_a_policy_with_no_model_runner_anywhere_fails_loudly(self) -> None:
        class Opaque:
            pass

        with self.assertRaises(AssertionError):
            _engine_policy(Opaque())


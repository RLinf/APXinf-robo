"""Startup preflight: does the checkpoint on disk match the preset serving it?

The delivered configuration that motivated :mod:`apxinf_robo.preflight` was a
Unitree G1 checkpoint (16-DoF, three cameras, delta joint actions) served with a
LIBERO ``norm_stats.json`` (8-dim state, 7-dim action) and ``--action-dim 7``.
Every layer accepted it. These tests cover both rejection of incompatible
checkpoints and acceptance of compatible ones.

The checks themselves are split across two repositories -- the checkpoint half
in :mod:`apxinf.checkpoints.preflight`, the preset half here -- so these tests
also pin that the seam between them still produces one coherent report.

Runs offline against synthesised checkpoint directories; no CUDA, no weights.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import struct
import sys
import tempfile
import unittest
import zipfile

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from apxinf_robo.preflight import (  # noqa: E402
    FAIL,
    INFO,
    WARN,
    check_checkpoint,
    format_findings,
)
from apxinf_robo.presets import get_robot_preset  # noqa: E402

G1_DIM = 16


def _stats(width: int, *, quantiles: bool = True) -> dict:
    entry = {"mean": [0.0] * width, "std": [1.0] * width}
    if quantiles:
        entry["q01"] = [-1.0] * width
        entry["q99"] = [1.0] * width
    return entry


class CheckpointFixture:
    """A checkpoint directory with exactly the files the preflight reads."""

    def __init__(self, case: unittest.TestCase, **norm_stats) -> None:
        self.path = pathlib.Path(tempfile.mkdtemp())
        case.addCleanup(self._cleanup)
        if norm_stats:
            self.write_norm_stats(**norm_stats)

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.path, ignore_errors=True)

    def write_norm_stats(self, *, nested: bool = True, **entries) -> "CheckpointFixture":
        document = {"norm_stats": entries} if nested else entries
        (self.path / "norm_stats.json").write_text(json.dumps(document))
        return self

    def write_tokenizer(self, name: str = "paligemma_tokenizer.model") -> "CheckpointFixture":
        (self.path / name).write_bytes(b"not a real sentencepiece model")
        return self

    def write_weights(self) -> "CheckpointFixture":
        """An empty ``model.safetensors``; nothing here reads a weight."""
        (self.path / "model.safetensors").write_bytes(b"")
        return self

    def write_metadata_pt(self, *, asset_id: str = "example-asset", action_dim: int = 32) -> "CheckpointFixture":
        """Write an openpi-shaped ``metadata.pt`` the way ``torch.save`` does.

        A zip whose ``<archive>/data.pkl`` member is an ordinary pickle — built
        by hand so these tests need neither torch nor a real checkpoint.
        """
        payload = {
            "global_step": 1,
            "config": {
                "exp_name": "pi05_example",
                "model": {
                    "action_dim": action_dim,
                    "action_horizon": 50,
                    "max_token_len": 200,
                    "discrete_state_input": True,
                    "pi05": True,
                },
                "data": {"assets": {"asset_id": asset_id}},
            },
        }
        with zipfile.ZipFile(self.path / "metadata.pt", "w") as archive:
            archive.writestr("metadata/data.pkl", pickle.dumps(payload, protocol=2))
            archive.writestr("metadata/version", "3\n")
        return self

    def write_asset_stats(self, asset_id: str, **entries) -> "CheckpointFixture":
        """Statistics where openpi actually writes them: ``assets/<asset_id>/``."""
        target = self.path / "assets" / asset_id / "norm_stats.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"norm_stats": entries}))
        return self

    def write_lerobot_processors(self, *, state_files: bool) -> "CheckpointFixture":
        """Write a current LeRobot PI0.5 processor layout for LIBERO."""
        self.write_weights()
        (self.path / "config.json").write_text(
            json.dumps(
                {
                    "type": "pi05",
                    "input_features": {
                        "observation.state": {"type": "STATE", "shape": [8]},
                    },
                    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
                }
            )
        )
        features = (
            {
                "observation.state": {"type": "STATE", "shape": [8]},
                "action": {"type": "ACTION", "shape": [7]},
            }
            if state_files
            else {}
        )
        pre = {
            "registry_name": "normalizer_processor",
            "config": {
                "features": features,
                "norm_map": {"STATE": "QUANTILES", "ACTION": "QUANTILES"},
            },
        }
        post = {
            "registry_name": "unnormalizer_processor",
            "config": {
                "features": {"action": features["action"]} if state_files else {},
                "norm_map": {"ACTION": "QUANTILES"},
            },
        }
        if state_files:
            pre["state_file"] = "pre.safetensors"
            post["state_file"] = "post.safetensors"
            self._write_safetensors(
                "pre.safetensors",
                {
                    "observation.state.q01": [-1.0] * 8,
                    "observation.state.q99": [1.0] * 8,
                    "action.q01": [-1.0] * 7,
                    "action.q99": [1.0] * 7,
                },
            )
            self._write_safetensors(
                "post.safetensors",
                {"action.q01": [-1.0] * 7, "action.q99": [1.0] * 7},
            )
        (self.path / "policy_preprocessor.json").write_text(
            json.dumps({"steps": [pre]})
        )
        (self.path / "policy_postprocessor.json").write_text(
            json.dumps({"steps": [post]})
        )
        return self

    def _write_safetensors(self, name: str, tensors: dict) -> None:
        header = {}
        payload = bytearray()
        for tensor_name, values in tensors.items():
            start = len(payload)
            payload.extend(struct.pack(f"<{len(values)}f", *values))
            header[tensor_name] = {
                "dtype": "F32",
                "shape": [len(values)],
                "data_offsets": [start, len(payload)],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        (self.path / name).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + payload
        )


def _levels(findings, check_prefix: str) -> list:
    return [f.level for f in findings if f.check.startswith(check_prefix)]


def _level(findings, check: str) -> list:
    """Levels for one exact check name.

    Needed where a check and its width check share a prefix -- ``action
    normalization`` and ``action normalization width`` are separate findings
    raised by separate repositories, and :func:`_levels` would merge them.
    """
    return [f.level for f in findings if f.check == check]


class NormStatsWidthTest(unittest.TestCase):
    """A1: the shipped statistics have to be *this robot's* statistics."""

    def test_libero_stats_under_the_g1_preset_are_fatal(self) -> None:
        """The exact delivered combination: G1 preset, LIBERO norm_stats."""
        ckpt = CheckpointFixture(
            self, actions=_stats(7), state=_stats(8)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        fatal = [f for f in findings if f.level == FAIL]
        self.assertEqual(len(fatal), 2, format_findings(findings))
        widths = {f.check for f in fatal}
        self.assertEqual(
            widths, {"norm_stats['actions'] width", "norm_stats['state'] width"}
        )
        # The remedy has to name the number the operator should be looking for;
        # "width mismatch" alone sends them to the wrong file.
        self.assertIn("16", fatal[0].detail + fatal[0].remedy)

    def test_matching_widths_pass(self) -> None:
        ckpt = CheckpointFixture(
            self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(
            [f for f in findings if f.level == FAIL], [], format_findings(findings)
        )
        self.assertEqual([f for f in findings if f.level == WARN], [])

    def test_dual_franka_accepts_padded_stats_without_state_tokens(self) -> None:
        # RLinf calculates these stats after padding both state and actions to
        # model width. The controller still consumes 20-value poses.
        ckpt = CheckpointFixture(
            self, actions=_stats(32), state=_stats(32)
        ).write_tokenizer()
        findings = check_checkpoint(ckpt.path, "dual_franka_tcp_rot6d")
        self.assertEqual(
            [f for f in findings if f.level == FAIL], [], format_findings(findings)
        )
        self.assertEqual(_levels(findings, "norm_stats['state']"), [])

        # An explicit state-token override must check the declared 20-value
        # physical state width, rather than silently accepting padded stats.
        overridden = check_checkpoint(
            ckpt.path, "dual_franka_tcp_rot6d", discrete_state=True
        )
        self.assertEqual(_levels(overridden, "norm_stats['state'] width"), [FAIL])

    def test_dual_franka_checks_effective_action_stats_coverage(self) -> None:
        for width in (19, 20, 24, 32, 64):
            with self.subTest(width=width):
                ckpt = CheckpointFixture(
                    self, actions=_stats(width), state=_stats(32)
                ).write_tokenizer()
                findings = check_checkpoint(ckpt.path, "dual_franka_tcp_rot6d")
                self.assertEqual(
                    _levels(findings, "norm_stats['actions'] width"),
                    [FAIL if width < 20 else INFO],
                )

    def test_dual_franka_checks_checkpoint_model_width(self) -> None:
        for model_width in (16, 20, 32, 64):
            with self.subTest(model_width=model_width):
                ckpt = CheckpointFixture(self, actions=_stats(20)).write_tokenizer()
                (ckpt.path / "config.json").write_text(json.dumps({
                    "model_type": "pi05", "action_dim": model_width,
                    "action_horizon": 50, "paligemma_variant": "gemma_2b",
                }))
                findings = check_checkpoint(ckpt.path, "dual_franka_tcp_rot6d")
                self.assertEqual(_levels(findings, "model action width"),
                                 [FAIL] if model_width < 20 else [])

    def test_dual_franka_reads_model_width_from_openpi_metadata(self) -> None:
        for model_width in (16, 64):
            with self.subTest(model_width=model_width):
                ckpt = (CheckpointFixture(self, actions=_stats(20)).write_tokenizer()
                        .write_weights().write_metadata_pt(action_dim=model_width))
                findings = check_checkpoint(ckpt.path, "dual_franka_tcp_rot6d")
                self.assertEqual(_levels(findings, "model action width"),
                                 [FAIL] if model_width < 20 else [])

    def test_dual_franka_action_override_uses_controller_width(self) -> None:
        ckpt = CheckpointFixture(
            self, actions=_stats(32), state=_stats(32)
        ).write_tokenizer()
        matching = check_checkpoint(
            ckpt.path, "dual_franka_tcp_rot6d", action_dim=20
        )
        self.assertEqual(_levels(matching, "action_dim"), [])
        padded = check_checkpoint(
            ckpt.path, "dual_franka_tcp_rot6d", action_dim=32
        )
        self.assertEqual(_levels(padded, "action_dim"), [WARN])

    def test_libero_preset_checks_its_own_asymmetric_widths(self) -> None:
        """LIBERO is 8-dim state / 7-dim action; the two are checked separately.

        A single "robot width" field would have to pick one and skip the other,
        so this pins that the preset carries both.
        """
        preset = get_robot_preset("franka_libero")
        self.assertEqual((preset.state_dim, preset.action_width), (8, 7))

        # LIBERO's default is discrete_state=False, so only "actions" is checked;
        # turn state injection on and the 8-dim entry has to be there and correct.
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(16)).write_tokenizer()
        off = check_checkpoint(ckpt.path, "franka_libero")
        self.assertEqual([f for f in off if f.level == FAIL], [], format_findings(off))

        on = check_checkpoint(ckpt.path, "franka_libero", discrete_state=True)
        self.assertEqual(_levels(on, "norm_stats['state'] width"), [FAIL])

    def test_flat_norm_stats_layout_is_read_too(self) -> None:
        """openpi writes ``{"norm_stats": {...}}``; some exports are flat."""
        ckpt = CheckpointFixture(self)
        ckpt.write_norm_stats(
            nested=False, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(
            [f for f in findings if f.level == FAIL], [], format_findings(findings)
        )

    def test_missing_quantiles_are_fatal_for_pi05(self) -> None:
        """pi05 always unnormalizes with q01/q99 regardless of any config field."""
        ckpt = CheckpointFixture(
            self,
            actions=_stats(G1_DIM, quantiles=False),
            state=_stats(G1_DIM),
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "norm_stats['actions'] quantiles"), [FAIL])

    def test_missing_state_entry_warns_only_when_state_is_used(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM)).write_tokenizer()

        used = check_checkpoint(ckpt.path, "unitree_g1")
        self.assertEqual(_levels(used, "norm_stats['state']"), [WARN])

        dropped = check_checkpoint(ckpt.path, "unitree_g1", discrete_state=False)
        self.assertEqual(_levels(dropped, "norm_stats['state']"), [])

    def test_missing_file_warns_about_identity_passthrough(self) -> None:
        ckpt = CheckpointFixture(self)
        ckpt.write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "norm_stats.json"), [WARN])


class TokenizerTest(unittest.TestCase):
    """A14: openpi downloads its tokenizer at runtime, so exports lack one."""

    def test_absent_tokenizer_is_fatal(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "tokenizer"), [FAIL])

    def test_either_accepted_filename_satisfies_the_check(self) -> None:
        for name in ("tokenizer.model", "paligemma_tokenizer.model"):
            with self.subTest(name=name):
                ckpt = CheckpointFixture(
                    self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
                ).write_tokenizer(name)
                findings = check_checkpoint(ckpt.path, "unitree_g1")
                self.assertEqual(_levels(findings, "tokenizer"), [INFO])

    def test_explicit_path_outranks_the_checkpoint_and_is_itself_checked(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))
        elsewhere = ckpt.path / "elsewhere.model"

        missing = check_checkpoint(ckpt.path, "unitree_g1", tokenizer_path=elsewhere)
        self.assertEqual(_levels(missing, "tokenizer"), [FAIL])

        elsewhere.write_bytes(b"x")
        present = check_checkpoint(ckpt.path, "unitree_g1", tokenizer_path=elsewhere)
        self.assertEqual(_levels(present, "tokenizer"), [INFO])


class OverrideTest(unittest.TestCase):
    """The flags that turn a correct checkpoint into a wrong deployment."""

    def setUp(self) -> None:
        self.ckpt = CheckpointFixture(
            self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

    def test_action_dim_override_that_contradicts_the_robot_warns(self) -> None:
        """``--action-dim 7`` on a 16-DoF robot: the other half of the delivery."""
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", action_dim=7)

        self.assertEqual(_levels(findings, "action_dim"), [WARN])

    def test_action_dim_matching_the_robot_is_silent(self) -> None:
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", action_dim=G1_DIM)

        self.assertEqual(_levels(findings, "action_dim"), [])

    def test_turning_state_off_on_a_joint_space_robot_warns(self) -> None:
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", discrete_state=False)

        warned = [f for f in findings if f.check == "discrete_state"]
        self.assertEqual([f.level for f in warned], [WARN])
        # State is *dropped*, not left continuous — the remedy has to say so,
        # because "no discrete state" reads like a precision choice.
        self.assertIn("dropped", warned[0].remedy)

    def test_reordered_camera_keys_warn_even_at_the_right_count(self) -> None:
        preset = get_robot_preset("unitree_g1")
        swapped = (preset.image_keys[1], preset.image_keys[0], preset.image_keys[2])

        findings = check_checkpoint(self.ckpt.path, "unitree_g1", image_keys=swapped)

        self.assertEqual(_levels(findings, "cameras"), [WARN])

    def test_fewer_cameras_than_the_preset_warns(self) -> None:
        preset = get_robot_preset("unitree_g1")

        findings = check_checkpoint(self.ckpt.path, "unitree_g1", image_keys=preset.image_keys[:2])

        self.assertEqual(_levels(findings, "cameras"), [WARN])


class ReportTest(unittest.TestCase):
    def test_findings_are_sorted_most_severe_first(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(8))

        findings = check_checkpoint(ckpt.path, "unitree_g1", action_dim=7)

        levels = [f.level for f in findings]
        self.assertEqual(levels, sorted(levels, key={FAIL: 0, WARN: 1, INFO: 2}.get))

    def test_quiet_rendering_drops_info_but_keeps_problems(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(G1_DIM))

        findings = check_checkpoint(ckpt.path, "unitree_g1")
        quiet = format_findings(findings, include_info=False)

        self.assertNotIn("[INFO", quiet)
        self.assertIn("[FAIL", quiet)


class CheckpointLayoutTest(unittest.TestCase):
    """OpenPI statistics are resolved from ``assets/<asset_id>/``."""

    def test_the_asset_path_is_checked_not_the_root_file(self) -> None:
        # The failure shape: correct statistics where openpi puts them, a
        # leftover 7-dim LIBERO file in the root. Reading the root one produces
        # a spurious FAIL here and wrong actions on the robot.
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(8))
        ckpt.write_weights().write_metadata_pt(asset_id="example-asset")
        ckpt.write_asset_stats(
            "example-asset", actions=_stats(G1_DIM), state=_stats(G1_DIM)
        )

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertNotIn(FAIL, _levels(findings, "norm_stats["))
        chosen = [f for f in findings if f.check == "norm_stats.json"]
        self.assertEqual([f.level for f in chosen], [INFO])
        self.assertIn("assets/example-asset/norm_stats.json", chosen[0].detail)

    def test_the_root_fallback_is_a_warning_not_a_silent_success(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))
        ckpt.write_weights().write_metadata_pt(asset_id="example-asset")

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        warned = [f for f in findings if f.check == "norm_stats.json"]
        self.assertEqual([f.level for f in warned], [WARN])
        # Both paths, so the operator can see what to go and fetch.
        self.assertIn("assets/example-asset/norm_stats.json", warned[0].detail)
        self.assertIn(str(ckpt.path / "norm_stats.json"), warned[0].detail)

    def test_absent_statistics_name_every_path_tried(self) -> None:
        ckpt = CheckpointFixture(self)
        ckpt.write_weights().write_metadata_pt(asset_id="example-asset")

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        warned = [f for f in findings if f.check == "norm_stats.json"]
        self.assertEqual([f.level for f in warned], [WARN])
        self.assertIn("identity passthrough", warned[0].detail)

    def test_the_layout_and_asset_id_are_reported(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))
        ckpt.write_weights().write_metadata_pt(asset_id="example-asset")

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        reported = {f.check: f.detail for f in findings}
        layout = [f.detail for f in findings if f.check == "checkpoint layout"]
        self.assertIn("openpi_pytorch", layout[0])
        self.assertIn("example-asset", reported["asset_id"])
        # The architecture comes out of metadata.pt, which is the only place an
        # openpi export states it: no config.json exists to read it from.
        self.assertIn("action_horizon=50", reported["architecture"])

    def test_a_flat_directory_is_still_checked_the_old_way(self) -> None:
        """No metadata.pt, no config.json: the hand-assembled layout still loads."""
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertNotIn(FAIL, _levels(findings, "norm_stats"))
        self.assertNotIn("checkpoint layout", {f.check for f in findings})

    def test_an_explicit_norm_stats_path_works_without_any_layout(self) -> None:
        """A flat directory plus --norm-stats: no metadata.pt, no config.json.

        The statistics for a hand-assembled directory often live outside it, so
        naming the file must not require the directory to declare a layout — and
        the root file it does have must then be ignored, not checked.
        """
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(8))
        elsewhere = ckpt.path / "elsewhere.json"
        elsewhere.write_text(
            json.dumps({"norm_stats": {"actions": _stats(G1_DIM), "state": _stats(G1_DIM)}})
        )

        findings = check_checkpoint(ckpt.path, "unitree_g1", norm_stats=elsewhere)

        self.assertNotIn(FAIL, _levels(findings, "norm_stats"))
        chosen = [f for f in findings if f.check == "norm_stats.json"]
        self.assertIn(str(elsewhere), chosen[0].detail)

    def test_a_missing_explicit_norm_stats_path_is_fatal(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))

        findings = check_checkpoint(
            ckpt.path, "unitree_g1", norm_stats=ckpt.path / "nope.json"
        )

        self.assertEqual(_levels(findings, "norm_stats.json"), [FAIL])

    def test_lerobot_base_identity_is_warned_but_not_rejected(self) -> None:
        ckpt = CheckpointFixture(self).write_lerobot_processors(state_files=False)
        ckpt.write_tokenizer()

        findings = check_checkpoint(ckpt.path, "franka_libero")

        self.assertEqual([f for f in findings if f.level == FAIL], [], format_findings(findings))
        self.assertEqual(_level(findings, "action normalization"), [WARN])
        # Identity, but identity at the *right* width: the engine reports the
        # passthrough, the preset confirms the shape it passes through.
        self.assertEqual(_level(findings, "action normalization width"), [INFO])

    def test_lerobot_finetune_sidecars_are_reported_as_resolved(self) -> None:
        ckpt = CheckpointFixture(self).write_lerobot_processors(state_files=True)
        ckpt.write_tokenizer()

        findings = check_checkpoint(ckpt.path, "franka_libero", discrete_state=True)

        self.assertEqual([f for f in findings if f.level == FAIL], [], format_findings(findings))
        self.assertEqual(_level(findings, "action normalization"), [INFO])
        self.assertEqual(_level(findings, "state normalization"), [INFO])


if __name__ == "__main__":
    unittest.main()

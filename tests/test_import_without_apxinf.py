"""apxinf-robo runs on the S600 board, where ApxInf cannot be built.

Most of these run with an import *attempt* of ``apxinf`` raising ``AssertionError``
(``_no_apxinf.py``), so a ``try/except ImportError`` probe cannot hide one: what is
asserted is that the public layer never tries, not that it copes with failing.
``test_selecting_a_missing_engine_says_so`` is the exception on purpose, using the
``absent`` mode that behaves like a machine without the package.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

from _require import needs_apxinf

BLOCKER = pathlib.Path(__file__).with_name("_no_apxinf.py")


def _run(body: str, tmp_path, *, absent: bool = False) -> subprocess.CompletedProcess:
    script = BLOCKER.read_text().split('if __name__')[0] + textwrap.dedent(body)
    env = {**os.environ, "APXINF_ROBO_BLOCK_MODE": "absent" if absent else "strict"}
    return subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
                          text=True, timeout=60, env=env)


def test_package_and_every_preset_need_no_engine(tmp_path):
    result = _run(
        """
        import apxinf_robo
        from apxinf_robo.presets import available_robots, get_robot_preset
        assert available_robots() == (
            "franka_libero", "unitree_g1", "dual_franka_tcp_rot6d", "yam_joint",
        ), available_robots()
        preset = get_robot_preset("franka_libero")
        assert preset.image_keys == ("observation/image", "observation/wrist_image")
        assert preset.state_dim == 8 and preset.action_dim == 7
        assert not preset.has_transform
        g1 = get_robot_preset("unitree_g1")
        assert g1.has_transform and g1.state_dim == 16
        assert apxinf_robo.UNITREE_G1 is g1 and apxinf_robo.YAM and apxinf_robo.DUAL_FRANKA
        assert [m for m in sys.modules if m.split(".")[0] == "apxinf"] == []
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_robot_transforms_wrap_a_policy_without_the_engine(tmp_path):
    result = _run(
        """
        import numpy as np
        from apxinf_robo import TransformedPolicy
        from apxinf_robo.presets import get_robot_preset

        class Engine:
            metadata = {}
            action_dim, action_horizon = 32, 2
            def infer(self, observation, *, noise=None):
                # An identity rotation in both dual-Franka arms (cols 3:9 and 13:19).
                row = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 4)[:32].astype(np.float32)
                return {"actions": np.tile(row, (2, 1)), "timing": {"total_ms": 0.0}}
            def close(self):
                pass

        for name, state_dim, width in (("unitree_g1", 16, 16), ("yam_joint", 14, 14),
                                       ("dual_franka_tcp_rot6d", 20, 20)):
            preset = get_robot_preset(name)
            transform = preset.transform(preset.state_key)
            state = np.eye(1, state_dim)[0]
            if name == "dual_franka_tcp_rot6d":
                arm = [0, 0, 0, 1, 0, 0, 0, 1, 0]  # xyz + identity rot6d
                state = np.array([0.0, 0.0, *arm, *arm])
            policy = TransformedPolicy(Engine(), transform)
            out = policy.infer({preset.state_key: state})
            assert out["actions"].shape == (2, width), (name, out["actions"].shape)
        assert [m for m in sys.modules if m.split(".")[0] == "apxinf"] == []
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_commands_used_on_the_board_need_no_engine(tmp_path):
    result = _run(
        """
        from apxinf_robo.cli import eval_libero, serve
        args = serve.build_parser().parse_args(["--engine", "drobotics"])
        assert args.engine == "drobotics"
        args = eval_libero.parse_args(
            ["--backend", "websocket", "--precision", "w8",
             "--results-jsonl", "r", "--summary-json", "s"])
        assert eval_libero.libero_convention().name == "libero"
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_selecting_a_missing_engine_says_so(tmp_path):
    result = _run(
        """
        import apxinf_robo
        from apxinf_robo import build_robot_policy, load_policy

        def hint(call):
            try:
                call()
            except ImportError as error:
                assert "apxinf is not importable" in str(error), error
            else:
                raise SystemExit("resolved without ApxInf")

        # Only what needs ApxInf fails, and only when it is actually used.
        hint(lambda: apxinf_robo.check_checkpoint)
        hint(lambda: apxinf_robo.load_bare_model)
        hint(lambda: load_policy("/nowhere"))
        hint(lambda: build_robot_policy("unitree_g1", "/nowhere"))
        hint(lambda: apxinf_robo.build_unitree_g1_policy(
            "/nowhere", state_key="s", image_keys=("a", "b", "c")))
        try:
            apxinf_robo.no_such_name
        except AttributeError:
            pass
        """,
        tmp_path,
        absent=True,
    )
    assert result.returncode == 0, result.stderr


def test_star_imports_work_without_the_engine(tmp_path):
    result = _run(
        """
        from apxinf_robo import *
        import apxinf_robo
        # Always-present names are exported; lazy ApxInf-backed ones stay out of __all__,
        # so a star import never forces ApxInf.
        for name in ("load_bare_model", "check_checkpoint", "format_findings"):
            assert name not in apxinf_robo.__all__, name
        for name in ("build_robot_policy", "load_policy", "websocket_server", "UNITREE_G1"):
            assert name in apxinf_robo.__all__, name
        from apxinf_robo.embodiments import *
        from apxinf_robo.presets import *
        assert FRANKA_LIBERO.name == "franka_libero"
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


@needs_apxinf
def test_installed_engine_changes_no_preset():
    from apxinf_robo.presets import available_robots

    assert available_robots() == (
        "franka_libero", "unitree_g1", "dual_franka_tcp_rot6d", "yam_joint",
    )

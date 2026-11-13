"""apxinf-robo runs on the S600 board, where ApxInf cannot be built."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

BLOCKER = pathlib.Path(__file__).with_name("_no_apxinf.py")


def _run(body: str, tmp_path) -> subprocess.CompletedProcess:
    script = BLOCKER.read_text().split('if __name__')[0] + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
                          text=True, timeout=60)


def test_package_and_the_libero_preset_need_no_engine(tmp_path):
    result = _run(
        """
        import apxinf_robo
        from apxinf_robo.presets import available_robots, get_robot_preset
        assert not apxinf_robo.apxinf_available()
        assert available_robots() == ("franka_libero",), available_robots()
        preset = get_robot_preset("franka_libero")
        assert preset.image_keys == ("observation/image", "observation/wrist_image")
        assert preset.state_dim == 8 and preset.action_dim == 7
        assert not preset.has_robot_steps
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


def test_what_needs_apxinf_says_so(tmp_path):
    result = _run(
        """
        import apxinf_robo, apxinf_robo.presets, apxinf_robo.embodiments
        for module, name in (
            (apxinf_robo, "UNITREE_G1"),
            (apxinf_robo, "check_checkpoint"),
            # ApxInf's own implementations are not offered where it is absent.
            (apxinf_robo, "load_policy"),
            (apxinf_robo, "load_bare_model"),
            (apxinf_robo, "websocket_server"),
            (apxinf_robo.presets, "YAM"),
            (apxinf_robo.embodiments, "build_yam_policy"),
        ):
            try:
                getattr(module, name)
            except ImportError as error:
                assert "apxinf is not importable" in str(error), error
            else:
                raise SystemExit(f"{name} resolved without ApxInf")
        try:
            apxinf_robo.no_such_name
        except AttributeError:
            pass
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_star_imports_work_without_the_engine(tmp_path):
    result = _run(
        """
        from apxinf_robo import *
        import apxinf_robo
        for name in ("load_policy", "load_bare_model", "websocket_server"):
            assert name not in apxinf_robo.__all__, name
        assert "build_robot_policy" in apxinf_robo.__all__
        from apxinf_robo.embodiments import *
        from apxinf_robo.presets import *
        assert FRANKA_LIBERO.name == "franka_libero"
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_installed_engine_keeps_every_preset():
    import pytest

    pytest.importorskip("apxinf.policies.base")
    from apxinf_robo.presets import available_robots

    assert available_robots() == (
        "franka_libero", "unitree_g1", "dual_franka_tcp_rot6d", "yam_joint",
    )

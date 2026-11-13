"""Exercise engine import diagnostics without the test runner's installed engine."""

import pathlib
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("namespace", [False, True])
def test_missing_engine_keeps_install_instructions(tmp_path, namespace):
    result = _probe_import(tmp_path, namespace=namespace, installed=False)
    assert result.returncode != 0
    assert "pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict" in result.stderr
    assert "git submodule update --init --recursive" in result.stderr
    assert "instead of the installed engine" not in result.stderr
    if namespace:
        assert "resolved to a namespace containing" in result.stderr
        assert "If the engine is already installed editable" in result.stderr


def test_regular_engine_takes_precedence_over_checkout_namespace(tmp_path):
    result = _probe_import(tmp_path, namespace=True, installed=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "installed engine"


def test_shadowing_is_its_own_error_and_is_not_swallowed_silently(tmp_path):
    shadowed = _probe_import(tmp_path, namespace=True, installed=False,
                             expr="engine.require_apxinf()")
    assert "ApxinfShadowedError" in shadowed.stderr
    # Not installed at all stays a plain ImportError that is not the shadow subclass.
    plain = _probe_import(tmp_path / "plain", namespace=False, installed=False,
                          expr="engine.require_apxinf()")
    assert "ImportError" in plain.stderr and "ApxinfShadowedError" not in plain.stderr


def test_available_is_false_for_shadowing_but_warns_once(tmp_path):
    """On the board an empty apxinf/ directory must not break import -- but not hide it."""
    result = _probe_import(
        tmp_path, namespace=True, installed=False,
        expr="(__import__('warnings').simplefilter('always'), "
             "[engine.apxinf_available() for _ in range(3)])[1]",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[False, False, False]"
    assert result.stderr.count("resolved to a namespace containing") == 1


def test_available_is_a_quiet_false_when_not_installed(tmp_path):
    result = _probe_import(tmp_path, namespace=False, installed=False,
                           expr="engine.apxinf_available()")
    assert result.stdout.strip() == "False"
    assert "namespace" not in result.stderr


def _probe_import(tmp_path, *, namespace, installed, expr="engine.require_apxinf().MARKER"):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    if namespace:
        (checkout / "apxinf").mkdir()
    site = tmp_path / "site"
    site.mkdir()
    if installed:
        package = site / "apxinf"
        package.mkdir()
        (package / "__init__.py").write_text('MARKER = "installed engine"\n')

    source = pathlib.Path(__file__).resolve().parents[1] / "src/apxinf_robo/engine.py"
    # Load engine.py directly: importing robo first would require an engine and
    # prevent us from exercising its missing-install diagnostic. -I -S removes
    # ambient installs, editable finders, PYTHONPATH, and pytest's module cache.
    script = textwrap.dedent("""\
        import importlib.util
        import pathlib
        import sys

        source, checkout, site, expr = sys.argv[1:]
        spec = importlib.util.spec_from_file_location("engine_under_test", source)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        engine.__file__ = str(pathlib.Path(checkout) / "src/apxinf_robo/engine.py")
        sys.path[:0] = [checkout, site]
        print(eval(expr))
        """)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(source),
            str(checkout),
            str(site),
            expr,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

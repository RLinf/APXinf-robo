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


def _probe_import(tmp_path, *, namespace, installed):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
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

        source, checkout, site = sys.argv[1:]
        spec = importlib.util.spec_from_file_location("engine_under_test", source)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        engine.__file__ = str(pathlib.Path(checkout) / "src/apxinf_robo/engine.py")
        sys.path[:0] = [checkout, site]
        print(engine.require_apxinf().MARKER)
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
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

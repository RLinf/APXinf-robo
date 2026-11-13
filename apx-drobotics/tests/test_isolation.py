"""apx-drobotics stands next to ApxInf and apxinf-robo, so it must import without either."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_imports_without_apxinf_or_robo(tmp_path):
    script = textwrap.dedent(
        """
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("apxinf", "apxinf_py", "apxinf_robo"):
                    raise ModuleNotFoundError(f"blocked {name}", name=name)

        sys.meta_path.insert(0, Block())
        import apx_drobotics
        import apx_drobotics.contract, apx_drobotics.postprocess, apx_drobotics.preprocess
        import apx_drobotics.runtime, apx_drobotics.serving, apx_drobotics.testing
        print("imported")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_contract_module_needs_only_the_standard_library(tmp_path):
    from apx_drobotics import contract

    result = subprocess.run(
        [sys.executable, "-I", "-S", contract.__file__, "print-metadata"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "contract_sha256" in result.stdout

"""Run a command as it runs on the S600 board, where ApxInf cannot be imported.

``python tests/_no_apxinf.py <module> [args...]`` runs ``module.main(args)``.

By default an import *attempt* of a blocked package raises ``AssertionError``, not
``ImportError``: code that probes with ``try: import apxinf / except ImportError`` would
swallow the latter, and "the public layer never tries to import an engine" is the thing
being checked. ``APXINF_ROBO_BLOCK_MODE=absent`` raises ``ModuleNotFoundError`` instead,
which is how a machine without the package really behaves; use it to test what happens
when a caller *does* select a missing engine. ``APXINF_ROBO_BLOCK`` names the packages
(default ``apxinf,apxinf_py``).
"""

import importlib
import os
import sys

_NAMES = frozenset(os.environ.get("APXINF_ROBO_BLOCK", "apxinf,apxinf_py").split(","))
_ABSENT = os.environ.get("APXINF_ROBO_BLOCK_MODE") == "absent"


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _NAMES:
            if _ABSENT:
                raise ModuleNotFoundError(f"blocked {name}", name=name)
            raise AssertionError(f"import of {name!r} was attempted, and must not be")


sys.meta_path.insert(0, _Block())

if __name__ == "__main__":
    sys.exit(importlib.import_module(sys.argv[1]).main(sys.argv[2:]))

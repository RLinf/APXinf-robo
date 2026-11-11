"""Run a command as it runs on the S600 board, where ApxInf cannot be imported.

``python tests/_no_apxinf.py <module> [args...]`` runs ``module.main(args)``.
"""

import importlib
import sys


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("apxinf", "apxinf_py"):
            raise ModuleNotFoundError(f"blocked {name}", name=name)


sys.meta_path.insert(0, _Block())

if __name__ == "__main__":
    sys.exit(importlib.import_module(sys.argv[1]).main(sys.argv[2:]))

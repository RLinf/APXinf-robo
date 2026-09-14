"""Keep the repository root off ``sys.path`` during collection.

The engine submodule is checked out at ``apxinf/``, which is also the name of the
package it installs. Anything that puts this directory on ``sys.path`` -- ``python
-m pytest``, a REPL started here, ``python -c`` -- therefore resolves ``import
apxinf`` to the submodule *directory* as an empty implicit namespace package
instead of to the installed engine. The failure surfaces later and elsewhere, as
``cannot import name 'Pi05Policy' from 'apxinf' (unknown location)``, which reads
like a version mismatch rather than a shadowed name.

Bare ``pytest`` never had the problem (rootdir is not prepended), so this only
makes the two invocations agree. Dropping the entry is safe: nothing in this
repository is imported from the root -- the package lives under ``src/`` and the
tests import it by name.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parent)

for _entry in ("", ".", _ROOT):
    while _entry in sys.path:
        sys.path.remove(_entry)

# Drop a namespace package already bound to the submodule directory, so a later
# ``import apxinf`` re-resolves against the cleaned path.
_apxinf = sys.modules.get("apxinf")
if _apxinf is not None and getattr(_apxinf, "__file__", None) is None:
    for _name in [name for name in sys.modules if name == "apxinf" or name.startswith("apxinf.")]:
        del sys.modules[_name]

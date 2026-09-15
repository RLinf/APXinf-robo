"""Keep the checkout namespace off the test import path.

The apxinf/ submodule can interfere with editable-install finders. Removing the
repository root makes collection independent of the pytest invocation.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

for _entry in ("", ".", _ROOT):
    while _entry in sys.path:
        sys.path.remove(_entry)

# Drop a namespace package already bound to the submodule directory, so a later
# ``import apxinf`` re-resolves against the cleaned path.
_apxinf = sys.modules.get("apxinf")
if _apxinf is not None and getattr(_apxinf, "__file__", None) is None:
    for _name in [name for name in sys.modules if name == "apxinf" or name.startswith("apxinf.")]:
        del sys.modules[_name]

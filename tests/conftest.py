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


# --- forbidden engines ---------------------------------------------------------------
#
# ``APXINF_ROBO_FORBID_IMPORT=apxinf,apxinf_py`` makes any *attempt* to import those
# top-level packages fail the test with an AssertionError. It is not an ImportError on
# purpose: code that probes with ``try: import apxinf / except ImportError`` would
# otherwise swallow the attempt, and "the public layer never tries to import the
# engine" is exactly what this guards. CI sets it for the run without ApxInf (and
# ``apx_drobotics`` for the reverse run).
import os  # noqa: E402


class ForbiddenImport:
    """A ``sys.meta_path`` finder that turns an import attempt into an assertion error."""

    def __init__(self, names):
        self.names = frozenset(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise AssertionError(f"import of forbidden engine package {name!r} was attempted")
        return None


_forbidden = [n for n in os.environ.get("APXINF_ROBO_FORBID_IMPORT", "").split(",") if n]
if _forbidden:
    sys.meta_path.insert(0, ForbiddenImport(_forbidden))

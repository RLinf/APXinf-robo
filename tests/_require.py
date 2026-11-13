"""Skip a test module when an optional dependency is missing -- unless CI forbids that.

``pytest.importorskip`` turns a missing dependency into a green run with the tests
silently gone, which is how an end-to-end suite can stop running without anyone
noticing. Set ``APXINF_ROBO_REQUIRE_OPTIONAL=1`` in CI (with the ``drobotics-test``
extra and ``pip install -e ./apx-drobotics[serve]`` installed) and a missing module
becomes a failure instead.
"""

from __future__ import annotations

import importlib
import os

import pytest


def require(name: str):
    """Import ``name``; skip if absent, or fail if ``APXINF_ROBO_REQUIRE_OPTIONAL`` is set."""
    if os.environ.get("APXINF_ROBO_REQUIRE_OPTIONAL"):
        return importlib.import_module(name)
    return pytest.importorskip(name)

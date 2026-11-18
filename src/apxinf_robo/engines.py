"""The one place that picks an engine, and the only module that names more than one.

Everything above this (presets, bodies, conventions, the CLIs' option handling) is written
against :class:`~apxinf_robo.protocol.Policy` and never imports an engine. A caller says
``engine="apxinf"`` or ``engine="drobotics"`` here and gets a policy; the chosen engine's
module is imported at that moment and not before, so selecting one needs nothing from the
other, and a missing engine raises its own install hint only when it is the one selected.

Nothing here probes whether an engine is importable to decide what to offer: the set of
engines is the same on every machine, and an unavailable one fails when used.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ENGINES", "load_engine_policy", "websocket_server"]

#: Accepted ``engine=`` values.
ENGINES = ("apxinf", "drobotics")


def _check(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; known: {list(ENGINES)}")


def load_engine_policy(engine: str, model_dir, **kwargs: Any):
    """Load ``model_dir`` on ``engine``; ``kwargs`` reach that engine's loader."""
    _check(engine)
    if engine == "drobotics":
        from .engine_drobotics import load_policy
    else:
        from .engine import load_policy
    return load_policy(model_dir, **kwargs)


def websocket_server(policy, host: str, port: int, *, engine: str = "apxinf"):
    """The OpenPI websocket server for a ``policy`` that ``engine`` produced."""
    _check(engine)
    if engine == "drobotics":
        from .engine_drobotics import websocket_server as make
    else:
        from .engine import websocket_server as make
    return make(policy, host, port)

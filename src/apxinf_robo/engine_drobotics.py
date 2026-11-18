"""The one module that loads and runs the D-Robotics S600, through ``apx_drobotics``.

This is the S600 counterpart of :mod:`apxinf_robo.engine`, and there is little in it
because :class:`apx_drobotics.DroboticsPolicy` already has the shape of an ApxInf policy. It also
has the same place: the runtime and the policy loop are the engine's, and only the
wire keys and the choice of robot come from here.

The package is imported lazily, so nothing needs it unless the S600 is selected.
"""

from __future__ import annotations

__all__ = ["load_policy", "require_apx_drobotics", "websocket_server"]

_INSTALL_HINT = (
    "apx_drobotics is not importable. It lives in this repository as its own package, and "
    "it is installed separately because it only runs on an S600 board:\n"
    "    pip install -e './apx-drobotics[serve]'"
)


def require_apx_drobotics():
    """Import and return the ``apx_drobotics`` package, with an actionable error."""
    try:
        import apx_drobotics
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return apx_drobotics


def load_policy(model_dir=None, **kwargs):
    """Load the S600 policy. ``model_dir`` is the SDK's ``oellm_runtime_s600`` directory.

    It holds the model and the runtime config, so it is the closest thing this engine
    has to a checkpoint directory. ``None`` selects the location ``scripts/deploy.sh``
    uses. ``kwargs`` reach :meth:`apx_drobotics.DroboticsPolicy.from_deployment`.
    """
    return require_apx_drobotics().DroboticsPolicy.from_deployment(model_dir, **kwargs)


def websocket_server(policy, host: str, port: int):
    """The S600 OpenPI server around ``policy``; ``.serve_forever()`` returns an exit status."""
    require_apx_drobotics()
    from apx_drobotics.serving import WebsocketPolicyServer

    return WebsocketPolicyServer(policy, host, port)

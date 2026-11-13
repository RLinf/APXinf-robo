"""apx-drobotics: Pi0.5 on the D-Robotics S600, usable wherever an ApxInf policy is.

The package wraps the SDK, which it treats as a black box, in three layers::

    Apx-drobotics-preprocess   preprocess.py   raw observation -> what the SDK should receive
    SDK oellm-runtime     runtime.py      the C++ worker, driven synchronously
    Apx-drobotics-postprocess  postprocess.py  the SDK's reply -> {actions, timing}

:class:`DroboticsPolicy` joins them behind the ``infer(observation)`` shape of an ApxInf
policy, and :mod:`apx_drobotics.serving` is its websocket server. The package stands next to
ApxInf and imports neither that nor apxinf-robo.
"""

from __future__ import annotations

from .contract import DEFAULT_CONTRACT, load_contract, model_metadata
from .errors import DeploymentError, RequestError, RuntimeFailure, DroboticsError, StartupFailure
from .policy import DroboticsPolicy
from .runtime import DEFAULT_SDK_ROOT, DEFAULT_WORKER

__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_SDK_ROOT",
    "DEFAULT_WORKER",
    "DeploymentError",
    "RequestError",
    "RuntimeFailure",
    "DroboticsError",
    "DroboticsPolicy",
    "StartupFailure",
    "load_contract",
    "model_metadata",
]

from ._version import __version__  # noqa: E402

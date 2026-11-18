"""apxinf-robo: the robot-facing layer over ApxInf and the S600.

ApxInf is an inference engine. It knows checkpoints, tensors, and the numeric
chain between them. It deliberately does not know that a Unitree G1 has sixteen
degrees of freedom, that LIBERO calls its wrist camera
``observation/wrist_image``, or that robosuite's ``OSC_POSE`` controller wants
axis-angle. Those facts change when you swap the *body* or the *environment*,
not when you swap the checkpoint, and they belong to whoever owns the robot.

That is this package. It supplies four separable things:

:mod:`~apxinf_robo.conventions`
    Wire-key dialects -- what a client calls its cameras, state, and prompt.

:mod:`~apxinf_robo.embodiments`
    Bodies -- DoF layout, gripper conventions, delta masks, and the pre/post
    steps that turn a body's numbers into the model's and back.

:mod:`~apxinf_robo.presets`
    Named ``Embodiment x Convention`` pairings, and :func:`build_robot_policy`
    to load one.

:mod:`~apxinf_robo.envs`
    Evaluation-environment glue (LIBERO, robosuite, IsaacLab).

Everything reaches an engine through one module: :mod:`apxinf_robo.engine` for
ApxInf, and :mod:`apxinf_robo.engine_drobotics` for the D-Robotics S600 (``apx_drobotics``).
Both give a policy of the :class:`~apxinf_robo.protocol.Policy` shape, so the layers
above do not depend on either, and this package imports without ApxInf, as it must on
the S600 board, and without ``apx_drobotics``. Every body and preset is available
everywhere; only checkpoint preflight and ``load_bare_model`` need ApxInf, and touching
them there raises its install hint.

    >>> from apxinf_robo import build_robot_policy, check_checkpoint, format_findings
    >>> print(format_findings(check_checkpoint("/ckpt/pi05_g1", "unitree_g1")))  # doctest: +SKIP
    >>> policy = build_robot_policy("unitree_g1", "/ckpt/pi05_g1")              # doctest: +SKIP
"""

from __future__ import annotations

from . import conventions, embodiments, protocol
from .conventions import (
    CONVENTIONS,
    Convention,
    available_conventions,
    get_convention,
    register_convention,
)
from .embodiments import (
    ROBOT_ALIASES,
    ROBOT_PRESETS,
    VIEW_SLOTS,
    Embodiment,
    RobotPreset,
    available_robots,
    get_robot_preset,
    build_unitree_g1_policy,
    register_robot_preset,
)
from .engine import ApxInfEngine, apxinf_available
from .engines import ENGINES, load_engine_policy
from .engines import websocket_server as _websocket_server
from .presets import DUAL_FRANKA, FRANKA_LIBERO, UNITREE_G1, YAM, build_robot_policy
from .protocol import Policy, RobotTransform, TransformedPolicy, apply_transform


def load_policy(model_dir, *, engine: str = "apxinf", **kwargs):
    """Load ``model_dir`` on ``engine`` (``"apxinf"`` or ``"drobotics"``).

    ``kwargs`` reach that engine's loader. The engine is imported only now, so a missing
    one raises its own install hint here and nothing else is affected.
    """
    return load_engine_policy(engine, model_dir, **kwargs)


def websocket_server(policy, host: str, port: int, *, engine: str = "apxinf"):
    """The OpenPI websocket server around ``policy``, for the ``engine`` that produced it."""
    return _websocket_server(policy, host, port, engine=engine)


#: ApxInf-backed names, resolved on first access. They are always *present*: what this
#: package offers does not depend on what is installed, and touching one without ApxInf
#: raises the install hint. They stay out of ``__all__`` so ``import *`` never forces them.
_LAZY = {
    "load_bare_model": (".engine", "load_bare_model"),
    "FAIL": (".preflight", "FAIL"),
    "WARN": (".preflight", "WARN"),
    "INFO": (".preflight", "INFO"),
    "Finding": (".preflight", "Finding"),
    "check_checkpoint": (".preflight", "check_checkpoint"),
    "inspect_for_robot": (".preflight", "inspect_for_robot"),
    "format_findings": (".preflight", "format_findings"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    from .engine import require_apxinf

    require_apxinf()
    return getattr(importlib.import_module(target[0], __name__), target[1])


__version__ = "0.1.0"

__all__ = [
    # sub-packages
    "conventions",
    "embodiments",
    "protocol",
    # conventions
    "Convention",
    "CONVENTIONS",
    "available_conventions",
    "get_convention",
    "register_convention",
    # bodies and presets
    "VIEW_SLOTS",
    "Embodiment",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "FRANKA_LIBERO",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "build_robot_policy",
    # engine access
    "ENGINES",
    "Policy",
    "RobotTransform",
    "TransformedPolicy",
    "apply_transform",
    "ApxInfEngine",
    "apxinf_available",
    "load_policy",
    "websocket_server",
    "UNITREE_G1",
    "DUAL_FRANKA",
    "YAM",
    "build_unitree_g1_policy",
    "__version__",
]

"""apxinf-robo: the robot-facing layer over the ApxInf inference engine.

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

Everything reaches the engine through :mod:`apxinf_robo.engine`, the one module
that imports ``apxinf``.

    >>> from apxinf_robo import build_robot_policy, check_checkpoint, format_findings
    >>> print(format_findings(check_checkpoint("/ckpt/pi05_g1", "unitree_g1")))  # doctest: +SKIP
    >>> policy = build_robot_policy("unitree_g1", "/ckpt/pi05_g1")              # doctest: +SKIP
"""

from __future__ import annotations

from . import conventions, embodiments
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
    build_unitree_g1_policy,
    get_robot_preset,
    register_robot_preset,
)
from .engine import ApxInfEngine, load_bare_model, load_policy, websocket_server
from .preflight import (
    FAIL,
    INFO,
    WARN,
    Finding,
    check_checkpoint,
    format_findings,
    inspect_for_robot,
)
from .presets import DUAL_FRANKA, FRANKA_LIBERO, UNITREE_G1, YAM, build_robot_policy

__version__ = "0.1.0"

__all__ = [
    # sub-packages
    "conventions",
    "embodiments",
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
    "UNITREE_G1",
    "DUAL_FRANKA",
    "YAM",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "build_robot_policy",
    "build_unitree_g1_policy",
    # engine access
    "ApxInfEngine",
    "load_policy",
    "load_bare_model",
    "websocket_server",
    # preflight
    "FAIL",
    "WARN",
    "INFO",
    "Finding",
    "check_checkpoint",
    "inspect_for_robot",
    "format_findings",
    "__version__",
]

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
the S600 board. There, only what ApxInf provides is missing: the bodies that need its
processor chain, and checkpoint preflight. Touching those raises its install hint.

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
    register_robot_preset,
)
from .engine import ApxInfEngine, apxinf_available, require_apxinf
from .presets import ENGINES, FRANKA_LIBERO, build_robot_policy
from .protocol import Policy

#: Exports that exist only where ApxInf does. ``load_policy``, ``load_bare_model`` and
#: ``websocket_server`` are ApxInf's implementations, not an engine-neutral facade, so
#: they are not offered where ApxInf is absent; use ``build_robot_policy`` (which takes
#: ``engine=``) or ``apxinf_robo.engine_drobotics`` there.
_APXINF_NAMES = (
    "load_policy",
    "load_bare_model",
    "websocket_server",
    "UNITREE_G1",
    "DUAL_FRANKA",
    "YAM",
    "build_unitree_g1_policy",
    "FAIL",
    "WARN",
    "INFO",
    "Finding",
    "check_checkpoint",
    "inspect_for_robot",
    "format_findings",
)

if apxinf_available():
    from .embodiments import build_unitree_g1_policy
    from .engine import load_bare_model, load_policy, websocket_server
    from .preflight import (
        FAIL,
        INFO,
        WARN,
        Finding,
        check_checkpoint,
        format_findings,
        inspect_for_robot,
    )
    from .presets import DUAL_FRANKA, UNITREE_G1, YAM
else:

    def __getattr__(name: str):
        if name in _APXINF_NAMES:
            require_apxinf()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "ApxInfEngine",
    "apxinf_available",
    # needs ApxInf
    *(_APXINF_NAMES if apxinf_available() else ()),
    "__version__",
]

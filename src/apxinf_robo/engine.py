"""The one module that loads and runs the ApxInf engine.

Other modules here import ApxInf for *vocabulary* -- the ``Policy`` protocol, the
``ProcessorStep`` base class, ``Finding``, ``VIEW_SLOTS`` -- which is shared type
information and belongs where it is used. What they do not do is construct an
engine, allocate a device, or serve one. That happens here and only here, so the
dependency on a specific engine build is a single file: pinning a new submodule
SHA means reading one diff, and a compatibility shim for an engine API change
has exactly one place to live.

ApxInf exposes two interfaces, and this is where a caller picks one:

``interface="policy"`` (**default**, L2)
    ``AutoPolicy.from_pretrained(...)`` -> ``Policy.infer(obs) -> {actions, ...}``.
    The engine runs its own resize / tokenize / normalize / noise chain — the
    numerically anchored one, covered by ApxInf's golden tests. Use this unless
    you have a reason not to.

``interface="bare"`` (L1)
    ``apxinf.Model.load(...)`` -> ``Model.infer_rgb(rgb_u8, layout, token_ids,
    noise=)``. The caller supplies already-preprocessed tensors. This exists for
    frameworks that bring their own transforms (RLinf's vendored openpi, for
    instance) and do not want a second implementation of them in the loop.

The two paths must agree. That is not automatic — nothing in either repository
checks it — which is why ``tests/test_parity.py`` exists and is the reason this
package is worth having.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional, Sequence

__all__ = [
    "ApxInfEngine",
    "load_policy",
    "load_bare_model",
    "load_random_model",
    "policy_from_random",
    "resolve_tactics",
    "websocket_server",
    "require_apxinf",
    "INTERFACES",
]

#: Accepted ``interface=`` values, in the order they are documented above.
INTERFACES = ("policy", "bare")

_INSTALL_HINT = (
    "apxinf is not importable. It is a git submodule of this repository, not a "
    "pip dependency, because it is a Rust/PyO3/CUDA build that has to be "
    "compiled for the target machine:\n"
    "    git submodule update --init --recursive\n"
    "    pip install -e apxinf/python/apxinf\n"
    "and build the apxinf_py extension with maturin for anything that runs a model."
)


def _shadowing_checkout_root():
    """The submodule directory shadowing ``apxinf``, if that is what happened."""
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    submodule = repo_root / "apxinf"
    if not submodule.is_dir():
        return None
    for entry in sys.path:
        try:
            resolved = pathlib.Path(entry or ".").resolve()
        except OSError:  # pragma: no cover - unreadable sys.path entry
            continue
        if resolved == repo_root:
            return submodule
    return None


def require_apxinf():
    """Import and return the ``apxinf`` package, with an actionable error.

    A bare ``ModuleNotFoundError: apxinf`` reads like a typo. It is almost always
    an uninitialized submodule, so say that instead.

    The submodule directory is itself named ``apxinf``, so anything that puts the
    repository root on ``sys.path`` -- ``python -c``, ``python -m``, a REPL
    started there -- resolves the import to that *directory* as an empty implicit
    namespace package instead of to the installed engine. A real package has a
    ``__file__``; a namespace package does not, which is what separates "not
    installed" from "shadowed by the checkout". Left undetected the failure
    resurfaces later as a missing *submodule* of ``apxinf``, which reads like a
    version mismatch.
    """
    try:
        import apxinf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_INSTALL_HINT) from exc
    if getattr(apxinf, "__file__", None) is None:  # pragma: no cover
        shadow = _shadowing_checkout_root()
        if shadow is not None:
            raise ImportError(
                f"`import apxinf` resolved to the submodule directory {shadow} "
                "instead of the installed engine, because the repository root is "
                "on sys.path. Run this from any other directory -- `pytest` and "
                "the `apxinf-robo` console script are unaffected -- or drop the "
                "root from sys.path the way conftest.py does."
            )
        raise ImportError(_INSTALL_HINT)
    return apxinf


def load_policy(model_dir, **kwargs):
    """L2: dispatch ``model_dir`` to its concrete policy.

    ``kwargs`` reach the policy's ``from_pretrained`` unchanged — ``image_keys``
    / ``state_key`` / ``prompt_key`` / ``discrete_state`` / ``action_dim`` /
    ``device`` / ``precision`` / ``model_type`` / ``metadata`` / ...
    """
    apxinf = require_apxinf()
    return apxinf.AutoPolicy.from_pretrained(model_dir, **kwargs)


# Model families whose L2 loader resolves tuned GEMM tactics for itself.
#
# The tuning database is *not* per-model -- its records are keyed by the physical
# GEMM contract (shapes, dtypes, epilogue, device), and a shape with no record
# falls back to the provider default. What is per-model is whether the family's
# ``from_pretrained`` opts in: ``Pi05Policy`` calls ``resolve_pi05_tactics``,
# ``WallossPolicy`` only forwards an explicit ``tactics=``.
#
# L1 mirrors that list rather than always resolving, because the invariant this
# package sells is that the two interfaces return the same numbers (tests/
# test_parity.py). Tuning L1 for a family whose L2 is untuned would fix pi05's
# divergence by introducing WallOSS's. Deleting this set is the right change
# *after* the engine resolves tactics for every family.
_TACTICS_RESOLVED_BY_L2 = frozenset({"pi05"})


def load_bare_model(
    model_dir,
    *,
    model: str = "pi05",
    device: str = "cuda:0",
    precision: str = "auto",
    **kwargs: Any,
):
    """L1: load the bare model handle, bypassing the engine's numpy chain.

    The returned handle is ``unsendable`` — its CUDA context is bound to the
    thread that created it. Load it on the thread that will call it.

    For families whose L2 loader auto-selects tuned GEMM tactics, they are
    selected here too. Without that the two interfaces are not comparable: the
    policy interface runs the tuned kernels and a bare handle runs the defaults,
    so the same checkpoint on byte-identical inputs returns different numbers
    (measured at 2**-8 in the normalized domain on Thor / bf16). Both answers are
    finite and plausible, which is what makes the difference expensive to find
    later.

    ``tactics=<path>`` overrides the selection; ``tactics=None`` forces the
    provider defaults, which is how you measure what the tuning is worth.
    """
    apxinf = require_apxinf()
    if "tactics" not in kwargs:
        if model in _TACTICS_RESOLVED_BY_L2:
            tactics = resolve_tactics(
                device, precision, model_dir=model_dir, allow_missing=True
            )
            if tactics:
                kwargs["tactics"] = str(tactics)
    elif kwargs["tactics"] is None:
        del kwargs["tactics"]
    return apxinf.Model.load(model, str(model_dir), device, precision, **kwargs)


# --- checkpoint-free path ----------------------------------------------------
#
# ``--random-weights`` serves an engine with deterministic random weights so a
# client can measure latency and preview the wire contract without a checkpoint.
# It bypasses ``from_pretrained``, so the three pieces below have to be reached
# individually rather than through one loader.


def resolve_tactics(
    device: str,
    precision: str,
    *,
    model_dir=None,
    override=None,
    allow_missing: bool = False,
):
    """Pick the GEMM tactics file for ``device``/``precision``.

    ``from_pretrained`` does this internally; ``load_bare_model`` and the random
    path have to ask. ``model_dir`` is honoured because a checkpoint-local
    ``tactics.json`` takes precedence over the source-tree tuning files.

    This is the single place where this package touches an ApxInf *private*
    module (``apxinf._tactics``). It is wrapped here rather than imported at the
    call site so that promoting it to public API upstream is a one-line change,
    and so a submodule bump that moves it breaks exactly one function.
    """
    require_apxinf()
    from apxinf._tactics import resolve_pi05_tactics

    extra = {} if model_dir is None else {"model_dir": pathlib.Path(model_dir)}
    return resolve_pi05_tactics(
        device, precision, override=override, allow_missing=allow_missing, **extra
    )


def load_random_model(**kwargs):
    """Build a checkpoint-free engine handle with deterministic random weights.

    ``kwargs`` are ``apxinf_py.Model.random``'s: ``model`` / ``device`` /
    ``precision`` / ``num_views`` / ``image_size`` / ``action_horizon`` /
    ``action_dim`` / ``num_flow_steps`` / ``max_token_len`` / ``calibration`` /
    ``tactics`` / ``autotune`` / ``seed``.
    """
    import apxinf_py  # lazy: only this path needs the CUDA binding at import time

    return apxinf_py.Model.random(**kwargs)


def policy_from_random(handle, **kwargs):
    """Wrap a random engine handle in a policy with synthetic processors.

    The actions are numerically meaningless by construction -- the tokenizer
    emits a fixed stream and never reads state. What *is* real is the wire
    contract: keys, view count, and shapes.
    """
    return require_apxinf().Pi05Policy.from_random(handle, **kwargs)


def websocket_server(policy, host: str, port: int):
    """An OpenPI-protocol WebSocket server around ``policy``.

    The transport lives in ApxInf because it is model-shaped, not robot-shaped:
    it duck-types ``Policy`` and never looks at a wire key. Call
    ``.serve_forever()`` on the result.
    """
    from apxinf.serving import WebsocketPolicyServer

    return WebsocketPolicyServer(policy, host, port)


class ApxInfEngine:
    """A loaded ApxInf handle at a chosen interface.

    This is a thin holder, not an abstraction over the two interfaces: they take
    genuinely different inputs and pretending otherwise would hide the one thing
    a caller has to understand. ``policy`` and ``bare`` are exposed directly and
    exactly one of them is non-``None``.

    >>> engine = ApxInfEngine.open("/ckpt/pi05_libero", interface="policy",
    ...                            image_keys=("observation/image",
    ...                                        "observation/wrist_image"),
    ...                            state_key="observation/state")
    >>> engine.policy.infer(observation)          # doctest: +SKIP
    """

    def __init__(self, *, interface: str, policy=None, bare=None) -> None:
        if interface not in INTERFACES:
            raise ValueError(
                f"unknown interface {interface!r}; known: {list(INTERFACES)}"
            )
        if (policy is None) == (bare is None):
            raise ValueError("exactly one of policy= / bare= must be given")
        self.interface = interface
        self.policy = policy
        self.bare = bare

    @classmethod
    def open(
        cls, model_dir, *, interface: str = "policy", **kwargs: Any
    ) -> "ApxInfEngine":
        """Load ``model_dir`` at the requested interface."""
        if interface == "policy":
            return cls(interface=interface, policy=load_policy(model_dir, **kwargs))
        if interface == "bare":
            return cls(interface=interface, bare=load_bare_model(model_dir, **kwargs))
        raise ValueError(f"unknown interface {interface!r}; known: {list(INTERFACES)}")

    @property
    def handle(self):
        """Whichever of ``policy`` / ``bare`` this engine holds."""
        return self.policy if self.policy is not None else self.bare

    @property
    def metadata(self) -> dict:
        """Served metadata, or ``{}`` at the bare interface (which publishes none)."""
        return dict(getattr(self.policy, "metadata", {}) or {})

    @property
    def action_horizon(self) -> Optional[int]:
        value = getattr(self.handle, "action_horizon", None)
        return int(value) if value is not None else None

    @property
    def action_dim(self) -> Optional[int]:
        value = getattr(self.handle, "action_dim", None)
        return int(value) if value is not None else None

    def infer(self, observation, *, noise=None) -> dict:
        """L2 inference. Raises at the bare interface, which has no observation."""
        if self.policy is None:
            raise RuntimeError(
                "infer(observation) needs the policy interface; this engine was "
                'opened with interface="bare", whose entry point is '
                "infer_rgb(rgb_u8, layout, token_ids). Reopen with "
                "interface=\"policy\" to get the engine's own preprocessing chain."
            )
        return self.policy.infer(observation, noise=noise)

    def infer_rgb(self, rgb_u8, layout: str, token_ids: Sequence[int], *, noise=None):
        """L1 inference. Raises at the policy interface, to keep the choice explicit."""
        if self.bare is None:
            raise RuntimeError(
                "infer_rgb needs the bare interface; this engine was opened with "
                'interface="policy", which owns preprocessing and takes an '
                'observation dict. Reopen with interface="bare" to feed tensors '
                "directly."
            )
        return self.bare.infer_rgb(rgb_u8, layout, token_ids, noise=noise)

    def close(self) -> None:
        close = getattr(self.handle, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ApxInfEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

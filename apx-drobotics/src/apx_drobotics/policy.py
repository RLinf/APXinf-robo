"""``DroboticsPolicy``: Apx-drobotics-preprocess, the SDK, and Apx-drobotics-postprocess as one Policy.

It has the same shape as an ApxInf policy -- ``infer(observation) -> {actions, timing}``,
``metadata``, ``action_dim``, ``action_horizon``, ``close()`` -- so whatever drives one
drives the other. It imports neither ApxInf nor apxinf-robo, and the shape is not
declared anywhere here: ApxInf's ``Policy`` is a structural protocol.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import contract as contract_lib
from .errors import DeploymentError, RuntimeFailure
from .postprocess import Postprocess
from .preprocess import Preprocess
from .runtime import DEFAULT_SDK_ROOT, DEFAULT_WORKER, Worker

__all__ = ["DroboticsPolicy"]

logger = logging.getLogger(__name__)


def _comparable(value: Any) -> Any:
    return list(value) if isinstance(value, (list, tuple)) else value


class DroboticsPolicy:
    """Pi0.5 LIBERO on the S600 BPU. Build one with :meth:`from_deployment`."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        worker: Worker,
        *,
        preprocess: Preprocess,
        postprocess: Postprocess,
        metadata: Mapping[str, Any],
        infer_deadline: float,
    ) -> None:
        self.contract = contract
        self.metadata = dict(metadata)
        self._worker = worker
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._infer_deadline = infer_deadline

    @classmethod
    def from_deployment(
        cls,
        sdk_root=None,
        *,
        image_keys: Sequence[str],
        state_key: str,
        prompt_key: str = "prompt",
        contract=None,
        worker=None,
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        discrete_state: Optional[bool] = None,
        precision: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        verify_hashes: bool = True,
        ready_timeout: Optional[float] = None,
        infer_deadline: Optional[float] = None,
    ) -> "DroboticsPolicy":
        """Check the deployment against the contract, then load the model.

        ``sdk_root`` is the SDK's ``oellm_runtime_s600`` directory. ``contract`` is a
        path or an already loaded mapping. ``action_dim``, ``state_dim``,
        ``discrete_state`` and ``precision`` are what the caller expects; none of them
        selects anything, because the contract fixes them, so a different value is an
        error rather than a setting.

        Raises ``ValueError`` for a caller that does not fit the contract,
        :class:`~apx_drobotics.errors.DeploymentError` for a deployment that does not, and
        :class:`~apx_drobotics.errors.StartupFailure` if the runtime does not come up.
        """
        loaded = (dict(contract) if isinstance(contract, Mapping)
                  else contract_lib.load_contract(contract))
        facts = contract_lib.model_metadata(loaded)
        views = contract_lib.unmasked_views(loaded)
        image_keys = tuple(image_keys)
        problems = []
        if len(image_keys) != views or len(set(image_keys)) != views:
            problems.append(f"image_keys {list(image_keys)}: need {views} distinct camera keys")
        for name, value in (("action_dim", action_dim), ("state_dim", state_dim),
                            ("precision", precision)):
            if value is not None and value != facts[name]:
                problems.append(f"{name}={value!r}: this model has {facts[name]!r}")
        if discrete_state:
            problems.append("discrete_state=True: this runtime does not read state")
        own = {**facts, "engine": "apx-drobotics", "runtime": "d-robotics-oellm",
               "image_keys": list(image_keys), "state_key": state_key, "prompt_key": prompt_key}
        extra = dict(metadata or {})
        for name, value in extra.items():
            if name in own and value is not None and _comparable(value) != own[name]:
                problems.append(f"metadata {name}={value!r} contradicts {own[name]!r}")
        if problems:
            raise ValueError("cannot serve this configuration on the S600: " + "; ".join(problems))

        root = Path(sdk_root) if sdk_root is not None else DEFAULT_SDK_ROOT
        executable = Path(worker) if worker is not None else DEFAULT_WORKER
        logger.info("checking the deployment under %s", root)
        problems = contract_lib.check_deployment(loaded, root, verify_hashes=verify_hashes)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            problems.append(f"{executable}: worker executable is missing; run scripts/deploy.sh")
        if problems:
            raise DeploymentError(problems)

        timeouts = loaded["timeouts"]
        logger.info("starting worker %s", executable)
        process = Worker.start(
            executable, root / loaded["runtime_config"]["path"],
            ready_timeout=float(timeouts["ready_seconds"] if ready_timeout is None else ready_timeout),
        )
        return cls(
            loaded,
            process,
            preprocess=Preprocess(loaded, image_keys=image_keys, state_key=state_key,
                                  prompt_key=prompt_key),
            postprocess=Postprocess(loaded),
            metadata={**extra, **own, "oellm_version": process.version},
            infer_deadline=float(
                timeouts["infer_deadline_seconds"] if infer_deadline is None else infer_deadline
            ),
        )

    @property
    def action_dim(self) -> int:
        return int(self.metadata["action_dim"])

    @property
    def action_horizon(self) -> int:
        return int(self.metadata["action_horizon"])

    def infer(
        self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None
    ) -> dict:
        """Map one raw observation to ``{actions, timing}``.

        ``actions`` is ``float32 [horizon, dim]``. In ``timing``, ``model_ms`` is the
        time in the HBMs and ``total_ms`` the whole call; ``sdk_*_ms`` is the runtime's
        own profile. Raises :class:`~apx_drobotics.errors.RequestError` for an observation
        the contract refuses, and :class:`~apx_drobotics.errors.RuntimeFailure` if the
        runtime can no longer serve.
        """
        if noise is not None:
            raise ValueError("the S600 runtime samples its own noise; noise= is not supported")
        started = time.perf_counter()
        request = self._preprocess(observation)
        reply, payload = self._worker.infer(request, self._infer_deadline)
        try:
            actions, timing = self._postprocess(reply, payload)
        except RuntimeFailure as error:
            self._worker.fail(str(error))
            raise
        timing["total_ms"] = (time.perf_counter() - started) * 1000.0
        return {"actions": actions, "timing": timing}

    def close(self) -> None:
        self._worker.close()

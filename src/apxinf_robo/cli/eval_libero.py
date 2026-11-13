#!/usr/bin/env python3
"""``apxinf-robo eval-libero`` -- resumable LIBERO evaluation, dual transport.

One evaluator for every way of reaching the model and every LIBERO suite:

* ``--backend websocket`` reaches a running OpenPI-compatible server through the
  unmodified ``openpi_client``.
* ``--backend in-process`` builds the policy in this process via
  :func:`apxinf_robo.engine.load_policy` and calls ``policy.infer`` directly --
  no socket, no server, no subprocess. ``--engine drobotics`` builds the S600 policy
  instead, on the board.

Both transports drive the *same* rollout loop through a small :class:`Backend`
abstraction, so the LIBERO harness (rollout constants, the resumable fsync'd
JSONL ledger, timing split) is shared verbatim. Native observation conversion
lives in :mod:`apxinf_robo.envs.libero` so evaluation and calibration cannot
silently diverge on camera orientation or robot-state layout. Model-specific
resize remains inside the selected policy, whichever engine runs it, so the evaluator
sends every engine the same oriented frames.

``--suite`` selects one LIBERO task suite (``libero_10`` / ``libero_90`` /
``libero_spatial`` / ``libero_object`` / ``libero_goal``) or ``all``. The ledger
key is ``(suite, task_id, trial_id)`` so multiple suites share one resumable
account without ``task_id=0`` colliding across suites.

The evaluator selects the LIBERO state and action conversion from the model type.
PI0.5 keeps its seven-value state; PI0-FAST preserves both finger joints; GR00T
uses named state and converts its decoded gripper before ``env.step``. A new
family needs a LIBERO conversion only when its contract differs from these.

    # websocket (server already running)
    apxinf-robo eval-libero --backend websocket --precision bf16 \
        --suite libero_10 --results-jsonl r.jsonl --summary-json s.json

    # in-process (no server)
    apxinf-robo eval-libero --backend in-process --model-dir /path/ckpt \
        --precision bf16 --action-dim 7 --suite libero_10 \
        --results-jsonl r.jsonl --summary-json s.json

    # websocket, with `apxinf-robo serve --engine drobotics` running on an S600 board
    apxinf-robo eval-libero --backend websocket --host <board> --precision w8 \
        --suite libero_10 --results-jsonl r.jsonl --summary-json s.json
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import logging
import os
import pathlib
import sys
import time
import traceback
from typing import Optional, Protocol, Tuple

import numpy as np

from ._engine_options import explicit_flags, refusal

from ..envs.libero import (
    libero_gr00t_action,
    libero_gr00t_state,
    libero_images,
    libero_state,
    make_env,
)

# --- rollout protocol constants (OpenPI's public PI0.5 LIBERO configuration) ---
LIBERO_ACTION_DIM = 7

#: The preset whose wire contract this evaluator drives. The keys themselves are
#: *not* restated here: they are read from the table below, so the in-process
#: backend cannot drift from what ``--robot franka_libero`` serves. For the
#: websocket backend the table is only half the answer — the server resolved its
#: own contract — so ``_assert_server_speaks_the_same_dialect`` compares the two
#: at connect.
LIBERO_PRESET = "franka_libero"
MAX_STEPS = 520
WAIT_STEPS = 10
REPLAN_STEPS = 5


def _two_finger_state(observation):
    return libero_state(observation, finger_joints=2)


_LIBERO_STATES = {
    "pi0_fast": _two_finger_state,
    "pi0fast": _two_finger_state,
    "pi05_drobotics_w8": _two_finger_state,
    "gr00t": libero_gr00t_state,
    "gr00tn1d7": libero_gr00t_state,
    "Gr00tN1d7": libero_gr00t_state,
}
_GR00T_TYPES = frozenset(("gr00t", "gr00tn1d7", "Gr00tN1d7"))

#: The five LIBERO task suites ``--suite all`` expands to, in a stable order.
ALL_SUITES = (
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
)

LedgerKey = Tuple[str, int, int]  # (suite, task_id, trial_id)


@functools.lru_cache(maxsize=1)
def libero_convention():
    """LIBERO's wire dialect, read from the preset table rather than restated.

    Both backends send the same observation, so both resolve it here: the keys
    the evaluator writes are the keys ``--robot franka_libero`` is defined by.
    Whether a *remote* server was started under that preset is a separate
    question, answered at connect. This costs no CUDA (the engine binding is
    loaded lazily by the policy loader), so the websocket backend pays nothing
    for it.
    """
    from ..presets import get_robot_preset

    return get_robot_preset(LIBERO_PRESET).convention


# --- LIBERO harness: resumable ledger + summary aggregation -------------------


#: In-process options that only ApxInf can honour; ``--engine drobotics`` refuses them
#: instead of dropping them. ``--warm-start`` is here because it feeds the model noise,
#: which the S600 runtime samples for itself.
_APXINF_ONLY = (
    "model_type", "checkpoint", "device", "calibration", "tactics", "tokenizer",
    "norm_stats", "norm_key", "num_views", "num_flow_steps", "flow_start_time",
    "action_horizon", "model_seed", "warm_start",
)

#: Metadata that identifies a contract-pinned engine, recorded in the run identity.
_CONTRACT_IDENTITY = (
    "engine", "runtime", "sdk_version", "model_type", "contract_id", "contract_sha256",
    # The contract hash covers the model files, not the code that drives them: a rebuilt
    # worker or a newer apx_drobotics must not resume an older ledger.
    "worker_sha256", "apx_drobotics_version",
)


def run_identity(args: argparse.Namespace, metadata: Optional[dict] = None) -> dict:
    """What a ledger row's result depends on, beyond precision.

    A resumed evaluation must not mix rows produced by different settings into
    one success rate. Every row records this, and :func:`completed_runs` refuses
    a ledger whose rows disagree with the current run. An engine that publishes a
    ``contract_sha256`` in its ``metadata`` is pinned by it: the hash stands for
    the SDK, model files and pre/post-processing, so it covers the model checksums.
    """
    identity = {
        "replan_steps": args.replan_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "settle_gripper": args.settle_gripper,
        "wait_steps": WAIT_STEPS,
    }
    if metadata and "contract_sha256" in metadata:
        identity.update((key, metadata.get(key)) for key in _CONTRACT_IDENTITY)
    return identity


def _check_identity(item: dict, identity: dict, line_number: int) -> None:
    recorded = item.get("identity")
    if recorded is None:
        # Rows written before identities existed. They are accepted for runs
        # without a contract, so older ledgers stay resumable.
        if "contract_sha256" in identity:
            raise ValueError(
                f"ledger line {line_number} has no run identity, so it cannot be "
                "shown to match the engine's contract; use a new --results-jsonl"
            )
        return
    if recorded != identity:
        changed = sorted(
            key for key in set(recorded) | set(identity)
            if recorded.get(key) != identity.get(key)
        )
        details = ", ".join(
            f"{key}: ledger {recorded.get(key)!r}, now {identity.get(key)!r}" for key in changed
        )
        raise ValueError(
            f"ledger line {line_number} was written by a different run ({details}); "
            "use a new --results-jsonl"
        )


def completed_runs(
    path: pathlib.Path, precision: str, identity: Optional[dict] = None
) -> dict[LedgerKey, dict]:
    """Load the ``status == "completed"`` rows from a resumable ledger.

    Keyed by ``(suite, task_id, trial_id)`` so one ledger can hold several suites
    without ``task_id=0`` colliding across them. The precision guard rejects a
    ledger written at a different precision than the one requested now, and,
    when ``identity`` is given, the identity guard rejects rows written with
    other settings (see :func:`run_identity`).
    """
    result: dict[LedgerKey, dict] = {}
    if not path.exists():
        return result
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_precision = item.get("precision", "fp8")
        if item_precision != precision:
            raise ValueError(
                f"ledger precision is {item_precision!r}, requested {precision!r} "
                f"at line {line_number}"
            )
        if identity is not None:
            _check_identity(item, identity, line_number)
        if item.get("status") == "completed":
            key: LedgerKey = (
                str(item["suite"]),
                int(item["task_id"]),
                int(item["trial_id"]),
            )
            if key in result:
                raise ValueError(f"duplicate completed run {key} at line {line_number}")
            result[key] = item
    return result


def append_record(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_summary(
    path: pathlib.Path,
    ledger: dict[LedgerKey, dict],
    expected_keys: set[LedgerKey],
    precision: str,
    transport: str,
    identity: Optional[dict] = None,
) -> None:
    """Write the aggregate summary, grouped per-suite then per-task."""
    per_suite: dict[str, dict] = {}
    for suite in sorted({key[0] for key in expected_keys}):
        per_task = {}
        for task_id in sorted({t for (s, t, _) in expected_keys if s == suite}):
            rows = [
                row
                for (s, t, _), row in ledger.items()
                if s == suite and t == task_id
            ]
            successes = sum(bool(row["success"]) for row in rows)
            per_task[str(task_id)] = {
                "completed": len(rows),
                "successes": successes,
                "success_rate": (successes / len(rows) if rows else None),
            }
        suite_rows = [row for (s, _, _), row in ledger.items() if s == suite]
        suite_successes = sum(bool(row["success"]) for row in suite_rows)
        per_suite[suite] = {
            "completed": len(suite_rows),
            "successes": suite_successes,
            "success_rate": (
                suite_successes / len(suite_rows) if suite_rows else None
            ),
            "per_task": per_task,
        }
    rows = list(ledger.values())
    document = {
        "schema": "apxinf.libero-eval.v2",
        "suites": sorted({key[0] for key in expected_keys}),
        "transport": transport,
        "precision": precision,
        "identity": identity,
        "expected_runs": len(expected_keys),
        "completed_runs": len(rows),
        "missing_runs": [
            {"suite": suite, "task_id": task, "trial_id": trial}
            for suite, task, trial in sorted(expected_keys - set(ledger))
        ],
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": (
            sum(bool(row["success"]) for row in rows) / len(rows) if rows else None
        ),
        "per_suite": per_suite,
        "timing": _aggregate_timing(rows),
        "updated_unix_seconds": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _aggregate_timing(rows: list[dict]) -> dict:
    """Roll the per-episode latency split up to totals + per-inference means.

    Each ``replan`` is one inference call, so per-call means divide a segment's
    summed seconds by the total replan count. Segments are the four
    non-overlapping parts recorded by :func:`run_episode`: ``preprocess``
    (client), ``model`` (pure bare-model), ``server_processor`` (server/policy
    pre/post pipeline) and ``websocket_transport`` (send/recv/serialize; ~0 for
    the in-process backend). ``inference`` is the client's full round-trip wall
    clock, reported for cross-check, not summed into the split. Rows missing a
    segment contribute 0 to it.
    """
    completed = [row for row in rows if row.get("status") == "completed"]
    total_replans = sum(int(row.get("replans", 0)) for row in completed)
    segments = (
        "preprocess_seconds",
        "model_seconds",
        "server_processor_seconds",
        "websocket_transport_seconds",
        "inference_seconds",
    )
    totals = {
        name: sum(float(row.get(name, 0.0)) for row in completed) for name in segments
    }
    per_call_ms = {
        name.replace("_seconds", "_ms"): (
            (value / total_replans * 1000.0) if total_replans else None
        )
        for name, value in totals.items()
    }
    return {
        "episodes": len(completed),
        "total_inference_calls": total_replans,
        "total_seconds": {name: round(value, 6) for name, value in totals.items()},
        "per_call_ms": {
            name: (round(value, 4) if value is not None else None)
            for name, value in per_call_ms.items()
        },
    }


# --- transport backends -------------------------------------------------------


class Backend(Protocol):
    """A transport that maps one observation to an action chunk + timing.

    The rollout is transport-agnostic: it only calls :meth:`infer` and reads the
    two internal timing segments (model vs server/policy processor). Everything
    transport-specific (a socket vs an in-process policy handle) lives behind
    this contract.
    """

    #: Static description sent by / read from the underlying policy.
    metadata: dict
    model_type: str

    def infer(
        self,
        base: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        prompt: str,
        noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Return ``(actions, normalized_actions, timing)``."""
        ...

    def close(self) -> None:
        ...


def _observation(base, wrist, state, prompt) -> dict:
    """The OpenPI LIBERO observation both backends consume, identical on the wire
    and in-process. Keys come from the preset table, not from constants here."""
    convention = libero_convention()
    return {
        convention.image_keys[0]: base,
        convention.image_keys[1]: wrist,
        convention.state_key: state,
        convention.prompt_key: prompt,
    }


def _assert_server_speaks_the_same_dialect(metadata) -> None:
    """Check published input fields against the evaluator's preset."""
    convention = libero_convention()
    expected = {
        "image_keys": list(convention.image_keys),
        "state_key": convention.state_key,
        "prompt_key": convention.prompt_key,
    }
    mismatches = []
    for field, want in expected.items():
        if field not in metadata:
            continue
        got = metadata[field]
        if isinstance(want, list):
            got = list(got) if isinstance(got, (list, tuple)) else got
        if got != want:
            mismatches.append(f"{field}: server {got!r}, evaluator {want!r}")
    if mismatches:
        raise RuntimeError(
            "server input fields do not match the LIBERO evaluator preset "
            f"--robot {LIBERO_PRESET} -- " + "; ".join(mismatches)
        )


class WebsocketBackend:
    """Reach an OpenPI-compatible server through the unmodified ``openpi_client``."""

    def __init__(self, host: str, port: int, expected_precision: str) -> None:
        from openpi_client import websocket_client_policy

        # WebsocketClientPolicy honours the ambient proxy; exempt the target host
        # so a loopback / LAN server is reached directly (the old evaluator did
        # the same).
        for variable in ("NO_PROXY", "no_proxy"):
            entries = [item for item in os.environ.get(variable, "").split(",") if item]
            if host not in entries:
                entries.append(host)
            os.environ[variable] = ",".join(entries)
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        # LIBERO scene initialization can monopolize the client process for longer
        # than websockets' 20 s Pong deadline. Keep sending Pings, but do not mistake
        # a slow simulator reset for a dead server. openpi_client offers no public
        # setting for this, so it reaches a private attribute: guard it, and say so
        # if a newer client no longer has it rather than failing on a missing name
        # or silently running with the default deadline.
        connection = getattr(self._client, "_ws", None)
        if connection is not None and hasattr(connection, "ping_timeout"):
            connection.ping_timeout = None
        else:
            logging.warning(
                "openpi_client no longer exposes _ws.ping_timeout; a LIBERO reset that "
                "blocks this process for over 20 s may drop the connection"
            )
        self.metadata = self._client.get_server_metadata()
        self.model_type = self.metadata.get("model_type", "pi05")
        actual_precision = self.metadata.get("precision")
        if actual_precision != expected_precision:
            self.close()
            raise RuntimeError(
                f"server precision is {actual_precision!r}, expected {expected_precision!r}"
            )
        try:
            _assert_server_speaks_the_same_dialect(self.metadata)
        except RuntimeError:
            self.close()
            raise

    def infer(
        self, base, wrist, state, prompt, noise=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        if noise is not None:
            raise RuntimeError("warm-start noise requires --backend in-process")
        response = self._client.infer(_observation(base, wrist, state, prompt))
        actions = np.asarray(response["actions"], dtype=np.float32)
        # Only OpenPI-contract keys ride the wire; ``policy_timing`` /
        # ``server_timing`` are the server's tolerated diagnostic namespaces.
        #   policy_timing.infer_ms  = pure model (bare-model infer_rgb)
        #   policy_timing.policy_ms = whole policy (pre + model + post)
        #   server_timing.infer_ms  = event-loop wall clock of policy.infer
        # Missing keys degrade gracefully to zero / to the coarser field.
        policy_timing = response.get("policy_timing", {}) or {}
        server_timing = response.get("server_timing", {}) or {}
        model_ms = float(policy_timing.get("infer_ms", 0.0))
        server_total_ms = float(policy_timing.get("policy_ms", model_ms))
        server_compute_ms = float(server_timing.get("infer_ms", server_total_ms))
        return actions, None, {
            "model_seconds": model_ms / 1000.0,
            # Server compute the client can see beyond the pure model = the
            # server-side processor pipeline (event-loop wall clock minus model).
            "server_processor_seconds": max(0.0, server_compute_ms - model_ms) / 1000.0,
        }

    def close(self) -> None:
        connection = getattr(self._client, "_ws", None)
        if connection is not None:
            connection.close()


class InProcessBackend:
    """Build the policy in this process and call ``policy.infer`` directly.

    Model-agnostic: :func:`apxinf_robo.engine.load_policy` dispatches on the
    checkpoint's ``config.json`` model type (overridable with ``--model-type``),
    so a future ``GrootPolicy`` runs here with no change to this file. ``--engine
    drobotics`` builds the S600 policy, which has the same shape.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        if args.engine == "drobotics":
            from ..presets import build_robot_policy

            # parse_args has already refused the options this engine cannot honour.
            # action_dim and discrete_state are passed on instead: the contract fixes
            # both, so a value that disagrees with it is an error there, not a setting.
            self._policy = build_robot_policy(
                LIBERO_PRESET,
                args.model_dir,
                engine="drobotics",
                precision=args.precision,
                action_dim=args.action_dim,
                discrete_state=args.discrete_state,
            )
            self.metadata = dict(self._policy.metadata)
            self.model_type = self.metadata["model_type"]
            return

        # Lazy: loading the engine pulls in the policy stack; websocket-only
        # users never pay for it.
        from ..engine import load_policy
        from apxinf.policies.auto import _read_model_type

        self.model_type = args.model_type or _read_model_type(args.model_dir)
        if self.model_type == "qwen_drive":
            raise ValueError("Qwen-Drive is a driving planner, not a LIBERO robot policy")

        options = {
            "checkpoint": args.checkpoint,
            "calibration": args.calibration,
            "tactics": args.tactics,
            "tokenizer_path": args.tokenizer,
            "norm_stats": args.norm_stats,
            "norm_key": args.norm_key,
            "action_horizon": args.action_horizon,
            "num_views": args.num_views,
            "num_flow_steps": args.num_flow_steps,
            "flow_start_time": args.flow_start_time,
            "discrete_state": args.discrete_state,
            "seed": args.model_seed if args.model_seed is not None else args.seed,
        }
        convention = libero_convention()
        self._policy = load_policy(
            args.model_dir,
            model_type=args.model_type,
            device=args.device,
            precision=args.precision,
            action_dim=(args.action_dim or None),
            # LIBERO's wire keys, read from the ``franka_libero`` preset rather
            # than restated: the policy layer holds no dataset's convention as a
            # default, and this evaluator should not become a second source of
            # truth for what "LIBERO keys" means.
            image_keys=convention.image_keys,
            state_key=convention.state_key,
            prompt_key=convention.prompt_key,
            metadata={"precision": args.precision, "policy": "libero", "model_type": self.model_type},
            **{name: value for name, value in options.items() if value is not None},
        )
        self.metadata = dict(getattr(self._policy, "metadata", {}))

    def infer(
        self, base, wrist, state, prompt, noise=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        result = self._policy.infer(_observation(base, wrist, state, prompt), noise=noise)
        actions = np.asarray(result["actions"], dtype=np.float32)
        normalized = result.get("normalized_actions")
        if normalized is not None:
            normalized = np.asarray(normalized, dtype=np.float32)
        timing = result.get("timing", {}) or {}
        model_ms = float(timing.get("model_ms", 0.0))
        total_ms = float(timing.get("total_ms", model_ms))
        return actions, normalized, {
            "model_seconds": model_ms / 1000.0,
            "server_processor_seconds": max(0.0, total_ms - model_ms) / 1000.0,
        }

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()


def build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "websocket":
        return WebsocketBackend(args.host, args.port, args.precision)
    return InProcessBackend(args)


# --- rollout ------------------------------------------------------------------


def run_episode(
    env,
    initial_state: np.ndarray,
    suite: str,
    task_id: int,
    trial_id: int,
    prompt: str,
    backend: Backend,
    transport: str,
    seed: int,
    warm_start: bool,
    warm_start_alpha: float,
    replan_steps: int = REPLAN_STEPS,
    settle_gripper: float = -1.0,
    max_steps: int | None = None,
    trace_path: pathlib.Path | None = None,
) -> dict:
    model_type = getattr(backend, "model_type", "pi05")
    episode_started = time.perf_counter()
    env.reset()
    observation = env.set_init_state(initial_state)
    dummy_action = [0.0] * 6 + [settle_gripper]
    for _ in range(WAIT_STEPS):
        observation, _, _, _ = env.step(dummy_action)

    def digest(array) -> str:
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    trace = None
    if trace_path is not None:
        trace = {
            "schema": "apxinf.libero-rollout-trace.v1",
            "suite": suite, "task_id": task_id, "trial_id": trial_id,
            "seed": seed, "prompt": prompt,
            "initial_state": np.asarray(initial_state).tolist(),
            "initial_state_sha256": digest(initial_state),
            "settled_observation": {
                "agentview_image_sha256": digest(observation["agentview_image"]),
                "wrist_image_sha256": digest(observation["robot0_eye_in_hand_image"]),
                "eef_pos": np.asarray(observation["robot0_eef_pos"]).tolist(),
                "eef_quat": np.asarray(observation["robot0_eef_quat"]).tolist(),
                "gripper_qpos": np.asarray(observation["robot0_gripper_qpos"]).tolist(),
            },
            "wait_steps": WAIT_STEPS, "settle_gripper": settle_gripper,
            "replan_steps": replan_steps,
            "inferences": [], "executed_actions": [],
        }

    action_plan: collections.deque[np.ndarray] = collections.deque()
    success = False
    action_steps = 0
    replans = 0
    # Timing is split into four non-overlapping segments so the delivery report
    # can attribute latency:
    #   preprocess  = client-side camera orientation + state assembly (this process)
    #   model       = pure bare-model infer_rgb (backend timing)
    #   server_proc = server/policy pre/post processor pipeline (backend timing)
    #   transport   = round_trip - server compute (~0 for in-process)
    # round_trip (== inference_seconds) is the wall clock around backend.infer and
    # equals transport + server_proc + model up to measurement noise.
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    model_seconds = 0.0
    server_processor_seconds = 0.0
    transport_seconds = 0.0
    first_action_checksum = None
    previous_normalized_chunk = None
    warm_start_replans = 0
    warm_noise_checksum = None
    rng = np.random.default_rng(seed + 1_000_003 * task_id + 10_007 * trial_id)

    while action_steps < (MAX_STEPS if max_steps is None else max_steps):
        if not action_plan:
            preprocess_started = time.perf_counter()
            images = libero_images(
                observation["agentview_image"],
                observation["robot0_eye_in_hand_image"],
            )
            state = _LIBERO_STATES.get(model_type, libero_state)(observation)
            preprocess_seconds += time.perf_counter() - preprocess_started

            noise = None
            if warm_start and previous_normalized_chunk is not None:
                shift = np.empty_like(previous_normalized_chunk)
                replan = min(replan_steps, shift.shape[0])
                if replan < shift.shape[0]:
                    shift[: shift.shape[0] - replan] = previous_normalized_chunk[replan:]
                shift[shift.shape[0] - replan :] = previous_normalized_chunk[-1]
                epsilon = rng.standard_normal(previous_normalized_chunk.shape).astype(np.float32)
                noise = np.ascontiguousarray(
                    warm_start_alpha * shift + (1.0 - warm_start_alpha) * epsilon,
                    dtype=np.float32,
                )
                warm_start_replans += 1
                if warm_noise_checksum is None:
                    warm_noise_checksum = float(np.abs(noise).sum())

            request_started = time.perf_counter()
            actions, normalized_actions, timing = backend.infer(
                images[0], images[1], state, prompt, noise=noise
            )
            if model_type in _GR00T_TYPES:
                actions = libero_gr00t_action(actions)
            round_trip_seconds = time.perf_counter() - request_started
            inference_seconds += round_trip_seconds
            # The action horizon is a checkpoint property (metadata ``action_horizon``),
            # not a fixed 10: the public OpenPI pi0.5 LIBERO config emits H=10, but the
            # native ``pi05_libero_base`` checkpoints emit H=50. The rollout only
            # consumes ``REPLAN_STEPS`` actions per chunk, so any horizon >=
            # REPLAN_STEPS is valid; we only require the correct action width.
            if (
                actions.ndim != 2
                or actions.shape[1] != LIBERO_ACTION_DIM
                or actions.shape[0] < replan_steps
            ):
                raise ValueError(
                    f"expected actions (>= {replan_steps}, {LIBERO_ACTION_DIM}), "
                    f"got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("backend returned non-finite actions")
            if warm_start:
                if normalized_actions is None:
                    raise RuntimeError("warm-start requires backend normalized_actions")
                if normalized_actions.ndim != 2:
                    raise ValueError(
                        f"expected normalized actions [H, D], got {normalized_actions.shape}"
                    )
                if not np.isfinite(normalized_actions).all():
                    raise FloatingPointError("backend returned non-finite normalized actions")
                previous_normalized_chunk = np.ascontiguousarray(
                    normalized_actions, dtype=np.float32
                )

            segment_model = float(timing.get("model_seconds", 0.0))
            segment_processor = float(timing.get("server_processor_seconds", 0.0))
            model_seconds += segment_model
            server_processor_seconds += segment_processor
            transport_seconds += max(
                0.0, round_trip_seconds - segment_model - segment_processor
            )
            if first_action_checksum is None:
                first_action_checksum = float(np.abs(actions).sum())
            if trace is not None:
                trace["inferences"].append({
                    "at_action_step": action_steps,
                    "base_sha256": digest(images[0]),
                    "wrist_sha256": digest(images[1]),
                    "state": np.asarray(state).tolist() if not isinstance(state, dict) else
                        {key: np.asarray(value).tolist() for key, value in state.items()},
                    "actions": actions.tolist(),
                    "model_seconds": segment_model,
                    "server_processor_seconds": segment_processor,
                    "round_trip_seconds": round_trip_seconds,
                })
            action_plan.extend(actions[:replan_steps])
            replans += 1

        action = action_plan.popleft()
        observation, _, done, _ = env.step(action.tolist())
        action_steps += 1
        if trace is not None:
            trace["executed_actions"].append(action.tolist())
        if done:
            success = True
            break

    if trace is not None:
        trace["success"] = success
        trace["action_steps"] = action_steps
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
        temporary.write_text(json.dumps(trace, sort_keys=True) + "\n")
        os.replace(temporary, trace_path)

    return {
        "status": "completed",
        "suite": suite,
        "task_id": task_id,
        "trial_id": trial_id,
        "prompt": prompt,
        "success": success,
        "action_steps": action_steps,
        "replans": replans,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "model_seconds": model_seconds,
        "server_processor_seconds": server_processor_seconds,
        "websocket_transport_seconds": transport_seconds,
        "elapsed_seconds": time.perf_counter() - episode_started,
        "first_action_abs_checksum": first_action_checksum,
        "warm_start": warm_start,
        "warm_start_alpha": warm_start_alpha if warm_start else None,
        "warm_start_replans": warm_start_replans,
        "first_warm_noise_abs_checksum": warm_noise_checksum,
        "transport": transport,
        "image_input": "openpi_uint8_hwc",
        "seed": seed,
        **({"trace_path": str(trace_path)} if trace_path is not None else {}),
    }


# --- argument parsing + suite/task resolution ---------------------------------


def build_parser() -> argparse.ArgumentParser:
    from ..presets import ENGINES

    parser = argparse.ArgumentParser(
        prog="apxinf-robo eval-libero",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=("websocket", "in-process"), required=True,
        help="reach the model through a running server, or build it in-process",
    )
    parser.add_argument("--precision", choices=("fp8", "bf16", "int8", "w8"), required=True)
    parser.add_argument("--suite", default="libero_10", choices=(*ALL_SUITES, "all"))
    parser.add_argument(
        "--tasks", default="all",
        help="'all' (default) or a comma list of task ids, applied within each selected suite",
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-seed",
        type=int,
        help="in-process model sampling seed (default: reuse --seed)",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=REPLAN_STEPS,
        help=f"actions executed from each predicted chunk (default: {REPLAN_STEPS})",
    )
    parser.add_argument(
        "--settle-gripper",
        type=float,
        default=-1.0,
        help="seventh action used during the initial settle steps (default: -1)",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--results-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--summary-json", required=True, type=pathlib.Path)
    parser.add_argument(
        "--trace-dir", type=pathlib.Path,
        help="save initial state, observation hashes, each predicted chunk, executed actions and timings",
    )

    websocket = parser.add_argument_group("websocket backend")
    websocket.add_argument("--host", default="127.0.0.1")
    websocket.add_argument("--port", type=int, default=8000)

    in_process = parser.add_argument_group("in-process backend")
    in_process.add_argument(
        "--engine", choices=ENGINES, default="apxinf",
        help="what runs the model in this process; drobotics needs an S600 board. With "
        "--backend websocket the server decides.",
    )
    in_process.add_argument(
        "--model-dir", type=pathlib.Path,
        help="checkpoint directory; for --engine drobotics, the SDK's oellm_runtime_s600 directory",
    )
    in_process.add_argument("--model-type", default=None, help="override config.json model type")
    in_process.add_argument("--checkpoint", type=pathlib.Path)
    in_process.add_argument("--device", default=None, help="ApxInf only (default: cuda:0)")
    in_process.add_argument("--calibration", type=pathlib.Path)
    in_process.add_argument("--tactics", type=pathlib.Path, help=argparse.SUPPRESS)
    in_process.add_argument("--tokenizer", type=pathlib.Path,
                            help="text tokenizer override; PI0-FAST also needs APXINF_FAST_TOKENIZER")
    in_process.add_argument(
        "--norm-stats", type=pathlib.Path,
        help="explicit OpenPI-style norm_stats.json; overrides checkpoint normalization",
    )
    in_process.add_argument("--norm-key")
    in_process.add_argument(
        "--action-dim", type=int, default=None,
        help="deployable action width (default: 7); 0 keeps the full vector",
    )
    in_process.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="load fewer camera views than the checkpoint declares; LIBERO uses 2",
    )
    in_process.add_argument(
        "--num-flow-steps",
        type=int,
        default=None,
        help="override the checkpoint's diffusion/flow inference steps",
    )
    in_process.add_argument(
        "--flow-start-time",
        type=float,
        default=None,
        help=(
            "override reverse-flow start time; with --warm-start this defaults to "
            "1-alpha, and without --warm-start values below 1.0 run partial flow "
            "from pure noise"
        ),
    )
    in_process.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="override the checkpoint's chunk length (default: its config.json "
        f"value; must be >= the {REPLAN_STEPS}-step replan stride)",
    )
    in_process.add_argument("--discrete-state", action="store_true", default=None)

    warm = parser.add_argument_group("warm start")
    warm.add_argument(
        "--warm-start",
        action="store_true",
        help="enable shifted action cache + tail repeat + alpha noise blend + partial flow",
    )
    warm.add_argument(
        "--warm-start-alpha",
        type=float,
        default=0.5,
        help=(
            "cache/noise blend coefficient used only with --warm-start; partial "
            "flow starts at 1-alpha"
        ),
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.engine == "drobotics":
        if args.backend != "in-process":
            parser.error("--engine drobotics applies to --backend in-process; "
                         "with --backend websocket the server decides")
        if args.precision != "w8":
            parser.error("--engine drobotics requires --precision w8")
        used = explicit_flags(args, _APXINF_ONLY)
        if used:
            parser.error(refusal(used, "drobotics"))
    elif args.backend == "in-process":
        if args.model_dir is None:
            parser.error("--backend in-process requires --model-dir")
        if args.precision == "w8":
            parser.error("--precision w8 is the S600 model; pass --engine drobotics")
    if args.engine != "drobotics":
        # Defaults that must not look "given" to the check above; see _engine_options.
        if args.device is None:
            args.device = "cuda:0"
        if args.action_dim is None:
            args.action_dim = 7
    if args.norm_stats is not None:
        if args.backend == "websocket":
            parser.error(
                "--norm-stats applies to --backend in-process; for the websocket "
                "backend pass it to `apxinf-robo serve` instead"
            )
        if not args.norm_stats.is_file():
            parser.error("--norm-stats must name an existing file")
    if not (0.0 <= args.warm_start_alpha <= 1.0):
        parser.error("--warm-start-alpha must be in [0, 1]")
    if args.warm_start and args.warm_start_alpha >= 1.0:
        parser.error(
            "--warm-start-alpha must be < 1.0 when --warm-start is enabled "
            "(otherwise the default flow_start_time would be 0)"
        )
    if args.warm_start:
        if args.backend != "in-process":
            parser.error("--warm-start requires --backend in-process")
        if args.num_flow_steps is None:
            args.num_flow_steps = 1
        if args.flow_start_time is None:
            args.flow_start_time = 1.0 - args.warm_start_alpha
        if args.flow_start_time <= 0.0 or args.flow_start_time > 1.0:
            parser.error("--flow-start-time must be in (0, 1]")
    elif args.flow_start_time is not None and args.flow_start_time < 1.0:
        print(
            "warning: --flow-start-time < 1.0 without --warm-start runs partial "
            "flow from pure noise",
            file=sys.stderr,
            flush=True,
        )
    if args.action_horizon is not None:
        # The websocket backend gets its horizon from whatever the server loaded,
        # so accepting the flag here would silently do nothing.
        if args.backend == "websocket":
            parser.error(
                "--action-horizon applies to --backend in-process; for the websocket "
                "backend pass it to `apxinf-robo serve` instead"
            )
        if args.action_horizon < args.replan_steps:
            parser.error(
                f"--action-horizon must be >= {args.replan_steps} (the rollout consumes "
                f"{args.replan_steps} actions per chunk)"
            )
    if args.replan_steps <= 0:
        parser.error("--replan-steps must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.trials_per_task <= 0 or args.trials_per_task > 50:
        parser.error("--trials-per-task must be in 1..=50")
    return args


def resolve_suites(name: str) -> list[str]:
    return list(ALL_SUITES) if name == "all" else [name]


def resolve_task_ids(spec: str, n_tasks: int, suite: str) -> list[int]:
    """Task ids for one suite: ``all`` -> ``0..n_tasks-1``; else a validated list."""
    if spec.strip() == "all":
        return list(range(n_tasks))
    task_ids = [int(value) for value in spec.split(",") if value.strip()]
    if sorted(set(task_ids)) != sorted(task_ids):
        raise ValueError(f"--tasks has duplicate ids: {task_ids}")
    out_of_range = [t for t in task_ids if not 0 <= t < n_tasks]
    if out_of_range:
        raise ValueError(
            f"--tasks {out_of_range} out of range for suite {suite!r} (0..{n_tasks - 1})"
        )
    return task_ids


# --- main ---------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    transport = "openpi_websocket" if args.backend == "websocket" else "in_process_api"

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    # Resolve every (suite -> instance, task_ids) up front so the expected-key set
    # and the rollout iterate the exact same scope.
    suites: dict[str, object] = {}
    task_ids_by_suite: dict[str, list[int]] = {}
    for name in resolve_suites(args.suite):
        suite = benchmark_dict[name]()
        suites[name] = suite
        task_ids_by_suite[name] = resolve_task_ids(args.tasks, suite.n_tasks, name)

    expected_keys: set[LedgerKey] = {
        (name, task_id, trial_id)
        for name, task_ids in task_ids_by_suite.items()
        for task_id in task_ids
        for trial_id in range(args.trials_per_task)
    }
    # The backend comes first: the run identity includes what the engine publishes.
    backend = build_backend(args)
    try:
        if args.warm_start and backend.model_type != "pi05":
            raise ValueError("--warm-start is currently defined for PI0.5 flow policies only")
        identity = run_identity(args, backend.metadata)
        ledger = completed_runs(args.results_jsonl, args.precision, identity)
        unexpected = set(ledger) - expected_keys
        if unexpected:
            raise ValueError(
                f"ledger contains runs outside requested scope: {sorted(unexpected)}"
            )
        write_summary(
            args.summary_json, ledger, expected_keys, args.precision, transport, identity
        )
        print(f"backend={args.backend} metadata={backend.metadata}", flush=True)
        for name, suite in suites.items():
            for task_id in task_ids_by_suite[name]:
                task = suite.get_task(task_id)
                prompt = str(task.language)
                pending = [
                    trial_id
                    for trial_id in range(args.trials_per_task)
                    if (name, task_id, trial_id) not in ledger
                ]
                if not pending:
                    print(f"{name} task {task_id}: already complete", flush=True)
                    continue
                print(f"{name} task {task_id}: pending trials {pending}", flush=True)
                initial_states = suite.get_task_init_states(task_id)
                env = make_env(task, args.seed)
                try:
                    for trial_id in pending:
                        for attempt in range(1, args.max_attempts + 1):
                            try:
                                record = run_episode(
                                    env,
                                    initial_states[trial_id],
                                    name,
                                    task_id,
                                    trial_id,
                                    prompt,
                                    backend,
                                    transport,
                                    args.seed,
                                    args.warm_start,
                                    args.warm_start_alpha,
                                    args.replan_steps,
                                    args.settle_gripper,
                                    args.max_steps,
                                    (args.trace_dir / name / f"task-{task_id}-trial-{trial_id}-attempt-{attempt}.json"
                                     if args.trace_dir is not None else None),
                                )
                                record["attempt"] = attempt
                                record["precision"] = args.precision
                                record["identity"] = identity
                                append_record(args.results_jsonl, record)
                                ledger[(name, task_id, trial_id)] = record
                                write_summary(
                                    args.summary_json, ledger, expected_keys,
                                    args.precision, transport, identity,
                                )
                                print(
                                    f"{name} task={task_id} trial={trial_id} "
                                    f"success={record['success']} steps={record['action_steps']} "
                                    f"replans={record['replans']} "
                                    f"completed={len(ledger)}/{len(expected_keys)}",
                                    flush=True,
                                )
                                break
                            except Exception as error:
                                failure = {
                                    "status": "technical_error",
                                    "suite": name,
                                    "task_id": task_id,
                                    "trial_id": trial_id,
                                    "attempt": attempt,
                                    "precision": args.precision,
                                    "identity": identity,
                                    "transport": transport,
                                    "error": repr(error),
                                    "traceback": traceback.format_exc(),
                                    "time_unix_seconds": time.time(),
                                }
                                append_record(args.results_jsonl, failure)
                                print(
                                    f"{name} task={task_id} trial={trial_id} attempt={attempt} "
                                    f"ERROR: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if attempt == args.max_attempts:
                                    raise
                finally:
                    env.close()
    finally:
        backend.close()

    write_summary(
        args.summary_json, ledger, expected_keys, args.precision, transport, identity
    )
    missing = expected_keys - set(ledger)
    if missing:
        raise RuntimeError(f"evaluation incomplete; missing {sorted(missing)}")
    successes = sum(bool(record["success"]) for record in ledger.values())
    suites_label = ",".join(suites)
    print(
        f"LIBERO [{suites_label}] complete: {successes}/{len(expected_keys)} successes",
        flush=True,
    )


if __name__ == "__main__":
    main()

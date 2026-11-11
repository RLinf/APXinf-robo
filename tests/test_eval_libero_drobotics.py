"""Pin the evaluator against a contract-pinned engine.

The evaluator holds no S600 knowledge any more: it sends the frames it would send
ApxInf, checks the dialect, and records what the server publishes. The server is
faked at the ``openpi_client`` boundary, so no socket is opened.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("openpi_client")

from openpi_client import websocket_client_policy  # noqa: E402

from apxinf_robo import engine_drobotics  # noqa: E402
from apxinf_robo.cli import eval_libero  # noqa: E402

METADATA = {
    "engine": "apx-drobotics",
    "runtime": "d-robotics-oellm",
    "sdk_version": "2.0.0",
    "model_type": "pi05_drobotics_w8",
    "contract_id": "drobotics-pi05-libero-w8",
    "contract_sha256": "a" * 64,
    "precision": "w8",
    "image_keys": ["observation/image", "observation/wrist_image"],
    "state_key": "observation/state",
    "prompt_key": "prompt",
}


class FakeSocket:
    ping_timeout = 20

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeServer:
    """Stands in for ``WebsocketClientPolicy``."""

    metadata: dict = {}
    response: dict = {}
    last = None

    def __init__(self, host, port):
        FakeServer.last = self
        self._ws = FakeSocket()
        self.received = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, observation):
        self.received.append(observation)
        return self.response


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", FakeServer)
    FakeServer.metadata = dict(METADATA)
    FakeServer.response = {
        "actions": np.zeros((10, 7), np.float32),
        "policy_timing": {"infer_ms": 146.0, "policy_ms": 148.0},
        "server_timing": {"infer_ms": 150.0},
    }
    return FakeServer


def frames():
    rng = np.random.default_rng(0)
    return (
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
    )


def test_every_engine_gets_the_same_raw_observation(server):
    """Resizing and prompt cleaning belong to the policy behind the server."""
    backend = eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    base, wrist = frames()
    state = np.zeros(8, np.float32)
    backend.infer(base, wrist, state, "  pick_up the soup\n")
    sent = server.last.received[0]
    assert sent["observation/image"] is base
    assert sent["observation/wrist_image"] is wrist
    assert sent["observation/state"] is state
    assert sent["prompt"] == "  pick_up the soup\n"


def test_timing_is_split_as_for_any_other_server(server):
    backend = eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    base, wrist = frames()
    actions, normalized, timing = backend.infer(base, wrist, np.zeros(8, np.float32), "go")
    assert actions.shape == (10, 7)
    assert normalized is None
    assert timing["model_seconds"] == pytest.approx(0.146)
    assert timing["server_processor_seconds"] == pytest.approx(0.004)


def test_slow_simulator_resets_do_not_read_as_a_dead_server(server):
    eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    assert server.last._ws.ping_timeout is None


def test_drobotics_model_gets_both_finger_joints(server):
    backend = eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    state = eval_libero._LIBERO_STATES[backend.model_type]({
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    })
    assert state.shape == (8,)


def test_server_of_another_precision_is_refused_and_disconnected(server):
    server.metadata = {**METADATA, "precision": "bf16"}
    with pytest.raises(RuntimeError, match="precision"):
        eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    assert server.last._ws.closed is True


def test_server_of_another_dialect_is_refused_and_disconnected(server):
    server.metadata = {**METADATA, "state_key": "state"}
    with pytest.raises(RuntimeError, match="state_key"):
        eval_libero.WebsocketBackend("127.0.0.1", 8000, "w8")
    assert server.last._ws.closed is True


COMMON = ["--results-jsonl", "r.jsonl", "--summary-json", "s.json"]


def _error(capsys, *argv) -> str:
    with pytest.raises(SystemExit):
        eval_libero.parse_args([*argv, *COMMON])
    return capsys.readouterr().err


def test_w8_over_a_websocket_needs_nothing_else():
    args = eval_libero.parse_args(["--backend", "websocket", "--precision", "w8", *COMMON])
    assert args.precision == "w8"


def test_w8_in_process_needs_the_drobotics_engine(capsys):
    err = _error(capsys, "--backend", "in-process", "--model-dir", "/m", "--precision", "w8")
    assert "--precision w8 is the S600 model" in err


def test_drobotics_engine_needs_w8(capsys):
    err = _error(capsys, "--backend", "in-process", "--engine", "drobotics", "--precision", "bf16")
    assert "--engine drobotics requires --precision w8" in err


def test_drobotics_engine_is_for_the_in_process_backend(capsys):
    err = _error(capsys, "--backend", "websocket", "--engine", "drobotics", "--precision", "w8")
    assert "--engine drobotics applies to --backend in-process" in err


def test_drobotics_engine_finds_the_sdk_itself():
    args = eval_libero.parse_args(
        ["--backend", "in-process", "--engine", "drobotics", "--precision", "w8", *COMMON]
    )
    assert args.model_dir is None


def test_in_process_drobotics_backend_uses_the_same_preset_keys(monkeypatch, tmp_path):
    testing = pytest.importorskip("apx_drobotics.testing")
    fake = testing.fake_deployment(tmp_path)
    load = engine_drobotics.load_policy
    monkeypatch.setattr(engine_drobotics, "load_policy",
                        lambda model_dir, **options: load(model_dir, **options, **fake.options))
    args = eval_libero.parse_args(
        ["--backend", "in-process", "--engine", "drobotics", "--precision", "w8",
         "--model-dir", str(fake.sdk_root), *COMMON]
    )
    backend = eval_libero.build_backend(args)
    try:
        assert backend.model_type == "pi05_drobotics_w8"
        base, wrist = frames()
        actions, normalized, timing = backend.infer(
            base, wrist, np.zeros(8, np.float32), " pick_up the soup\n"
        )
    finally:
        backend.close()
    assert actions.shape == (10, 7)
    assert normalized is None
    assert timing["model_seconds"] == pytest.approx(0.006)
    assert fake.request(0)[0]["prompt"] == "pick up the soup"


# --- ledger identity ------------------------------------------------------------


def _args(**overrides):
    values = {"replan_steps": 5, "max_steps": 520, "seed": 7, "settle_gripper": -1.0}
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_rows(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(trial_id=0, **extra):
    return {"status": "completed", "suite": "libero_object", "task_id": 0,
            "trial_id": trial_id, "precision": "w8", "success": True, **extra}


def test_identity_records_what_the_engine_publishes():
    assert eval_libero.run_identity(_args(), METADATA) == {
        "replan_steps": 5,
        "max_steps": 520,
        "seed": 7,
        "settle_gripper": -1.0,
        "wait_steps": 10,
        "engine": "apx-drobotics",
        "runtime": "d-robotics-oellm",
        "sdk_version": "2.0.0",
        "model_type": "pi05_drobotics_w8",
        "contract_id": "drobotics-pi05-libero-w8",
        "contract_sha256": "a" * 64,
    }


@pytest.mark.parametrize("metadata", [None, {}, {"model_type": "pi05", "precision": "bf16"}])
def test_identity_of_an_engine_without_a_contract_is_the_rollout_setting(metadata):
    assert eval_libero.run_identity(_args(replan_steps=8), metadata) == {
        "replan_steps": 8, "max_steps": 520, "seed": 7,
        "settle_gripper": -1.0, "wait_steps": 10,
    }


def test_resume_accepts_rows_of_the_same_run(tmp_path):
    identity = eval_libero.run_identity(_args(), METADATA)
    ledger = tmp_path / "r.jsonl"
    _write_rows(ledger, _row(0, identity=identity), _row(1, identity=identity))
    assert len(eval_libero.completed_runs(ledger, "w8", identity)) == 2


@pytest.mark.parametrize("field, value", [("contract_sha256", "0" * 64), ("replan_steps", 10)])
def test_resume_refuses_rows_of_another_run(tmp_path, field, value):
    identity = eval_libero.run_identity(_args(), METADATA)
    ledger = tmp_path / "r.jsonl"
    _write_rows(ledger, _row(0, identity={**identity, field: value}))
    with pytest.raises(ValueError, match=field):
        eval_libero.completed_runs(ledger, "w8", identity)


def test_failed_rows_of_another_run_are_refused_too(tmp_path):
    identity = eval_libero.run_identity(_args(), METADATA)
    ledger = tmp_path / "r.jsonl"
    failure = {"status": "technical_error", "suite": "libero_object", "task_id": 0,
               "trial_id": 0, "precision": "w8", "identity": {**identity, "replan_steps": 3}}
    _write_rows(ledger, failure)
    with pytest.raises(ValueError, match="replan_steps"):
        eval_libero.completed_runs(ledger, "w8", identity)


def test_contract_run_refuses_rows_without_identity(tmp_path):
    identity = eval_libero.run_identity(_args(), METADATA)
    ledger = tmp_path / "r.jsonl"
    _write_rows(ledger, _row(0))
    with pytest.raises(ValueError, match="no run identity"):
        eval_libero.completed_runs(ledger, "w8", identity)


def test_older_ledgers_without_identity_still_resume(tmp_path):
    ledger = tmp_path / "r.jsonl"
    _write_rows(ledger, _row(0, precision="bf16"))
    identity = eval_libero.run_identity(_args())
    assert len(eval_libero.completed_runs(ledger, "bf16", identity)) == 1
    assert len(eval_libero.completed_runs(ledger, "bf16")) == 1


def test_summary_records_the_identity(tmp_path):
    identity = eval_libero.run_identity(_args(), METADATA)
    summary = tmp_path / "s.json"
    key = ("libero_object", 0, 0)
    row = {**_row(0, identity=identity), "action_steps": 1, "replans": 1}
    eval_libero.write_summary(summary, {key: row}, {key}, "w8", "openpi_websocket", identity)
    assert json.loads(summary.read_text())["identity"] == identity

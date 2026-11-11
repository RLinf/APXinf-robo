"""``apxinf-robo serve --engine drobotics``, end to end, as it runs on the board.

The command runs as a real subprocess with ApxInf blocked, against a fake SDK and
worker, and the evaluator's own backend connects to it.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("msgpack")
pytest.importorskip("websockets")
pytest.importorskip("openpi_client")
testing = pytest.importorskip("apx_drobotics.testing")

from openpi_client import image_tools, websocket_client_policy  # noqa: E402

from apxinf_robo.cli import eval_libero, serve  # noqa: E402

NO_APXINF = pathlib.Path(__file__).with_name("_no_apxinf.py")


class Running:
    def __init__(self, deployment, *extra: str, cwd) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        command = [
            sys.executable, str(NO_APXINF), "apxinf_robo.cli.serve",
            "--engine", "drobotics", "--robot", "franka_libero",
            "--model-dir", str(deployment.sdk_root),
            "--policy-options", json.dumps(deployment.options),
            "--host", "127.0.0.1", "--port", str(self.port), *extra,
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=cwd)
        self.lines: list[str] = []
        self.listening = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.append(line.rstrip())
            if "listening on" in line:
                self.listening.set()

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def wait_listening(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.listening.wait(0.1):
                return True
            if self.process.poll() is not None:
                self._reader.join(5)
                return self.listening.is_set()
        return False

    def wait_exit(self, timeout: float = 30.0) -> int:
        code = self.process.wait(timeout)
        self._reader.join(5)
        return code

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


@pytest.fixture
def launch(tmp_path):
    running = []

    def start(mode: str = "ok", *extra: str, prepare=None):
        root = tmp_path / f"deployment-{len(running)}"
        root.mkdir()
        fake = testing.fake_deployment(root, mode)
        if prepare is not None:
            prepare(fake)
        running.append(Running(fake, *extra, cwd=tmp_path))
        return running[-1], fake

    yield start
    for item in running:
        item.stop()


def frames():
    rng = np.random.default_rng(2)
    return (
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
    )


def test_evaluator_talks_to_the_server_without_apxinf_on_the_board(launch):
    server, fake = launch()
    assert server.wait_listening(), server.output
    backend = eval_libero.WebsocketBackend("127.0.0.1", server.port, "w8")
    base, wrist = frames()
    actions, _, timing = backend.infer(base, wrist, np.zeros(8, np.float32), " pick_up the soup\n")
    assert backend.model_type == "pi05_drobotics_w8"
    assert backend.metadata["robot"] == "franka_libero"
    assert backend.metadata["policy"] == "franka_libero"
    backend.close()
    assert np.array_equal(actions, testing.EXPECTED_ACTIONS.astype(np.float32))
    assert timing["model_seconds"] == pytest.approx(0.006)

    header, payload = fake.request(0)
    assert header["prompt"] == "pick up the soup"
    assert header["images"] == [{"width": 224, "height": 224}] * 2
    # The bytes the runtime sees are what the PC used to send after its own resize.
    assert payload == b"".join(
        np.transpose(image_tools.resize_with_pad(image, 224, 224), (2, 0, 1)).tobytes()
        for image in (base, wrist)
    )
    assert "worker: fake runtime log on stdout" in server.output


def test_bad_request_is_named_and_the_server_keeps_serving(launch):
    server, fake = launch()
    assert server.wait_listening(), server.output
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", server.port)
    base, wrist = frames()
    good = {"observation/image": base, "observation/wrist_image": wrist,
            "observation/state": np.zeros(8, np.float32), "prompt": "go"}
    with pytest.raises(RuntimeError, match="observation/state"):
        client.infer({**good, "observation/state": np.zeros(7, np.float32)})
    assert fake.request_count() == 0
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", server.port)
    assert client.infer(good)["actions"].shape == (10, 7)
    assert server.process.poll() is None


def test_server_does_not_listen_when_the_deployment_differs(launch):
    def change(fake):
        config = fake.sdk_root / fake.contract["runtime_config"]["path"]
        document = json.loads(config.read_text())
        document["processing"]["inject_state"] = True
        config.write_text(json.dumps(document))

    server, _ = launch("ok", prepare=change)
    assert server.wait_exit() == 2
    assert not server.listening.is_set()
    assert "processing.inject_state" in server.output


def test_server_does_not_listen_when_the_runtime_fails_to_load(launch):
    server, _ = launch("init_fail")
    assert server.wait_exit() == 3
    assert not server.listening.is_set()
    assert "init_failed" in server.output


@pytest.mark.parametrize(
    "mode, needle", [("crash", "worker exited"), ("error", "runtime error: boom")]
)
def test_runtime_failure_stops_the_server(launch, mode, needle):
    server, _ = launch(mode)
    assert server.wait_listening(), server.output
    backend = eval_libero.WebsocketBackend("127.0.0.1", server.port, "w8")
    base, wrist = frames()
    with pytest.raises(RuntimeError, match="server failure"):
        backend.infer(base, wrist, np.zeros(8, np.float32), "go")
    assert server.wait_exit() == 1
    assert needle in server.output


def test_sigterm_stops_cleanly_and_releases_the_worker(launch):
    server, _ = launch()
    assert server.wait_listening(), server.output
    os.kill(server.process.pid, signal.SIGTERM)
    assert server.wait_exit() == 0


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["--checkpoint", "/ckpt"], "--checkpoint applies to ApxInf"),
        (["--autotune", "--action-horizon", "50"], "--autotune, --action-horizon"),
        (["--random-weights"], "--random-weights applies to ApxInf"),
        (["--robot", "none"], "requires a robot preset"),
    ],
)
def test_options_the_drobotics_cannot_honour_are_refused(argv, needle):
    args = serve.build_parser().parse_args(["--engine", "drobotics", *argv])
    with pytest.raises(ValueError, match=needle):
        serve.run(args)


def test_w8_without_the_drobotics_engine_is_refused():
    args = serve.build_parser().parse_args(["--model-dir", "/m", "--precision", "w8"])
    with pytest.raises(ValueError, match="--engine drobotics"):
        serve.run(args)


def test_apxinf_default_precision_is_unchanged(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "pi0_fast"}))
    captured = {}
    monkeypatch.setattr(serve, "build_robot_policy", lambda robot, model_dir, **options: (
        captured.update(options) or type("P", (), {"action_horizon": 1, "action_dim": 1,
                                                   "close": lambda self: None})()
    ))
    monkeypatch.setattr(serve, "websocket_server", lambda *args: type(
        "S", (), {"serve_forever": lambda self: None})())
    serve.run(serve.build_parser().parse_args(["--model-dir", str(tmp_path)]))
    assert captured["precision"] == "bf16"

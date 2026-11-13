"""Pin the S600 websocket server, with the real ``openpi_client`` as the client."""

from __future__ import annotations

import threading
import urllib.request

import numpy as np
import pytest

pytest.importorskip("msgpack")
pytest.importorskip("websockets")
pytest.importorskip("openpi_client")

from openpi_client import websocket_client_policy  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from apx_drobotics.serving import WebsocketPolicyServer, unpackb  # noqa: E402
from apx_drobotics.testing import EXPECTED_ACTIONS  # noqa: E402

from conftest import observation  # noqa: E402


class Running:
    def __init__(self, policy):
        self.server = WebsocketPolicyServer(policy, "127.0.0.1", 0)
        self.status = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.server.listening.wait(10)

    def _run(self):
        self.status = self.server.serve_forever()

    def client(self):
        return websocket_client_policy.WebsocketClientPolicy("127.0.0.1", self.server.port)


@pytest.fixture
def serve(make_policy):
    running = []

    def start(mode="ok", **options):
        policy, fake = make_policy(mode, **options)
        running.append(Running(policy))
        return running[-1], fake

    yield start
    for item in running:
        item.server.stop()
        item.thread.join(10)


def test_metadata_is_sent_on_connect(serve):
    running, fake = serve()
    metadata = running.client().get_server_metadata()
    assert metadata["contract_id"] == fake.contract["contract_id"]
    assert metadata["image_keys"] == ["cam/base", "cam/wrist"]


def test_actions_come_back_with_timing(serve):
    running, _ = serve()
    response = running.client().infer(observation(256))
    assert response["actions"].dtype == np.float32
    assert np.array_equal(response["actions"], EXPECTED_ACTIONS.astype(np.float32))
    assert response["policy_timing"]["infer_ms"] == 6.0
    assert response["policy_timing"]["policy_ms"] > 0
    assert response["server_timing"]["infer_ms"] > 0


def test_bad_request_is_named_and_the_server_keeps_serving(serve):
    running, fake = serve()
    with pytest.raises(RuntimeError, match="proprio"):
        running.client().infer({**observation(), "proprio": np.zeros(7, np.float32)})
    assert fake.request_count() == 0
    assert running.client().infer(observation())["actions"].shape == (10, 7)
    assert running.thread.is_alive()


def test_text_frame_is_rejected(serve):
    running, _ = serve()
    with connect(f"ws://127.0.0.1:{running.server.port}", compression=None) as socket:
        unpackb(socket.recv())
        socket.send("not msgpack")
        assert "expected a binary msgpack frame" in socket.recv()


@pytest.mark.parametrize(
    "mode, options, needle",
    [
        ("crash", {}, "worker exited"),
        ("hang", {"infer_deadline": 0.5}, "did not answer within 0.5 s"),
        ("error", {}, "runtime error: boom"),
        ("nan", {}, "non-finite"),
    ],
)
def test_runtime_failure_reaches_the_client_and_stops_the_server(serve, mode, options, needle):
    running, _ = serve(mode, **options)
    with pytest.raises(RuntimeError, match="server failure"):
        running.client().infer(observation())
    running.thread.join(10)
    assert running.status == 1
    assert needle in running.server.failure


def test_stopping_is_not_a_failure(serve):
    running, _ = serve()
    running.server.stop()
    running.thread.join(10)
    assert running.status == 0


def test_health_check(serve):
    running, _ = serve()
    url = f"http://127.0.0.1:{running.server.port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as reply:
        assert reply.status == 200

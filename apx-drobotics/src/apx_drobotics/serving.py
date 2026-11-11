"""OpenPI-compatible websocket server for :class:`~apx_drobotics.policy.DroboticsPolicy`.

The S600 counterpart of ``apxinf.serving``. Its wire protocol is the same, so the
unmodified ``openpi_client`` connects: metadata on connect, msgpack-numpy frames,
``/healthz``. Two things differ, both on purpose:

* A request the contract refuses gets a text frame naming every problem, then the
  connection closes with code 1011. The server goes on serving.
* A runtime failure also stops the server, and :meth:`serve_forever` returns non-zero.
  Restarting the worker would hide the failure and restart the runtime's noise
  sequence, so that is left to the operator.

Requires the ``serve`` extra (``msgpack``, ``websockets``).
"""

from __future__ import annotations

import asyncio
import functools
import http
import logging
import signal
import threading
import time
from typing import Any, Mapping, Optional

import msgpack
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .errors import RequestError

__all__ = ["WebsocketPolicyServer", "packb", "unpackb", "wire_response"]

logger = logging.getLogger(__name__)


# The same msgpack extension as openpi_client.msgpack_numpy, which clients use.


def _pack_default(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_hook(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]),
                          shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


packb = functools.partial(msgpack.packb, default=_pack_default)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_hook)


def wire_response(result: Mapping[str, Any], infer_ms: float) -> dict:
    """Project a policy result onto the wire; timing uses ``apxinf.serving``'s names."""
    timing = result.get("timing", {})
    return {
        "actions": np.ascontiguousarray(result["actions"]),
        "policy_timing": {"infer_ms": float(timing.get("model_ms", 0.0)),
                          "policy_ms": float(timing.get("total_ms", 0.0))},
        "server_timing": {"infer_ms": infer_ms},
    }


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WebsocketPolicyServer:
    """Serve one policy until it is stopped or its runtime fails."""

    def __init__(self, policy: Any, host: str, port: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self.failure: Optional[str] = None
        #: The bound port, once :attr:`listening` is set; ``port=0`` picks a free one.
        self.port: Optional[int] = None
        self.listening = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _reject(self, websocket, problems) -> None:
        message = "S600 server rejected the request:\n" + "\n".join(f"  - {p}" for p in problems)
        logger.warning("%s %s", websocket.remote_address, message.replace("\n", " "))
        await websocket.send(message)
        await websocket.close(1011, "request rejected")

    def _fail(self, message: str) -> None:
        logger.error("stopping: %s", message)
        self.failure = message
        self._stop.set()

    async def handler(self, websocket) -> None:
        logger.info("connected: %s", websocket.remote_address)
        try:
            await websocket.send(packb(dict(self._policy.metadata)))
            async for message in websocket:
                if isinstance(message, str):
                    await self._reject(websocket, ["expected a binary msgpack frame"])
                    return
                try:
                    observation = unpackb(message)
                except Exception as error:  # malformed client data, not a server fault
                    await self._reject(websocket, [f"cannot decode msgpack: {error}"])
                    return
                started = time.perf_counter()
                try:
                    # On the event loop, as apxinf.serving does: inference is one at
                    # a time, and the worker's own deadline bounds this call.
                    result = self._policy.infer(observation)
                except RequestError as error:
                    await self._reject(websocket, error.problems)
                    return
                except Exception as error:
                    await self._reject(websocket, [f"server failure: {error}"])
                    self._fail(str(error))
                    return
                infer_ms = (time.perf_counter() - started) * 1000.0
                await websocket.send(packb(wire_response(result, infer_ms)))
        except ConnectionClosed:
            pass
        finally:
            logger.info("disconnected: %s", websocket.remote_address)

    async def run(self) -> int:
        self._stop = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._loop.add_signal_handler(signum, self._stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass  # not the main thread: the caller stops the server with stop()
        # ping_timeout=None: a LIBERO client can hold its GIL for ~35 s while it
        # rebuilds the MuJoCo render context on reset (seen under WSL's D3D12 GL), so
        # it misses the 20 s Pong deadline although it is alive. Pings are still sent;
        # a client that really goes away closes its TCP connection.
        async with serve(self.handler, self._host, self._port, compression=None, max_size=None,
                         ping_timeout=None, process_request=_health_check) as server:
            self.port = server.sockets[0].getsockname()[1]
            logger.info("listening on ws://%s:%d (contract %s, sha256 %s)", self._host, self.port,
                        self._policy.metadata.get("contract_id"),
                        self._policy.metadata.get("contract_sha256"))
            self.listening.set()
            await self._stop.wait()
        return 1 if self.failure else 0

    def serve_forever(self) -> int:
        """Serve until stopped. Returns the exit status: non-zero after a runtime failure."""
        return asyncio.run(self.run())

    def stop(self) -> None:
        """Stop the server from another thread. Harmless if it has already stopped."""
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:  # the loop has closed
                pass

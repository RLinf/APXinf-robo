"""The S600 runtime: the C++ worker process and its framed pipe protocol.

The SDK is a C++ library and prints to stdout as soon as it is loaded, so it runs in
its own process (``worker/pi05_worker.cc``, which documents the frames). Calls are
synchronous, because :meth:`apx_drobotics.policy.DroboticsPolicy.infer` is, and they carry a
deadline, because a hung BPU call must not hang the caller.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .errors import RuntimeFailure, StartupFailure
from .preprocess import ModelRequest

__all__ = ["DEFAULT_SDK_ROOT", "DEFAULT_WORKER", "L2M_SIZES", "Worker"]

DEFAULT_SDK_ROOT = Path(
    "/opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600"
)
#: Where ``scripts/deploy.sh`` builds the worker, for a source checkout of this package.
DEFAULT_WORKER = Path(__file__).resolve().parents[2] / "build/worker/drobotics_pi05_worker"
#: The official run.sh sets this before starting the demo; runtime init fails without it.
L2M_SIZES = "6:6:6:6"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MAX_HEADER_BYTES = 1 << 20
_MAX_PAYLOAD_BYTES = 64 << 20

logger = logging.getLogger(__name__)


class _Timeout(Exception):
    pass


class _Closed(Exception):
    pass


def encode_request(request: ModelRequest) -> bytes:
    """One request frame. The runtime's raw-image path takes CHW."""
    header = {
        "op": "infer",
        "prompt": request.prompt,
        "state": [float(value) for value in np.asarray(request.state, dtype=np.float64)],
        "images": [{"width": int(image.shape[1]), "height": int(image.shape[0])}
                   for image in request.images],
    }
    text = json.dumps(header).encode("utf-8")
    payload = b"".join(
        np.ascontiguousarray(np.transpose(image, (2, 0, 1))).tobytes() for image in request.images
    )
    return struct.pack("<I", len(text)) + text + struct.pack("<I", len(payload)) + payload


class Worker:
    """One running worker process. Requests are served one at a time."""

    def __init__(self, process: subprocess.Popen, replies: int) -> None:
        self._process = process
        self._replies: Optional[int] = replies
        self._lock = threading.Lock()
        self._failure: Optional[str] = None
        self._logs = threading.Thread(target=self._forward_logs, daemon=True)
        self._logs.start()
        self.version: Optional[str] = None

    @classmethod
    def start(cls, executable, config, *, ready_timeout: float) -> "Worker":
        """Start the worker, and return once the model has loaded."""
        config = Path(config)
        read_end, write_end = os.pipe()
        try:
            process = subprocess.Popen(
                [str(executable), "--config", config.name, "--reply-fd", str(write_end)],
                cwd=config.parent,
                env=dict(os.environ, HB_DNN_USER_DEFINED_L2M_SIZES=L2M_SIZES),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                pass_fds=(write_end,),
            )
        except OSError as error:
            os.close(read_end)
            raise StartupFailure(f"cannot start worker {executable}: {error}") from error
        finally:
            os.close(write_end)
        os.set_blocking(read_end, False)
        os.set_blocking(process.stdin.fileno(), False)
        worker = cls(process, read_end)
        try:
            header, _ = worker._read_frame(time.monotonic() + ready_timeout)
        except _Timeout:
            worker.fail("not ready")
            worker.close()
            raise StartupFailure(f"worker was not ready within {ready_timeout:g} s") from None
        except _Closed:
            status = worker._exit_status()
            worker.close()
            raise StartupFailure(f"worker exited (status {status}) before it was ready") from None
        except RuntimeFailure as error:
            worker.close()
            raise StartupFailure(str(error)) from error
        if header.get("event") != "ready":
            worker.close()
            raise StartupFailure(f"worker failed to start: {header}")
        worker.version = header.get("oellm_version")
        return worker

    def _forward_logs(self) -> None:
        for line in self._process.stdout:
            text = _ANSI.sub("", line.decode("utf-8", "replace")).rstrip()
            if text:
                logger.info("worker: %s", text)

    def _exit_status(self):
        try:
            return self._process.wait(5)
        except subprocess.TimeoutExpired:
            return "still running"

    def _read_exact(self, size: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Timeout
            ready, _, _ = select.select([self._replies], [], [], remaining)
            if not ready:
                continue
            try:
                chunk = os.read(self._replies, size - len(data))
            except BlockingIOError:
                continue
            if not chunk:
                raise _Closed
            data += chunk
        return bytes(data)

    def _read_sized(self, limit: int, deadline: float) -> bytes:
        size = struct.unpack("<I", self._read_exact(4, deadline))[0]
        if size > limit:
            raise RuntimeFailure(f"worker sent a {size}-byte field, over the {limit}-byte limit")
        return self._read_exact(size, deadline)

    def _read_frame(self, deadline: float) -> Tuple[dict, bytes]:
        text = self._read_sized(_MAX_HEADER_BYTES, deadline)
        try:
            header = json.loads(text)
        except ValueError as error:
            raise RuntimeFailure(f"worker sent an unreadable header: {error}") from error
        return header, self._read_sized(_MAX_PAYLOAD_BYTES, deadline)

    def _write_all(self, data: bytes, deadline: float) -> None:
        pipe = self._process.stdin.fileno()
        view = memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Timeout
            _, ready, _ = select.select([], [pipe], [], remaining)
            if not ready:
                continue
            try:
                view = view[os.write(pipe, view):]
            except BlockingIOError:
                continue
            except (BrokenPipeError, ConnectionResetError):
                raise _Closed from None

    def fail(self, reason: str) -> None:
        """Refuse every later request; the worker's state can no longer be trusted."""
        self._failure = reason

    def infer(self, request: ModelRequest, deadline_seconds: float) -> Tuple[dict, bytes]:
        """Run one request; return the reply header and the action bytes."""
        with self._lock:
            if self._failure is not None:
                raise RuntimeFailure(f"the runtime already failed: {self._failure}")
            deadline = time.monotonic() + deadline_seconds
            try:
                self._write_all(encode_request(request), deadline)
                header, payload = self._read_frame(deadline)
                if not header.get("ok"):
                    raise RuntimeFailure(
                        f"runtime error: {header.get('error')} (code {header.get('error_code')})"
                    )
            except _Timeout:
                self._failure = f"worker did not answer within {deadline_seconds:g} s"
            except _Closed:
                self._failure = f"worker exited (status {self._exit_status()})"
            except RuntimeFailure as error:
                self._failure = str(error)
            if self._failure is not None:
                raise RuntimeFailure(self._failure)
            return header, payload

    def close(self) -> None:
        """Stop the worker. Closing stdin lets it release the BPU memory itself."""
        if self._replies is None:
            return
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            # A failed worker may be stuck in a BPU call and will not notice stdin.
            self._process.wait(1 if self._failure is not None else 10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._logs.join(5)
        os.close(self._replies)
        self._replies = None

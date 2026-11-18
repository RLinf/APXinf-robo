"""A fake deployment, for testing this package and its callers without a board.

The C++ worker is replaced by a small Python program that speaks the same framed
protocol, and the SDK by a tiny file tree that satisfies a copy of the contract. The
real policy, server and contract checks run unchanged against them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys
import textwrap
from typing import Tuple

import numpy as np

from .contract import load_contract

__all__ = ["EXPECTED_ACTIONS", "FakeDeployment", "fake_deployment"]

#: What the fake worker returns in ``ok`` mode.
EXPECTED_ACTIONS = np.arange(70, dtype=np.float64).reshape(10, 7) / 100.0

_FAKE_WORKER = textwrap.dedent(
    '''
    import json, os, struct, sys, time

    mode, record = sys.argv[1:3]
    args = sys.argv[3:]
    config = args[args.index("--config") + 1]
    reply = os.fdopen(int(args[args.index("--reply-fd") + 1]), "wb", buffering=0)
    # The real SDK prints to stdout; the protocol must not care.
    print("fake runtime log on stdout", flush=True)

    def send(header, payload=b""):
        text = json.dumps(header).encode()
        reply.write(struct.pack("<I", len(text)) + text + struct.pack("<I", len(payload)) + payload)

    def read():
        data = sys.stdin.buffer.read(4)
        if len(data) < 4:
            return None
        header = json.loads(sys.stdin.buffer.read(struct.unpack("<I", data)[0]))
        payload = sys.stdin.buffer.read(struct.unpack("<I", sys.stdin.buffer.read(4))[0])
        return header, payload

    if mode == "init_fail":
        send({"event": "init_failed", "error_code": 7})
        sys.exit(2)
    if mode == "silent":
        time.sleep(60)
    if not os.path.isfile(config):
        sys.exit(f"config {config} not found from {os.getcwd()}")
    if os.environ.get("HB_DNN_USER_DEFINED_L2M_SIZES") != "6:6:6:6":
        sys.exit("the runtime needs HB_DNN_USER_DEFINED_L2M_SIZES")
    send({"event": "ready", "oellm_version": "2.0.0"})
    count = 0
    while True:
        frame = read()
        if frame is None:
            break
        header, payload = frame
        with open(os.path.join(record, f"request_{count}.json"), "w") as stream:
            json.dump(header, stream)
        with open(os.path.join(record, f"request_{count}.bin"), "wb") as stream:
            stream.write(payload)
        count += 1
        if mode == "crash":
            os._exit(9)
        if mode == "hang":
            time.sleep(60)
        if mode == "error":
            send({"ok": False, "error": "boom", "error_code": -3})
            continue
        values = [i / 100.0 for i in range(70)]
        if mode == "nan":
            values[11] = float("nan")
        shape = [1, 50, 7] if mode == "bad_shape" else [1, 10, 7]
        send({"ok": True, "action_shape": shape,
              "timing_ms": {"preproc": 0.1, "visual": 1.0, "lm": 2.0, "action": 3.0,
                            "postproc": 0.1, "total": 7.5, "runtime_wall": 7.6}},
             struct.pack("<70d", *values))
    '''
)


@dataclass(frozen=True)
class FakeDeployment:
    sdk_root: Path
    contract_path: Path
    contract: dict
    worker: Path
    record: Path

    @property
    def options(self) -> dict:
        """Keyword arguments for ``DroboticsPolicy.from_deployment``, JSON-serializable."""
        return {"contract": str(self.contract_path), "worker": str(self.worker)}

    def request(self, index: int) -> Tuple[dict, bytes]:
        """The header and payload of the ``index``-th request the worker received."""
        return (
            json.loads((self.record / f"request_{index}.json").read_text()),
            (self.record / f"request_{index}.bin").read_bytes(),
        )

    def request_count(self) -> int:
        return len(list(self.record.glob("request_*.json")))


def _entry(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fake_deployment(root, mode: str = "ok") -> FakeDeployment:
    """Write a fake SDK, worker and matching contract copy under ``root``.

    ``mode`` selects the worker's behaviour: ``ok``, ``init_fail``, ``silent`` (never
    becomes ready), ``crash``, ``hang``, ``error``, ``nan``, or ``bad_shape``.
    """
    root = Path(root)
    contract = load_contract()
    expect = contract["runtime_config"]["expect"]
    official = {
        "runtime_type": "Pi0Sdk",
        "model": {
            "version": "0.5", "denoise_num": 10, "work_path": "../../../model/Pi05_libero",
            "model_runner": {part: expect[f"model.model_runner.{part}"]
                             for part in ("visual", "lm", "action")},
        },
        "interface": {"action_horizon": 10, "raw_state_size": 8, "raw_action_size": 7,
                      "image_mask": [True, True, False]},
        "processing": {"preproc": True, "postproc": True, "filter": {"type": "none", "fs": 0},
                       "use_absolute_action": True, "use_quantiles_norm": True},
    }
    norm = {"norm_stats": {"state": {"q01": [0.0] * 8, "q99": [1.0] * 8},
                           "actions": {"q01": [0.0] * 7, "q99": [1.0] * 7}}}
    sdk = root / "sdk"

    def write(relative: str, data: bytes) -> dict:
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return _entry(data)

    contract["runtime_config"].update(
        write(contract["runtime_config"]["path"], json.dumps(official).encode())
    )
    for name in contract["sdk"]["files"]:
        contract["sdk"]["files"][name] = write(name, name.encode())
    for name in contract["model"]["files"]:
        data = json.dumps(norm).encode() if name == "norm_stats.json" else name.encode()
        contract["model"]["files"][name] = write(f"{contract['model']['directory']}/{name}", data)
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    record = root / "requests"
    record.mkdir()
    script = root / "fake_worker.py"
    script.write_text(_FAKE_WORKER)
    worker = root / "fake_worker"
    command = shlex.join([sys.executable, str(script), mode, str(record)])
    worker.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    worker.chmod(0o755)
    return FakeDeployment(sdk, contract_path, contract, worker, record)

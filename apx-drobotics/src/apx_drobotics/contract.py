"""The S600 Pi05 LIBERO contract: what is deployed, and what crosses the runtime boundary.

The contract (``contracts/pi05_libero_w8.json``) pins the SDK, the model files and the
runtime config, and states what the runtime takes and returns. Wire keys are not in it:
what a client calls its cameras belongs to the caller, as it does with ApxInf.

Each check returns a list of problems, so a caller can report every mismatch at once.
An empty list means the check passed. This file uses only the standard library, so it
can also run on the board outside any virtual environment::

    python3 contract.py check-deployment --sdk-root \\
        /opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600
    python3 contract.py print-metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

__all__ = [
    "DEFAULT_CONTRACT",
    "check_actions",
    "check_deployment",
    "check_model_input",
    "contract_sha256",
    "sha256_file",
    "load_contract",
    "model_metadata",
    "unmasked_views",
]

DEFAULT_CONTRACT = Path(__file__).resolve().parent / "contracts/pi05_libero_w8.json"

#: Marks a dotted config path as absent during lookup.
_MISSING = object()


def load_contract(path=None) -> dict:
    return json.loads(Path(path or DEFAULT_CONTRACT).read_text(encoding="utf-8"))


def contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash the contract's content, independent of file formatting and key order."""
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lookup(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _check_file(path: Path, entry: Mapping[str, Any], *, verify_hash: bool) -> list[str]:
    if not path.is_file():
        return [f"{path}: file is missing"]
    size = path.stat().st_size
    if size != entry["bytes"]:
        return [f"{path}: {size} bytes, contract says {entry['bytes']}"]
    if verify_hash:
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            return [f"{path}: sha256 {actual}, contract says {entry['sha256']}"]
    return []


def _check_config(contract: Mapping[str, Any], path: Path) -> list[str]:
    expected = contract["runtime_config"]
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{path}: cannot read runtime config: {error}"]
    problems = []
    for dotted, want in expected["expect"].items():
        got = _lookup(config, dotted)
        if got is _MISSING:
            problems.append(f"runtime config {dotted}: missing, contract says {want!r}")
        elif got != want:
            problems.append(f"runtime config {dotted}: {got!r}, contract says {want!r}")
    for dotted in expected["absent"]:
        got = _lookup(config, dotted)
        if got is not _MISSING:
            problems.append(f"runtime config {dotted}: set to {got!r}, contract says unset")
    return problems


def _check_norm_stats(contract: Mapping[str, Any], path: Path) -> list[str]:
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
    except (OSError, ValueError, KeyError) as error:
        return [f"{path}: cannot read norm stats: {error}"]
    widths = {"state": contract["state"]["dim"], "actions": contract["action"]["dim"]}
    problems = []
    for section, width in widths.items():
        for field in ("q01", "q99"):
            got = len(stats.get(section, {}).get(field, ()))
            if got != width:
                problems.append(f"norm stats {section}.{field}: {got} values, contract says {width}")
    return problems


def check_deployment(contract: Mapping[str, Any], sdk_root, *, verify_hashes: bool = True) -> list[str]:
    """Check the deployed SDK, model and runtime config before the runtime starts."""
    root = Path(sdk_root)
    model_dir = root / contract["model"]["directory"]
    config_entry = contract["runtime_config"]
    problems = _check_file(root / config_entry["path"], config_entry, verify_hash=verify_hashes)
    problems += _check_config(contract, root / config_entry["path"])
    for name, entry in contract["sdk"]["files"].items():
        problems += _check_file(root / name, entry, verify_hash=verify_hashes)
    for name, entry in contract["model"]["files"].items():
        problems += _check_file(model_dir / name, entry, verify_hash=verify_hashes)
    problems += _check_norm_stats(contract, model_dir / "norm_stats.json")
    return problems


def unmasked_views(contract: Mapping[str, Any]) -> int:
    """How many cameras the runtime reads."""
    return sum(not slot["masked"] for slot in contract["camera"]["slots"])


def model_metadata(contract: Mapping[str, Any]) -> dict:
    """What the contract fixes about the served model, published to clients."""
    return {
        "contract_id": contract["contract_id"],
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(contract),
        **contract["metadata"],
    }


def _shape(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def _dtype(value: Any) -> Any:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else getattr(dtype, "name", str(dtype))


def _nonfinite(value: Any) -> int:
    def walk(item: Any) -> int:
        if isinstance(item, list):
            return sum(walk(child) for child in item)
        return 0 if math.isfinite(float(item)) else 1

    return walk(value.tolist())


def check_model_input(
    contract: Mapping[str, Any], images: Sequence[Any], state: Any, prompt: Any
) -> list[str]:
    """Check what is about to enter the runtime, after preprocessing."""
    camera = contract["camera"]
    want = contract["state"]
    problems = []
    views = unmasked_views(contract)
    if len(images) != views:
        problems.append(f"images: {len(images)} views, contract says {views}")
    for index, image in enumerate(images):
        if _shape(image) != camera["model_shape"]:
            problems.append(f"image {index}: shape {_shape(image)}, contract says {camera['model_shape']}")
        if _dtype(image) != camera["input"]["dtype"]:
            problems.append(f"image {index}: dtype {_dtype(image)}, contract says {camera['input']['dtype']}")
    if _shape(state) != [want["dim"]]:
        problems.append(f"state: shape {_shape(state)}, contract says [{want['dim']}]")
    else:
        if _dtype(state) != want["input_dtype"]:
            problems.append(f"state: dtype {_dtype(state)}, contract says {want['input_dtype']}")
        if _nonfinite(state):
            problems.append("state: contains non-finite values")
    if not isinstance(prompt, str) or not prompt:
        problems.append(f"prompt: must be a non-empty string, got {prompt!r}")
    return problems


def check_actions(contract: Mapping[str, Any], actions: Any) -> list[str]:
    """Check an action chunk before it leaves the policy."""
    action = contract["action"]
    want = [action["horizon"], action["dim"]]
    shape = _shape(actions)
    if shape != want:
        return [f"actions: shape {shape}, contract says {want}"]
    bad = _nonfinite(actions)
    return [f"actions: {bad} non-finite values"] if bad else []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check an S600 deployment against its contract")
    commands = parser.add_subparsers(dest="command", required=True)
    deployment = commands.add_parser("check-deployment", help="check the deployed SDK and model")
    deployment.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    deployment.add_argument("--sdk-root", type=Path, required=True)
    deployment.add_argument("--skip-hashes", action="store_true",
                            help="compare file sizes only; faster, weaker")
    metadata = commands.add_parser("print-metadata", help="print what the contract publishes")
    metadata.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)

    contract = load_contract(args.contract)
    if args.command == "print-metadata":
        print(json.dumps(model_metadata(contract), indent=2, ensure_ascii=False))
        return 0
    problems = check_deployment(contract, args.sdk_root, verify_hashes=not args.skip_hashes)
    for problem in problems:
        print(f"CONTRACT VIOLATION: {problem}", file=sys.stderr)
    print(json.dumps({"check": args.command, "contract_id": contract["contract_id"],
                      "contract_sha256": contract_sha256(contract),
                      "ok": not problems, "problems": len(problems)}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

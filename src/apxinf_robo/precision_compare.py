"""Backend-agnostic fixed-input action comparison for APXinf-robo.

Two aligned bundles -- same observations, same saved noise -- are compared
action-domain by action-domain, then by horizon slice. Nothing here imports
torch, lerobot or apxinf, so the comparison runs on a CPU-only checkout and
applies to any pair of backends (``bf16``/``int8_dynamic``, ``bf16``/``fp8_static``,
``bf16``/``w4a16``, ...).

Report axes
-----------
* action **domains** are the outer axis and come from the model, not from the clock:
  ``normalized_model`` ``[H, 32]`` (``valid_action_dim`` values followed by padding),
  ``policy_actions`` ``[H, 7]`` and ``env_step_actions`` ``[H, 7]``. A domain the
  bundles do not carry is reported as ``null``.
* horizon **slices** are the inner axis: ``executed_prefix`` ``[0, R)``,
  ``unused_suffix`` ``[R, H)`` and ``full_horizon`` ``[0, H)``.

Sign classes are neutral -- ``neg`` / ``zero`` / ``pos`` -- because gripper polarity is
a property of the controller, not of this comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "pi05_cross_precision_v1"

#: Action domains, outer axis of every report.
DOMAINS: tuple[str, ...] = ("normalized_model", "policy_actions", "env_step_actions")

#: Horizon slices, inner axis of every report.
SLICES: tuple[str, ...] = ("executed_prefix", "unused_suffix", "full_horizon")

#: Neutral sign classes. ``-0.0`` and ``+0.0`` both land in ``zero``.
SIGN_LABELS: tuple[str, ...] = ("neg", "zero", "pos")

#: Control-semantics grouping of the first seven action dimensions. The names follow
#: the command decomposition the LIBERO evaluator and the robot presets already use.
CONTROL_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("translation_command", 0, 3),
    ("rotation_command", 3, 6),
    ("gripper_command", 6, 7),
)


class ContractError(ValueError):
    """The two bundles are not a valid pair for this contract."""


@dataclass(frozen=True)
class ComparisonContract:
    """What both bundles promised to produce."""

    horizon: int = 50
    executed_prefix: int = 5
    full_action_dim: int = 32
    valid_action_dim: int = 7
    gripper_dim: int = 6
    gripper_domain: str = "policy_actions"
    baseline_precision: str = "bf16"
    candidate_precision: str = "int8_dynamic"

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.executed_prefix < 1:
            raise ContractError("horizon and executed_prefix must be positive")
        if self.executed_prefix > self.horizon:
            raise ContractError("executed_prefix cannot exceed the horizon")
        if self.full_action_dim < 1 or self.valid_action_dim < 1:
            raise ContractError("action dimensions must be positive")
        if self.full_action_dim < self.valid_action_dim:
            raise ContractError("full_action_dim cannot be smaller than valid_action_dim")
        if not 0 <= self.gripper_dim < self.valid_action_dim:
            raise ContractError("gripper_dim must index a valid action dimension")
        if self.gripper_domain not in DOMAINS:
            raise ContractError(f"unknown gripper domain {self.gripper_domain!r}")

    def domain_width(self, domain: str) -> int:
        """Widths differ per domain: only the model domain carries padding."""
        return self.full_action_dim if domain == "normalized_model" else self.valid_action_dim

    def slices(self) -> dict[str, tuple[int, int]]:
        return {
            "executed_prefix": (0, self.executed_prefix),
            "unused_suffix": (self.executed_prefix, self.horizon),
            "full_horizon": (0, self.horizon),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "executed_prefix": self.executed_prefix,
            "full_action_dim": self.full_action_dim,
            "valid_action_dim": self.valid_action_dim,
            "gripper_dim": self.gripper_dim,
            "gripper_domain": self.gripper_domain,
            "baseline_precision": self.baseline_precision,
            "candidate_precision": self.candidate_precision,
        }


@dataclass
class SampleRecord:
    """One observation's actions from one backend.

    ``arrays`` maps an action domain to ``[horizon, width]`` float actions;
    ``hashes`` carries whatever inputs the producer can vouch for
    (``observation`` / ``noise`` / ``state``), so a pair that did not see the same
    inputs is rejected instead of being silently compared.
    """

    sample_id: str
    precision: str
    arrays: Mapping[str, np.ndarray]
    hashes: Mapping[str, str] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.dtype.str}:{contiguous.shape}:".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_contract(
    baseline: Sequence[SampleRecord],
    candidate: Sequence[SampleRecord],
    contract: ComparisonContract,
) -> None:
    """Reject anything that would make the comparison mean something else."""
    expected = {
        "baseline": (baseline, contract.baseline_precision),
        "candidate": (candidate, contract.candidate_precision),
    }
    for side, (records, precision) in expected.items():
        if not records:
            raise ContractError(f"{side} bundle is empty")
        seen = set()
        for record in records:
            if record.sample_id in seen:
                raise ContractError(f"{side} bundle repeats sample_id {record.sample_id!r}")
            seen.add(record.sample_id)
            if record.precision != precision:
                raise ContractError(
                    f"{side} sample {record.sample_id!r} is labelled {record.precision!r}, "
                    f"expected {precision!r}"
                )
            if not record.arrays:
                raise ContractError(f"{side} sample {record.sample_id!r} carries no action domain")
            for domain, array in record.arrays.items():
                if domain not in DOMAINS:
                    raise ContractError(f"unknown action domain {domain!r}")
                values = np.asarray(array)
                if values.dtype.kind != "f":
                    raise ContractError(
                        f"{side} sample {record.sample_id!r} domain {domain!r} has "
                        f"non-float dtype {values.dtype}"
                    )
                width = contract.domain_width(domain)
                if values.shape != (contract.horizon, width):
                    raise ContractError(
                        f"{side} sample {record.sample_id!r} domain {domain!r} has shape "
                        f"{values.shape}, expected {(contract.horizon, width)}"
                    )

    if [r.sample_id for r in baseline] != [r.sample_id for r in candidate]:
        raise ContractError("sample order differs between the two bundles")
    for base, cand in zip(baseline, candidate):
        if set(base.arrays) != set(cand.arrays):
            raise ContractError(
                f"sample {base.sample_id!r} carries different domains: "
                f"{sorted(base.arrays)} vs {sorted(cand.arrays)}"
            )
        if dict(base.hashes) != dict(cand.hashes):
            raise ContractError(
                f"sample {base.sample_id!r} input hashes differ; the pair did not see "
                "the same observation/noise"
            )


def sign_class(values: np.ndarray) -> np.ndarray:
    """Map values to ``-1``/``0``/``+1``. ``-0.0`` and ``+0.0`` are both zero."""
    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0.0, 1, np.where(array < 0.0, -1, 0)).astype(np.int8)


def _class_counts(classes: np.ndarray) -> dict[str, int]:
    return {label: int((classes == value).sum()) for value, label in zip((-1, 0, 1), SIGN_LABELS)}


def _stats(diff: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(diff)
    good = int(finite.sum())
    if good == 0:
        return {"n": 0, "non_finite": int(diff.size), "mae": None, "rmse": None, "max_abs": None, "bias": None}
    values = diff[finite]
    magnitude = np.abs(values)
    return {
        "n": good,
        "non_finite": int(diff.size) - good,
        "mae": float(magnitude.mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "max_abs": float(magnitude.max()),
        "bias": float(values.mean()),
    }


def _per_dim_mae(window: np.ndarray) -> list[float | None]:
    mae = []
    for index in range(window.shape[-1]):
        column = window[:, :, index]
        finite = np.isfinite(column)
        mae.append(float(np.abs(column[finite]).mean()) if finite.any() else None)
    return mae


def _control_groups(window: np.ndarray) -> dict[str, Any] | None:
    """Group MAE/RMSE over the command semantics, when the width covers all three."""
    if window.shape[-1] < CONTROL_GROUPS[-1][2]:
        return None
    groups = {}
    for name, low, high in CONTROL_GROUPS:
        stats = _stats(window[:, :, low:high])
        groups[name] = {"rows": [low, high], **{key: stats[key] for key in ("n", "non_finite", "mae", "rmse", "max_abs")}}
    return groups


def _domain_block(
    base: np.ndarray, cand: np.ndarray, contract: ComparisonContract
) -> dict[str, Any]:
    valid = min(contract.valid_action_dim, base.shape[-1])
    diff = cand.astype(np.float64) - base.astype(np.float64)
    slices = {}
    for name, (low, high) in contract.slices().items():
        window = diff[:, low:high, :valid]
        slices[name] = {
            "rows": [low, high],
            **_stats(window),
            "groups": _control_groups(window),
            "per_dim_mae": _per_dim_mae(window),
        }

    padding = None
    if diff.shape[-1] > valid:
        pad = diff[:, :, valid:]
        finite = np.isfinite(pad)
        padding = {
            "dims": [valid, diff.shape[-1]],
            "count": int(diff.shape[-1] - valid),
            "max_abs": float(np.abs(pad[finite]).max()) if finite.any() else None,
            "non_finite": int(pad.size - finite.sum()),
        }
    return {"width": int(diff.shape[-1]), "valid_dims": valid, "slices": slices, "padding": padding}


def _gripper_block(
    baseline: Sequence[SampleRecord], candidate: Sequence[SampleRecord], contract: ComparisonContract
) -> dict[str, Any] | None:
    domain = contract.gripper_domain
    if not all(domain in record.arrays for record in baseline):
        return None
    if contract.gripper_dim >= contract.valid_action_dim:
        return None
    slices = {}
    for name, (low, high) in contract.slices().items():
        base = np.stack([np.asarray(r.arrays[domain])[low:high, contract.gripper_dim] for r in baseline])
        cand = np.stack([np.asarray(r.arrays[domain])[low:high, contract.gripper_dim] for r in candidate])
        base = base.astype(np.float64)
        cand = cand.astype(np.float64)
        usable = np.isfinite(base) & np.isfinite(cand)
        base_class = sign_class(base)
        cand_class = sign_class(cand)
        pairs = int(usable.sum())
        disagreements = int(((base_class != cand_class) & usable).sum())
        slices[name] = {
            "pairs": pairs,
            "disagreements": disagreements,
            "disagreement_rate": (disagreements / pairs) if pairs else None,
            "non_finite": int(base.size - pairs),
            "class_counts": {
                "baseline": _class_counts(base_class[usable]),
                "candidate": _class_counts(cand_class[usable]),
            },
        }
    return {"domain": domain, "dim": contract.gripper_dim, "slices": slices}


def _mae_prefix(record_base: np.ndarray, record_cand: np.ndarray, contract: ComparisonContract) -> float | None:
    low, high = contract.slices()["executed_prefix"]
    window = record_cand[low:high, : contract.valid_action_dim].astype(np.float64) - record_base[
        low:high, : contract.valid_action_dim
    ]
    finite = np.isfinite(window)
    return float(np.abs(window[finite]).mean()) if finite.any() else None


def compare_outputs(
    baseline: Sequence[SampleRecord],
    candidate: Sequence[SampleRecord],
    contract: ComparisonContract,
) -> dict[str, Any]:
    """Compare two validated bundles and return the strict report."""
    validate_contract(baseline, candidate, contract)

    available = [domain for domain in DOMAINS if all(domain in r.arrays for r in baseline)]
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        if domain not in available:
            domains[domain] = None
            continue
        base = np.stack([np.asarray(r.arrays[domain]) for r in baseline])
        cand = np.stack([np.asarray(r.arrays[domain]) for r in candidate])
        domains[domain] = _domain_block(base, cand, contract)

    metric_domain = next(
        (domain for domain in ("policy_actions", "env_step_actions", "normalized_model") if domain in available),
        available[0],
    )

    samples = []
    for base, cand in zip(baseline, candidate):
        finite = all(np.isfinite(np.asarray(v)).all() for r in (base, cand) for v in r.arrays.values())
        samples.append(
            {
                "sample_id": base.sample_id,
                "status": "ok" if finite else "non_finite",
                "mae_executed_prefix": _mae_prefix(
                    np.asarray(base.arrays[metric_domain]), np.asarray(cand.arrays[metric_domain]), contract
                ),
            }
        )

    non_finite_total = sum(
        block["slices"][name]["non_finite"]
        for block in domains.values()
        if block
        for name in SLICES
    )
    status = "failed" if non_finite_total or any(s["status"] != "ok" for s in samples) else "ok"

    return {
        "schema": SCHEMA_VERSION,
        "status": status,
        "contract": contract.to_dict(),
        "metric_domain": metric_domain,
        "domains": domains,
        "gripper": _gripper_block(baseline, candidate, contract),
        "samples": samples,
    }


def evaluate_gates(
    report: Mapping[str, Any],
    *,
    max_prefix_mae: float | None = None,
    fail_on_gripper_flip: bool = False,
    domain: str | None = None,
) -> dict[str, Any]:
    """CI gate. A report that is not ``ok`` fails closed even if its metrics are finite."""
    reasons: list[str] = []
    if report.get("status", "failed") != "ok":
        reasons.append("report_not_ok")

    metric_domain = domain or report.get("metric_domain")
    block = (report.get("domains") or {}).get(metric_domain) or {}
    prefix = block.get("slices", {}).get("executed_prefix", {})
    mae = prefix.get("mae")
    if max_prefix_mae is not None:
        if not isinstance(mae, (int, float)) or not np.isfinite(mae):
            reasons.append("missing_prefix_mae")
        elif float(mae) > float(max_prefix_mae):
            reasons.append("prefix_mae_exceeds")

    if fail_on_gripper_flip:
        gripper = (report.get("gripper") or {}).get("slices", {}).get("executed_prefix", {})
        rate = gripper.get("disagreement_rate")
        if not isinstance(rate, (int, float)) or not np.isfinite(rate):
            reasons.append("missing_gripper_rate")
        elif float(rate) > 0.0:
            reasons.append("gripper_flip")

    return {
        "status": "failed" if reasons else "passed",
        "reasons": reasons,
        "domain": metric_domain,
        "max_prefix_mae": max_prefix_mae,
        "fail_on_gripper_flip": fail_on_gripper_flip,
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Strict JSON: unmeasured metrics are ``null`` and NaN can never be serialised."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def save_bundle(path: str | Path, records: Sequence[SampleRecord]) -> None:
    """One ``.npz`` per precision: ``<sample_id>|<domain>`` plus a JSON ``meta``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    meta = {"schema": SCHEMA_VERSION, "records": []}
    for record in records:
        meta["records"].append(
            {
                "sample_id": record.sample_id,
                "precision": record.precision,
                "hashes": dict(record.hashes),
                "domains": sorted(record.arrays),
            }
        )
        for domain, array in record.arrays.items():
            payload[f"{record.sample_id}|{domain}"] = np.ascontiguousarray(array)
    payload["meta"] = np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False))
    np.savez_compressed(target.with_name(target.name + ".tmp.npz"), **payload)
    os.replace(target.with_name(target.name + ".tmp.npz"), target)


def load_bundle(path: str | Path) -> list[SampleRecord]:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        records = []
        for item in meta["records"]:
            arrays = {
                domain: np.array(data[f"{item['sample_id']}|{domain}"]) for domain in item["domains"]
            }
            records.append(
                SampleRecord(
                    sample_id=item["sample_id"],
                    precision=item["precision"],
                    arrays=arrays,
                    hashes=item.get("hashes") or {},
                )
            )
    return records


def environment_manifest(files: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Identity of the interpreter and of the files a comparison trusts."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "files": {str(Path(path)): sha256_file(path) for path in files if Path(path).is_file()},
    }

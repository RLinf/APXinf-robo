"""CPU-only contract tests for the cross-precision comparator.

Everything here runs in a couple of seconds with NumPy alone: the comparator's whole
job is arithmetic over two aligned arrays, so the interesting cases are shape/identity
mistakes, non-finite values, the executed-prefix versus suffix split, padding
isolation, and the sign-class boundaries of the gripper command -- not model weights.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apxinf_robo.precision_compare import (
    ComparisonContract,
    ContractError,
    SampleRecord,
    compare_outputs,
    evaluate_gates,
    load_bundle,
    save_bundle,
    sign_class,
    validate_contract,
    write_report,
)

REPO = Path(__file__).resolve().parents[1]


def contract(**overrides) -> ComparisonContract:
    values = {"horizon": 4, "executed_prefix": 2, "full_action_dim": 10, "valid_action_dim": 7, "gripper_dim": 6}
    values.update(overrides)
    return ComparisonContract(**values)


def record(sample_id: str, precision: str, actions: np.ndarray, **hashes) -> SampleRecord:
    return SampleRecord(sample_id=sample_id, precision=precision, arrays={"policy_actions": actions}, hashes=hashes)


def pair(actions: np.ndarray, candidate: np.ndarray | None = None, **kwargs):
    base = [record("s0", "bf16", actions, observation="o", noise="n")]
    cand = [record("s0", "int8_dynamic", actions if candidate is None else candidate, observation="o", noise="n")]
    return compare_outputs(base, cand, contract(**kwargs))


def test_identical_actions_pass_with_zero_error() -> None:
    actions = np.arange(4 * 7, dtype=np.float32).reshape(4, 7) * 1e-3
    report = pair(actions)
    assert report["status"] == "ok"
    assert report["metric_domain"] == "policy_actions"
    assert report["domains"]["policy_actions"]["slices"]["executed_prefix"]["mae"] == 0.0
    assert evaluate_gates(report, max_prefix_mae=0.0)["status"] == "passed"


def test_prefix_and_suffix_are_reported_separately() -> None:
    actions = np.zeros((4, 7), dtype=np.float32)
    candidate = actions.copy()
    candidate[2:, :] = 0.5
    report = pair(actions, candidate)
    slices = report["domains"]["policy_actions"]["slices"]
    assert slices["executed_prefix"]["mae"] == 0.0
    assert slices["unused_suffix"]["mae"] == pytest.approx(0.5)
    assert slices["full_horizon"]["mae"] == pytest.approx(0.25)


def test_control_groups_split_translation_rotation_gripper() -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = base.copy()
    cand[:, 0:3] = 1.0
    cand[:, 3:6] = 0.5
    cand[:, 6] = 0.25
    report = pair(base, cand)
    groups = report["domains"]["policy_actions"]["slices"]["executed_prefix"]["groups"]
    assert groups["translation_command"]["rows"] == [0, 3]
    assert groups["translation_command"]["mae"] == pytest.approx(1.0)
    assert groups["rotation_command"]["mae"] == pytest.approx(0.5)
    assert groups["gripper_command"]["mae"] == pytest.approx(0.25)
    assert set(groups) == {"translation_command", "rotation_command", "gripper_command"}


def test_real_horizon_and_padding_width() -> None:
    """The shapes a real PI0.5 chunk uses: H=50, 32-wide model space, 7 valid dims."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((50, 32)).astype(np.float32) * 0.01
    cand = base.copy()
    cand[0:5, 0:3] += 0.2  # executed prefix, translation
    cand[:, 7:] += 5.0  # padding only
    report = compare_outputs(
        [SampleRecord("s0", "bf16", {"normalized_model": base})],
        [SampleRecord("s0", "int8_dynamic", {"normalized_model": cand})],
        ComparisonContract(horizon=50, executed_prefix=5, full_action_dim=32, valid_action_dim=7),
    )
    block = report["domains"]["normalized_model"]
    assert block["slices"]["executed_prefix"]["rows"] == [0, 5]
    assert block["slices"]["unused_suffix"]["rows"] == [5, 50]
    assert block["slices"]["full_horizon"]["rows"] == [0, 50]
    assert block["slices"]["executed_prefix"]["groups"]["translation_command"]["mae"] == pytest.approx(
        0.2, rel=1e-6
    )
    assert block["slices"]["unused_suffix"]["groups"]["translation_command"]["mae"] == pytest.approx(0.0)
    assert block["padding"]["count"] == 25
    assert block["padding"]["max_abs"] == pytest.approx(5.0)
    assert len(block["slices"]["full_horizon"]["per_dim_mae"]) == 7


def test_padding_differences_never_leak_into_valid_metrics() -> None:
    base = np.zeros((4, 10), dtype=np.float32)
    cand = base.copy()
    cand[:, 7:] = 9.0
    report = compare_outputs(
        [SampleRecord("s0", "bf16", {"normalized_model": base})],
        [SampleRecord("s0", "int8_dynamic", {"normalized_model": cand})],
        contract(),
    )
    block = report["domains"]["normalized_model"]
    assert block["slices"]["executed_prefix"]["mae"] == 0.0
    assert block["padding"] == {"dims": [7, 10], "count": 3, "max_abs": 9.0, "non_finite": 0}
    assert report["domains"]["policy_actions"] is None


def test_missing_domain_is_null_not_zero() -> None:
    report = pair(np.zeros((4, 7), dtype=np.float32))
    assert report["domains"]["normalized_model"] is None
    assert report["domains"]["env_step_actions"] is None


def test_non_finite_candidate_fails_closed_even_with_finite_metrics() -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = base.copy()
    cand[3, 0] = np.nan
    report = pair(base, cand)
    assert report["status"] == "failed"
    assert report["samples"][0]["status"] == "non_finite"
    gate = evaluate_gates(report, max_prefix_mae=1.0)
    assert gate["status"] == "failed"
    assert gate["reasons"] == ["report_not_ok"]


def test_unmeasured_metrics_serialise_as_null_not_nan(tmp_path: Path) -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = np.full((4, 7), np.nan, dtype=np.float32)
    report = pair(base, cand)
    prefix = report["domains"]["policy_actions"]["slices"]["executed_prefix"]
    assert prefix["n"] == 0
    assert prefix["non_finite"] == 2 * 7
    assert prefix["mae"] is None
    write_report(report, tmp_path / "report.json")
    loaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert loaded["domains"]["policy_actions"]["slices"]["executed_prefix"]["mae"] is None
    with pytest.raises(ValueError):
        write_report({"bad": float("nan")}, tmp_path / "bad.json")


@pytest.mark.parametrize(
    "base_value,cand_value,disagrees",
    [
        (-0.01, 0.01, True),
        (0.01, -0.01, True),
        (0.0, -0.0, False),
        (-0.0, 0.0, False),
        (0.0, 1e-9, True),
        (0.5, 0.25, False),
    ],
)
def test_gripper_sign_classes(base_value: float, cand_value: float, disagrees: bool) -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = base.copy()
    base[:, 6] = base_value
    cand[:, 6] = cand_value
    report = pair(base, cand)
    rate = report["gripper"]["slices"]["executed_prefix"]["disagreement_rate"]
    assert rate == (1.0 if disagrees else 0.0)


def test_gripper_classes_are_neutral_labels() -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = base.copy()
    base[:, 6] = [-1.0, 0.0, 1.0, 0.0]
    cand[:, 6] = [1.0, -0.0, 1.0, -1.0]
    report = pair(base, cand)
    counts = report["gripper"]["slices"]["full_horizon"]["class_counts"]
    assert counts["baseline"] == {"neg": 1, "zero": 2, "pos": 1}
    assert counts["candidate"] == {"neg": 1, "zero": 1, "pos": 2}
    assert set(counts["baseline"]) == {"neg", "zero", "pos"}


def test_sign_class_handles_negative_zero() -> None:
    values = np.array([-0.0, 0.0, -1e-9, 1e-9], dtype=np.float64)
    assert sign_class(values).tolist() == [0, 0, -1, 1]


def test_gripper_gate_flags_a_single_flip() -> None:
    base = np.zeros((4, 7), dtype=np.float32)
    cand = base.copy()
    cand[0, 6] = 0.2
    report = pair(base, cand)
    assert evaluate_gates(report, fail_on_gripper_flip=True)["status"] == "failed"
    assert evaluate_gates(report, fail_on_gripper_flip=True)["reasons"] == ["gripper_flip"]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda records: records[0].__setattr__("sample_id", "other"), "sample order"),
        (lambda records: records.append(records[0]), "repeats sample_id"),
        (lambda records: records[0].__setattr__("precision", "fp8_static"), "labelled"),
        (lambda records: records[0].hashes.__setitem__("noise", "different"), "input hashes differ"),
    ],
)
def test_contract_rejects_mismatches(mutate, message: str) -> None:
    base = [record("s0", "bf16", np.zeros((4, 7), dtype=np.float32), observation="o", noise="n")]
    cand = [record("s0", "int8_dynamic", np.zeros((4, 7), dtype=np.float32), observation="o", noise="n")]
    mutate(cand)
    with pytest.raises(ContractError, match=message):
        validate_contract(base, cand, contract())


def test_contract_rejects_wrong_shape_and_integer_input() -> None:
    base = [record("s0", "bf16", np.zeros((4, 6), dtype=np.float32))]
    with pytest.raises(ContractError, match="shape"):
        validate_contract(base, [record("s0", "int8_dynamic", np.zeros((4, 6), dtype=np.float32))], contract())
    ints = [record("s0", "bf16", np.zeros((4, 7), dtype=np.int32))]
    with pytest.raises(ContractError, match="dtype"):
        validate_contract(ints, ints, contract())


def test_backend_labels_are_free_form() -> None:
    base = [record("s0", "fp8_static", np.zeros((4, 7), dtype=np.float32))]
    cand = [record("s0", "w4a16", np.zeros((4, 7), dtype=np.float32))]
    report = compare_outputs(base, cand, contract(baseline_precision="fp8_static", candidate_precision="w4a16"))
    assert report["contract"]["baseline_precision"] == "fp8_static"
    assert report["contract"]["candidate_precision"] == "w4a16"


def test_bundle_round_trip(tmp_path: Path) -> None:
    actions = np.arange(4 * 7, dtype=np.float32).reshape(4, 7)
    records = [record("a", "bf16", actions, observation="o", noise="n")]
    path = tmp_path / "bundle.npz"
    save_bundle(path, records)
    restored = load_bundle(path)
    assert [r.sample_id for r in restored] == ["a"]
    assert restored[0].hashes == {"observation": "o", "noise": "n"}
    np.testing.assert_array_equal(restored[0].arrays["policy_actions"], actions)


def _cli():
    spec = importlib.util.spec_from_file_location("compare_cli", REPO / "scripts" / "compare_pi05_precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_compare_exit_codes(tmp_path: Path) -> None:
    cli = _cli()
    actions = np.zeros((4, 7), dtype=np.float32)
    save_bundle(tmp_path / "base.npz", [record("s0", "bf16", actions, observation="o", noise="n")])
    save_bundle(tmp_path / "cand.npz", [record("s0", "int8_dynamic", actions, observation="o", noise="n")])
    args = ["--horizon", "4", "--executed-prefix", "2", "--full-action-dim", "10"]
    assert cli.main(["compare", "--baseline", str(tmp_path / "base.npz"), "--candidate", str(tmp_path / "cand.npz"), "--out", str(tmp_path / "ok"), *args]) == 0

    worse = actions.copy()
    worse[:2] = 0.5
    save_bundle(tmp_path / "worse.npz", [record("s0", "int8_dynamic", worse, observation="o", noise="n")])
    assert cli.main(["compare", "--baseline", str(tmp_path / "base.npz"), "--candidate", str(tmp_path / "worse.npz"), "--out", str(tmp_path / "gate"), "--max-prefix-mae", "0.1", *args]) == 3
    assert cli.main(["compare", "--baseline", str(tmp_path / "base.npz"), "--candidate", str(tmp_path / "missing.npz"), "--out", str(tmp_path / "bad"), *args]) == 1


def test_comparator_never_imports_the_engine() -> None:
    script = (
        "import sys, apxinf_robo.precision_compare as m;"
        "assert 'apxinf' not in sys.modules and 'apxinf_py' not in sys.modules;"
        "print(m.SCHEMA_VERSION)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, cwd=REPO, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "pi05_cross_precision_v1"

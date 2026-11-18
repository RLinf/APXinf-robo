"""Pin the checks that run on the deployment, and on what crosses the runtime boundary.

A mismatch here is silent at run time: the runtime accepts a seven-value state or a
256-pixel image, and the rollout only reports a lower success rate. These tests pin
the rejections, and that the contract file itself stays consistent.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from apx_drobotics import contract as contract_module
from apx_drobotics.testing import fake_deployment


@pytest.fixture
def contract():
    return contract_module.load_contract()


def _entry(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _model_input(**change):
    values = {
        "images": [np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8)],
        "state": np.zeros(8, np.float32),
        "prompt": "pick up the alphabet soup",
    }
    return {**values, **change}


def test_contract_is_internally_consistent(contract):
    """The sections restate a few facts; they must not drift apart."""
    metadata = contract["metadata"]
    slots = contract["camera"]["slots"]
    expect = contract["runtime_config"]["expect"]
    assert metadata["sdk_version"] == contract["sdk"]["version"]
    assert metadata["precision"] == contract["model"]["precision"]
    assert metadata["num_views"] == contract_module.unmasked_views(contract)
    assert metadata["image_shape"] == contract["camera"]["model_shape"]
    assert metadata["state_dim"] == contract["state"]["dim"] == len(contract["state"]["components"])
    assert metadata["action_horizon"] == contract["action"]["horizon"]
    assert metadata["action_dim"] == contract["action"]["dim"] == len(contract["action"]["components"])
    assert metadata["discrete_state"] is False
    assert contract["state"]["used_by_model"] is False
    assert "processing.inject_state" in contract["runtime_config"]["absent"]
    assert expect["interface.image_mask"] == [not s["masked"] for s in slots]
    assert expect["interface.raw_state_size"] == contract["state"]["dim"]
    assert expect["interface.raw_action_size"] == contract["action"]["dim"]
    assert expect["interface.action_horizon"] == contract["action"]["horizon"]
    for name in (expect[f"model.model_runner.{part}"] for part in ("visual", "lm", "action")):
        assert name in contract["model"]["files"]


def test_wire_keys_are_not_part_of_the_contract(contract):
    """What a client calls its cameras is the caller's, as with ApxInf."""
    text = json.dumps(contract)
    assert "observation/" not in text
    assert "wire_key" not in text


def test_contract_hash_ignores_key_order(contract):
    reordered = json.loads(json.dumps(contract, sort_keys=True))
    assert contract_module.contract_sha256(reordered) == contract_module.contract_sha256(contract)


def test_contract_hash_changes_with_content(contract):
    changed = copy.deepcopy(contract)
    changed["action"]["horizon"] = 50
    assert contract_module.contract_sha256(changed) != contract_module.contract_sha256(contract)


def test_metadata_identifies_the_contract(contract):
    metadata = contract_module.model_metadata(contract)
    assert metadata["contract_id"] == contract["contract_id"]
    assert metadata["contract_version"] == contract["contract_version"]
    assert metadata["contract_sha256"] == contract_module.contract_sha256(contract)
    assert metadata["model_type"] == "pi05_drobotics_w8"


# --- deployment ---------------------------------------------------------------


def test_matching_deployment_passes(deployment):
    assert contract_module.check_deployment(deployment.contract, deployment.sdk_root) == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda c: c["processing"].update(inject_state=True), "processing.inject_state"),
        (lambda c: c["interface"].update(raw_state_size=7), "interface.raw_state_size"),
        (lambda c: c["interface"].update(action_horizon=50), "interface.action_horizon"),
        (lambda c: c["interface"].update(image_mask=[True, True, True]), "interface.image_mask"),
        (lambda c: c["processing"].update(use_absolute_action=False), "use_absolute_action"),
        (lambda c: c["model"]["model_runner"].update(action="other.hbm"), "model_runner.action"),
        (lambda c: c["processing"].pop("use_quantiles_norm"), "use_quantiles_norm"),
    ],
)
def test_changed_runtime_config_is_rejected(deployment, mutate, needle):
    local = copy.deepcopy(deployment.contract)
    path = deployment.sdk_root / local["runtime_config"]["path"]
    changed = json.loads(path.read_text())
    mutate(changed)
    data = json.dumps(changed).encode()
    path.write_bytes(data)
    # Match the recorded hash so only the semantic check can catch the change.
    local["runtime_config"].update(_entry(data))
    problems = contract_module.check_deployment(local, deployment.sdk_root)
    assert any(needle in problem for problem in problems), problems


def test_model_file_with_same_size_but_other_content_is_rejected(deployment):
    local = deployment.contract
    name = local["runtime_config"]["expect"]["model.model_runner.action"]
    path = deployment.sdk_root / local["model"]["directory"] / name
    path.write_bytes(b"x" * path.stat().st_size)
    problems = contract_module.check_deployment(local, deployment.sdk_root)
    assert any("sha256" in problem and name in problem for problem in problems), problems


def test_skipping_hashes_compares_sizes_only(deployment):
    local = deployment.contract
    name = local["runtime_config"]["expect"]["model.model_runner.action"]
    path = deployment.sdk_root / local["model"]["directory"] / name
    path.write_bytes(b"x" * path.stat().st_size)
    assert contract_module.check_deployment(local, deployment.sdk_root, verify_hashes=False) == []


def test_missing_model_file_is_rejected(deployment):
    local = deployment.contract
    name = local["runtime_config"]["expect"]["model.model_runner.lm"]
    (deployment.sdk_root / local["model"]["directory"] / name).unlink()
    problems = contract_module.check_deployment(local, deployment.sdk_root)
    assert any("missing" in problem and name in problem for problem in problems), problems


def test_norm_stats_of_another_width_are_rejected(deployment):
    local = copy.deepcopy(deployment.contract)
    path = deployment.sdk_root / local["model"]["directory"] / "norm_stats.json"
    data = json.dumps({"norm_stats": {
        "state": {"q01": [0.0] * 14, "q99": [1.0] * 14},
        "actions": {"q01": [0.0] * 14, "q99": [1.0] * 14},
    }}).encode()
    path.write_bytes(data)
    local["model"]["files"]["norm_stats.json"] = _entry(data)
    problems = contract_module.check_deployment(local, deployment.sdk_root)
    assert any("norm stats state.q01" in problem for problem in problems), problems
    assert any("norm stats actions.q99" in problem for problem in problems), problems


# --- what enters the runtime ---------------------------------------------------


def test_good_model_input_passes(contract):
    assert contract_module.check_model_input(contract, **_model_input()) == []


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"images": [np.zeros((224, 224, 3), np.uint8)]}, "1 views"),
        ({"images": [np.zeros((256, 256, 3), np.uint8)] * 2}, "image 0: shape"),
        ({"images": [np.zeros((3, 224, 224), np.uint8)] * 2}, "image 1: shape"),
        ({"images": [np.zeros((224, 224, 3), np.float32)] * 2}, "image 0: dtype"),
        ({"state": np.zeros(7, np.float32)}, "state: shape"),
        ({"state": np.zeros(8, np.float64)}, "state: dtype"),
        ({"state": np.full(8, np.nan, np.float32)}, "non-finite"),
        ({"prompt": ""}, "prompt"),
        ({"prompt": 7}, "prompt"),
    ],
)
def test_bad_model_input_is_rejected(contract, change, needle):
    problems = contract_module.check_model_input(contract, **_model_input(**change))
    assert any(needle in problem for problem in problems), problems


# --- what leaves the runtime ---------------------------------------------------


def test_good_actions_pass(contract):
    assert contract_module.check_actions(contract, np.zeros((10, 7))) == []


@pytest.mark.parametrize("shape", [(1, 10, 7), (50, 14), (10, 32), (70,)])
def test_actions_of_another_shape_are_rejected(contract, shape):
    assert contract_module.check_actions(contract, np.zeros(shape))


def test_nonfinite_actions_are_rejected(contract):
    actions = np.zeros((10, 7))
    actions[3, 2] = np.inf
    assert contract_module.check_actions(contract, actions)


# --- command line --------------------------------------------------------------


def test_cli_exit_codes(tmp_path, capsys):
    fake = fake_deployment(tmp_path)
    command = ["check-deployment", "--contract", str(fake.contract_path),
               "--sdk-root", str(fake.sdk_root)]
    assert contract_module.main(command) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    next(iter((fake.sdk_root / "lib").iterdir())).unlink()
    assert contract_module.main(command) == 1
    assert "CONTRACT VIOLATION" in capsys.readouterr().err


def test_cli_prints_the_metadata(contract, capsys):
    assert contract_module.main(["print-metadata"]) == 0
    assert json.loads(capsys.readouterr().out) == contract_module.model_metadata(contract)

"""Pin ``DroboticsPolicy`` against a fake worker: the Policy shape, and how it fails."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from apx_drobotics import DeploymentError, RequestError, RuntimeFailure, DroboticsPolicy, StartupFailure
from apx_drobotics.testing import EXPECTED_ACTIONS, fake_deployment

from conftest import KEYS, observation


def test_result_has_the_policy_shape(make_policy):
    policy, _ = make_policy()
    result = policy.infer(observation())
    assert result["actions"].dtype == np.float32
    assert np.array_equal(result["actions"], EXPECTED_ACTIONS.astype(np.float32))
    assert result["timing"]["model_ms"] == 6.0
    assert result["timing"]["total_ms"] > 0
    assert result["timing"]["sdk_total_ms"] == 7.5
    assert policy.action_horizon == 10
    assert policy.action_dim == 7


def test_request_reaches_the_runtime_as_it_expects(make_policy):
    policy, fake = make_policy()
    raw = observation(224)
    raw["task"] = " pick_up the soup\n"
    policy.infer(raw)
    header, payload = fake.request(0)
    assert header["op"] == "infer"
    assert header["prompt"] == "pick up the soup"
    assert header["state"] == raw["proprio"].astype(np.float64).tolist()
    assert header["images"] == [{"width": 224, "height": 224}] * 2
    assert payload == b"".join(
        np.transpose(raw[key], (2, 0, 1)).tobytes() for key in KEYS["image_keys"]
    )


def test_frames_of_another_size_are_resized_before_the_runtime(make_policy):
    policy, fake = make_policy()
    policy.infer(observation(256))
    header, payload = fake.request(0)
    assert header["images"] == [{"width": 224, "height": 224}] * 2
    assert len(payload) == 2 * 3 * 224 * 224


def test_metadata_describes_model_and_caller(make_policy):
    policy, fake = make_policy()
    metadata = policy.metadata
    assert metadata["engine"] == "apx-drobotics"
    assert metadata["model_type"] == "pi05_drobotics_w8"
    assert metadata["precision"] == "w8"
    assert metadata["oellm_version"] == "2.0.0"
    assert metadata["image_keys"] == list(KEYS["image_keys"])
    assert metadata["state_key"] == "proprio"
    assert metadata["prompt_key"] == "task"
    assert metadata["contract_id"] == fake.contract["contract_id"]
    json.dumps(metadata)


def test_caller_metadata_is_kept(make_policy):
    policy, _ = make_policy(metadata={"robot": "franka_libero", "state_dim": 8})
    assert policy.metadata["robot"] == "franka_libero"


def test_noise_cannot_be_supplied(make_policy):
    policy, _ = make_policy()
    with pytest.raises(ValueError, match="own noise"):
        policy.infer(observation(), noise=np.zeros((10, 32)))


def test_bad_observation_never_reaches_the_runtime_and_leaves_the_policy_usable(make_policy):
    policy, fake = make_policy()
    with pytest.raises(RequestError):
        policy.infer({**observation(), "proprio": np.zeros(7, np.float32)})
    assert fake.request_count() == 0
    assert policy.infer(observation())["actions"].shape == (10, 7)


@pytest.mark.parametrize(
    "mode, options, needle",
    [
        ("crash", {}, "worker exited"),
        ("hang", {"infer_deadline": 0.5}, "did not answer within 0.5 s"),
        ("error", {}, "runtime error: boom"),
        ("nan", {}, "non-finite"),
        ("bad_shape", {}, "action shape"),
    ],
)
def test_runtime_failure_is_reported_and_final(make_policy, mode, options, needle):
    policy, fake = make_policy(mode, **options)
    with pytest.raises(RuntimeFailure, match=needle):
        policy.infer(observation())
    time.sleep(0.05)
    count = fake.request_count()
    with pytest.raises(RuntimeFailure, match="already failed"):
        policy.infer(observation())
    assert fake.request_count() == count


def test_close_is_idempotent_and_stops_the_worker(make_policy):
    policy, _ = make_policy()
    process = policy._worker._process
    policy.close()
    policy.close()
    assert process.poll() == 0


def test_close_does_not_wait_long_for_a_stuck_worker(make_policy):
    policy, _ = make_policy("hang", infer_deadline=0.2)
    with pytest.raises(RuntimeFailure):
        policy.infer(observation())
    started = time.monotonic()
    policy.close()
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "options, needle",
    [
        ({"image_keys": ("cam/base",)}, "need 2 distinct"),
        ({"image_keys": ("cam/base", "cam/base")}, "need 2 distinct"),
        ({"action_dim": 14}, "action_dim=14"),
        ({"state_dim": 16}, "state_dim=16"),
        ({"precision": "bf16"}, "precision='bf16'"),
        ({"discrete_state": True}, "does not read state"),
        ({"metadata": {"precision": "bf16"}}, "metadata precision"),
        ({"metadata": {"image_keys": ["a", "b"]}}, "metadata image_keys"),
    ],
)
def test_a_caller_that_does_not_fit_is_refused_before_anything_starts(deployment, options, needle):
    with pytest.raises(ValueError, match=needle):
        DroboticsPolicy.from_deployment(
            deployment.sdk_root, **{**KEYS, **options}, **deployment.options
        )


@pytest.mark.parametrize("options", [{"action_dim": 7}, {"state_dim": 8}, {"precision": "w8"},
                                     {"discrete_state": False}, {"metadata": {"state_dim": None}}])
def test_matching_expectations_are_accepted(make_policy, options):
    make_policy(**options)


def test_changed_deployment_is_refused_before_the_worker_starts(tmp_path):
    fake = fake_deployment(tmp_path)
    config = fake.sdk_root / fake.contract["runtime_config"]["path"]
    changed = json.loads(config.read_text())
    changed["processing"]["inject_state"] = True
    config.write_text(json.dumps(changed))
    with pytest.raises(DeploymentError, match="processing.inject_state") as error:
        DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options)
    assert error.value.exit_code == 2
    assert fake.request_count() == 0


def test_missing_worker_is_a_deployment_problem(tmp_path):
    fake = fake_deployment(tmp_path)
    options = {**fake.options, "worker": str(tmp_path / "absent")}
    with pytest.raises(DeploymentError, match="worker executable is missing"):
        DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **options)


def test_runtime_that_fails_to_load_is_a_startup_failure(tmp_path):
    fake = fake_deployment(tmp_path, "init_fail")
    with pytest.raises(StartupFailure, match="init_failed") as error:
        DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options)
    assert error.value.exit_code == 3


def test_runtime_that_never_gets_ready_is_a_startup_failure(tmp_path):
    fake = fake_deployment(tmp_path, "silent")
    with pytest.raises(StartupFailure, match="not ready within 0.3 s"):
        DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options, ready_timeout=0.3)


def test_runtime_logs_are_forwarded_at_debug(make_policy, caplog):
    """SDK chatter is noisy on the board, so it is DEBUG; problem lines stay visible."""
    with caplog.at_level("DEBUG", logger="apx_drobotics.runtime"):
        make_policy()
        deadline = time.monotonic() + 5
        while "fake runtime log" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
    assert "worker: fake runtime log on stdout" in caplog.text
    record = next(r for r in caplog.records if "fake runtime log" in r.getMessage())
    assert record.levelname == "DEBUG"


def test_hashes_are_not_recomputed_by_default_but_can_be(make_policy, tmp_path):
    """Sizes are always checked; a same-size edit needs verify_hashes=True to be found."""
    policy, fake = make_policy()
    assert policy.metadata["hashes_verified"] is False
    policy.close()
    victim = next(fake.sdk_root.rglob("config.json"), None) or next(
        path for path in (fake.sdk_root / fake.contract["model"]["directory"]).iterdir()
        if path.name != "norm_stats.json"
    )
    victim.write_bytes(bytes(len(victim.read_bytes())))  # same size, other content
    DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options).close()
    with pytest.raises(DeploymentError, match="sha256"):
        DroboticsPolicy.from_deployment(fake.sdk_root, **KEYS, **fake.options, verify_hashes=True)


def test_metadata_identifies_the_code_that_ran(make_policy):
    import hashlib
    import apx_drobotics

    policy, fake = make_policy()
    assert policy.metadata["apx_drobotics_version"] == apx_drobotics.__version__
    assert policy.metadata["worker_sha256"] == hashlib.sha256(fake.worker.read_bytes()).hexdigest()


def _with_keys(tmp_path, image_keys):
    fake = fake_deployment(tmp_path)
    options = {**KEYS, "image_keys": image_keys}
    return DroboticsPolicy.from_deployment(fake.sdk_root, **options, **fake.options)


def test_wrist_camera_in_the_base_slot_is_refused(tmp_path):
    with pytest.raises(ValueError, match="names a wrist camera but fills the base slot"):
        _with_keys(tmp_path, ("cam/wrist", "cam/base"))


def test_wrist_slot_without_a_wrist_key_only_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        policy = _with_keys(tmp_path, ("cam/a", "cam/b"))
    assert "does not mention a wrist camera" in caplog.text
    policy.close()


def test_contract_without_required_metadata_is_a_deployment_error(tmp_path):
    fake = fake_deployment(tmp_path)
    broken = json.loads(fake.contract_path.read_text())
    del broken["metadata"]["action_dim"]
    with pytest.raises(DeploymentError, match="missing 'action_dim'"):
        DroboticsPolicy.from_deployment(
            fake.sdk_root, **KEYS, contract=broken, worker=str(fake.worker)
        )

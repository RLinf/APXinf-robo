"""``eval-libero --dump-observations``: an opt-in seam, not a second pipeline.

The rollout loop is driven by a stub environment and a stub backend, so these run
offline with no LIBERO, MuJoCo or engine. What is held still: the dump is off unless
asked for, the stride and caps are honoured, the frames use the same NPZ dialect as
``capture-libero`` (so one reader serves both), and the manifest hash is the file's
own bytes.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import ClassVar

import numpy as np

from apxinf_robo.cli import eval_libero
from apxinf_robo.precision_compare import sha256_array


class StubEnv:
    def __init__(self, frames: int = 8) -> None:
        self.frames = frames
        self.steps = 0

    def _observation(self):
        return {
            "agentview_image": np.full((6, 8, 3), self.steps % 255, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
        }

    def reset(self):
        self.steps = 0
        return self._observation()

    def set_init_state(self, state):
        return self._observation()

    def step(self, action):
        self.steps += 1
        return self._observation(), 0.0, self.steps >= self.frames, {}

    def close(self):
        pass


class StubBackend:
    model_type = "pi05"
    metadata: ClassVar[dict] = {"action_horizon": 2, "action_dim": 3}

    def __init__(self) -> None:
        self.noises = []

    def infer(self, base, wrist, state, prompt, noise=None):
        self.noises.append(None if noise is None else np.array(noise))
        return np.zeros((2, 7), dtype=np.float32), None, {
            "model_seconds": 0.0,
            "server_processor_seconds": 0.0,
        }


def _episode(tmp_path, recorder, *, max_steps: int = 8, backend=None, task_id: int = 0):
    backend = backend or StubBackend()
    # The settle window steps the env before the rollout starts, so the stub's done
    # counter has to account for it or the episode ends on its first real step.
    record = eval_libero.run_episode(
        StubEnv(frames=eval_libero.WAIT_STEPS + max_steps + 2),
        np.zeros(7, dtype=np.float32),
        "libero_10",
        task_id,
        0,
        "pick up the bowl",
        backend,
        "in_process_api",
        0,
        False,
        1.0,
        1,
        -1.0,
        max_steps,
        None,
        recorder,
    )
    return record, backend


def test_no_dump_by_default(tmp_path: pathlib.Path) -> None:
    dump = tmp_path / "dumps"
    record, backend = _episode(tmp_path, None, max_steps=4)
    assert record["status"] == "completed"
    assert not dump.exists()
    assert all(noise is None for noise in backend.noises)


def test_stride_and_per_episode_cap(tmp_path: pathlib.Path) -> None:
    recorder = eval_libero.ObservationRecorder(
        tmp_path / "dumps", skip=1, every=2, max_per_episode=2, max_total=10
    )
    record, _ = _episode(tmp_path, recorder, max_steps=8)
    recorder.close()
    names = sorted(path.name for path in (tmp_path / "dumps").glob("*.npz"))
    # replans 0..7; skip=1 every=2 allows 1, 3, 5, 7; the per-episode cap keeps 1 and 3.
    assert names == [
        "mid-libero_10-task0-trial0-replan0001.npz",
        "mid-libero_10-task0-trial0-replan0003.npz",
    ], names
    assert record["action_steps"] == 8


def test_frames_use_the_capture_dialect_and_a_byte_hash(tmp_path: pathlib.Path) -> None:
    recorder = eval_libero.ObservationRecorder(tmp_path / "dumps", skip=0, every=1, max_total=2)
    _episode(tmp_path, recorder, max_steps=3)
    recorder.close()

    entries = [json.loads(line) for line in (tmp_path / "dumps" / "manifest.jsonl").read_text().splitlines()]
    assert len(entries) == 2
    entry = entries[0]
    assert entry["noise_sha256"] is None
    payload = np.load(tmp_path / "dumps" / entry["path"])
    assert set(payload.files) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert payload["observation/image"].dtype == np.uint8
    assert payload["observation/state"].dtype == np.float32
    assert payload["observation/state"].shape == (7,)
    assert str(payload["prompt"]) == "pick up the bowl"
    assert entry["sha256"] == hashlib.sha256((tmp_path / "dumps" / entry["path"]).read_bytes()).hexdigest()
    assert (entry["trial_id"], entry["replan"], entry["action_step"]) == (0, 0, 0)


def test_noise_bank_drives_the_chunk_and_is_saved(tmp_path: pathlib.Path) -> None:
    recorder = eval_libero.ObservationRecorder(
        tmp_path / "dumps", skip=0, every=1, max_total=1, noise_bank=tmp_path / "noise.npz", noise_shape=(2, 3)
    )
    _, backend = _episode(tmp_path, recorder, max_steps=2)
    recorder.close()

    assert backend.noises[0] is not None
    assert backend.noises[0].shape == (2, 3)
    saved = np.load(tmp_path / "noise.npz")
    sample_id = "mid-libero_10-task0-trial0-replan0000"
    assert sample_id in saved.files
    np.testing.assert_array_equal(saved[sample_id], backend.noises[0])
    entry = json.loads((tmp_path / "dumps" / "manifest.jsonl").read_text().splitlines()[0])
    # The manifest hash must be the comparator's canonical array hash, or a manifest
    # entry and a comparison report could never be cross-checked.
    assert entry["noise_sha256"] == sha256_array(backend.noises[0])
    # The recipe is (seed, task, trial, replan), so a second recorder reproduces it.
    again = eval_libero.ObservationRecorder(
        tmp_path / "dumps2", skip=0, every=1, max_total=1, noise_bank=tmp_path / "noise2.npz", noise_shape=(2, 3)
    )
    _, backend2 = _episode(tmp_path, again, max_steps=2)
    again.close()
    np.testing.assert_array_equal(backend2.noises[0], backend.noises[0])


def _stable(record: dict) -> dict:
    """The episode result minus wall-clock fields, which are not comparable across runs."""
    return {
        key: value
        for key, value in record.items()
        if "second" not in key and "utc" not in key
    }


def test_dumping_without_a_noise_bank_leaves_the_episode_untouched(tmp_path: pathlib.Path) -> None:
    plain, plain_backend = _episode(tmp_path, None, max_steps=4)
    recorder = eval_libero.ObservationRecorder(tmp_path / "dumps", skip=0, every=1, max_total=10)
    dumped, dumped_backend = _episode(tmp_path, recorder, max_steps=4)
    recorder.close()

    assert sorted(path.name for path in (tmp_path / "dumps").glob("*.npz")), "the recorder dumped nothing"
    assert _stable(plain) == _stable(dumped)
    assert plain_backend.noises == dumped_backend.noises
    assert all(noise is None for noise in dumped_backend.noises)


def test_resumed_run_keeps_existing_noise_entries(tmp_path: pathlib.Path) -> None:
    bank = tmp_path / "noise.npz"
    first = eval_libero.ObservationRecorder(
        tmp_path / "d1", skip=0, every=1, max_total=1, noise_bank=bank, noise_shape=(2, 3)
    )
    _episode(tmp_path, first, max_steps=1)
    first.close()
    before = set(np.load(bank).files)

    second = eval_libero.ObservationRecorder(
        tmp_path / "d2", skip=0, every=1, max_total=1, noise_bank=bank, noise_shape=(2, 3)
    )
    _episode(tmp_path, second, max_steps=1, task_id=1)
    second.close()
    after = set(np.load(bank).files)

    assert before, "the first run recorded no noise"
    assert before < after, f"resume dropped entries: {before} -> {after}"

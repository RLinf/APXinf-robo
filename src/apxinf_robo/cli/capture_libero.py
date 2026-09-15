#!/usr/bin/env python3
r"""Capture LIBERO observations as NPZ files for FP8 calibration.

Run from the APXinf-robo repository root. Requires LIBERO, MuJoCo, and the
Robo Python package; no model checkpoint is needed. Each NPZ contains one
observation using the selected robot preset's input field names.

    apxinf-robo capture-libero --robot franka_libero --suite libero_10 \
        --output-dir devlocal/fp8-calibration/observations

After capture, run the printed scripts/calibrate_pi05.py command from the Robo
root, replacing <ckpt> with your checkpoint path. If calibrating on another
machine, copy the NPZ directory and update --input-dir in that command.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..envs.libero import make_env, to_apxinf_observation

#: Default input field names, shared with LIBERO evaluation.
LIBERO_PRESET = "franka_libero"

ALL_SUITES = (
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
)

#: Steps of a neutral action after ``set_init_state``, matching ``eval-libero``'s
#: settle window: a freshly-set state is still visually transient, and calibration
#: wants the frames a policy would actually see.
WAIT_STEPS = 10

#: Open gripper, no motion -- the same dummy used while eval-libero settles.
NEUTRAL_ACTION = [0.0] * 6 + [-1.0]

#: Calibrator path relative to the Robo repository root.

CALIBRATE_SCRIPT = "scripts/calibrate_pi05.py"


def _progress(message: str) -> None:
    print(f"[capture] {message}", file=sys.stderr, flush=True)


def _calibrate_command(convention, output_dir) -> str:
    """Build a calibration command using the captured observation field names."""
    flags = [f"--image-key {key}" for key in convention.image_keys]
    flags.append(f"--state-key {convention.state_key}")
    flags.append(f"--prompt-key {convention.prompt_key}")
    return (
        f"python {CALIBRATE_SCRIPT} --model-dir <ckpt> "
        f"{' '.join(flags)} --input-dir {output_dir}"
    )


def load_suite(suite_name: str):
    try:
        from libero.libero import benchmark
    except ImportError as error:
        raise ImportError(
            "capturing native LIBERO observations requires the LIBERO and MuJoCo "
            "dependencies; install them with `pip install apxinf-robo[libero]` plus "
            "LIBERO itself, as described in README.md"
        ) from error
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    return benchmark_dict[suite_name]()


def task_stratified_indices(
    task_indices: Sequence[object], *, sample_count: int, seed: int
) -> list[int]:
    """Choose deterministic, balanced frame indices across every task group.

    Round-robin over tasks before taking a second frame from any of them, so a
    small ``--samples`` still covers the whole suite rather than over-sampling
    whichever task happens to come first.
    """
    if sample_count < 1:
        raise ValueError("calibration sample count must be positive")
    if sample_count > len(task_indices):
        raise ValueError(
            f"requested {sample_count} calibration samples from {len(task_indices)} frames"
        )
    groups: dict[object, list[int]] = {}
    for index, raw_task in enumerate(task_indices):
        task = raw_task.item() if hasattr(raw_task, "item") else raw_task
        groups.setdefault(task, []).append(index)
    if sample_count < len(groups):
        raise ValueError(
            f"--samples={sample_count} cannot cover all {len(groups)} tasks; "
            "increase --samples"
        )

    rng = np.random.default_rng(seed)
    tasks = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    queues = {task: list(rng.permutation(groups[task])) for task in tasks}
    selected: list[int] = []
    while len(selected) < sample_count:
        progressed = False
        for task in tasks:
            if queues[task] and len(selected) < sample_count:
                selected.append(int(queues[task].pop()))
                progressed = True
        if not progressed:
            break
    return selected


def capture_observations(
    suite_name: str,
    *,
    image_keys: Sequence[str],
    prompt_key: str,
    state_key: str,
    sample_count: Optional[int],
    seed: int,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Mapping[str, object], ...]:
    """Capture task-balanced observations from native LIBERO initial states."""
    if len(image_keys) != 2:
        raise ValueError(
            "native LIBERO capture requires exactly two configured image views, "
            f"got {len(image_keys)}: {list(image_keys)}"
        )
    progress = progress or (lambda _message: None)
    suite = load_suite(suite_name)
    states_by_task = [
        suite.get_task_init_states(task_id) for task_id in range(suite.n_tasks)
    ]
    task_indices = [
        task_id
        for task_id, initial_states in enumerate(states_by_task)
        for _ in range(len(initial_states))
    ]
    selected_count = sample_count if sample_count is not None else suite.n_tasks
    flat_indices = task_stratified_indices(
        task_indices, sample_count=selected_count, seed=seed
    )

    # Flat frame index -> (task, per-task state index), so each task's environment
    # is built once and every selected state for it is captured in that one pass.
    offsets = []
    offset = 0
    for initial_states in states_by_task:
        offsets.append(offset)
        offset += len(initial_states)
    selected_by_task: dict[int, list[int]] = {}
    for flat_index in flat_indices:
        task_id = task_indices[flat_index]
        selected_by_task.setdefault(task_id, []).append(flat_index - offsets[task_id])

    observations = []
    for task_id in sorted(selected_by_task):
        task = suite.get_task(task_id)
        prompt = str(task.language)
        progress(
            f"Capturing {suite_name} task {task_id} "
            f"({len(selected_by_task[task_id])} observation(s))..."
        )
        env = make_env(task, seed)
        try:
            for initial_state_index in selected_by_task[task_id]:
                env.reset()
                raw_observation = env.set_init_state(
                    states_by_task[task_id][initial_state_index]
                )
                for _ in range(WAIT_STEPS):
                    raw_observation, _, _, _ = env.step(NEUTRAL_ACTION)
                observations.append(
                    to_apxinf_observation(
                        raw_observation,
                        prompt=prompt,
                        image_keys=(image_keys[0], image_keys[1]),
                        prompt_key=prompt_key,
                        state_key=state_key,
                    )
                )
        finally:
            env.close()
    return tuple(observations)


def write_npz_observations(
    observations: Sequence[Mapping[str, object]],
    output_dir: pathlib.Path,
    *,
    prefix: str,
    force: bool = False,
) -> list[pathlib.Path]:
    """Write one NPZ per observation, in a zero-padded, sort-stable order.

    ``calibrate_pi05 --input-dir`` globs ``*.npz`` and sorts the names, so the
    padding is what keeps sample 10 from preceding sample 2 and silently changing
    the calibration data identity between runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.npz"))
    if existing and not force:
        raise ValueError(
            f"{output_dir} already holds {len(existing)} *.npz file(s); pass --force "
            "to replace them (calibration reads the whole directory)"
        )
    for path in existing:
        path.unlink()

    width = max(3, len(str(len(observations) - 1)))
    written = []
    for index, observation in enumerate(observations):
        path = output_dir / f"{prefix}-{index:0{width}d}.npz"
        # A str prompt becomes a 0-d unicode array, which is what the engine's
        # loader unwraps back into a str via ``ndarray.item()``.
        np.savez(path, **{name: np.asarray(value) for name, value in observation.items()})
        written.append(path)
    return written


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--suite",
        default="libero_10",
        choices=ALL_SUITES,
        help="LIBERO task suite to capture initial states from",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="directory to write Observation *.npz files into",
    )
    parser.add_argument(
        "--robot",
        default=LIBERO_PRESET,
        help=(
            "robot preset whose wire keys the NPZ fields are named after "
            f"(default {LIBERO_PRESET}); must match how the checkpoint is served"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="task-balanced sample count (default: one initial state per task)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="libero", help="NPZ filename prefix")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace *.npz files already in --output-dir",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apxinf-robo capture-libero",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_arguments(parser)


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.samples is not None and args.samples < 1:
        build_parser().error("--samples must be positive")
    if args.seed < 0:
        build_parser().error("--seed must be non-negative")
    return args


def run(args) -> int:
    from ..presets import get_robot_preset

    convention = get_robot_preset(args.robot).convention
    if convention.state_key is None:
        raise ValueError(
            f"preset {args.robot!r} records no state key, so a captured observation "
            "would carry no proprioception; pick a preset whose convention has one"
        )
    observations = capture_observations(
        args.suite,
        image_keys=convention.image_keys,
        prompt_key=convention.prompt_key,
        state_key=convention.state_key,
        sample_count=args.samples,
        seed=args.seed,
        progress=_progress,
    )
    written = write_npz_observations(
        observations, args.output_dir, prefix=args.prefix, force=args.force
    )
    _progress(f"Wrote {len(written)} observation(s) to {args.output_dir}.")
    _progress("Feed them to the engine with:")
    _progress(f"    {_calibrate_command(convention, args.output_dir)}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

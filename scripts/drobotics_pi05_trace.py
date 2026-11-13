#!/usr/bin/env python3
"""Trace one LIBERO observation through the client and the S600 Pi05 runtime.

The file runs in two places, so board-side code uses only the standard library:

    # PC: build the input variants with the evaluator's own conversion functions.
    python scripts/drobotics_pi05_trace.py prepare \
        --fixture devlocal/s600-pi05-stage3/fixture \
        --out devlocal/s600-pi05-stage4/variants

    # S600: run each variant through the official demo with tensor dumps enabled.
    python3 drobotics_pi05_trace.py run --variants variants --work-dir .

    # PC: compare the dumped tensors against the contract's stated preprocessing.
    python scripts/drobotics_pi05_trace.py analyze \
        --contract apx-drobotics/src/apx_drobotics/contracts/pi05_libero_w8.json \
        --run-dir <fetched work dir> --variants devlocal/s600-pi05-stage4/variants \
        --fixture devlocal/s600-pi05-stage3/fixture \
        --model-zip /tmp/Pi05_libero.zip --out trace.json

    # PC: step the returned actions in LIBERO to confirm what they mean physically.
    MUJOCO_GL=egl python scripts/drobotics_pi05_trace.py replay \
        --contract apx-drobotics/src/apx_drobotics/contracts/pi05_libero_w8.json \
        --fixture devlocal/s600-pi05-stage3/fixture \
        --action-bin <fetched work dir>/runs/raw224_pil/output/action.bin --out replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time


DEFAULT_SDK = Path(
    "/opt/d-robotics-llm/2.0.0/"
    "D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("raw256", "jpeg256", "raw224_pil", "state7", "prompt_dirty")
#: Large dumps the analysis does not read; removed so a trace stays small.
DISCARD_PREFIXES = ("lm_o", "visual0_o", "visual1_o", "visual2_o")
DIRTY_PROMPT = "pick_up the alphabet soup\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_libero_conversion():
    """Load the evaluator's conversion module without importing the package.

    This module needs only NumPy, and this script also runs where the package is not
    installed. Loading it by path means the trace exercises the evaluator's real
    functions.
    """
    path = REPO_ROOT / "src/apxinf_robo/envs/libero.py"
    spec = importlib.util.spec_from_file_location("_drobotics_trace_libero", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_prepare(args: argparse.Namespace) -> None:
    import numpy as np
    from openpi_client import image_tools

    conversion = load_libero_conversion()
    fixture = args.fixture.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = dict(np.load(fixture / "observation.npz"))
    prompt = (fixture / "prompt.txt").read_text(encoding="utf-8")

    images = conversion.libero_images(raw["agentview_image"], raw["robot0_eye_in_hand_image"])
    state8 = conversion.libero_state(raw, finger_joints=2)
    state7 = conversion.libero_state(raw)
    fixture_state = np.frombuffer((fixture / "state.bin").read_bytes(), dtype="<f8")

    def write_common(directory: Path, state, text: str) -> None:
        directory.mkdir(parents=True)
        (directory / "state.bin").write_bytes(np.asarray(state, dtype="<f8").tobytes())
        (directory / "prompt.txt").write_bytes(text.encode("utf-8"))

    def write_raw(directory: Path, batch) -> None:
        for index, image in enumerate(batch):
            height, width, _ = image.shape
            chw = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
            (directory / f"image_{index}_{width}x{height}.bin").write_bytes(chw.tobytes())

    resized = image_tools.resize_with_pad(images, 224, 224)
    for name in ("raw256", "state7", "prompt_dirty"):
        directory = out / name
        write_common(
            directory,
            state7 if name == "state7" else state8,
            DIRTY_PROMPT if name == "prompt_dirty" else prompt,
        )
        write_raw(directory, images)
    write_common(out / "raw224_pil", state8, prompt)
    write_raw(out / "raw224_pil", resized)
    write_common(out / "jpeg256", state8, prompt)
    for index in range(2):
        shutil.copyfile(fixture / f"image_{index}.jpg", out / "jpeg256" / f"image_{index}.jpg")

    np.save(out / "client_images_256.npy", images)
    np.save(out / "client_images_224_pil.npy", resized)
    manifest = {
        "fixture_manifest": json.loads((fixture / "manifest.json").read_text(encoding="utf-8")),
        "client_conversion": "src/apxinf_robo/envs/libero.py",
        "images_flipped_sha256": sha256_bytes(images.tobytes()),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "state8_from_evaluator": [float(value) for value in state8],
        "state7_from_evaluator": [float(value) for value in state7],
        "state8_vs_fixture_max_abs_diff": float(np.abs(state8 - fixture_state).max()),
        "variant_files": {
            name: {
                path.name: sha256_file(path)
                for path in sorted((out / name).iterdir())
            }
            for name in VARIANTS
        },
    }
    (out / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "fixture_manifest"},
                     indent=2)[:2000])


def run_variant(name: str, variants: Path, sdk: Path, work: Path) -> dict:
    demo = sdk / "examples/vla_demo/pi0"
    run_dir = work / "runs" / name
    dump_dir = run_dir / "dump"
    output_dir = run_dir / "output"
    dump_dir.mkdir(parents=True)
    output_dir.mkdir()

    config = json.loads((demo / "pi05_libero_config.json").read_text(encoding="utf-8"))
    # Only these two fields differ from the official config: where the model is
    # (the config now lives outside the demo directory), and tensor dumping.
    config["model"]["work_path"] = str(sdk / "model/Pi05_libero")
    config["dfx"]["dump"] = True
    config["dfx"]["dump_dir"] = str(dump_dir) + "/"
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    launch = {
        "default_mode": "local", "modes": ["local"],
        "io": {"input_dir": str((variants / name).resolve()),
               "output_dir": str(output_dir), "multi_chunk": False},
        "demo": {"infer_nums": 1, "image_count": 2, "language_count": 1, "state_count": 1},
    }
    launch_path = run_dir / "launch.json"
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")

    command = ["bash", "run.sh", os.path.relpath(config_path, demo), "local",
               "--demo_config", str(launch_path)]
    started = time.perf_counter()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        try:
            returncode = subprocess.run(
                command, cwd=demo, stdout=log, stderr=subprocess.STDOUT,
                text=True, timeout=300,
            ).returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"

    dumped_bytes = 0
    for path in sorted(dump_dir.rglob("*")):
        if path.is_file():
            dumped_bytes += path.stat().st_size
            if path.name.startswith(DISCARD_PREFIXES):
                path.unlink()
    return {
        "variant": name, "returncode": returncode,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "dumped_bytes": dumped_bytes,
        "kept_files": {
            str(path.relative_to(dump_dir)): path.stat().st_size
            for path in sorted(dump_dir.rglob("*")) if path.is_file()
        },
        "action_written": (output_dir / "action.bin").is_file(),
    }


def cmd_run(args: argparse.Namespace) -> None:
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.only or VARIANTS:
        print(f"[{time.strftime('%H:%M:%S')}] {name} start", flush=True)
        result = run_variant(name, args.variants.resolve(), args.sdk_root.resolve(), work)
        results.append(result)
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={result['returncode']} "
              f"action={result['action_written']} dumped={result['dumped_bytes']}", flush=True)
        (work / "run_summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    print("TRACE_RUN_FINISHED", flush=True)


def _checker(checks: list):
    def check(check_id: str, ok, **detail) -> None:
        checks.append({"id": check_id, "ok": bool(ok), **detail})
    return check


def _tokenizer(model_dir: Path):
    """Load tokenizer.json for encoding; its decoder section is not needed here."""
    from tokenizers import Tokenizer

    data = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    data["decoder"] = None
    return Tokenizer.from_str(json.dumps(data))


def _libero_prompts() -> list:
    from libero.libero import benchmark

    table = benchmark.get_benchmark_dict()
    prompts = set()
    for name in ("libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"):
        suite = table[name]()
        prompts.update(suite.get_task(index).language for index in range(suite.n_tasks))
    return sorted(prompts)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compare the dumped tensors with what the contract says should happen."""
    import tempfile
    import zipfile

    import cv2
    import numpy as np
    from openpi_client import image_tools
    from PIL import Image

    conversion = load_libero_conversion()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    runs = args.run_dir.resolve() / "runs"
    variants = args.variants.resolve()
    fixture = args.fixture.resolve()
    checks: list = []
    check = _checker(checks)

    def dump(variant: str, name: str) -> Path:
        return runs / variant / "dump" / name

    def f16(path: Path, shape) -> "np.ndarray":
        return np.frombuffer(path.read_bytes(), dtype="<f2").reshape(shape).astype(np.float64)

    def pixels(variant: str, camera: int) -> "np.ndarray":
        # The runtime's tensor is CHW fp16 holding pixel * 2 / 255 - 1.
        values = f16(dump(variant, f"visual{camera}_i0.bin"), (3, 224, 224))
        return np.transpose((values + 1) * 127.5, (1, 2, 0))

    def action_output(variant: str) -> "np.ndarray":
        data = (runs / variant / "output/action.bin").read_bytes()
        return np.frombuffer(data, dtype="<f8").reshape(
            contract["action"]["horizon"], contract["action"]["dim"]
        )

    def tokens_of(variant: str) -> "np.ndarray":
        return np.frombuffer(dump(variant, "lm_i0.bin").read_bytes(), dtype="<i4")

    summary = json.loads((args.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    check("run.all_variants_succeeded",
          all(item["returncode"] == 0 and item["action_written"] for item in summary),
          returncodes={item["variant"]: item["returncode"] for item in summary})

    with tempfile.TemporaryDirectory() as temporary:
        model_dir = Path(temporary)
        with zipfile.ZipFile(args.model_zip) as archive:
            for name, entry in contract["model"]["files"].items():
                data = archive.read(f"{contract['model']['name']}/{name}")
                actual = {"bytes": len(data), "sha256": sha256_bytes(data)}
                check(f"model.file.{name}", actual == entry, **actual)
                if not name.endswith(".hbm"):
                    (model_dir / name).write_bytes(data)
        norm = json.loads((model_dir / "norm_stats.json").read_text(encoding="utf-8"))["norm_stats"]
        tokenizer = _tokenizer(model_dir)

    # --- client side ---------------------------------------------------------
    raw = dict(np.load(fixture / "observation.npz"))
    images = conversion.libero_images(raw["agentview_image"], raw["robot0_eye_in_hand_image"])
    resized = image_tools.resize_with_pad(images, *contract["camera"]["model_shape"][:2])
    state = conversion.libero_state(raw, finger_joints=2)
    sent_state = np.frombuffer((variants / "raw224_pil/state.bin").read_bytes(), dtype="<f8")
    fixture_state = np.frombuffer((fixture / "state.bin").read_bytes(), dtype="<f8")
    check("client.images_from_evaluator_functions",
          np.array_equal(images, np.load(variants / "client_images_256.npy"))
          and np.array_equal(resized, np.load(variants / "client_images_224_pil.npy")))
    check("client.state_from_evaluator_function",
          np.array_equal(sent_state, state.astype(np.float64))
          and np.abs(sent_state - fixture_state).max() < 1e-6,
          max_abs_diff_vs_fixture=float(np.abs(sent_state - fixture_state).max()))
    check("client.state_dim_and_range",
          sent_state.size == contract["state"]["dim"] and all(
              norm["state"]["q01"][i] <= sent_state[i] <= norm["state"]["q99"][i]
              for i in range(sent_state.size)
          ),
          note="Inside q01..q99 for every component, consistent with the stated order and signs.")

    # --- camera --------------------------------------------------------------
    kernels = {
        "cv2.INTER_LINEAR": lambda image: cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR),
        "PIL.BILINEAR": lambda image: np.asarray(
            Image.fromarray(image).resize((224, 224), Image.BILINEAR)),
    }
    metrics: dict = {"camera_kernels_vs_runtime_at_256": {}}
    for camera in range(2):
        passthrough = np.abs(pixels("raw224_pil", camera) - resized[camera].astype(np.float64))
        check(f"camera.{camera}.contract_path_passes_through",
              passthrough.max() < args.fp16_tolerance, max_abs_diff=float(passthrough.max()),
              note="A client-resized 224 image reaches the model unchanged, apart from fp16 rounding.")
        swapped = np.abs(pixels("raw224_pil", camera)[..., ::-1] - resized[camera].astype(np.float64))
        check(f"camera.{camera}.rgb_order", swapped.mean() > 100 * max(passthrough.mean(), 1e-3),
              mean_abs_diff=float(passthrough.mean()), mean_abs_diff_if_swapped=float(swapped.mean()))
        at_256 = pixels("raw256", camera)
        metrics["camera_kernels_vs_runtime_at_256"][str(camera)] = {
            name: {"max": float(np.abs(at_256 - fn(images[camera])).max()),
                   "mean": float(np.abs(at_256 - fn(images[camera])).mean())}
            for name, fn in kernels.items()
        }
        linear = metrics["camera_kernels_vs_runtime_at_256"][str(camera)]["cv2.INTER_LINEAR"]["max"]
        pil = metrics["camera_kernels_vs_runtime_at_256"][str(camera)]["PIL.BILINEAR"]["max"]
        check(f"camera.{camera}.runtime_resize_is_not_the_client_kernel",
              linear < args.fp16_tolerance and pil > 1.0,
              cv2_linear_max=linear, pil_bilinear_max=pil)
    third = f16(dump("raw224_pil", "visual2_i0.bin"), (3, 224, 224))
    attention = f16(dump("raw256", "lm_i2.bin"), (968, 968))
    check("camera.third_slot_black_and_masked",
          (third == -1).all() and (attention[0, 512:768] < 0).all()
          and (attention[0, 0:512] == 0).all(),
          note="Slot 2 holds only -1 and is masked for prompt and image rows; slots 0 and 1 are visible.")

    # --- prompt --------------------------------------------------------------
    prompt_contract = contract["prompt"]

    def encode(text: str) -> list:
        return tokenizer.encode(text, add_special_tokens=False).ids

    prompt = (variants / "raw224_pil/prompt.txt").read_text(encoding="utf-8")
    expected = [prompt_contract["bos_token"], *encode(prompt + "\n")]
    tokens = tokens_of("raw224_pil")
    check("prompt.tokens_match_contract",
          tokens.size == prompt_contract["token_buffer"]
          and tokens[: len(expected)].tolist() == expected
          and (tokens[len(expected):] == prompt_contract["pad_token"]).all(),
          tokens=expected)
    all_prompts = _libero_prompts() if args.libero_prompts else []
    differing = [
        text for text in all_prompts
        if encode(text + "\n") != encode(text.strip().replace("_", " ").replace("\n", " "))
        + [prompt_contract["newline_token"]]
    ]
    unclean = [text for text in all_prompts
               if text != text.strip().replace("_", " ").replace("\n", " ")]
    check("prompt.all_libero_prompts_encode_like_the_reference",
          bool(all_prompts) and not differing and not unclean,
          prompts=len(all_prompts), token_mismatches=len(differing), changed_by_cleaning=len(unclean),
          max_tokens=max(len(encode(text + "\n")) + 1 for text in all_prompts) if all_prompts else None)
    dirty = tokens_of("prompt_dirty")
    check("prompt.runtime_does_not_clean",
          235298 in dirty.tolist() and dirty.tolist() != tokens.tolist(),
          tokens=dirty[dirty != 0].tolist(),
          note="An underscore reaches the model as token 235298, so cleaning belongs to the client.")

    # --- state ---------------------------------------------------------------
    log = (runs / "raw224_pil/run.log").read_text(encoding="utf-8", errors="replace")
    action_inputs = sorted(
        path.name for path in (runs / "raw224_pil/dump").glob("action_i*") if "_0." in path.name
    )
    check("state.not_read_by_model",
          "[Model State Size]: 0" in log and "state processing disabled" in log
          and np.array_equal(action_output("state7"), action_output("raw256"))
          and contract["state"]["used_by_model"] is False,
          note="A seven-value state is accepted and gives byte-identical actions.",
          step_zero_action_inputs=action_inputs)
    check("state.reaches_the_runtime",
          dump("raw224_pil", "state.bin").read_bytes() == sent_state.tobytes())

    # --- action --------------------------------------------------------------
    action = contract["action"]
    low = np.array(norm["actions"]["q01"])
    high = np.array(norm["actions"]["q99"])
    eps = action["normalization"]["eps"]
    errors = {}
    finite = {}
    for item in summary:
        name = item["variant"]
        inner = f16(dump(name, "action_inner.bin"), (action["horizon"], action["model_output_dim"]))
        predicted = (inner[:, : action["dim"]] + 1) / 2 * (high - low + eps) + low
        errors[name] = float(np.abs(predicted - action_output(name)).max())
        finite[name] = bool(np.isfinite(action_output(name)).all())
    check("action.denormalization_formula", max(errors.values()) < 1e-9, max_abs_error=errors)
    check("action.shape_and_finite", all(finite.values()), shape=[action["horizon"], action["dim"]])

    # --- noise ---------------------------------------------------------------
    noise_hashes = {
        item["variant"]: sha256_file(dump(item["variant"], "action_i0_0.bin")) for item in summary
    }
    noise = f16(dump("raw224_pil", "action_i0_0.bin"), (action["horizon"], action["model_output_dim"]))
    check("noise.fixed_sequence_per_process", len(set(noise_hashes.values())) == 1,
          sha256=next(iter(noise_hashes.values())), mean=float(noise.mean()), std=float(noise.std()))

    reference = action_output("raw224_pil")
    metrics["action_change_vs_contract_path"] = {
        name: {"max_abs": float(np.abs(action_output(name) - reference).max()),
               "mean_abs": float(np.abs(action_output(name) - reference).mean())}
        for name in ("raw256", "jpeg256", "prompt_dirty")
    }
    chain = {
        "client_images_sha256": sha256_bytes(resized.tobytes()),
        "client_state_sha256": sha256_bytes(sent_state.tobytes()),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "runtime_pixels_sha256": [
            sha256_file(dump("raw224_pil", f"visual{camera}_i0.bin")) for camera in range(3)
        ],
        "runtime_tokens_sha256": sha256_file(dump("raw224_pil", "lm_i0.bin")),
        "runtime_noise_sha256": noise_hashes["raw224_pil"],
        "runtime_action_inner_sha256": sha256_file(dump("raw224_pil", "action_inner.bin")),
        "action_sha256": sha256_bytes(reference.tobytes()),
    }
    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "contract_id": contract["contract_id"], "contract_version": contract["contract_version"],
        "primary_variant": "raw224_pil", "fixture": str(fixture.name), "checks": checks,
        "failed": failed, "metrics": metrics, "chain": chain,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


def cmd_replay(args: argparse.Namespace) -> None:
    """Step the board's actions in LIBERO to confirm their physical meaning."""
    import numpy as np
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    fixture = args.fixture.resolve()
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    saved = dict(np.load(fixture / "observation.npz"))
    task = benchmark.get_benchmark_dict()[manifest["suite"]]().get_task(manifest["task_id"])
    states = torch.load(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        weights_only=False,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256,
    )
    action = contract["action"]
    chunk = np.frombuffer(args.action_bin.read_bytes(), dtype="<f8").reshape(action["horizon"], action["dim"])
    checks: list = []
    check = _checker(checks)

    def reset():
        env.seed(manifest["seed"])
        env.reset()
        observation = env.set_init_state(states[manifest["initial_state_id"]])
        for _ in range(manifest["settle_steps"]):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        return observation

    def play(actions) -> dict:
        observation = reset()
        start = observation["robot0_eef_pos"].copy()
        target_start = observation[args.target_key].copy()
        for row in actions:
            observation, _, _, _ = env.step(list(map(float, row)))
        end = observation["robot0_eef_pos"]
        return {
            "distance_start_m": float(np.linalg.norm(start - target_start)),
            "distance_end_m": float(np.linalg.norm(end - observation[args.target_key])),
            "eef_delta_m": [float(value) for value in end - start],
            "finger_qpos": [float(value) for value in observation["robot0_gripper_qpos"]],
        }

    def constant(index: int, value: float, gripper: float = -1.0):
        rows = np.zeros((action["horizon"], action["dim"]))
        rows[:, 6] = gripper
        rows[:, index] = value
        return rows

    try:
        first = reset()
        controller = env.env.robots[0].controller
        check("replay.reproduces_the_fixture",
              np.array_equal(first["agentview_image"], saved["agentview_image"])
              and np.array_equal(first["robot0_eye_in_hand_image"], saved["robot0_eye_in_hand_image"])
              and np.array_equal(first["robot0_eef_pos"], saved["robot0_eef_pos"]))
        limits = action["controller"]
        check("controller.matches_contract",
              controller.name == "OSC_POSE" and env.env.action_dim == action["dim"]
              and list(controller.input_max) == [1.0] * 6
              and list(controller.output_max) == [limits["output_max_position_m"]] * 3
              + [limits["output_max_rotation_rad"]] * 3,
              name=controller.name, output_max=[float(v) for v in controller.output_max])
        runs = {"board_actions": play(chunk)}
        negated = chunk.copy()
        negated[:, :3] *= -1
        runs["position_negated"] = play(negated)
        for axis, name in enumerate(("dx", "dy", "dz")):
            runs[f"{name}_positive"] = play(constant(axis, 0.5))
        runs["gripper_plus_one"] = play(constant(6, 1.0, gripper=1.0))
        runs["gripper_minus_one"] = play(constant(6, -1.0, gripper=-1.0))
    finally:
        env.close()

    board = runs["board_actions"]
    flipped = runs["position_negated"]
    check("action.moves_toward_the_named_object",
          board["distance_end_m"] < board["distance_start_m"] < flipped["distance_end_m"],
          target=args.target_key, start=board["distance_start_m"],
          with_board_actions=board["distance_end_m"], with_position_negated=flipped["distance_end_m"])
    for axis, name in enumerate(("dx", "dy", "dz")):
        delta = runs[f"{name}_positive"]["eef_delta_m"]
        dominant = int(np.argmax(np.abs(delta)))
        check(f"action.{name}_is_positive_world_axis", dominant == axis and delta[axis] > 0, eef_delta_m=delta)
    closed = runs["gripper_plus_one"]["finger_qpos"][0]
    opened = runs["gripper_minus_one"]["finger_qpos"][0]
    check("action.gripper_sign",
          action["gripper"]["close"] == 1.0 and action["gripper"]["open"] == -1.0 and closed < opened,
          finger_qpos_at_plus_one=closed, finger_qpos_at_minus_one=opened)
    check("action.gripper_open_while_approaching",
          (chunk[:, action["gripper"]["index"]] < 0).all() and board["finger_qpos"][0] > 0.03,
          model_gripper_min=float(chunk[:, 6].min()), model_gripper_max=float(chunk[:, 6].max()))
    failed = [item["id"] for item in checks if not item["ok"]]
    report = {
        "contract_id": contract["contract_id"], "action_bin_sha256": sha256_file(args.action_bin),
        "actions": chunk.tolist(), "checks": checks, "failed": failed, "runs": runs,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--fixture", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.set_defaults(handler=cmd_prepare)
    run = commands.add_parser("run")
    run.add_argument("--variants", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK)
    run.add_argument("--only", nargs="*", choices=VARIANTS)
    run.set_defaults(handler=cmd_run)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--contract", type=Path, required=True)
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--variants", type=Path, required=True)
    analyze.add_argument("--fixture", type=Path, required=True)
    analyze.add_argument("--model-zip", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--fp16-tolerance", type=float, default=0.1,
                         help="largest pixel difference attributed to fp16 rounding")
    analyze.add_argument("--no-libero-prompts", dest="libero_prompts", action="store_false",
                         help="skip the check over every LIBERO task prompt")
    analyze.set_defaults(handler=cmd_analyze)
    replay = commands.add_parser("replay")
    replay.add_argument("--contract", type=Path, required=True)
    replay.add_argument("--fixture", type=Path, required=True)
    replay.add_argument("--action-bin", type=Path, required=True)
    replay.add_argument("--target-key", default="alphabet_soup_1_pos")
    replay.add_argument("--out", type=Path, required=True)
    replay.set_defaults(handler=cmd_replay)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

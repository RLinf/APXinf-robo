#!/usr/bin/env python3
"""Run the official Pi05 LIBERO demo on S600 with three saved observations.

Copy this file and a fixture made by drobotics_pi05_libero_fixture.py to S600, then:

    python3 drobotics_pi05_stage3_verify.py --fixture fixture --work-dir .

The work directory receives launch.json, run.log, timeline.jsonl, three action
files, and result.json. No nonstandard Python packages are needed on S600.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time


DEFAULT_SDK = Path(
    "/opt/d-robotics-llm/2.0.0/"
    "D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600"
)
INPUT_NAMES = ("image_0.jpg", "image_1.jpg", "state.bin", "prompt.txt")
HBM_NAMES = (
    "pi05_vision_224x224_w8_nash-p_corenum_1.hbm",
    "pi05_language_w8_nash-p_corenum_4.hbm",
    "pi05_action_ah_10_w8_nash-p_corenum_4.hbm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_f64(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    if len(data) % 8:
        raise ValueError(f"{path}: byte count is not a multiple of fp64")
    return struct.unpack("<" + "d" * (len(data) // 8), data)


def ion_regions() -> dict[str, int]:
    root = Path("/proc/device-tree/reserved-memory")
    if not root.is_dir():
        return {}
    address_cells = int.from_bytes((root / "#address-cells").read_bytes(), "big")
    size_cells = int.from_bytes((root / "#size-cells").read_bytes(), "big")
    regions = {}
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith("ion_"):
            raw = (entry / "reg").read_bytes()
            start = 4 * address_cells
            regions[entry.name.split("@")[0]] = int.from_bytes(
                raw[start:start + 4 * size_cells], "big"
            )
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true",
                        help="summarize an existing run without starting inference")
    args = parser.parse_args()

    sdk = args.sdk_root.resolve()
    fixture = args.fixture.resolve()
    work = args.work_dir.resolve()
    demo = sdk / "examples/vla_demo/pi0"
    config_path = demo / "pi05_libero_config.json"
    model_dir = sdk / "model/Pi05_libero"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    for name in INPUT_NAMES:
        actual = sha256(fixture / name)
        expected = manifest["file_sha256"][name]
        if actual != expected:
            raise ValueError(f"fixture checksum mismatch: {name}")
    state = read_f64(fixture / "state.bin")
    if len(state) != 8 or not all(map(math.isfinite, state)):
        raise ValueError("fixture must contain eight finite fp64 state values")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["runtime_type"] != "Pi0Sdk" or
            config["interface"]["action_horizon"] != 10 or
            config["interface"]["raw_action_size"] != 7 or
            config["interface"]["raw_state_size"] != 8 or
            config["interface"]["image_mask"] != [True, True, False]):
        raise ValueError("official Pi05 LIBERO config does not match expected shapes")
    if (demo / config["model"]["work_path"]).resolve() != model_dir:
        raise ValueError("official config work_path does not resolve to Pi05_libero")
    for key, name in zip(("visual", "lm", "action"), HBM_NAMES):
        if config["model"]["model_runner"][key] != name or not (model_dir / name).is_file():
            raise ValueError(f"config {key} does not resolve to LIBERO HBM {name}")

    work.mkdir(parents=True, exist_ok=True)
    input_root = work / "input"
    output_root = work / "output"
    samples = [f"run_{i:03d}" for i in range(3)]
    command = ["bash", "run.sh", "pi05_libero_config.json", "local",
               "--demo_config", str(work / "launch.json"), "--verbose"]
    timeline_path = work / "timeline.jsonl"
    if args.analyze_only:
        joined_log = (work / "run.log").read_text(encoding="utf-8")
        returncode = 1 if "OellmRuntime init failed" in joined_log else 0
        total_wall_ms = None
        started_at = "unknown (existing run)"
    else:
        input_root.mkdir(exist_ok=True)
        output_root.mkdir(exist_ok=True)
        for sample in samples:
            directory = input_root / sample
            directory.mkdir(exist_ok=True)
            for name in INPUT_NAMES:
                shutil.copyfile(fixture / name, directory / name)
        launch = {
            "default_mode": "local", "modes": ["local"],
            "io": {"input_dir": str(input_root), "output_dir": str(output_root),
                   "multi_chunk": True},
            "demo": {"infer_nums": 1, "image_count": 2,
                     "language_count": 1, "state_count": 1},
        }
        (work / "launch.json").write_text(
            json.dumps(launch, indent=2) + "\n", encoding="utf-8"
        )
        start = time.perf_counter()
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with (work / "run.log").open("w", encoding="utf-8") as log, \
                timeline_path.open("w", encoding="utf-8") as timeline:
            process = subprocess.Popen(
                command, cwd=demo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log.write(line)
                log.flush()
                timeline.write(json.dumps({"elapsed_ms": round(elapsed_ms, 3),
                                           "line": line.rstrip("\n")}) + "\n")
                timeline.flush()
            returncode = process.wait()
        total_wall_ms = (time.perf_counter() - start) * 1000
        joined_log = (work / "run.log").read_text(encoding="utf-8")

    shapes = re.findall(r"action_shape:\s*(\d+)\*(\d+)\*(\d+)", joined_log)
    profiles = []
    for section in joined_log.split("VLA PROFILE SUMMARY")[1:]:
        metrics = {
            name.lower(): float(value)
            for name, value in re.findall(
                r"\[(Preproc|Visual|LM|Action|Postproc|Total)\]\s*:\s*([\d.]+) ms",
                section.split("VLA PROFILE SUMMARY", 1)[0],
            )
        }
        if metrics:
            profiles.append(metrics)
    timeline = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    ready_line = next((item for item in timeline
                       if "VLA SDK runtime is ready" in item["line"]), None)
    failure_line = next((item for item in timeline
                         if "Cannot malloc bpu memory" in item["line"]), None)
    results = []
    for sample in samples:
        path = output_root / sample / "action.bin"
        if not path.is_file():
            results.append({"sample": sample, "status": "missing"})
            continue
        values = read_f64(path)
        if not values:
            results.append({"sample": sample, "status": "empty"})
            continue
        results.append({
            "sample": sample, "shape": [10, 7], "dtype": "float64 little-endian",
            "elements": len(values), "finite": all(map(math.isfinite, values)),
            "min": min(values), "max": max(values), "mean": sum(values) / len(values),
            "sha256": sha256(path),
        })
    version_paths = [config_path, demo / "dist/vla_sdk_demo",
                     demo / "dist/libvla_pi0_sdk.so",
                     sdk / "lib/liboellm_runtime.so", sdk / "lib/libvla_sdk.so",
                     sdk / "lib/libdnn.so", sdk / "lib/libhbrt4.so",
                     sdk / "lib/libhbucp.so"]
    version_paths += [model_dir / name for name in HBM_NAMES]
    version_paths += [model_dir / name for name in
                      ("norm_stats.json", "tokenizer.model", "tokenizer.json",
                       "tokenizer_config.json")]
    report = {
        "started_at": started_at, "hostname": os.uname().nodename,
        "kernel": os.uname().release, "sdk_root": str(sdk),
        "model_dir": str(model_dir), "sdk_version": "2.0.0",
        "command": command, "exit_code": returncode,
        "last_logged_elapsed_ms": timeline[-1]["elapsed_ms"] if timeline else None,
        "bpu_allocation_failure_elapsed_ms": (
            failure_line["elapsed_ms"] if failure_line else None
        ),
        "cold_init_to_ready_wall_ms": ready_line["elapsed_ms"] if ready_line else None,
        "total_process_wall_ms": round(total_wall_ms, 3) if total_wall_ms else None,
        "fixture": manifest,
        "input_runtime": {"image_shape": [2, 256, 256, 3],
                          "image_dtype": "uint8 RGB (JPEG decoded)",
                          "state_shape": [8], "state_dtype": "float64",
                          "prompt_type": "UTF-8 string"},
        "runtime_reported_shapes": [list(map(int, shape)) for shape in shapes],
        "profiles_ms": profiles,
        "outputs": results,
        "bpu_hbm_log_mentions": {name: name in joined_log for name in HBM_NAMES},
        "ion_region_bytes": ion_regions(),
        "host_boot_dtb_visible": Path("/boot/hobot").is_dir(),
        "files": {str(path): {"bytes": path.stat().st_size,
                              "sha256": sha256(path)} for path in version_paths},
    }
    (work / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in
                      ("exit_code", "cold_init_to_ready_wall_ms",
                       "total_process_wall_ms", "runtime_reported_shapes",
                       "profiles_ms", "outputs", "bpu_hbm_log_mentions")}, indent=2))
    if returncode != 0 or shapes != [("1", "10", "7")] * 3:
        raise RuntimeError("inference failed or action shapes differ; see run.log")
    if not all(result.get("elements") == 70 and result.get("finite") for result in results):
        raise RuntimeError("action chunks have invalid shape or nonfinite values")


if __name__ == "__main__":
    main()

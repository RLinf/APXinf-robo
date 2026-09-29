#!/usr/bin/env python3
"""Fit PI0-FAST L1 prefix and per-token latency with Robo's policy loader.

Input preparation and the autoregressive fit use the benchmark implementation
at the pinned ApxInf revision. The timed call starts at prepared RGB/token IDs
and ends after native action tokens return.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from apxinf_robo import load_policy


def _engine_benchmark():
    path = Path(__file__).resolve().parents[1] / "apxinf/scripts/bench_pi0_fast.py"
    spec = importlib.util.spec_from_file_location("apxinf_pi0_fast_benchmark", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load pinned ApxInf benchmark: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-variant", choices=("flipped", "raw"), default="flipped")
    parser.add_argument("--frames-limit", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--survey", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.survey < 2 or args.repeats < 1:
        parser.error("--survey must be at least 2 and --repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    bench = _engine_benchmark()
    policy = load_policy(
        args.model_dir,
        precision=args.precision,
        state_key=args.state_key,
        prompt_key=args.prompt_key,
    )
    try:
        frames = bench._load_frames(args.frames, args.frame_variant, args.frames_limit)
        observations = bench._observations(policy, frames, bench.DEFAULT_PROMPT)
        prepared = [bench._prepare(policy, observation) for observation in observations]
        stop_token = int(policy.tokenizer.pipe_token_id)

        def infer(payload):
            rgb, token_ids = payload
            return policy.model.infer_action_tokens_rgb(
                rgb, "nhwc", token_ids, stop_token=stop_token
            )

        result = {
            "schema": "apxinf.pi0_fast.latency.v1",
            "precision": args.precision,
            "model_dir": str(args.model_dir),
            "frames": str(args.frames),
            "frame_count": len(prepared),
            "ar": {"layer": "l1", **bench._run_ar(infer, prepared, args.survey, args.repeats)},
        }
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        policy.close()


if __name__ == "__main__":
    main()

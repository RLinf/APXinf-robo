#!/usr/bin/env python3
"""Compare two action bundles produced from identical inputs, with an optional CI gate.

Offline (no torch, no engine)::

    scripts/compare_pi05_precision.py compare \
        --baseline runs/bf16.npz --candidate runs/int8_dynamic.npz --out runs/cmp \
        --max-prefix-mae 0.02 --fail-on-gripper-flip

Paired inference (loads one checkpoint at two precisions over the same captured
observations and saved ``[H, 32]`` noise, then writes both bundles and the report)::

    scripts/compare_pi05_precision.py run \
        --model-dir /path/ckpt --observations devlocal/obs_50 --noise noise_50x32.npz \
        --precisions bf16,int8_dynamic --out runs/cmp

Exit codes: ``0`` gate passed, ``1`` contract or data error, ``2`` inference failure,
``3`` gate failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apxinf_robo.precision_compare import (
    ComparisonContract,
    ContractError,
    SampleRecord,
    compare_outputs,
    environment_manifest,
    evaluate_gates,
    load_bundle,
    save_bundle,
    sha256_array,
    sha256_file,
    write_report,
)

EXIT_OK = 0
EXIT_CONTRACT = 1
EXIT_INFER = 2
EXIT_GATE = 3


def _contract_from(
    args: argparse.Namespace, *, baseline: str | None = None, candidate: str | None = None
) -> ComparisonContract:
    return ComparisonContract(
        horizon=args.horizon,
        executed_prefix=args.executed_prefix,
        full_action_dim=args.full_action_dim,
        valid_action_dim=args.valid_action_dim,
        gripper_dim=args.gripper_dim,
        gripper_domain=args.gripper_domain,
        baseline_precision=baseline or getattr(args, "baseline_precision", "bf16"),
        candidate_precision=candidate or getattr(args, "candidate_precision", "int8_dynamic"),
    )


def _finish(report: dict, args: argparse.Namespace) -> int:
    gate = evaluate_gates(
        report,
        max_prefix_mae=args.max_prefix_mae,
        fail_on_gripper_flip=args.fail_on_gripper_flip,
        domain=args.gate_domain,
    )
    report["gates"] = gate
    out = Path(args.out)
    write_report(report, out / "report.json" if out.suffix != ".json" else out)
    prefix = report["domains"].get(report["metric_domain"]) or {}
    mae = prefix.get("slices", {}).get("executed_prefix", {}).get("mae")
    gripper = (report.get("gripper") or {}).get("slices", {}).get("executed_prefix", {})
    print(
        json.dumps(
            {
                "status": report["status"],
                "metric_domain": report["metric_domain"],
                "prefix_mae": mae,
                "gripper_disagreement_rate": gripper.get("disagreement_rate"),
                "gate": gate,
            },
            sort_keys=True,
        )
    )
    if gate["status"] != "passed":
        return EXIT_GATE
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    contract = _contract_from(args)
    try:
        baseline = load_bundle(args.baseline)
        candidate = load_bundle(args.candidate)
        report = compare_outputs(baseline, candidate, contract)
    except (ContractError, ValueError, KeyError, OSError) as error:
        print(f"compare: {error}", file=sys.stderr)
        return EXIT_CONTRACT
    report["environment"] = environment_manifest([args.baseline, args.candidate])
    return _finish(report, args)


def _load_observations(directory: Path, limit: int | None) -> list[tuple[str, Path, dict]]:
    paths = sorted(Path(directory).glob("*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no *.npz observations under {directory}")
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            loaded.append((path.stem, path, {key: np.array(data[key]) for key in data.files}))
    return loaded


def _noise_bank(path: Path | None, names: list[str], shape: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    if path is None:
        bank = {}
        for index, name in enumerate(sorted(names)):
            bank[name] = np.random.default_rng([seed, index]).standard_normal(shape).astype(np.float32)
        return bank
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    if all(name in arrays for name in names):
        return {name: np.ascontiguousarray(arrays[name], dtype=np.float32) for name in names}
    single = [value for value in arrays.values() if value.shape == shape]
    if len(single) == 1:
        return {name: np.ascontiguousarray(single[0], dtype=np.float32) for name in names}
    raise ValueError(f"{path} does not key noise by sample_id and holds no single {shape} array")


def _payload(observation: dict) -> dict:
    """0-d unicode prompt arrays become ``str``; everything else stays an array."""
    return {key: (value.item() if value.dtype.kind == "U" else value) for key, value in observation.items()}


def _domain_of(actions: np.ndarray, contract: ComparisonContract) -> str:
    width = actions.shape[-1]
    if width == contract.full_action_dim and width != contract.valid_action_dim:
        return "normalized_model"
    return "policy_actions"


def _arrays_for(actions: np.ndarray, result: dict, contract: ComparisonContract) -> dict[str, np.ndarray]:
    """Record the deployable chunk, plus the model-width chunk when the engine returns it."""
    domain = _domain_of(actions, contract)
    arrays = {domain: actions[:, : contract.domain_width(domain)]}
    normalized = result.get("normalized_actions") if isinstance(result, dict) else None
    if normalized is not None:
        value = np.asarray(normalized, dtype=np.float32)
        if value.ndim == 2 and value.shape[1] == contract.full_action_dim and value.shape[0] >= contract.horizon:
            arrays["normalized_model"] = value[: contract.horizon]
    return arrays


def cmd_run(args: argparse.Namespace) -> int:
    precisions = [piece.strip() for piece in args.precisions.split(",") if piece.strip()]
    if len(precisions) != 2:
        print("run: --precisions needs exactly two labels, e.g. bf16,int8_dynamic", file=sys.stderr)
        return EXIT_CONTRACT
    contract = _contract_from(args, baseline=precisions[0], candidate=precisions[1])
    out = Path(args.out)
    try:
        observations = _load_observations(Path(args.observations), args.limit)
        shape = (contract.horizon, contract.full_action_dim)
        noise = _noise_bank(Path(args.noise) if args.noise else None, [n for n, _, _ in observations], shape, args.seed)
    except (ValueError, OSError, KeyError) as error:
        print(f"run: {error}", file=sys.stderr)
        return EXIT_CONTRACT

    generated_noise = None
    if args.noise is None:
        # A generated bank is an input asset: keep it beside the bundles, so the run
        # can be replayed without regenerating noise and hoping the recipe is stable.
        out.mkdir(parents=True, exist_ok=True)
        generated_noise = out / "noise.npz"
        temporary = generated_noise.with_name(generated_noise.name + ".tmp.npz")
        np.savez_compressed(temporary, **noise)
        os.replace(temporary, generated_noise)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from apxinf_robo.engine import load_policy
    except Exception as error:  # noqa: BLE001 - the engine reports its own install hint
        print(f"run: cannot import the engine: {error}", file=sys.stderr)
        return EXIT_INFER

    # Role-based names keep a same-precision repeatability run (bf16,bf16) from
    # overwriting the first bundle with the second and "measuring" zero error.
    bundles: list[Path] = []
    for role, precision in zip(("baseline", "candidate"), precisions):
        try:
            policy = load_policy(args.model_dir, precision=precision)
        except Exception as error:  # noqa: BLE001 - surfaced as a runner failure
            print(f"run: cannot load {precision}: {error}", file=sys.stderr)
            return EXIT_INFER
        records = []
        try:
            for name, path, observation in observations:
                sample_noise = noise[name]
                # The engine may write into the array it is handed; keep the bank intact.
                result = policy.infer(_payload(observation), noise=sample_noise.copy())
                actions = np.asarray(result["actions"], dtype=np.float32)
                records.append(
                    SampleRecord(
                        sample_id=name,
                        precision=precision,
                        arrays=_arrays_for(actions, result, contract),
                        hashes={
                            "observation": sha256_file(path),
                            "noise": sha256_array(sample_noise),
                        },
                    )
                )
        except Exception as error:  # noqa: BLE001 - any inference fault is EXIT_INFER
            print(f"run: inference failed at {precision}: {error}", file=sys.stderr)
            return EXIT_INFER
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        bundle = out / f"{role}_{precision}.npz"
        save_bundle(bundle, records)
        bundles.append(bundle)

    try:
        report = compare_outputs(load_bundle(bundles[0]), load_bundle(bundles[1]), contract)
    except (ContractError, ValueError, KeyError, OSError) as error:
        print(f"run: {error}", file=sys.stderr)
        return EXIT_CONTRACT
    assets = list(bundles) + ([generated_noise] if generated_noise else [])
    report["environment"] = environment_manifest(assets)
    return _finish(report, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, help="output directory (or a .json file)")
        p.add_argument("--horizon", type=int, default=50)
        p.add_argument("--executed-prefix", type=int, default=5, help="R: actions the environment executes")
        p.add_argument("--full-action-dim", type=int, default=32, help="model action width, padding included")
        p.add_argument("--valid-action-dim", type=int, default=7)
        p.add_argument("--gripper-dim", type=int, default=6)
        p.add_argument("--gripper-domain", default="policy_actions")
        p.add_argument("--max-prefix-mae", type=float, default=None)
        p.add_argument("--fail-on-gripper-flip", action="store_true")
        p.add_argument("--gate-domain", default=None, choices=("normalized_model", "policy_actions", "env_step_actions"))

    compare = sub.add_parser("compare", help="compare two exported .npz bundles")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline-precision", default="bf16", help="label recorded in the report")
    compare.add_argument("--candidate-precision", default="int8_dynamic")
    common(compare)
    compare.set_defaults(handler=cmd_compare)

    run = sub.add_parser("run", help="paired inference over captured observations")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--observations", required=True)
    run.add_argument("--noise", default=None, help="npz of saved [H, 32] noise keyed by sample_id")
    run.add_argument("--precisions", default="bf16,int8_dynamic")
    run.add_argument("--seed", type=int, default=0, help="seed used when --noise is omitted")
    run.add_argument("--limit", type=int, default=None)
    common(run)
    run.set_defaults(handler=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""``apxinf-robo serve`` -- an OpenPI-compatible WebSocket policy server.

All reusable logic lives in libraries: the transport in :mod:`apxinf.serving`,
the policy in ApxInf, and the per-embodiment
wire contract in :mod:`apxinf_robo.presets`. This file is argument parsing and
wiring -- load an **in-process** policy through the ``apxinf_py`` PyO3 binding
and serve it.

**Embodiment:** ``--robot`` selects the wire keys and the robot pre/post steps,
the way openpi's ``serve_policy.py --policy.config <TrainConfig>`` does. It
defaults to ``franka_libero``; a checkpoint fine-tuned for another robot **must**
name it, because the wire keys, the state routing, and the action encoding all
differ and a mismatch degrades silently rather than failing. ``--image-keys`` /
``--state-key`` override individual fields for a client that already speaks a
fixed dialect.

**State:** the concrete policy owns state encoding. PI0.5 can drop or discretize
it using ``--discrete-state``; PI0-FAST and GR00T use their own state contracts.
``--robot none`` serves a policy such as Qwen-Drive without a robot preset.

**Images are RGB.** Neither this server nor openpi converts colour: an
``H x W x 3`` uint8 array is taken as RGB as-is. A client reading frames with
OpenCV must ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` first. Resizing *is* done
here (aspect-preserving pad to the model's edge), so any resolution is fine.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

from apxinf.checkpoints import FORMATS as CHECKPOINT_FORMATS

from ..embodiments.base import ROBOT_PRESETS, available_robots, get_robot_preset
from ..engine import (
    load_policy,
    load_random_model,
    policy_from_random,
    resolve_tactics,
    websocket_server,
)
from ..preflight import FAIL, WARN, check_checkpoint, format_findings
from ..presets import build_robot_policy

DEFAULT_ROBOT = "franka_libero"


def _json_options(value: str) -> dict:
    try:
        options = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid policy options: {error.msg}") from error
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("--policy-options must be a JSON object")
    return options


def _split_keys(value: str) -> tuple:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise argparse.ArgumentTypeError("--image-keys needs at least one camera key")
    return keys


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Populate ``parser`` with the serving options.

    Split out from :func:`main` so ``apxinf-robo serve`` and a bare
    ``python -m apxinf_robo.cli.serve`` share one definition.
    """
    parser.add_argument("--model-dir", type=pathlib.Path, help="checkpoint directory")
    parser.add_argument(
        "--robot",
        choices=(*available_robots(include_aliases=True), "none"),
        default=DEFAULT_ROBOT,
        help="embodiment preset: wire keys + robot pre/post steps + action width "
        f"(default: {DEFAULT_ROBOT}). openpi's --policy.config equivalent; a "
        "checkpoint fine-tuned for another robot must name it. Presets are named "
        "<arm>_<key convention>, since the arm alone does not fix the contract.",
    )
    parser.add_argument(
        "--image-keys",
        type=_split_keys,
        default=None,
        help="comma-separated camera wire keys, overriding the preset. Order is "
        "significant: key i fills model view slot i (base, left wrist, right "
        "wrist). Nested client layouts are written as a path, e.g. "
        "'images/cam_high,images/cam_left_wrist'.",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="observation key holding the state vector, overriding the preset",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="serve a checkpoint-free engine with deterministic random weights and "
        "synthetic processors (latency-only; actions are numerically meaningless). "
        "Reproduces --robot's wire keys and view count but not its robot pre/post "
        "steps, which need a checkpoint; the served metadata says robot_steps=false "
        "and startup warns per gap. No --model-dir needed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="checkpoint or index (default: MODEL_DIR/model.safetensors)",
    )
    parser.add_argument(
        "--model-type",
        help="policy model_type; default reads MODEL_DIR/config.json, or metadata.pt "
        "for an openpi export, which has no config.json (e.g. pi05)",
    )
    parser.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        help="text tokenizer override (PI0-FAST also needs its FAST action tokenizer)",
    )
    parser.add_argument(
        "--ckpt-format",
        choices=CHECKPOINT_FORMATS,
        default="auto",
        help="how to read MODEL_DIR. 'auto' (default) picks openpi_pytorch when a "
        "metadata.pt is present and lerobot when a config.json is. Pin it when a "
        "directory carries both and the wrong one wins.",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="override the asset_id the checkpoint names, which is what selects "
        "assets/<asset_id>/norm_stats.json. For assets reorganized after export.",
    )
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=None,
        help="explicit OpenPI-style norm_stats.json, outranking every path convention. "
        "LeRobot processor state is discovered from policy_preprocessor.json and "
        "policy_postprocessor.json instead; a base checkpoint with no declared "
        "processor state uses LeRobot-compatible identity transforms.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--policy-options", type=_json_options, default={},
        help="model-specific loader options as JSON; for example Qwen-Drive planner/mode",
    )
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument(
        "--calibration",
        type=pathlib.Path,
        help="FP8 activation calibration JSON; required only for --precision fp8",
    )
    parser.add_argument("--tactics", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing exact GEMM tactics from real requests and persist them",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="deployable action width to trim to, overriding the preset "
        "(LIBERO=7; 0 keeps the full vector)",
    )
    parser.add_argument("--norm-key", default="actions")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="chunk length to serve. Default: the checkpoint's own value (config.json, "
        "or metadata.pt for an openpi export), or 50 with --random-weights. An "
        "explicit value outranks the checkpoint "
        "(the horizon is a sequence length, not a weight dimension).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="serve fewer cameras than the checkpoint declares (must equal the "
        "number of image keys). Drops the trailing view slots at load time -- "
        "equivalent to openpi zero-padding and masking them, minus their patch "
        "tokens. Required to be explicit: a short image_keys list on its own is "
        "an error, so a forgotten camera fails instead of degrading. Also sets "
        "the synthetic view count under --random-weights.",
    )
    # Synthetic-shape knobs, used only with --random-weights (a checkpoint runs its
    # native config). They mirror apxinf_py.ModelRunner.random.
    parser.add_argument("--image-size", type=int, default=224, help="random: image edge")
    parser.add_argument("--num-flow-steps", type=int, default=10, help="random: flow steps")
    parser.add_argument("--max-token-len", type=int, default=200, help="random: max prompt tokens")
    parser.add_argument(
        "--token-count", type=int, default=10, help="random: synthetic prompt length"
    )
    parser.add_argument(
        "--discrete-state",
        dest="discrete_state",
        action="store_true",
        default=None,
        help="inject discretized state into the prompt (state normalized to "
        "[-1, 1] from norm_stats), overriding the preset. Without it state is "
        "silently dropped -- a joint-space robot needs this on.",
    )
    parser.add_argument(
        "--no-discrete-state",
        dest="discrete_state",
        action="store_false",
        help="drop state even if the preset injects it",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_parser() -> argparse.ArgumentParser:
    robot_help = "\n".join(f"  {p.describe()}" for p in ROBOT_PRESETS.values())
    parser = argparse.ArgumentParser(
        prog="apxinf-robo serve",
        description="Serve an ApxInf policy through OpenPI's websocket API "
        "(in-process; no subprocess)",
        epilog=f"robot presets (--robot):\n{robot_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """Serve until interrupted. ``args`` comes from :func:`build_parser`."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.random_weights and args.model_dir is None:
        raise ValueError("pass --model-dir, or --random-weights for a checkpoint-free engine")
    if args.random_weights and args.model_dir is not None:
        raise ValueError("--random-weights is checkpoint-free; do not also pass --model-dir")

    if not args.random_weights:
        from apxinf.policies.auto import _read_model_type

        model_type = args.model_type or _read_model_type(args.model_dir)
        if model_type != "pi05":
            _serve_registered_policy(args, model_type)
            return
        if args.robot == "none":
            raise ValueError("PI0.5 serving requires a robot preset")
        if args.policy_options:
            raise ValueError("--policy-options is supported by registered non-PI0.5 policies")

    preset = get_robot_preset(args.robot)
    image_keys = args.image_keys if args.image_keys is not None else preset.image_keys
    state_key = args.state_key if args.state_key is not None else preset.state_key
    discrete_state = preset.discrete_state if args.discrete_state is None else args.discrete_state
    if args.num_views is not None and args.num_views != len(image_keys):
        raise ValueError(
            f"--num-views {args.num_views} disagrees with the {len(image_keys)} "
            f"camera keys being served ({list(image_keys)}); they name the same "
            "cameras, so they must match"
        )

    metadata = {
        "protocol": "openpi.websocket_policy",
        "precision": args.precision,
        "policy": preset.name,
        "autotune": args.autotune,
    }
    if args.random_weights:
        # Random engines bypass from_pretrained, and there is no checkpoint whose
        # local tactics.json could take precedence, so this synthetic server
        # resolves the package default itself.
        tactics = resolve_tactics(
            args.device,
            args.precision,
            override=args.tactics,
            allow_missing=args.autotune,
        )
        if tactics is not None:
            logging.info("using %s tactics for %s: %s", args.precision, args.device, tactics)
        # Synthetic FP8 has no calibration file; a uniform activation scale keeps the
        # FP8 path on. bf16/int8 need neither calibration nor tactics.
        calibration = None
        if args.precision == "fp8":
            calibration = str(args.calibration) if args.calibration is not None else "uniform:1.0"
        action_horizon = args.action_horizon if args.action_horizon is not None else 50
        # The preset's cameras define the synthetic view count unless --num-views is
        # given explicitly, so --robot alone yields a servable synthetic engine.
        num_views = args.num_views if args.num_views is not None else len(image_keys)
        # A checkpoint-free engine still serves the preset's *deployable* width, so a
        # client previews the action shape it will get in production. --action-dim
        # outranks it (and also sets the synthetic model's own width).
        model_dim = args.action_dim or 32
        trim_dim = args.action_dim if args.action_dim is not None else preset.action_dim
        served_dim = trim_dim or model_dim
        # The synthetic path reproduces the preset's wire keys and view count and
        # nothing else: the tokenizer emits a fixed token stream and never reads
        # state, and preset.builder never runs. The published metadata is the wire
        # contract, so name every gap and mark robot_steps=False below -- silently
        # serving half an embodiment under its own name is precisely the mismatch
        # --robot exists to prevent.
        gaps = preset.synthetic_gaps(discrete_state=discrete_state, served_action_dim=served_dim)
        if gaps:
            logging.warning(
                "--random-weights cannot honour %s: %s. The wire keys and view count "
                "are real; the action semantics are not. Use a checkpoint to preview "
                "the full contract.",
                preset.name,
                "; ".join(gaps),
            )
        logging.info(
            "serving checkpoint-free %s random-weights engine (views=%d, H=%d, T=%d) "
            "-- actions are latency-only",
            args.precision,
            num_views,
            action_horizon,
            args.token_count,
        )
        handle = load_random_model(
            model=(args.model_type or "pi05"),
            device=args.device,
            precision=args.precision,
            num_views=num_views,
            image_size=args.image_size,
            action_horizon=action_horizon,
            action_dim=model_dim,
            num_flow_steps=args.num_flow_steps,
            max_token_len=args.max_token_len,
            calibration=calibration,
            tactics=(str(tactics) if tactics is not None else None),
            autotune=args.autotune,
            seed=args.seed,
        )
        policy = policy_from_random(
            handle,
            token_count=args.token_count,
            action_dim=(trim_dim or None),
            seed=args.seed,
            image_keys=image_keys[:num_views],
            state_key=state_key,
            metadata={**metadata, "robot": preset.name, "robot_steps": False},
        )
    else:
        # Validate checkpoint and embodiment metadata before loading weights.
        findings = check_checkpoint(
            args.model_dir,
            preset.name,
            norm_key=args.norm_key,
            discrete_state=discrete_state,
            image_keys=image_keys,
            action_dim=args.action_dim,
            tokenizer_path=args.tokenizer,
            checkpoint_format=args.ckpt_format,
            asset_id=args.asset_id,
            norm_stats=args.norm_stats,
        )
        fatal = [f for f in findings if f.level == FAIL]
        if fatal:
            raise SystemExit(
                f"preflight: {args.model_dir} does not match --robot {preset.name}\n"
                + format_findings(findings, include_info=False)
            )
        for finding in findings:
            level = (
                logging.ERROR
                if finding.level == FAIL
                else (logging.WARNING if finding.level == WARN else logging.INFO)
            )
            logging.log(level, "preflight %s", finding)

        logging.info(
            "loading %s policy in-process from %s as robot=%s",
            args.precision,
            args.model_dir,
            preset.describe(),
        )
        policy = build_robot_policy(
            preset.name,
            args.model_dir,
            image_keys=image_keys,
            state_key=state_key,
            discrete_state=discrete_state,
            action_dim=args.action_dim,
            num_views=args.num_views,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
            calibration=args.calibration,
            tactics=args.tactics,
            autotune=args.autotune,
            tokenizer_path=args.tokenizer,
            checkpoint_format=args.ckpt_format,
            asset_id=args.asset_id,
            norm_stats=args.norm_stats,
            norm_key=args.norm_key,
            action_horizon=args.action_horizon,
            seed=args.seed,
            metadata=metadata,
        )
    # Clients read the served wire contract off this metadata rather than assuming
    # one: a key mismatch is silent on the wire but visible here. A null state_key
    # is not a gap -- it says this policy drops state, so there is no key to send
    # one under; rendered as "(dropped)" so the log line cannot read as an omission.
    served_state_key = policy.metadata["state_key"]
    logging.info(
        "serving robot=%s robot_steps=%s H=%d x D=%d image_keys=%s state=%s discrete_state=%s",
        policy.metadata.get("robot", preset.name),
        policy.metadata.get("robot_steps"),
        policy.metadata["action_horizon"],
        policy.metadata["action_dim"],
        policy.metadata["image_keys"],
        served_state_key if served_state_key is not None else "(dropped)",
        policy.metadata["discrete_state"],
    )
    server = websocket_server(policy, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


def _serve_registered_policy(args: argparse.Namespace, model_type: str) -> None:
    """Serve a policy using its own loader contract instead of PI0.5 CLI defaults."""
    options = dict(args.policy_options)
    explicit = {
        "calibration": args.calibration,
        "tactics": args.tactics,
        "action_horizon": args.action_horizon,
        "tokenizer_path": args.tokenizer,
        "checkpoint": args.checkpoint,
        "num_views": args.num_views,
        "discrete_state": args.discrete_state,
        "norm_stats": args.norm_stats,
        "image_keys": args.image_keys,
        "state_key": args.state_key,
        "action_dim": args.action_dim,
    }
    options.update((key, value) for key, value in explicit.items() if value is not None)
    if args.norm_key != "actions":
        options["norm_key"] = args.norm_key
    if args.autotune:
        options["autotune"] = True
    options.update(model_type=model_type, device=args.device, precision=args.precision)
    if args.robot == "none":
        policy = load_policy(args.model_dir, **options)
    else:
        if model_type == "qwen_drive":
            raise ValueError("Qwen-Drive has no LIBERO robot preset; pass --robot none")
        options["metadata"] = {**options.get("metadata", {}), "model_type": model_type}
        policy = build_robot_policy(args.robot, args.model_dir, **options)
    try:
        logging.info("serving model=%s H=%d x D=%d", model_type,
                     policy.action_horizon, policy.action_dim)
        websocket_server(policy, args.host, args.port).serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


def main(argv=None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

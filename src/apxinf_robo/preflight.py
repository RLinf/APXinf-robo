"""Validate a checkpoint against the robot preset that will serve it.

This is the preset-dependent half of the startup preflight. The other half —
layout detection, where the normalization statistics resolved from, pi05's
mandatory quantiles, the tokenizer — is
:func:`apxinf.checkpoints.preflight.inspect_checkpoint`, and lives in the engine
because those are facts about a checkpoint, true regardless of what is on the
other end of the wire.

What is added here is the comparison: 7-wide statistics are unremarkable until
you say they will drive a 16-DoF G1, at which point they are fatal. The engine
has no robots and so cannot make that call; this module has the presets and does.

Both halves emit :class:`~apxinf.checkpoints.preflight.Finding` objects, so
:func:`check_checkpoint` merges them into one sorted report and callers see a
single list.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from apxinf.checkpoints.preflight import (
    FAIL,
    INFO,
    WARN,
    CheckpointReport,
    Finding,
    NormFacts,
    format_findings,
    inspect_checkpoint,
    sort_findings,
)

from .embodiments.base import RobotPreset, get_robot_preset

_PI0_FAST_TYPES = frozenset(("pi0_fast", "pi0fast"))
_GR00T_TYPES = frozenset(("gr00t", "gr00tn1d7", "Gr00tN1d7"))


def _registered_policy_report(
    model_dir: Path, preset: RobotPreset, model_type: str, image_keys: Sequence[str]
) -> Tuple[CheckpointReport, Tuple[Finding, ...]]:
    """Check cheap disk facts; the concrete policy validates its own assets on load."""
    if model_type == "qwen_drive":
        findings = [Finding(FAIL, "robot preset", "Qwen-Drive is a driving planner",
                            "load or serve it without a LIBERO robot preset")]
    else:
        required = (
            "model.safetensors",
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ) if model_type in _PI0_FAST_TYPES else ("assets/cosmos/apxinf_assets.json",)
        findings = []
        for name in required:
            present = (model_dir / name).is_file()
            findings.append(Finding(
                INFO if present else FAIL, name, "present" if present else "missing",
                "" if present else "prepare the complete checkpoint before loading",
            ))
        if model_type in _GR00T_TYPES and not any(
            (model_dir / name).is_file() for name in
            ("model.safetensors", "model.safetensors.index.json")
        ):
            findings.append(Finding(FAIL, "model weights", "missing",
                                    "supply the GR00T weights and index"))
        findings.extend(_check_cameras(preset, image_keys))
        findings.append(Finding(WARN, "policy resources",
                                "model-specific tokenizer and processor checks run at load time",
                                "load the policy to validate all resources"))
    report = CheckpointReport(tuple(findings), None, {}, None, None)
    return report, sort_findings(findings)

__all__ = [
    "FAIL",
    "WARN",
    "INFO",
    "Finding",
    "check_checkpoint",
    "inspect_for_robot",
    "format_findings",
]


def _check_width(facts: NormFacts, preset: RobotPreset, expected: int) -> List[Finding]:
    """Are the shipped statistics *this robot's* statistics?

    This is the check that catches a G1 preset served a LIBERO checkpoint: 16-dim
    robot, 7-dim stats. Both directions are fatal -- too narrow and the
    unnormalizer refuses the array outright, too wide and it silently
    unnormalizes dimensions the robot does not have.
    """
    if not facts.present or facts.width is None:
        # Absent, or present with no vector-valued stat. inspect_checkpoint
        # already reported it; there is no width to compare against.
        return []
    if facts.width != expected:
        return [
            Finding(
                FAIL,
                f"{facts.check} width",
                f"{facts.width}, but preset {preset.name!r} is a {expected}-dim robot",
                f"these statistics are not this robot's. Serving them maps the "
                f"model's output through the wrong physical range -- the actions "
                f"stay in range and stay wrong. Point --model-dir at a checkpoint "
                f"whose norm_stats.json is {expected} wide.",
            )
        ]
    return [Finding(INFO, f"{facts.check} width", f"{facts.width}, matches preset")]


def _check_cameras(preset: RobotPreset, image_keys: Sequence[str]) -> List[Finding]:
    """Camera count and ordering against the preset's declared view slots."""
    keys = tuple(image_keys)
    if keys == preset.image_keys:
        return [Finding(INFO, "cameras", f"{len(keys)} views, preset keys unchanged")]
    if len(keys) != preset.num_views:
        return [
            Finding(
                WARN,
                "cameras",
                f"serving {len(keys)} views {list(keys)} but preset {preset.name!r} "
                f"declares {preset.num_views} {list(preset.image_keys)}",
                "the checkpoint was trained with a fixed number of view slots; pass "
                "--num-views to load it for fewer, or drop the --image-keys override",
            )
        ]
    return [
        Finding(
            WARN,
            "cameras",
            f"--image-keys overrides the preset: {list(keys)} instead of "
            f"{list(preset.image_keys)}",
            "entry i fills model view slot i, so a reordered tuple silently feeds "
            "the wrong camera to each slot. Confirm the order matches "
            f"{[slot for slot, _ in preset.slots]}.",
        )
    ]


def _check_overrides(
    preset: RobotPreset, *, action_dim: Optional[int], discrete: bool
) -> List[Finding]:
    """Flag serving knobs that were set away from what the preset declares."""
    findings: List[Finding] = []
    if (
        action_dim is not None
        and preset.action_width is not None
        and action_dim != preset.action_width
    ):
        findings.append(
            Finding(
                WARN,
                "action_dim",
                f"--action-dim {action_dim} overrides preset {preset.name!r}'s "
                f"{preset.action_width}",
                "a width the robot does not have gets truncated or padded somewhere "
                "downstream without an error. Drop the flag unless the deployed "
                "client genuinely expects this width.",
            )
        )
    if discrete and not preset.discrete_state:
        findings.append(
            Finding(INFO, "discrete_state", "enabled by override; preset default is off")
        )
    elif not discrete and preset.discrete_state:
        findings.append(
            Finding(
                WARN,
                "discrete_state",
                f"disabled by override, but preset {preset.name!r} discretizes state "
                "into the prompt",
                "state is not merely un-discretized, it is *dropped*: the model sees "
                "no proprioception, and any delta->absolute output step becomes a "
                "no-op because it has no base to add.",
            )
        )
    return findings


def check_checkpoint(
    model_dir,
    robot: str,
    *,
    norm_key: str = "actions",
    discrete_state: Optional[bool] = None,
    image_keys: Optional[Sequence[str]] = None,
    action_dim: Optional[int] = None,
    tokenizer_path=None,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
) -> Tuple[Finding, ...]:
    """Check a checkpoint directory against the preset that will serve it.

    Takes the *resolved* serving knobs (what the server settled on after applying
    its overrides), not the raw command line, so what is checked is what will
    actually run. Returns findings sorted most-severe first; an empty tuple is
    impossible because passing checks are reported as :data:`INFO`.

    ``checkpoint_format`` / ``asset_id`` / ``norm_stats`` must match the values
    passed to :func:`~apxinf_robo.presets.build_robot_policy`.
    """
    _, findings = inspect_for_robot(
        model_dir,
        robot,
        norm_key=norm_key,
        discrete_state=discrete_state,
        image_keys=image_keys,
        action_dim=action_dim,
        tokenizer_path=tokenizer_path,
        checkpoint_format=checkpoint_format,
        asset_id=asset_id,
        norm_stats=norm_stats,
    )
    return findings


def inspect_for_robot(
    model_dir,
    robot: str,
    *,
    norm_key: str = "actions",
    discrete_state: Optional[bool] = None,
    image_keys: Optional[Sequence[str]] = None,
    action_dim: Optional[int] = None,
    tokenizer_path=None,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
) -> Tuple[CheckpointReport, Tuple[Finding, ...]]:
    """:func:`check_checkpoint`, also returning the engine's raw report.

    For callers that want the resolved facts -- tokenizer path, detected layout,
    observed widths -- and not only the rendered verdict.
    """
    model_dir = Path(model_dir)
    preset = get_robot_preset(robot)
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    keys = preset.image_keys if image_keys is None else tuple(image_keys)

    if (model_dir / "config.json").is_file():
        from apxinf.policies.auto import _read_model_type

        model_type = _read_model_type(model_dir)
        if model_type in _PI0_FAST_TYPES | _GR00T_TYPES | {"qwen_drive"}:
            return _registered_policy_report(model_dir, preset, model_type, keys)

    # Only inspect the statistics this deployment will actually read. A server
    # that drops proprioception asks about no state key at all, so no state
    # finding is produced -- a warning about numbers nobody loads is noise.
    state_expected = preset.state_dim if discrete else None
    report = inspect_checkpoint(
        model_dir,
        norm_key=norm_key,
        state_norm_key="state" if state_expected is not None else None,
        tokenizer_path=tokenizer_path,
        checkpoint_format=checkpoint_format,
        asset_id=asset_id,
        norm_stats=norm_stats,
    )

    findings: List[Finding] = list(report.findings)
    for role, expected in (("action", preset.action_width), ("state", state_expected)):
        facts = report.norm.get(role)
        if facts is not None and expected is not None:
            findings.extend(_check_width(facts, preset, expected))
    findings.extend(_check_cameras(preset, keys))
    findings.extend(_check_overrides(preset, action_dim=action_dim, discrete=discrete))

    return report, sort_findings(findings)

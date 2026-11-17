"""Dual-YAM body for RLinf's pi05_yam_joint config: geometry and its transform.

The arithmetic is :mod:`~apxinf_robo.embodiments.yam_transform`; nothing here imports an
inference engine except :func:`build_yam_policy`, which loads ApxInf when called.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..engines import load_engine_policy
from ..protocol import Policy, apply_transform
from .base import Embodiment
from .yam_transform import YAM_DELTA_MASK, YAM_DIM, YamTransform, yam_absolute_actions  # noqa: F401

__all__ = ["YAM_BODY", "build_yam_policy"]


def build_yam_policy(
    model_dir,
    *,
    state_key: str,
    image_keys: Sequence[str],
    prompt_key: str = "prompt",
    action_dim: int = YAM_DIM,
    metadata: Optional[dict] = None,
    **load_kwargs: Any,
) -> Policy:
    """Load the RLinf delta-joint YAM policy on ApxInf and expose absolute qpos14."""
    if action_dim != YAM_DIM:
        raise ValueError("YAM controller actions must have 14 values")
    base = load_engine_policy(
        "apxinf",
        model_dir,
        state_key=state_key,
        image_keys=tuple(image_keys),
        prompt_key=prompt_key,
        action_dim=YAM_DIM,
        **load_kwargs,
    )
    return apply_transform(
        base,
        YamTransform(state_key),
        metadata={"robot": "yam_joint", **(metadata or {})},
    )


YAM_BODY = Embodiment(
    name="yam",
    num_cameras=3,
    action_dim=YAM_DIM,
    state_dim=YAM_DIM,
    action_width=YAM_DIM,
    transform=YamTransform,
    engine_options={
        # Match OpenPI's float64 normalization. State values near a discretization
        # bin edge can otherwise produce different token ids.
        "apxinf": {"norm_dtype": "float64"},
    },
)

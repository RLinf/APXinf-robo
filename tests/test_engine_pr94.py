"""Cross-repository loader contract for the PR #94 engine pin."""

import json
from types import SimpleNamespace

import pytest

from apxinf_robo import engine


@pytest.mark.parametrize(
    ("precision", "variant"),
    [("auto", "auto"), ("bf16", "bf16"), ("fp8", "fp8_static"), ("int8", "int8_dynamic")],
)
def test_pi05_cli_precision_reaches_model_variant(tmp_path, monkeypatch, precision, variant):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    seen = []
    monkeypatch.setattr(
        engine, "require_apxinf",
        lambda: SimpleNamespace(
            AutoPolicy=SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append(kw))
        ),
    )
    engine.load_policy(tmp_path, precision=precision)
    assert seen == [{"model_variant": variant}]


def test_pi05_uses_upstream_model_type_fallback(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "", "model_type": "", "model": "pi05"})
    )
    seen = []
    monkeypatch.setattr(
        engine, "require_apxinf",
        lambda: SimpleNamespace(
            AutoPolicy=SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append(kw))
        ),
    )
    engine.load_policy(tmp_path, precision="bf16")
    assert seen == [{"model_variant": "bf16"}]


def test_qwen_drive_options_pass_through(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen_drive"}))
    seen = []
    monkeypatch.setattr(
        engine, "require_apxinf",
        lambda: SimpleNamespace(
            AutoPolicy=SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append(kw))
        ),
    )
    engine.load_policy(
        tmp_path, model_variant="bf16", mode="direct_planning", planner="/ckpt/planner"
    )
    assert seen == [{"model_variant": "bf16", "mode": "direct_planning", "planner": "/ckpt/planner"}]


def test_walloss_keeps_precision(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_vl",
                "experts": [{}, {}],
                "action_hidden_size": 1024,
                "noise_scheduler": {},
            }
        )
    )
    from apxinf.policies.auto import _read_model_type

    assert _read_model_type(tmp_path) == "walloss"
    seen = []
    monkeypatch.setattr(
        engine, "require_apxinf",
        lambda: SimpleNamespace(
            AutoPolicy=SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append(kw))
        ),
    )
    engine.load_policy(tmp_path, precision="bf16")
    assert seen == [{"precision": "bf16"}]


def test_random_handle_uses_model_runner(monkeypatch):
    import sys

    seen = []
    monkeypatch.setitem(
        sys.modules,
        "apxinf_py",
        SimpleNamespace(ModelRunner=SimpleNamespace(random=lambda **kw: seen.append(kw))),
    )
    engine.load_random_model(model="pi05", precision="fp8")
    assert seen == [{"model": "pi05", "model_variant": "fp8_static"}]

"""Exercise the evaluator CLI-to-checkpoint path without CUDA or a simulator.

``--norm-stats`` is the one evaluator flag that changes what the *model* computes
rather than how the rollout is scored, so a silent no-op here would look like a
model regression. These tests pin it to the loader call and to the checkpoint
layer that consumes it.
"""

import json
from types import SimpleNamespace

import pytest

from apxinf.checkpoints import detect_checkpoint
from apxinf_robo import engine
from apxinf_robo.cli import eval_libero


def parse(tmp_path, *extra, backend="in-process"):
    return eval_libero.parse_args([
        "--backend", backend, "--precision", "bf16",
        "--model-dir", str(tmp_path),
        "--results-jsonl", str(tmp_path / "results.jsonl"),
        "--summary-json", str(tmp_path / "summary.json"), *extra,
    ])


def test_explicit_norm_stats_reaches_checkpoint_loader(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    stats = tmp_path / "external-norms.json"
    stats.write_text(json.dumps({"norm_stats": {
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "actions": {"q01": [2.0] * 7, "q99": [4.0] * 7},
    }}))
    loaded = []

    def load(model_dir, **options):
        loaded.append(detect_checkpoint(model_dir, norm_stats=options["norm_stats"]))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(engine, "load_policy", load)
    eval_libero.InProcessBackend(parse(tmp_path, "--norm-stats", str(stats)))
    assert loaded[0].norm_stats == stats
    assert loaded[0].normalization.action.values["q01"] == (2.0,) * 7


def test_omitted_norm_stats_preserves_checkpoint_defaults(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(engine, "load_policy", load)
    eval_libero.InProcessBackend(parse(tmp_path))
    assert "norm_stats" not in options


def test_libero_wire_keys_come_from_the_preset(monkeypatch, tmp_path):
    """The evaluator must not become a second definition of "LIBERO keys"."""
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(engine, "load_policy", load)
    eval_libero.InProcessBackend(parse(tmp_path))

    convention = eval_libero.libero_convention()
    assert options["image_keys"] == convention.image_keys
    assert options["state_key"] == convention.state_key
    assert options["prompt_key"] == convention.prompt_key


def test_websocket_norm_stats_is_rejected(tmp_path, capsys):
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        parse(tmp_path, "--norm-stats", str(stats), backend="websocket")
    assert exc.value.code == 2
    assert "pass it to `apxinf-robo serve`" in capsys.readouterr().err


def test_missing_norm_stats_is_rejected_before_rollout(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(tmp_path, "--norm-stats", str(tmp_path / "missing.json"))
    assert exc.value.code == 2
    assert "--norm-stats must name an existing file" in capsys.readouterr().err

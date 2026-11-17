"""``build_robot_policy`` resolves preset defaults and honours overrides.

The preset table is the single definition of a robot's wire contract, so what
matters is that the *right* values reach the policy loader -- not that the engine
runs. The loader is stubbed out, which keeps this test CPU-only and instant.
"""

from __future__ import annotations


class _CapturedPolicy:
    metadata: dict = {}


def _stub_loader(monkeypatch, captured: dict):
    """Replace the engine loader that ``build_robot_policy`` calls."""
    from apxinf_robo import presets

    def fake_load(engine, model_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured.update(kwargs)
        return _CapturedPolicy()

    monkeypatch.setattr(presets, "load_engine_policy", fake_load)


def test_franka_libero_leaves_checkpoint_state_semantics_unchanged(monkeypatch):
    from apxinf_robo.presets import build_robot_policy

    captured: dict = {}
    _stub_loader(monkeypatch, captured)

    build_robot_policy("franka_libero", "/checkpoint")

    assert captured["model_dir"] == "/checkpoint"
    assert captured["image_keys"] == (
        "observation/image",
        "observation/wrist_image",
    )
    assert captured["state_key"] == "observation/state"
    assert captured["prompt_key"] == "prompt"
    assert captured["action_dim"] == 7
    # The LIBERO convention takes no position on state discretization, so the
    # preset must not invent one: the checkpoint's own setting stands.
    assert "discrete_state" not in captured


def test_robot_policy_user_overrides_take_precedence(monkeypatch):
    from apxinf_robo.presets import build_robot_policy

    captured: dict = {}
    _stub_loader(monkeypatch, captured)

    build_robot_policy(
        "franka_libero",
        "/checkpoint",
        image_keys=("front", "hand"),
        state_key="state",
        prompt_key="instruction",
        action_dim=9,
        discrete_state=True,
    )

    assert captured["image_keys"] == ("front", "hand")
    assert captured["state_key"] == "state"
    assert captured["prompt_key"] == "instruction"
    assert captured["action_dim"] == 9
    assert captured["discrete_state"] is True


def test_metadata_publishes_the_served_wire_contract(monkeypatch):
    """A client reads the contract off metadata rather than assuming one."""
    from apxinf_robo.presets import build_robot_policy

    captured: dict = {}
    _stub_loader(monkeypatch, captured)

    build_robot_policy("franka_libero", "/checkpoint")

    metadata = captured["metadata"]
    assert metadata["robot"] == "franka_libero"
    assert metadata["state_dim"] == 8
    # Slot order is baked into the weights: key i fills model view slot i.
    assert metadata["robot_slots"] == [
        ["base_0_rgb", "observation/image"],
        ["left_wrist_0_rgb", "observation/wrist_image"],
    ]

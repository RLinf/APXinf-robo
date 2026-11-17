"""Wire-key lookup for observation dicts, with no engine behind it.

Clients spell a key flat (``obs["observation/image"]``) or nested one level
(``obs["images"]["cam_high"]``), and robot transforms address both with one string. This
is the same resolution rule ApxInf applies to its own pipelines (``apxinf.processors
.transforms.lookup_key``), restated here so the robot layer needs no engine to read or
rewrite a state vector. A flat hit always wins; only an absent key is split on ``/``.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["lookup_key", "set_key"]

_NO_DEFAULT = object()


def lookup_key(observation: Mapping[str, Any], key: str, default: Any = _NO_DEFAULT) -> Any:
    """Resolve ``key`` flat first, then as a ``/`` path. ``KeyError`` without a default."""
    if key in observation:
        return observation[key]
    if "/" in key:
        node: Any = observation
        for part in key.split("/"):
            if not isinstance(node, Mapping) or part not in node:
                break
            node = node[part]
        else:
            return node
    if default is _NO_DEFAULT:
        raise KeyError(key)
    return default


def set_key(observation: Mapping[str, Any], key: str, value: Any) -> dict:
    """Return a copy of ``observation`` with ``key`` set, mirroring :func:`lookup_key`.

    Writes flat when the key exists flat (or has no ``/``); otherwise walks the nested
    path, shallow-copying only the mappings along it. The caller's dict is never mutated.
    """
    if key in observation or "/" not in key:
        return {**observation, key: value}
    head, *rest = key.split("/")
    node = observation.get(head)
    if isinstance(node, Mapping):
        return {**observation, head: set_key(node, "/".join(rest), value)}
    return {**observation, key: value}

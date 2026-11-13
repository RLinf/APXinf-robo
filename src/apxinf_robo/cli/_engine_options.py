"""Options only ApxInf can honour, and how the CLIs refuse them under another engine.

``serve`` and ``eval-libero`` both accept ApxInf's loader options and both can run
the S600 engine, which has none of them. An option the S600 engine ignores is worse
than one it rejects: ``--seed`` looks like it makes a run reproducible, and nothing
in the S600 runtime reads it. So each CLI lists the options and refuses any that was
given.

"Given" means "not at its default", and the default of every listed option is
``None`` or ``False``. An option whose real default is a value (``--device cuda:0``)
must therefore parse to ``None`` and have the value applied *after* this check, only
on the ApxInf path; otherwise ``--device cuda:0`` and no ``--device`` look the same
and the check could never fire for it.
"""

from __future__ import annotations

from typing import Iterable, List


def explicit_flags(args, names: Iterable[str]) -> List[str]:
    """The ``--flags`` among ``names`` that the user set on ``args``."""
    return [
        "--" + name.replace("_", "-")
        for name in names
        if getattr(args, name, None) is not None and getattr(args, name, None) is not False
    ]


def refusal(flags: List[str], engine: str) -> str:
    """The error text for ``flags`` given under ``engine``."""
    return f"{', '.join(flags)} applies to ApxInf, not to --engine {engine}"

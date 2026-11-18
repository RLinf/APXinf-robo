"""What can go wrong, split by who has to act on it.

:class:`RequestError` is the caller's fault and leaves the policy usable. Everything
else is an :class:`DroboticsError`. ``exit_code`` is what a service should exit with when
that error stops it.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["DeploymentError", "RequestError", "RuntimeFailure", "DroboticsError", "StartupFailure"]


def _render(summary: str, problems: Sequence[str]) -> str:
    return "\n".join([summary, *(f"  - {problem}" for problem in problems)])


class RequestError(ValueError):
    """An observation does not meet the contract. The policy can still serve others."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(_render("the observation does not meet the S600 contract:", self.problems))


class DroboticsError(RuntimeError):
    """The S600 policy cannot start or cannot go on."""

    exit_code = 1


class DeploymentError(DroboticsError):
    """The SDK, model or worker on disk is not what the contract pins."""

    exit_code = 2

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(_render("the deployment does not match the contract:", self.problems))


class StartupFailure(DroboticsError):
    """The deployment matched, but the runtime did not come up."""

    exit_code = 3


class RuntimeFailure(DroboticsError):
    """The worker crashed, timed out, or returned something unusable.

    A restart would also restart the runtime's noise sequence, so nothing here
    retries. The policy stays failed, and the operator restarts the process.
    """

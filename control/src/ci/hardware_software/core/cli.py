"""
CLI invocation helpers for HITL tests.

Always invokes via typer CliRunner so output is captured and test failures
print a useful command + output snippet, not just an exit code.
"""

from __future__ import annotations

import logging

from typer.testing import CliRunner

from control.pseti import app

logger = logging.getLogger(__name__)


def invoke(runner: CliRunner, args: list[str], *, check: bool = True) -> str:
    """Invoke a pseti command, returning stdout.

    Args:
        runner: Session-scoped typer CliRunner from conftest.
        args: Command arguments (e.g. ["start", "-y", "--no-hv"]).
        check: If True, assert exit_code == 0 and raise on failure.

    Returns:
        stdout as a single string.

    Raises:
        AssertionError: when check=True and exit_code != 0.
    """
    cmd_str = "pseti " + " ".join(args)
    logger.info("invoke: %s", cmd_str)
    result = runner.invoke(app, args)
    if check and result.exit_code != 0:
        raise AssertionError(
            f"`{cmd_str}` failed (exit {result.exit_code}):\n{result.output}"
        )
    return result.output


def invoke_ok(runner: CliRunner, args: list[str]) -> str:
    """Invoke and assert success. Alias for invoke(..., check=True)."""
    return invoke(runner, args, check=True)

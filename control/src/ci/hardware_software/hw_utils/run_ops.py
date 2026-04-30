"""
Observing run operations — primitive entrypoints.
Drives pseti start/stop via the Typer CLI runner so the full transaction
machinery (ledger, locking, gRPC calls) is exercised.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def start_run_via_cli(nsecs: int = 30, no_hv: bool = True, **kwargs) -> None:
    """
    Invoke 'pseti start' with the given parameters.

    Transitions: HV_ON / ACQ_CONFIGURED → ACQUIRING.
    The --no-hv flag is the default for safety during framework development;
    tests that need HV must explicitly pass no_hv=False.
    """
    from typer.testing import CliRunner

    from control.pseti import app

    runner = CliRunner()
    args = ["start", "--nsecs", str(nsecs)]
    if no_hv:
        args.append("--no-hv")
    logger.info("start_run_via_cli: invoking pseti %s", " ".join(args))
    result = runner.invoke(app, args, catch_exceptions=False)
    if result.exit_code != 0:
        raise RuntimeError(f"pseti start failed (exit {result.exit_code}):\n{result.stdout}")


def stop_run_via_cli(**kwargs) -> None:
    """
    Invoke 'pseti stop --yes'.

    Transitions: ACQUIRING → ACQ_CONFIGURED.
    """
    from typer.testing import CliRunner

    from control.pseti import app

    runner = CliRunner()
    logger.info("stop_run_via_cli: invoking pseti stop --yes")
    result = runner.invoke(app, ["stop", "--yes"], catch_exceptions=False)
    if result.exit_code != 0:
        raise RuntimeError(f"pseti stop failed (exit {result.exit_code}):\n{result.stdout}")

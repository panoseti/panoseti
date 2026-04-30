"""
Helpers for running 'pseti' CLI commands during HITL tests.
Ensures --yes is always passed and captures output for debugging.
"""

from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)

def start_run(yes: bool = True, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run 'pseti start --yes'."""
    cmd = ["pseti", "start"]
    if yes:
        cmd.append("--yes")
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)

def stop_run(yes: bool = True) -> subprocess.CompletedProcess:
    """Run 'pseti stop --yes'."""
    cmd = ["pseti", "stop"]
    if yes:
        cmd.append("--yes")
    
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)

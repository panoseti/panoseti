"""Tier 1 (Unit): CLI alias loading tests.

Verifies:
- cfg, led, start, stop, status aliases work as expected.
- Importing pseti doesn't install a root handler.
"""
from __future__ import annotations

import logging

from typer.testing import CliRunner

from control.pseti import app

runner = CliRunner()

def test_cfg_alias_help() -> None:
    result = runner.invoke(app, ["cfg", "--help"])
    assert result.exit_code == 0
    # Match some text from config help
    assert "Configure observatory hardware" in result.stdout

def test_led_alias_help() -> None:
    result = runner.invoke(app, ["obs", "led", "--help"])
    assert result.exit_code == 0
    assert "Inspect the run state ledger" in result.stdout

def test_start_stop_status_aliases_help() -> None:
    for cmd in ["start", "stop", "status"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        if cmd == "start":
             assert "start a recording run" in result.stdout
        elif cmd == "stop":
             assert "Stop an in-progress recording run" in result.stdout
        elif cmd == "status":
             assert "Show the status of a PSETI recording run" in result.stdout

def test_no_root_handler_on_pseti_import() -> None:
    from rich.logging import RichHandler
    root_logger = logging.getLogger()
    assert not any(isinstance(h, RichHandler) for h in root_logger.handlers)

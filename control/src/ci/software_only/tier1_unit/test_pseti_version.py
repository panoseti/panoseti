"""Unit tests for `pseti --version` (control/pseti.py)."""

from __future__ import annotations

from importlib.metadata import version

from typer.testing import CliRunner

from control.pseti import app

runner = CliRunner()


def test_version_flag_prints_installed_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert version("pseti-ctl") in result.output


def test_version_flag_is_eager_and_short_circuits_invalid_options() -> None:
    # --no-env is a bool flag; passing it a value it doesn't accept would only
    # fail argument parsing if --version weren't eager and short-circuiting first.
    result = runner.invoke(app, ["--version", "--no-env", "extra-arg"])
    assert result.exit_code == 0
    assert version("pseti-ctl") in result.output

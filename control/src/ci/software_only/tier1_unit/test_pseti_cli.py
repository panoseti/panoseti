"""
test_pseti_cli.py — Unit tests for the main pseti CLI entry point.
Verifies that all subcommands (recursively) can display their help without errors.
"""

from __future__ import annotations

import click
import pytest
import typer
from typer.testing import CliRunner

from control.pseti import app

runner = CliRunner()


def get_all_subcommand_paths(node: click.Group | click.Command, path: list[str] | None = None) -> list[list[str]]:
    """Recursively find all command paths in the CLI tree."""
    if path is None:
        path = []
    paths = []
    if path:
        paths.append(path)
    
    if isinstance(node, click.Group):
        # We need a context to list commands for PanoLazyGroup/GrpcLazyGroup
        ctx = click.Context(node)
        for cmd_name in node.list_commands(ctx):
            cmd = node.get_command(ctx, cmd_name)
            if cmd:
                paths.extend(get_all_subcommand_paths(cmd, [*path, cmd_name]))
    return paths


# Generate the list of all command paths at collection time
# This includes pseti start, pseti show paths, pseti grpc telemetry log, etc.
root_click_group = typer.main.get_command(app)
ALL_COMMAND_PATHS = get_all_subcommand_paths(root_click_group)


def test_pseti_top_level_help():
    """Verify pseti --help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PSETI Observatory Control CLI" in result.output


@pytest.mark.parametrize("command_path", ALL_COMMAND_PATHS, ids=lambda p: " ".join(p))
def test_pseti_subcommand_help(command_path: list[str]):
    """
    Verify every subcommand in the entire tree can display help.
    This ensures lazy-loading works and no sub-module has broken imports.
    """
    # Use -h for conciseness
    args = [*list(command_path), "-h"]
    result = runner.invoke(app, args)
    
    cmd_str = "pseti " + " ".join(command_path)
    assert result.exit_code == 0, f"Command '{cmd_str} -h' failed with exit code {result.exit_code}\nOutput: {result.output}"
    
    # Common help markers
    assert "Usage:" in result.output or "Options:" in result.output
    # Ensure no lazy-load error strings
    assert "Error loading command" not in result.output

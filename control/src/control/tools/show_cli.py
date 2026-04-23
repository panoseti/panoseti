from __future__ import annotations

import os
from typing import Annotated

import typer
from panoseti_grpc.util.cli import display_tree_callback
from rich.console import Console
from rich.table import Table

from control.utils.paths import PanoPaths

app = typer.Typer(help="Inspect and visualize PSETI system state.", no_args_is_help=True)


@app.callback()
def show_callback(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree for system inspection.", callback=display_tree_callback)] = False
) -> None:
    """Inspect and visualize system state."""
    pass


@app.command(name="paths")
def show_paths() -> None:
    """
    Display the current resolved paths for all key directories.
    
    To override any of these paths, set the corresponding environment variable.
    
    Examples:
      $ pseti show paths
      $ export PSETI_CONFIG=/tmp/custom_configs
      $ export PSETI_TMP=./local_tmp
    """
    console = Console()
    table = Table(title="PSETI Path Mapping")
    table.add_column("Directory", style="cyan", no_wrap=True)
    table.add_column("Resolved Path", style="green", overflow="fold")
    table.add_column("Override Variable", style="magenta", no_wrap=True)
    table.add_column("Source", style="blue", no_wrap=True)

    paths = [
        ("Repository Root", PanoPaths.software_root_dir(), "PSETI_ROOT"),
        ("Control Package", PanoPaths.base_dir(), "PSETI_CONTROL"),
        ("Configs", PanoPaths.config_dir(), "PSETI_CONFIG"),
        ("Transient (tmp)", PanoPaths.tmp_dir(), "PSETI_TMP"),
        ("Quabos Metadata", PanoPaths.quabos_dir(), "PSETI_QUABOS"),
        ("Logs", PanoPaths.logs_dir(), "PSETI_LOGS"),
        ("Firmware", PanoPaths.firmware_dir(), "PSETI_FIRMWARE"),
        ("White Rabbit", PanoPaths.wr_dir(), "PSETI_WR"),
        ("DAQ Scripts", PanoPaths.daq_scripts_dir(), "PSETI_DAQ_SCRIPTS"),
    ]

    for name, path, var in paths:
        source = f"[bold]{var}[/bold]" if os.environ.get(var) else "Default"
        table.add_row(name, str(path), var, source)

    console.print(table)
    console.print("\n[dim]Tip: Overriding PSETI_ROOT or PSETI_CONTROL will shift the default locations of all sub-directories.[/dim]")


@app.command(name="commands")
def show_commands(
    ctx: typer.Context,
    tree: Annotated[bool, typer.Option("--tree", "-t", help="Display the command tree.", callback=display_tree_callback)] = True
) -> None:
    """
    Display a tree-like view of all available PSETI commands and subcommands.
    """
    # This command is now just an alias for -t at this level
    pass


if __name__ == "__main__":
    app()

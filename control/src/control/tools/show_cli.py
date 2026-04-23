import os

import click
import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from control.utils.paths import PanoPaths

app = typer.Typer(help="Inspect and visualize PSETI system state.", no_args_is_help=True)


@app.command(name="paths")
def show_paths():
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


def _walk_commands(node: click.Group | click.Command, tree: Tree):
    """Recursively walk click commands and add them to the rich tree."""
    if isinstance(node, click.Group):
        # Sort subcommands for consistent output
        for cmd_name in sorted(node.list_commands(click.Context(node))):
            # get_command handles our lazy-loading logic automatically
            cmd = node.get_command(click.Context(node), cmd_name)
            if cmd:
                help_text = cmd.help.split("\n")[0] if cmd.help else ""
                # Truncate help text if too long
                if len(help_text) > 60:
                    help_text = help_text[:57] + "..."
                
                branch = tree.add(f"[bold cyan]{cmd_name}[/] [dim]— {help_text}[/]")
                _walk_commands(cmd, branch)


@app.command(name="commands")
def show_commands(ctx: typer.Context):
    """
    Display a tree-like view of all available PSETI commands and subcommands.
    """
    console = Console()
    
    # We need to find the root app. In Typer, it's accessible via ctx.parent
    root_ctx: click.Context | None = ctx
    while root_ctx and root_ctx.parent:
        root_ctx = root_ctx.parent
        
    if not root_ctx:
        return

    root_tree = Tree("[bold reverse] PSETI CLI Structure [/]")
    _walk_commands(root_ctx.command, root_tree)
    
    console.print("\n", root_tree, "\n")


if __name__ == "__main__":
    app()

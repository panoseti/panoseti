"""Read-only ledger inspection CLI — `pseti obs ledger`."""
from __future__ import annotations

import pathlib
from glob import glob
from typing import Annotated

import typer

app = typer.Typer(
    name="ledger",
    help="Inspect the run state ledger (read-only).",
    no_args_is_help=True,
)


def _ledger_path() -> pathlib.Path:
    from control.utils.paths import PanoPaths
    return PanoPaths.runs_dir() / "run_state.toml"


@app.command("show")
def show(
    run_name: Annotated[str | None, typer.Argument(help="Run name (omit for current ledger)")] = None,
) -> None:
    """Pretty-print the current run ledger (or a specific run's archived copy)."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        console = Console()
        _rich_available = True
    except ImportError:
        _rich_available = False

    if run_name is None:
        target = _ledger_path()
        if not target.exists():
            typer.echo("No active ledger found.", err=True)
            raise typer.Exit(1)
    else:
        from control.utils import config_file
        try:
            daq_config = config_file.get_daq_config()
            data_dir = daq_config.head_node_data_dir
        except Exception:
            typer.echo("Could not load daq_config.json; cannot locate archived ledger.", err=True)
            raise typer.Exit(1) from None
        candidates = glob(f"{data_dir}/_aborted/{run_name}*/stale_run_state.toml")
        if not candidates:
            typer.echo(f"No archived ledger found for run '{run_name}'.", err=True)
            raise typer.Exit(1)
        target = pathlib.Path(sorted(candidates)[-1])

    content = target.read_text()
    if _rich_available:
        console.print(f"[dim]{target}[/dim]")
        console.print(Syntax(content, "toml", theme="monokai", line_numbers=False))
    else:
        typer.echo(str(target))
        typer.echo(content)


# Default command when called with no subcommand: show current ledger.
@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """Show the current run ledger."""
    if ctx.invoked_subcommand is None:
        show(run_name=None)


@app.command("path")
def path_cmd() -> None:
    """Print the absolute path to the current ledger file."""
    typer.echo(str(_ledger_path()))


@app.command("history")
def history() -> None:
    """List archived (aborted/stale) ledger files under head_node_data_dir/_aborted/."""
    from control.utils import config_file
    try:
        daq_config = config_file.get_daq_config()
        data_dir = daq_config.head_node_data_dir
    except Exception:
        typer.echo("Could not load daq_config.json to find archived ledgers.", err=True)
        raise typer.Exit(1) from None

    candidates = sorted(glob(f"{data_dir}/_aborted/*/stale_run_state.toml"))
    if not candidates:
        typer.echo("No archived ledgers found.")
        return
    for c in candidates:
        typer.echo(c)

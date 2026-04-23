import os

import typer
from rich.console import Console
from rich.table import Table

from control.utils.paths import PanoPaths

app = typer.Typer(help="Manage and visualize PSETI directory paths.", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def show():
    """
    Display the current resolved paths for all key directories.
    
    To override any of these paths, set the corresponding environment variable.
    
    Examples:
      $ pseti path show
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


@app.command()
def init():
    """Ensure the required directory structure exists on disk."""
    PanoPaths.ensure_dirs()
    typer.echo("Standard PSETI directories initialized.")


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt.")
):
    """Clean transient files in the tmp/ directory (locks, state)."""
    tmp = PanoPaths.tmp_dir()
    if not tmp.exists():
        typer.echo("tmp/ directory does not exist. Nothing to clean.")
        return

    if not force and not typer.confirm(f"Are you sure you want to clean transient files in {tmp}?"):
        raise typer.Abort()

    files_to_remove = ["panoseti_control.lock", "run_state.toml", "current_run", "panoseti_transfer.lock"]
    for f in files_to_remove:
        path = tmp / f
        if path.exists():
            path.unlink()
            typer.echo(f"Removed {f}")

    typer.echo("Cleanup complete.")


if __name__ == "__main__":
    app()

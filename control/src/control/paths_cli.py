import os

import typer
from rich.console import Console
from rich.table import Table

from control.utils.paths import PanoPaths

app = typer.Typer(help="Manage and visualize PANOSETI directory paths.", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def show():
    """Display the current resolved paths for all key directories."""
    console = Console()
    table = Table(title="PANOSETI Path Mapping")
    table.add_column("Directory", style="cyan")
    table.add_column("Resolved Path", style="green")
    table.add_column("Source", style="magenta")

    paths = [
        ("Base (PANOSETI_HOME)", PanoPaths.base_dir(), "PANOSETI_HOME" if os.environ.get("PANOSETI_HOME") else "Default (CWD)"),
        ("Configs", PanoPaths.config_dir(), "PANOSETI_CONFIG_DIR" if os.environ.get("PANOSETI_CONFIG_DIR") else "Default"),
        ("Transient (tmp)", PanoPaths.tmp_dir(), "PANOSETI_TMP_DIR" if os.environ.get("PANOSETI_TMP_DIR") else "Default"),
        ("Quabos Metadata", PanoPaths.quabos_dir(), "PANOSETI_QUABOS_DIR" if os.environ.get("PANOSETI_QUABOS_DIR") else "Default"),
        ("Logs", PanoPaths.logs_dir(), "PANOSETI_LOGS_DIR" if os.environ.get("PANOSETI_LOGS_DIR") else "Default"),
        ("Firmware", PanoPaths.firmware_dir(), "PANOSETI_FIRMWARE_DIR" if os.environ.get("PANOSETI_LOGS_DIR") else "Default"),
    ]

    for name, path, source in paths:
        table.add_row(name, str(path), source)

    console.print(table)


@app.command()
def init():
    """Ensure the required directory structure exists on disk."""
    PanoPaths.ensure_dirs()
    typer.echo("Standard PANOSETI directories initialized.")


@app.command()
def clean():
    """Clean transient files in the tmp/ directory (locks, state)."""
    tmp = PanoPaths.tmp_dir()
    if not tmp.exists():
        typer.echo("tmp/ directory does not exist. Nothing to clean.")
        return

    files_to_remove = ["panoseti_control.lock", "run_state.toml", "current_run", "panoseti_transfer.lock"]
    for f in files_to_remove:
        path = tmp / f
        if path.exists():
            path.unlink()
            typer.echo(f"Removed {f}")

    typer.echo("Cleanup complete.")


if __name__ == "__main__":
    app()

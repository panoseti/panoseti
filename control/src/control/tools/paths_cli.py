
import typer

from control.utils.paths import PanoPaths

app = typer.Typer(help="Manage transient PSETI directory files.", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


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

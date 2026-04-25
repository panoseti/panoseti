"""Transfer queue CLI — `pseti obs transfer`."""
from __future__ import annotations

import typer

app = typer.Typer(name="transfer", help="Inspect and manage the transfer queue.")


@app.command()
def status(run: str | None = typer.Argument(None, help="Run name to inspect")) -> None:
    """Show transfer daemon health and queue status."""
    from control.transfer.service import get_queue_summary

    summary = get_queue_summary()
    for bucket, runs in summary.items():
        count = len(runs)
        typer.echo(f"{bucket}: {count} job(s)")
        for r in runs:
            typer.echo(f"  - {r}")

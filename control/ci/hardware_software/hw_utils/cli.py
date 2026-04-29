"""
HITL (Hardware-in-the-Loop) test orchestration CLI.
"""

import typer
from rich.console import Console

app = typer.Typer(help="Hardware-in-the-Loop (HITL) physical lab tests", no_args_is_help=True)
console = Console()

@app.command(name="plan")
def hw_plan():
    """Dry-run: print the batch plan + estimated wall clock. No hardware touched."""
    console.print("[yellow]HITL Plan: Not yet implemented.[/yellow]")

@app.command(name="run")
def hw_run(
    ctx: typer.Context,
    dev: bool = typer.Option(False, "--dev", help="Dev mode: no power cycle, keep running."),
    hw_class: str = typer.Option(None, "--class", "-c", help="Filter by test class."),
    hw_state: str = typer.Option(None, "--state", "-s", help="Filter by required state."),
):
    """Run HITL tests with state-aware batching."""
    console.print("[yellow]HITL Run: Not yet implemented.[/yellow]")

@app.command(name="preflight")
def hw_preflight():
    """Run only tests marked as preflight."""
    console.print("[yellow]HITL Preflight: Not yet implemented.[/yellow]")

@app.command(name="status")
def hw_status():
    """Report current believed hardware state + reachability."""
    console.print("[yellow]HITL Status: Not yet implemented.[/yellow]")

@app.command(name="safe-down")
def hw_safe_down():
    """Manually invoke emergency teardown (driving to safe state)."""
    console.print("[yellow]HITL Safe-Down: Not yet implemented.[/yellow]")

if __name__ == "__main__":
    app()

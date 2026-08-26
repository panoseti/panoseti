"""Transfer queue CLI — `pseti xfr`."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.text import Text

from control.transfer.models import TransferStatus
from control.utils.paths import PanoPaths

app = typer.Typer(
    name="transfer",
    help="Inspect and manage the background transfer queue.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daemon_pid() -> int | None:
    """Return the transfer daemon PID from its pid file, or None."""
    pid_path = PanoPaths.state_dir() / "transfer" / "daemon.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _daemon_heartbeat_age() -> float | None:
    """Return seconds since last heartbeat, or None if no heartbeat file."""
    hb = PanoPaths.state_dir() / "transfer" / "daemon.heartbeat"
    if not hb.exists():
        return None
    try:
        return time.time() - float(hb.read_text().strip())
    except (ValueError, OSError):
        return None


def _daemon_alive() -> bool:
    age = _daemon_heartbeat_age()
    return age is not None and age < 30.0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("stat")
def stat(
    run: Annotated[str | None, typer.Argument(help="Run name to inspect")] = None,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Periodically refresh the status display.")] = False,
    interval: Annotated[float, typer.Option("--interval", "-i", help="Refresh interval in seconds (requires --watch).")] = 1.0,
) -> None:
    """Show transfer daemon health and queue summary."""
    from control.transfer.service import get_queue_summary

    console = Console()

    def generate_layout() -> Group:
        """Generates a single Rich renderable combining text and progress bars."""
        renderables: list[Text | Progress] = []

        # 1. Daemon Health
        pid = _daemon_pid()
        age = _daemon_heartbeat_age()

        if pid is None:
            renderables.append(Text.from_markup("Daemon: [bold red]NOT RUNNING[/] (no pid file)"))
        elif age is None:
            renderables.append(Text.from_markup(f"Daemon: pid={pid}  heartbeat: [bold yellow]absent[/]"))
        elif age < 30:
            renderables.append(Text.from_markup(f"Daemon: [bold green]RUNNING[/]  pid={pid}  heartbeat {age:.0f}s ago"))
        else:
            renderables.append(Text.from_markup(f"Daemon: [bold yellow]STALE[/]    pid={pid}  heartbeat {age:.0f}s ago (>30s)"))

        renderables.append(Text(""))

        # 2. Queue Summary
        summary = get_queue_summary()
        bucket_colors = {
            "pending": "cyan", 
            "active": "blue", 
            "completed": "green", 
            "failed": "red"
        }

        queue_lines = []
        for bucket in TransferStatus:
            runs = summary.get(bucket, [])
            color = bucket_colors.get(bucket, "white")
            queue_lines.append(f"  [{color}]{bucket:12s}[/] {len(runs):3d} job(s)")
            
            if run:
                if run in runs:
                    queue_lines.append(f"    [bold green]✓[/] {run}")
            elif runs:
                for r in runs:
                    queue_lines.append(f"    - {r}")
                    
        renderables.append(Text.from_markup("\n".join(queue_lines)))

        # 3. Active Progress Bars
        active_runs = summary.get(TransferStatus.ACTIVE, [])
        if active_runs:
            renderables.append(Text.from_markup("\n[bold]Active Transfers:[/bold]"))
            active_d = PanoPaths.transfer_queue_dir() / TransferStatus.ACTIVE
            
            # Instantiate without context manager for embedding in Group
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
            
            for r_name in active_runs:
                sidecars = list(active_d.glob(f"{r_name}.*.progress.json"))
                if not sidecars:
                    progress.add_task(f"[yellow]{r_name}[/] (starting...)", total=None)
                    continue
                
                for s in sidecars:
                    node_ip = s.name.split(".")[1]
                    try:
                        with open(s) as f:
                            data = json.load(f)
                        progress.add_task(
                            f"{r_name} [{node_ip}]",
                            total=100,
                            completed=data.get("pct", 0),
                        )
                    except Exception:
                        progress.add_task(f"{r_name} [{node_ip}] (loading...)", total=None)
                        
            renderables.append(progress)

        return Group(*renderables)

    # Render once and exit if not watching
    if not watch:
        console.print(generate_layout())
        return

    # Use Rich's Live view to flawlessly update the terminal in-place
    try:
        # We wrap the dynamically generated layout in the Live view
        with Live(generate_layout(), console=console, refresh_per_second=1.0/interval) as live:
            while True:
                time.sleep(interval)
                # On each tick, regenerate the layout and update the live display
                live.update(generate_layout())
    except KeyboardInterrupt:
        pass

@app.command()
def queue(
    bucket: Annotated[str, typer.Argument(help="pending | active | completed | failed")] = "pending",
) -> None:
    """List jobs in a queue bucket (default: pending)."""
    from control.transfer.queue import TransferQueue

    console = Console()
    if bucket not in TransferStatus:
        console.print(f"[bold red]Unknown bucket '{bucket}'.[/] Choose from: {', '.join(TransferStatus)}")
        raise typer.Exit(1)
        
    tq = TransferQueue()
    jobs = tq.list_jobs(TransferStatus(bucket))
    if not jobs:
        console.print(f"No jobs in {bucket}/")
        return
        
    for j in jobs:
        console.print(f"- {j}")


@app.command()
def retry(run_name: Annotated[str, typer.Argument(help="Run name to retry")]) -> None:
    """Move a failed job back to pending/ (resets attempt counter)."""
    from control.transfer.queue import TransferQueue

    console = Console()
    tq = TransferQueue()
    
    if tq.retry(run_name):
        console.print(f"[bold green]Success:[/bold green] Moved {run_name} from failed/ → pending/")
    else:
        console.print(f"[bold red]Error:[/bold red] No failed job found for '{run_name}'")
        raise typer.Exit(1)


@app.command()
def clean(run_name: Annotated[str, typer.Argument(help="Run name to remove from pending/ or failed/")]) -> None:
    """Remove a job from pending/ or failed/ and try to delete its run directory on each DAQ node.

    Checks pending/ first, then failed/. The job never reached (or never completed)
    manifest generation, so there's no verified manifest to checksum against -- this
    uses CLEANUP_FULL (removes the whole run directory on each DAQ node) rather than
    the daemon's own selective, manifest-gated cleanup. DAQ-node deletion is
    best-effort: a node that's unreachable or errors out is reported as a warning but
    does not undo the local queue removal.
    """
    from control.transfer.daq_control import cleanup_daq_nodes
    from control.transfer.queue import TransferQueue

    console = Console()
    tq = TransferQueue()

    job = tq.clean(run_name)
    if job is None:
        console.print(f"[bold red]Error:[/bold red] No pending or failed job found for '{run_name}'")
        raise typer.Exit(1)

    console.print(f"[bold green]Success:[/bold green] Removed {run_name} from the transfer queue")

    errors = asyncio.run(cleanup_daq_nodes(job, mode="CLEANUP_FULL"))
    if errors:
        console.print("[bold yellow]Warning:[/bold yellow] Could not clean up on all DAQ nodes:")
        for err in errors:
            console.print(f"  [yellow]{err}[/yellow]")


@app.command("start")
def start_daemon() -> None:
    """Start the transfer daemon (idempotent: no-op if already running)."""
    from control.utils import util

    if _daemon_alive():
        typer.echo("Transfer daemon is already running.")
        return
    util.start_daemon([sys.executable, "-m", "control.transfer"])
    typer.echo("Transfer daemon started.")


@app.command("stop")
def stop_daemon(
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds to wait for graceful exit.")] = 60.0,
) -> None:
    """Send SIGTERM to the transfer daemon and wait for graceful exit."""
    pid = _daemon_pid()
    if pid is None:
        typer.echo("Transfer daemon is not running (no pid file).")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        typer.echo(f"Process {pid} not found — daemon may have already exited.")
        return
        
    typer.echo(f"Sent SIGTERM to pid={pid}. Waiting up to {timeout:.0f}s...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            typer.echo("Transfer daemon exited.")
            return
        time.sleep(1.0)
        
    typer.echo(f"Daemon still running after {timeout:.0f}s; sending SIGKILL.")
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


@app.command()
def tail(
    lines: Annotated[int, typer.Option("-n", help="Number of lines to show.")] = 40,
    follow: Annotated[bool, typer.Option("-f", help="Follow the log (like tail -f).")] = False,
) -> None:
    """Tail the transfer daemon log."""
    log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
    log_file = log_dir / "transfer_daemon.log"
    if not log_file.exists():
        stderr_log = log_dir / "stderr.log"
        typer.echo(
            f"Log file not found: {log_file}\n"
            f"  Check also: {stderr_log} (backstop for pre-logger crashes)",
            err=True,
        )
        raise typer.Exit(1)
        
    flags = ["-f"] if follow else []
    os.execvp("tail", ["tail", f"-n{lines}", *flags, str(log_file)])


@app.command()
def verify(run_name: Annotated[str, typer.Argument(help="Run name to verify")]) -> None:
    """Run manifest verification on a completed run (no state changes)."""
    from control.transfer.verify import verify_manifest
    
    console = Console()

    try:
        from control.utils import config_file
        daq_config = config_file.get_daq_config()
        head_data_dir = daq_config.head_node_data_dir
    except Exception:
        console.print("[bold red]Error:[/] Could not load daq_config.json; pass data dir manually.")
        raise typer.Exit(1) from None

    import pathlib
    run_dir = pathlib.Path(head_data_dir) / run_name
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/] Run directory not found: {run_dir}")
        raise typer.Exit(1)

    # 1. New format: dp_manifest.node_*.algo_*.txt
    manifest_files = list(run_dir.glob("dp_manifest.node_*.algo_*.txt"))
    # 2. Legacy format: manifest.<algo>
    for algo in ("blake3", "xxh3_128", "sha256"):
        legacy = run_dir / f"manifest.{algo}"
        if legacy.exists() and legacy not in manifest_files:
            manifest_files.append(legacy)
      
    if not manifest_files:
        console.print(f"[bold red]Error:[/] No manifest files found in {run_dir}")
        raise typer.Exit(1)
    
    all_ok = True
    for mf in manifest_files:
        ok, errs = verify_manifest(mf, run_dir)
        
        if ok:
            console.print(f"  {mf.name}: [bold green]OK[/]")
        else:
            console.print(f"  {mf.name}: [bold red]FAILED[/]")
            
        for e in errs:
            console.print(f"    [red]{e}[/]")
            
        if not ok:
            all_ok = False

    if not all_ok:
        raise typer.Exit(1)

    if not all_ok:
        raise typer.Exit(1)
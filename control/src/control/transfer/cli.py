"""Transfer queue CLI — `pseti obs transfer`."""
from __future__ import annotations

import os
import signal
import sys
import time
from typing import Annotated

import typer

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

@app.command()
def status(run: Annotated[str | None, typer.Argument(help="Run name to inspect")] = None) -> None:
    """Show transfer daemon health and queue summary."""
    from control.transfer.service import get_queue_summary

    pid = _daemon_pid()
    age = _daemon_heartbeat_age()

    if pid is None:
        typer.echo("Daemon: NOT RUNNING (no pid file)")
    elif age is None:
        typer.echo(f"Daemon: pid={pid}  heartbeat: absent")
    elif age < 30:
        typer.echo(f"Daemon: RUNNING  pid={pid}  heartbeat {age:.0f}s ago")
    else:
        typer.echo(f"Daemon: STALE    pid={pid}  heartbeat {age:.0f}s ago (>30s)")

    typer.echo("")
    summary = get_queue_summary()
    for bucket in ("pending", "active", "completed", "failed"):
        runs = summary.get(bucket, [])
        typer.echo(f"  {bucket:12s} {len(runs):3d} job(s)")
        if run:
            if run in runs:
                typer.echo(f"    ✓ {run}")
        elif runs:
            for r in runs:
                typer.echo(f"    - {r}")


@app.command()
def queue(
    bucket: Annotated[str, typer.Argument(help="pending | active | completed | failed")] = "pending",
) -> None:
    """List jobs in a queue bucket (default: pending)."""
    from control.transfer.queue import TransferQueue

    valid = ("pending", "active", "completed", "failed")
    if bucket not in valid:
        typer.echo(f"Unknown bucket '{bucket}'. Choose from: {', '.join(valid)}", err=True)
        raise typer.Exit(1)
    tq = TransferQueue()
    jobs = tq.list_jobs(bucket)
    if not jobs:
        typer.echo(f"No jobs in {bucket}/")
        return
    for j in jobs:
        typer.echo(j)


@app.command()
def retry(run_name: Annotated[str, typer.Argument(help="Run name to retry")]) -> None:
    """Move a failed job back to pending/ (resets attempt counter)."""
    from control.transfer.queue import TransferQueue

    tq = TransferQueue()
    if tq.retry(run_name):
        typer.echo(f"Moved {run_name} from failed/ → pending/")
    else:
        typer.echo(f"No failed job found for '{run_name}'", err=True)
        raise typer.Exit(1)


@app.command("start")
def start_daemon() -> None:
    """Start the transfer daemon (idempotent: no-op if already running)."""
    from control.utils import util

    if _daemon_alive():
        typer.echo("Transfer daemon is already running.")
        return
    util.start_daemon(["python", "-m", "control.transfer"])
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
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@app.command()
def tail(
    lines: Annotated[int, typer.Option("-n", help="Number of lines to show.")] = 40,
    follow: Annotated[bool, typer.Option("-f", help="Follow the log (like tail -f).")] = False,
) -> None:
    """Tail the transfer daemon log."""
    log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
    log_file = log_dir / "current.log"
    if not log_file.exists():
        typer.echo(f"Log file not found: {log_file}", err=True)
        raise typer.Exit(1)
    flags = ["-f"] if follow else []
    os.execvp("tail", ["tail", f"-n{lines}", *flags, str(log_file)])


@app.command()
def verify(run_name: Annotated[str, typer.Argument(help="Run name to verify")]) -> None:
    """Run manifest verification on a completed run (no state changes)."""
    from control.transfer.verify import verify_manifest

    daq_config_path = None
    try:
        from control.utils import config_file
        daq_config = config_file.get_daq_config()
        head_data_dir = daq_config.head_node_data_dir
    except Exception:
        typer.echo("Could not load daq_config.json; pass data dir manually.", err=True)
        raise typer.Exit(1)

    import pathlib
    run_dir = pathlib.Path(head_data_dir) / run_name
    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}", err=True)
        raise typer.Exit(1)

    found_any = False
    all_ok = True
    for algo in ("blake3", "xxh3_128", "sha256"):
        mf = run_dir / f"manifest.{algo}"
        if not mf.exists():
            continue
        found_any = True
        ok, errs = verify_manifest(mf, run_dir)
        status_str = "OK" if ok else "FAILED"
        typer.echo(f"  manifest.{algo}: {status_str}")
        for e in errs:
            typer.echo(f"    {e}", err=True)
        if not ok:
            all_ok = False

    if not found_any:
        typer.echo(f"No manifest files found in {run_dir}", err=True)
        raise typer.Exit(1)

    if not all_ok:
        raise typer.Exit(1)

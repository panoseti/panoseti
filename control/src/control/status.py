#! /usr/bin/env python3

# show the status of a recording run

import asyncio
import time
import sys
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from panoseti_grpc.telemetry.logger import get_logger
from panoseti_grpc.util.cli import BaseLazyGroup

from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


# ---------- logging setup ----------
def ut_now_str() -> str:
    """Return the current time as a formatted UTC string."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.Status", log_dir=str(log_dir), grpc_enabled=True)


# ---------- helpers ----------

def _transfer_daemon_age() -> float | None:
    hb = PanoPaths.state_dir() / "transfer" / "daemon.heartbeat"
    if not hb.exists():
        return None
    try:
        return time.time() - float(hb.read_text().strip())
    except (ValueError, OSError):
        return None


def _queue_counts() -> dict[str, int]:
    from control.transfer.queue import TransferQueue
    try:
        tq = TransferQueue()
        return {b: len(tq.list_jobs(b)) for b in ("pending", "active", "completed", "failed")}
    except Exception:
        return {}


def _local_summary() -> list[str]:
    lines: list[str] = []
    state_mgr = RunStateManager()
    ledger = state_mgr.load_state()

    if ledger:
        lines.append(f"Run:     {ledger.run_name}")
        lines.append(f"Status:  {ledger.status}")
        lines.append(f"Started: {ledger.start_time}")
    else:
        run_name = util.read_run_name()
        lines.append(f"Run:     {run_name or '(none)'}")
        lines.append("Status:  (no active ledger)")

    lines.append(f"HK rec:  {'running' if util.is_hk_recorder_running() else 'stopped'}")

    age = _transfer_daemon_age()
    if age is None:
        lines.append("Daemon:  not running (no heartbeat)")
    elif age < 30:
        lines.append(f"Daemon:  RUNNING  (heartbeat {age:.0f}s ago)")
    else:
        lines.append(f"Daemon:  STALE    (heartbeat {age:.0f}s ago)")

    counts = _queue_counts()
    if counts:
        q_str = "  ".join(f"{k}={v}" for k, v in counts.items())
        lines.append(f"Queue:   {q_str}")

    return lines


async def _remote_summary(daq_config=None, clients: dict[str, Any] | None = None) -> list[str]:
    """Query each DAQ node via gRPC and return detailed rows per node."""
    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    import os

    lines: list[str] = []
    
    if daq_config is None:
        try:
            daq_config = config_file.get_daq_config()
            network_config = config_file.get_network_config()
            util.attach_daq_config(daq_config, network_config)
        except Exception as e:
            return [f"ERROR loading daq_config: {e}"]

    async def _do_query(client: AsyncDaqControlClient, data_dir: str):
        return await asyncio.wait_for(
            client.StatusDaq({
                "data_dir": data_dir,
                "check_hashpipe_running": True,
                "check_disk_usage": True,
                "check_run_dirs": True,
            }),
            timeout=5.0,
        )

    async def probe(node: object) -> str:
        from control.utils.pydantic_config_models import DaqNode
        assert isinstance(node, DaqNode)
        ip_str = str(node.ip_addr)
        
        if not node.module_ids:
            return f"  • {ip_str:<14} | (no modules configured)"
            
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, daq_config)
        
        try:
            # Reuse cached client if available, otherwise spin up a new context
            if clients is not None and ip_str in clients:
                ok, status = await _do_query(clients[ip_str], node.data_dir)
            else:
                async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                    ok, status = await _do_query(client, node.data_dir)

            if not ok:
                return f"  • {ip_str:<14} | gRPC returned not-ok"

            # 1. Modules
            mod_str = f"Mods: {len(node.module_ids)}"

            # 2. Hashpipe Status & PID
            hp_state = "RUNNING" if status.get("hashpipe_running") else "STOPPED"
            pid = status.get('hashpipe_pid')
            pid_str = f" (PID:{pid})" if pid else ""
            hp_str = f"HP: {hp_state}{pid_str}"
            
            # 3. Disk Usage (in KB with commas)
            disk_usage = status.get("disk_usage", {})
            free_bytes = disk_usage.get("free_disk_space", 0)
            total_bytes = disk_usage.get("total_disk_space", 0)
            
            if total_bytes > 0:
                free_gb = free_bytes / 2**30
                total_gb = total_bytes / 2**30
                disk_str = f"Free: {free_gb:,.3f} /{total_gb:,.3f} GiB"
            else:
                disk_str = "Disk: ?"

            # 4. Run Directory (Base name only)
            run_dirs = status.get("run_dirs", [])
            if run_dirs:
                # Strip parent path, leave only the run folder name
                run_name = os.path.basename(run_dirs[0])
                runs_str = f"{run_name}"
            else:
                runs_str = "none"

            # Combine everything cleanly
            # Adjusted string padding lengths to accommodate larger KB numbers and PIDs
            return f"  • {ip_str:<14} | {hp_str:<16} | {disk_str:<28} | {runs_str}"

        except Exception as exc:
            return f"  • {ip_str:<14} | UNREACHABLE: {exc}"

    tasks = [probe(n) for n in daq_config.daq_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        lines.append(str(r))
    return lines


async def _sweep_summary() -> list[str]:
    """Full reachability sweep: Quabo ping + gRPC checks. Read-only."""
    from control.start import _check_daq_reachability, _quabo_reachability_report

    lines: list[str] = ["=== Network Sweep ==="]
    try:
        daq_config = config_file.get_daq_config()
        quabo_uids = config_file.get_quabo_uids()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)
    except Exception as e:
        return [f"ERROR loading config: {e}"]

    # DAQ gRPC
    try:
        await _check_daq_reachability(daq_config)
        lines.append("DAQ gRPC:  OK — all nodes reachable")
    except Exception as e:
        lines.append(f"DAQ gRPC:  FAILED — {e}")

    # Quabo reachability report
    try:
        results = await _quabo_reachability_report(quabo_uids, network_config)
        total = len(results)
        reachable = [r for r in results if r.reachable]
        up_count = len(reachable)
        
        if up_count == total:
            lines.append(f"Quabos:    OK    — {up_count}/{total} reachable")
        elif up_count > 0:
            down_uids = [r.uid for r in results if not r.reachable]
            lines.append(f"Quabos:    DEGRADED — {up_count}/{total} reachable; down: {', '.join(down_uids)}")
        else:
            lines.append(f"Quabos:    DOWN  — 0/{total} reachable")
    except Exception as e:
        lines.append(f"Quabos:    ERROR — {e}")

    return lines


def _render(local: list[str], remote: list[str] | None, sweep: list[str] | None) -> str:
    parts = [f"[{ut_now_str()}]", "--- Head Node ---", *local]
    if remote is not None:
        parts += ["", "--- DAQ Nodes ---", *remote]
    if sweep is not None:
        parts += ["", *sweep]
    return "\n".join(parts)


def status(no_remote: bool = False, sweep_mode: bool = False) -> None:
    """Synchronous single-shot status render."""
    local = _local_summary()
    remote_lines = None if no_remote else asyncio.run(_remote_summary())
    sweep_lines = asyncio.run(_sweep_summary()) if sweep_mode else None
    typer.echo(_render(local, remote_lines, sweep_lines))

async def _watch_loop(interval: float, no_remote: bool) -> None:
    """Continuously fetch and display status, maintaining persistent gRPC connections."""
    from panoseti_grpc.daq_control.client import AsyncDaqControlClient
    import sys

    try:
        daq_config = config_file.get_daq_config()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)
    except Exception as e:
        typer.echo(f"ERROR loading config for watch loop: {e}")
        return

    # Use an exit stack to gracefully manage the lifespans of all our gRPC clients
    async with AsyncExitStack() as stack:
        clients: dict[str, AsyncDaqControlClient] = {}
        
        if not no_remote:
            for node in daq_config.daq_nodes:
                if node.module_ids:
                    grpc_host, grpc_port = util.daq_grpc_endpoint(node, daq_config)
                    client = AsyncDaqControlClient(host=grpc_host, port=grpc_port)
                    await stack.enter_async_context(client)
                    # Ensure IP is saved as a string in the cache lookup
                    clients[str(node.ip_addr)] = client

        # Clear the entire screen on initialization
        sys.stdout.write("\033[2J")

        while True:
            local = _local_summary()
            remote_lines = None if no_remote else await _remote_summary(daq_config, clients)
            output = _render(local, remote_lines, None)
            
            # Cursor to top-left (\033[H) + Clear to end of screen (\033[0J)
            sys.stdout.write(f"\033[H\033[0J{output}\n")
            sys.stdout.flush()
            
            await asyncio.sleep(interval)



class StatLazyGroup(BaseLazyGroup):
    """
    Lazy-loading group for status subcommands like ledger.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        lazy_mapping = {
            "ledger": ("control.tools.ledger_cli", "app", "Inspect the run state ledger (read-only)."),
        }
        super().__init__(*args, lazy_mapping=lazy_mapping, **kwargs)

app = typer.Typer(
    cls=StatLazyGroup,
    help="Show observatory health, acquisition status, and ledger.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    no_remote: Annotated[bool, typer.Option("--no-remote", help="Skip querying DAQ nodes via gRPC.")] = False,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Refresh continuously.")] = False,
    interval: Annotated[float, typer.Option("--interval", "-i", help="Refresh interval in seconds (requires --watch).")] = 1.0,
) -> None:
    """Query and display the current status of the observatory control plane."""
    if ctx.invoked_subcommand is not None:
        return

    if watch:
        try:
            # We now pass the interval variable directly
            asyncio.run(_watch_loop(interval=interval, no_remote=no_remote))
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        status(no_remote=no_remote)


@app.command("sweep")
def sweep_cmd() -> None:
    """Full network reachability sweep (Quabo ping + DAQ gRPC). Read-only."""
    lines = asyncio.run(_sweep_summary())
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()
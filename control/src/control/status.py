#! /usr/bin/env python3

# show the status of a recording run

import asyncio
import time
from contextlib import AsyncExitStack, suppress
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from panoseti_grpc.telemetry.logger import get_logger
from panoseti_grpc.util.cli import BaseLazyGroup
from rich.console import Console
from rich.live import Live
from rich.text import Text

from control.transfer.models import TransferStatus
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
        return {b: len(tq.list_jobs(b)) for b in TransferStatus}
    except Exception:
        return {}


def _local_summary() -> list[Text]:
    lines: list[Text] = []
    state_mgr = RunStateManager()
    ledger = state_mgr.load_state()

    if ledger:
        lines.append(Text.assemble(("Run:     ", "bold"), f"{ledger.run_name}"))
        status_style = "green" if ledger.status == "recording" else "yellow"
        lines.append(Text.assemble(("Status:  ", "bold"), (f"{ledger.status}", status_style)))
        lines.append(Text.assemble(("Started: ", "bold"), f"{ledger.start_time}"))
    else:
        run_name = util.read_run_name()
        lines.append(Text.assemble(("Run:     ", "bold"), f"{run_name or '(none)'}"))
        lines.append(Text.assemble(("Status:  ", "bold"), ("(no active ledger)", "dim")))

    hk_running = util.is_hk_recorder_running()
    hk_style = "green" if hk_running else "red"
    lines.append(Text.assemble(("HK rec:  ", "bold"), (f"{'running' if hk_running else 'stopped'}", hk_style)))

    age = _transfer_daemon_age()
    if age is None:
        lines.append(Text.assemble(("Daemon:  ", "bold"), ("not running (no heartbeat)", "red")))
    elif age < 30:
        lines.append(Text.assemble(("Daemon:  ", "bold"), ("RUNNING", "green"), f"  (heartbeat {age:.0f}s ago)"))
    else:
        lines.append(Text.assemble(("Daemon:  ", "bold"), ("STALE", "yellow"), f"    (heartbeat {age:.0f}s ago)"))

    counts = _queue_counts()
    if counts:
        q_text = Text.assemble(("Queue:   ", "bold"))
        for k, v in counts.items():
            style = "green" if v == 0 else "yellow"
            q_text.append(f"{k}=")
            q_text.append(f"{v}", style=style)
            q_text.append("  ")
        lines.append(q_text)

    return lines


async def _remote_summary(daq_config: Any = None, clients: dict[str, Any] | None = None) -> list[Text]:
    """Query each DAQ node via gRPC and return detailed rows per node."""
    import os

    from panoseti_grpc.daq_control.client import AsyncDaqControlClient

    lines: list[Text] = []
    
    if daq_config is None:
        try:
            from control.utils import config_file, util
            daq_config = config_file.get_daq_config()
            network_config = config_file.get_network_config()
            util.attach_daq_config(daq_config, network_config)
        except Exception as e:
            return [Text(f"ERROR loading daq_config: {e}", style="red")]

    async def _do_query(client: AsyncDaqControlClient, data_dir: str) -> tuple[bool, dict[str, Any]]:
        return await asyncio.wait_for(
            client.StatusDaq({
                "data_dir": data_dir,
                "check_hashpipe_running": True,
                "check_disk_usage": True,
                "check_run_dirs": True,
            }),
            timeout=5.0,
        )

    async def probe(node: object) -> Text:
        from control.utils import util
        from control.utils.pydantic_config_models import DaqNode
        assert isinstance(node, DaqNode)
        ip_str = str(node.ip_addr)
        
        if not node.module_ids:
            return Text(f"  • {ip_str:<14} | (no modules configured)", style="dim")
            
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, daq_config)
        
        try:
            # Reuse cached client if available, otherwise spin up a new context
            if clients is not None and ip_str in clients:
                ok, status = await _do_query(clients[ip_str], node.data_dir)
            else:
                async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                    ok, status = await _do_query(client, node.data_dir)

            if not ok:
                return Text.assemble(f"  • {ip_str:<14} | ", ("gRPC returned not-ok", "red"))

            # 2. Hashpipe Status & PID
            hp_running = status.get("hashpipe_running")
            hp_state = "RUNNING" if hp_running else "STOPPED"
            hp_style = "green" if hp_running else "red"
            pid = status.get('hashpipe_pid')
            pid_str = f" (PID:{pid})" if pid else ""
            hp_text = Text.assemble((f"HP: {hp_state}", hp_style), f"{pid_str}")
            
            # 3. Disk Usage
            disk_usage = status.get("disk_usage", {})
            free_bytes = disk_usage.get("free_disk_space", 0)
            total_bytes = disk_usage.get("total_disk_space", 0)
            
            if total_bytes > 0:
                free_gb = free_bytes / 2**30
                total_gb = total_bytes / 2**30
                usage_ratio = free_bytes / total_bytes
                disk_style = "green" if usage_ratio > 0.2 else "yellow" if usage_ratio > 0.1 else "red"
                disk_str = f"Free: {free_gb:,.3f} /{total_gb:,.3f} GiB"
            else:
                disk_str = "Disk: ?"
                disk_style = "dim"

            # 4. Run Directory
            run_dirs = status.get("run_dirs", [])
            if run_dirs:
                run_name = os.path.basename(run_dirs[0])
                runs_str = f"{run_name}"
            else:
                runs_str = "none"

            # Combine
            return Text.assemble(
                f"  • {ip_str:<14} | ",
                hp_text,
                " " * max(0, 16 - len(hp_text.plain)),
                "| ",
                (disk_str, disk_style),
                " " * max(0, 28 - len(disk_str)),
                f" | {runs_str}"
            )

        except Exception as exc:
            return Text.assemble(f"  • {ip_str:<14} | ", (f"UNREACHABLE: {exc}", "red"))

    tasks = [probe(n) for n in daq_config.daq_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            lines.append(Text(f"ERROR probing node: {r}", style="red"))
        else:
            lines.append(r)
    return lines


async def _sweep_summary() -> list[Text]:
    """Full reachability sweep: Quabo ping + gRPC checks. Read-only."""
    from control.start import _check_daq_reachability, _quabo_reachability_report

    lines: list[Text] = [Text("=== Network Sweep ===", style="bold yellow")]
    try:
        daq_config = config_file.get_daq_config()
        quabo_uids = config_file.get_quabo_uids()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)
    except Exception as e:
        return [Text(f"ERROR loading config: {e}", style="red")]

    # DAQ gRPC
    try:
        await _check_daq_reachability(daq_config)
        lines.append(Text.assemble(("DAQ gRPC:  ", "bold"), ("OK — all nodes reachable", "green")))
    except Exception as e:
        lines.append(Text.assemble(("DAQ gRPC:  ", "bold"), (f"FAILED — {e}", "red")))

    # Quabo reachability report
    try:
        results = await _quabo_reachability_report(quabo_uids, network_config)
        total = len(results)
        reachable = [r for r in results if r.reachable]
        up_count = len(reachable)
        
        if up_count == total:
            lines.append(Text.assemble(("Quabos:    ", "bold"), (f"OK    — {up_count}/{total} reachable", "green")))
        elif up_count > 0:
            down_uids = [r.uid for r in results if not r.reachable]
            lines.append(Text.assemble(
                ("Quabos:    ", "bold"), 
                (f"DEGRADED — {up_count}/{total} reachable", "yellow"),
                f"; down: {', '.join(down_uids)}"
            ))
        else:
            lines.append(Text.assemble(("Quabos:    ", "bold"), (f"DOWN  — 0/{total} reachable", "red")))
    except Exception as e:
        lines.append(Text.assemble(("Quabos:    ", "bold"), (f"ERROR — {e}", "red")))

    return lines


def _render(local: list[Text], remote: list[Text] | None, sweep: list[Text] | None) -> Text:
    res = Text()
    res.append(f"[{ut_now_str()}]\n", style="dim")
    res.append("--- Head Node ---\n", style="bold blue")
    for line in local:
        res.append(line)
        res.append("\n")
    
    if remote is not None:
        res.append("\n--- DAQ Nodes ---\n", style="bold magenta")
        for line in remote:
            res.append(line)
            res.append("\n")
            
    if sweep is not None:
        res.append("\n")
        for line in sweep:
            res.append(line)
            res.append("\n")
    return res


def status(no_remote: bool = False, sweep_mode: bool = False) -> None:
    """Synchronous single-shot status render."""
    local = _local_summary()
    remote_lines = None if no_remote else asyncio.run(_remote_summary())
    sweep_lines = asyncio.run(_sweep_summary()) if sweep_mode else None
    
    console = Console()
    console.print(_render(local, remote_lines, sweep_lines))


async def _watch_loop(interval: float, no_remote: bool) -> None:
    """Continuously fetch and display status, maintaining persistent gRPC connections."""
    from panoseti_grpc.daq_control.client import AsyncDaqControlClient

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

        console = Console()
        
        # Use Rich's Live view.
        with Live(console=console, auto_refresh=True) as live:
            while True:
                local = _local_summary()
                remote_lines = None if no_remote else await _remote_summary(daq_config, clients)
                output = _render(local, remote_lines, None)
                
                live.update(output)
                
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
        with suppress(KeyboardInterrupt):
            # We now pass the interval variable directly
            asyncio.run(_watch_loop(interval=interval, no_remote=no_remote))
    else:
        status(no_remote=no_remote)


@app.command("sweep")
def sweep_cmd() -> None:
    """Full network reachability sweep (Quabo ping + DAQ gRPC). Read-only."""
    lines = asyncio.run(_sweep_summary())
    console = Console()
    for line in lines:
        console.print(line)


if __name__ == "__main__":
    app()
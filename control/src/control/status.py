#! /usr/bin/env python3

# show the status of a recording run

import asyncio
import time
from datetime import UTC, datetime
from typing import Annotated

import typer
from panoseti_grpc.telemetry.logger import get_logger

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


async def _remote_summary() -> list[str]:
    """Query each DAQ node via gRPC and return one row per node."""
    from panoseti_grpc.daq_control.client import AsyncDaqControlClient

    lines: list[str] = []
    try:
        daq_config = config_file.get_daq_config()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)
    except Exception as e:
        return [f"ERROR loading daq_config: {e}"]

    async def probe(node: object) -> str:
        from control.utils.pydantic_config_models import DaqNode
        assert isinstance(node, DaqNode)
        if not node.module_ids:
            return f"  {node.ip_addr}  (no modules)"
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        try:
            async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                ok, status = await asyncio.wait_for(
                    client.StatusDaq({
                        "data_dir": node.data_dir,
                        "check_hashpipe_running": True,
                        "check_disk_usage": True,
                        "check_run_dirs": False,
                    }),
                    timeout=5.0,
                )
            if not ok:
                return f"  {node.ip_addr}  gRPC returned not-ok"
            hp = "hashpipe=RUNNING" if status.get("hashpipe_running") else "hashpipe=stopped"
            pid = f"pid={status.get('hashpipe_pid', '?')}" if status.get("hashpipe_running") else ""
            vols = status.get("vols", {})
            free_strs = [
                f"{name}:{v.get('free', 0) / 1e9:.1f}GB"
                for name, v in vols.items()
            ] if vols else []
            disk = "  disk=" + ",".join(free_strs) if free_strs else ""
            return f"  {node.ip_addr}:{grpc_port}  {hp} {pid}{disk}".strip()
        except Exception as exc:
            return f"  {node.ip_addr}  UNREACHABLE: {exc}"

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


def status(remote: bool = False, sweep_mode: bool = False) -> None:
    """Synchronous single-shot status render."""
    local = _local_summary()
    remote_lines = asyncio.run(_remote_summary()) if remote else None
    sweep_lines = asyncio.run(_sweep_summary()) if sweep_mode else None
    typer.echo(_render(local, remote_lines, sweep_lines))


from panoseti_grpc.util.cli import BaseLazyGroup
from typing import Any

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
    remote: Annotated[bool, typer.Option("--remote", help="Also query DAQ nodes via gRPC.")] = False,
    watch: Annotated[bool, typer.Option("--watch", help="Refresh continuously.")] = False,
    interval: Annotated[float, typer.Option("--interval", help="Refresh interval in seconds (--watch).")] = 5.0,
) -> None:
    """Query and display the current status of the observatory control plane."""
    if ctx.invoked_subcommand is not None:
        return

    if watch:
        try:
            while True:
                local = _local_summary()
                remote_lines = asyncio.run(_remote_summary()) if remote else None
                output = _render(local, remote_lines, None)
                import os
                os.system("clear")
                typer.echo(output)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    else:
        status(remote=remote)


@app.command("remote")
def remote_cmd(
    watch: Annotated[bool, typer.Option("--watch", help="Refresh continuously.")] = False,
    interval: Annotated[float, typer.Option("--interval", help="Refresh interval in seconds.")] = 5.0,
) -> None:
    """Query each DAQ node via gRPC (auto-resolves port-forwarding)."""
    if watch:
        try:
            while True:
                local = _local_summary()
                remote_lines = asyncio.run(_remote_summary())
                output = _render(local, remote_lines, None)
                import os
                os.system("clear")
                typer.echo(output)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    else:
        local = _local_summary()
        remote_lines = asyncio.run(_remote_summary())
        typer.echo(_render(local, remote_lines, None))


@app.command("sweep")
def sweep_cmd() -> None:
    """Full network reachability sweep (Quabo ping + DAQ gRPC). Read-only."""
    lines = asyncio.run(_sweep_summary())
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()

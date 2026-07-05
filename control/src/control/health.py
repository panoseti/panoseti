"""
pseti health — unified all-systems-green check for the observatory.

Consolidates checks that otherwise live scattered across `pseti val`,
`pseti stat`, `pseti admin status`, and `pseti test hw check-env` into one
command with a single pass/fail verdict per category: config validity, WPS
power, Quabo network reachability, DAQ node + head node gRPC service health,
container status, and the transfer daemon.

Deliberately does NOT reuse control.utils.util.ping()/perform_network_ping_sweep
for Quabo reachability -- that UDP echo probe (opcode 0x82) has proven
unreliable against real hardware (confirmed live: reports Quabos DOWN
immediately after a TFTP round-trip to the same IP:port succeeded). Uses a
real TFTP get_flashuid() round-trip instead, the same mechanism `pseti uids`
already depends on for real work.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Unified health check for the observatory.", invoke_without_command=True)
console = Console()

# gRPC health-check service names, per proto full_name (package + service) --
# NOT the config_field/profile names ("daq_control"/"daq_data"/"telemetry")
# used elsewhere in this codebase. Confirmed live via
# {telemetry,daq_data,daq_control}_pb2.DESCRIPTOR.services_by_name[...].full_name
# rather than guessed -- daq_data's proto package is "daqdata", not
# "panoseti.daq_data" like the other two, so guessing consistently-prefixed
# names is exactly wrong here.
_SVC_TELEMETRY = "panoseti.telemetry.Telemetry"
_SVC_DAQ_DATA = "daqdata.DaqData"
_SVC_DAQ_CONTROL = "panoseti.daq_control.DaqControl"


def _check_quabo_tftp(real_ip: str, port: int) -> bool:
    """Reachability check via a real TFTP round-trip.

    Each call uses its own temp filename -- get_flashuid() defaults to a
    fixed 'flashuid' path in the cwd, which races when checked in parallel
    (confirmed live: concurrent callers hit "Failed to acquire write lock").
    """
    from control.driver.quabo_tftp import tftpw

    tmp_path = os.path.join(tempfile.gettempdir(), f"flashuid_{uuid.uuid4().hex}")
    try:
        tftpw(real_ip, port).get_flashuid(tmp_path)
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _check_config() -> tuple[bool, str]:
    from control.utils import config_file
    try:
        config_file.get_obs_config()
        config_file.get_daq_config()
        config_file.get_network_config()
        config_file.get_data_config()
        return True, "obs/daq/network/data configs all load and validate"
    except Exception as e:
        return False, str(e)


def _check_wps() -> list[tuple[str, bool, str]]:
    from control.power import quabo_power_query
    from control.utils import config_file
    from control.utils.pydantic_config_models import WpsConfig

    obs_config = config_file.get_obs_config()
    extra = obs_config.model_extra or {}
    results = []
    for key in [k for k in extra if "wps" in k.lower()]:
        wps_data = extra[key]
        wps = WpsConfig(**wps_data) if isinstance(wps_data, dict) else wps_data
        try:
            state = quabo_power_query(wps)
            results.append((key, state is not None, "reachable" if state is not None else "no response"))
        except Exception as e:
            results.append((key, False, str(e)))
    return results


def _check_quabos() -> list[tuple[str, bool]]:
    from control.utils import config_file, util

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    targets: list[tuple[str, str, int]] = []
    for dome in obs_config.domes:
        for module in dome.modules:
            for i in range(4):
                ip_ports = util.get_quabo_ip_port(module.ip_addr, i, network_config)
                quabo_ip = config_file.quabo_ip_addr(str(module.ip_addr), i)
                desc = f"{dome.name}: {quabo_ip} (Q{i})"
                targets.append((desc, str(ip_ports.ip_addr), ip_ports.reboot_port))

    if not targets:
        return []

    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        results = list(pool.map(lambda t: _check_quabo_tftp(t[1], t[2]), targets))
    return [(desc, ok) for (desc, _real_ip, _port), ok in zip(targets, results, strict=True)]


def _check_grpc_daq_nodes() -> list[tuple[str, bool, str]]:
    from panoseti_grpc.grpc_utils.health import HealthClient

    from control.utils import config_file, util

    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)

    results = []
    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        host, port = util.daq_grpc_endpoint(node, daq_config)
        try:
            hc = HealthClient(host=host, port=port)
            control_ok = hc.check(_SVC_DAQ_CONTROL, timeout=5.0)
            data_ok = hc.check(_SVC_DAQ_DATA, timeout=5.0)
            detail = f"daq_control={'up' if control_ok else 'down'} daq_data={'up' if data_ok else 'down'}"
            results.append((str(node.ip_addr), control_ok and data_ok, detail))
        except Exception as e:
            results.append((str(node.ip_addr), False, str(e)))
    return results


def _check_grpc_headnode() -> tuple[bool, str]:
    from panoseti_grpc.grpc_utils.health import HealthClient

    try:
        hc = HealthClient(host="localhost", port=50051)
        telemetry_ok = hc.check(_SVC_TELEMETRY, timeout=5.0)
        data_ok = hc.check(_SVC_DAQ_DATA, timeout=5.0)
        return telemetry_ok or data_ok, f"telemetry={'up' if telemetry_ok else 'down'} daq_data(gateway)={'up' if data_ok else 'down'}"
    except Exception as e:
        return False, str(e)


def _compose_ps_running(cmd: list[str]) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr.strip() or "compose ps failed")[:200]
    running = "running" in r.stdout.lower()
    return running, "" if running else "no running containers"


def _check_daqnode_containers() -> list[tuple[str, bool, str]]:
    from control.admin.cli import get_docker_context_for_node
    from control.utils import config_file
    from control.utils.paths import PanoPaths

    daq_config = config_file.get_daq_config()
    daqnode_compose = PanoPaths.software_root_dir() / "grpc" / "deploy" / "docker-compose.daqnode.yml"
    alloy_compose = PanoPaths.software_root_dir() / "grpc" / "deploy" / "alloy" / "docker-compose.alloy.yml"

    results = []
    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        host = str(node.ip_addr)
        context = get_docker_context_for_node(host)
        grpc_ok, grpc_detail = _compose_ps_running(
            ["docker", "--context", context, "compose", "-f", str(daqnode_compose), "ps", "--format", "json"]
        )
        alloy_ok, alloy_detail = _compose_ps_running(
            ["docker", "--context", context, "compose", "-f", str(alloy_compose), "ps", "--format", "json"]
        )
        detail = f"grpc={'up' if grpc_ok else 'down ' + grpc_detail} alloy={'up' if alloy_ok else 'down ' + alloy_detail}"
        results.append((host, grpc_ok and alloy_ok, detail))
    return results


def _check_headnode_containers() -> tuple[bool, str]:
    from control.utils.paths import PanoPaths

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    if not compose_file.exists():
        return False, f"{compose_file} not found"
    ok, detail = _compose_ps_running(["docker", "compose", "-f", str(compose_file), "ps", "--format", "json"])
    return ok, detail


def _check_transfer_daemon() -> tuple[bool, str]:
    from control.utils.paths import PanoPaths

    hb = PanoPaths.state_dir() / "transfer" / "daemon.heartbeat"
    if not hb.exists():
        return False, "no heartbeat file (daemon never started, or state dir is fresh)"
    try:
        age = time.time() - float(hb.read_text().strip())
    except (ValueError, OSError) as e:
        return False, str(e)
    if age < 30:
        return True, f"heartbeat {age:.0f}s ago"
    return False, f"heartbeat stale ({age:.0f}s ago)"


def _row(table: Table, category: str, name: str, ok: bool, detail: str) -> None:
    status = "[green]✔ UP[/green]" if ok else "[red]✖ DOWN[/red]"
    table.add_row(category, name, status, detail)


@app.callback(invoke_without_command=True)
def main(
    skip_quabos: Annotated[bool, typer.Option("--skip-quabos", help="Skip Quabo TFTP reachability (slowest check).")] = False,
    skip_containers: Annotated[bool, typer.Option("--skip-containers", help="Skip container status checks.")] = False,
) -> None:
    """Run every health check and print one pass/fail summary.

    Covers: config validity, WPS power reachability, Quabo network
    reachability (real TFTP round-trip, not the UDP echo probe used
    elsewhere -- see module docstring), DAQ node + head node gRPC service
    health, container status, and the transfer daemon.
    """
    table = Table(title="PSETI Observatory Health", show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True

    def mark(ok: bool) -> None:
        nonlocal all_ok
        if not ok:
            all_ok = False

    console.print("[dim]Checking configuration...[/dim]")
    ok, detail = _check_config()
    _row(table, "Config", "obs/daq/network/data", ok, detail)
    mark(ok)

    if ok:
        console.print("[dim]Checking WPS power...[/dim]")
        for name, wps_ok, detail in _check_wps():
            _row(table, "WPS", name, wps_ok, detail)
            mark(wps_ok)

        if not skip_quabos:
            console.print("[dim]Checking Quabo reachability (TFTP, parallel)...[/dim]")
            for name, quabo_ok in _check_quabos():
                _row(table, "Quabo", name, quabo_ok, "" if quabo_ok else "TFTP round-trip failed")
                mark(quabo_ok)

        console.print("[dim]Checking head node gRPC services...[/dim]")
        hn_ok, hn_detail = _check_grpc_headnode()
        _row(table, "gRPC", "headnode (localhost:50051)", hn_ok, hn_detail)
        mark(hn_ok)

        console.print("[dim]Checking DAQ node gRPC services...[/dim]")
        for name, daq_ok, detail in _check_grpc_daq_nodes():
            _row(table, "gRPC", f"DAQ {name}", daq_ok, detail)
            mark(daq_ok)

        if not skip_containers:
            console.print("[dim]Checking container status...[/dim]")
            hc_ok, hc_detail = _check_headnode_containers()
            _row(table, "Containers", "headnode", hc_ok, hc_detail)
            mark(hc_ok)

            for name, c_ok, detail in _check_daqnode_containers():
                _row(table, "Containers", f"DAQ {name}", c_ok, detail)
                mark(c_ok)

        console.print("[dim]Checking transfer daemon...[/dim]")
        td_ok, td_detail = _check_transfer_daemon()
        _row(table, "Transfer Daemon", "heartbeat", td_ok, td_detail)
        mark(td_ok)
    else:
        console.print("[yellow]Skipping hardware/service checks -- fix config errors first.[/yellow]")

    console.print(table)

    if all_ok:
        console.print("\n[bold green]✅ ALL SYSTEMS GREEN.[/bold green]")
    else:
        console.print("\n[bold red]❌ ISSUES FOUND[/bold red] -- see table above.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

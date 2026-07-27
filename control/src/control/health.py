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
            reachable = state is not None
            detail = f"reachable (power {'on' if state else 'off'})" if reachable else "no response"
            results.append((key, reachable, detail))
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
    """Probe the head node's gRPC services on the endpoint a real client would use.

    Resolving the port via ``util.resolve_grpc_port("headnode")`` (rather
    than a hardcoded 50051) is what makes this a genuine desync check: it
    tells us whether the server the operator's .env says should be
    listening on HEADNODE_GRPC_PORT actually is -- catching exactly the bug
    class where a server binds one port (stale TOML, forgotten --port-env)
    while every client still assumes the default.
    """
    from panoseti_grpc.grpc_utils.health import HealthClient

    from control.utils import util

    port = util.resolve_grpc_port("headnode")
    try:
        hc = HealthClient(host="localhost", port=port)
        telemetry_ok = hc.check(_SVC_TELEMETRY, timeout=5.0)
        data_ok = hc.check(_SVC_DAQ_DATA, timeout=5.0)
        return telemetry_ok or data_ok, f"telemetry={'up' if telemetry_ok else 'down'} daq_data(gateway)={'up' if data_ok else 'down'}"
    except Exception as e:
        return False, f"localhost:{port} -- {e}"


def _check_port_collision() -> list[tuple[str, bool, str]]:
    """Pure, no-network check for a co-located head+DAQ node port/data-dir collision.

    On a single-machine deployment (e.g. Lick), the head and DAQ unified
    servers both run with network_mode: host, so they MUST resolve to
    different ports -- if HEADNODE_GRPC_PORT and DAQNODE_GRPC_PORT resolve
    to the same value on a node that is local to the head, the two servers
    will fight over one port and one of them loses (see wiki_docs's
    "Co-locating Head Node and DAQ Node" section). Likewise DAQ_DATA_DIR and
    PSETI_DATA_DIR must not overlap, or the DAQ node's hashpipe output and
    the head node's own service state corrupt each other.

    Runs before any network I/O -- catches the misconfiguration that would
    otherwise surface later as "container keeps restarting" or silent data
    loss, per wiki_docs's debugging section for co-located nodes.
    """
    from control.utils import config_file, util

    results: list[tuple[str, bool, str]] = []
    try:
        daq_config = config_file.get_daq_config()
    except Exception as e:
        return [("config", False, f"could not load daq_config.json: {e}")]

    head_port = util.resolve_grpc_port("headnode")
    daq_port = util.resolve_grpc_port("daqnode")
    for node in daq_config.daq_nodes:
        if not util.is_local(node.ip_addr, daq_config):
            continue
        ok = head_port != daq_port
        results.append((
            str(node.ip_addr),
            ok,
            f"HEADNODE_GRPC_PORT={head_port} DAQNODE_GRPC_PORT={daq_port}"
            if ok else
            f"co-located with head node but HEADNODE_GRPC_PORT == DAQNODE_GRPC_PORT == {head_port} "
            "-- set distinct values in .env (see wiki_docs's co-location guide)",
        ))

        # DAQ_DATA_DIR/PSETI_DATA_DIR are compose-time env vars (see
        # docker-compose.daqnode.yml / docker-compose.headnode.yml), not
        # PanoPaths-resolved paths -- comparing them here is only meaningful
        # when both are actually set, i.e. an operator running pseti admin
        # deploy for a co-located node; skip silently otherwise rather than
        # inventing a comparison against unrelated defaults.
        daq_data_dir = os.environ.get("DAQ_DATA_DIR", str(node.data_dir))
        pseti_data_dir = os.environ.get("PSETI_DATA_DIR")
        if pseti_data_dir:
            dd_ok = os.path.normpath(daq_data_dir) != os.path.normpath(pseti_data_dir)
            results.append((
                f"{node.ip_addr} (data dirs)",
                dd_ok,
                f"DAQ_DATA_DIR={daq_data_dir} PSETI_DATA_DIR={pseti_data_dir}"
                if dd_ok else
                f"DAQ_DATA_DIR and PSETI_DATA_DIR are both {daq_data_dir!r} -- "
                "must be distinct or the transfer/cleanup pipeline can destroy head-node state",
            ))
    return results


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
        # Must match pseti admin's project name (admin/cli.py's deploy_node)
        # exactly -- `docker compose ps` without -p resolves its own default
        # project name (derived from the compose file's directory), which is
        # never what pseti admin actually deployed under. Confirmed live:
        # containers demonstrably Up reported "no running containers" here
        # before this fix, because it was querying the wrong project.
        project_name = f"pseti-daqnode-{host.replace('.', '-')}"
        grpc_ok, grpc_detail = _compose_ps_running(
            ["docker", "--context", context, "compose", "-p", project_name, "-f", str(daqnode_compose), "ps", "--format", "json"]
        )
        alloy_ok, alloy_detail = _compose_ps_running(
            ["docker", "--context", context, "compose", "-p", project_name, "-f", str(alloy_compose), "ps", "--format", "json"]
        )
        detail = f"grpc={'up' if grpc_ok else 'down ' + grpc_detail} alloy={'up' if alloy_ok else 'down ' + alloy_detail}"
        results.append((host, grpc_ok and alloy_ok, detail))
    return results


def _check_headnode_containers() -> tuple[bool, str]:
    from control.utils.paths import PanoPaths

    compose_file = PanoPaths.base_dir() / "deploy" / "docker-compose.headnode.yml"
    if not compose_file.exists():
        return False, f"{compose_file} not found"
    # -p pseti-headnode must match admin/cli.py's deploy_headnode() project
    # name -- same bug class as _check_daqnode_containers() above.
    ok, detail = _compose_ps_running(
        ["docker", "compose", "-p", "pseti-headnode", "-f", str(compose_file), "ps", "--format", "json"]
    )
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
    elsewhere -- see module docstring), co-located port/data-dir collisions,
    DAQ node + head node gRPC service health, container status, and the
    transfer daemon.
    """
    from control.utils import util

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

        console.print("[dim]Checking for co-located port/data-dir collisions...[/dim]")
        for name, pc_ok, detail in _check_port_collision():
            _row(table, "Port/Dir Collision", name, pc_ok, detail)
            mark(pc_ok)

        console.print("[dim]Checking head node gRPC services...[/dim]")
        hn_ok, hn_detail = _check_grpc_headnode()
        _row(table, "gRPC", f"headnode (localhost:{util.resolve_grpc_port('headnode')})", hn_ok, hn_detail)
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

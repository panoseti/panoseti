from __future__ import annotations

import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
)

console = Console()

# Pydantic Validation

## Validation graph
def print_compact_config(config_name: str, config_obj: Any) -> None:
    """Prints a configuration object but collapses massive lists like module_ids."""
    import copy
    
    # Dump model to dict for easier manipulation of the compact view
    compact_dict = copy.deepcopy(config_obj.model_dump())

    if config_name.lower() == 'daq':
        for node in compact_dict.get('daq_nodes', []):
            if isinstance(node.get('module_ids'), list):
                # Compress [0, 1, 2... 255] into a readable string
                ids = node['module_ids']
                if len(ids) > 5:
                    node['module_ids'] = f"[{ids[0]}, {ids[1]} ... {len(ids)} total IDs ... {ids[-1]}]"

    console.print(Panel(f"[bold green]Parsed {config_name} Config[/bold green]"))
    pprint(compact_dict, expand_all=True)



def _check_tcp_port(ip: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    """Fast TCP check to see if a port is accepting connections."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True, ""
    except OSError as e:
        return False, str(e)


def perform_network_ping_sweep(validated_configs: dict[str, Any]) -> bool:
    console.print("[bold cyan]Running Parallel Network Ping Sweep...[/bold cyan]")

    # targets: tuple of (Description, Target_IP, Port, Associated_IP_to_Mark_Up)
    targets: set[tuple[str, str, int, str | None]] = set()

    obs_cfg: ObsConfigValidator = validated_configs['obs']
    daq_cfg: DaqConfigValidator = validated_configs['daq']
    net_cfg: NetworkConfigValidator = validated_configs['network']

    # --- 1. Head Node ---
    head_ip = str(daq_cfg.head_node_ip_addr)
    if head_ip and not daq_cfg.head_node_container:
        targets.add(("Head Node", head_ip, 22, None))

    # --- 2. WPS Power Strips ---
    # Access wps config through model_extra
    extra_obs = obs_cfg.model_extra or {}
    for dome in obs_cfg.domes:
        for mod in dome.modules:
            wps_name = mod.wps or 'wps'
            if wps_name in extra_obs:
                wps_data = extra_obs[wps_name]
                wps_url = wps_data.get('url', '')
                parsed = urllib.parse.urlparse(wps_url)
                if parsed.hostname:
                    # HTTP standard port is 80
                    targets.add((f"WPS ({wps_name})", parsed.hostname, 80, None))

    # --- 3. DAQ Nodes ---
    pf_daq_map = {str(d.ip_addr): d.port_forwarding for d in net_cfg.daq_nodes}
    for daq in daq_cfg.daq_nodes:
        ip = str(daq.ip_addr)
        pf = pf_daq_map.get(ip)

        if pf and pf.status and pf.gw_ip:
            # It's behind a gateway. Check the gateway port specifically forwarded for this DAQ.
            gw_ip = str(pf.gw_ip)
            forwarded_port = pf.port or 22
            targets.add((f"DAQ Node ({ip}) via GW", gw_ip, forwarded_port, ip))
        else:
            # Direct connection
            targets.add((f"DAQ Node ({ip})", ip, 22, None))

    # --- 4. Modules/Quabos ---
    pf_mod_map = {str(m.ip_addr): m.port_forwarding for m in net_cfg.modules}
    
    for dome in obs_cfg.domes:
        for mod in dome.modules:
            ip = str(mod.ip_addr)
            pf = pf_mod_map.get(ip)
            dome_name = dome.name

            if pf and pf.status and pf.gw_ip:
                # Module is behind a gateway. Check the first CMD port on the gateway.
                gw_ip = str(pf.gw_ip)
                cmd_ports = pf.cmd_port or [60000]
                first_cmd_port = cmd_ports[0] if cmd_ports else 60000
                targets.add((f"Module ({dome_name}: {ip}) via GW", gw_ip, first_cmd_port, ip))
            else:
                # Direct connection to the module's Quabo 0 CMD port
                targets.add((f"Module ({dome_name}: {ip})", ip, 60000, None))

    # --- Execute Parallel Sweep ---
    up_hosts = set()
    all_passed = True
    results: list[tuple[str, bool, str]] = []

    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_target = {
            executor.submit(_check_tcp_port, target_ip, port): (desc, target_ip, assoc_ip)
            for desc, target_ip, port, assoc_ip in targets
        }

        for future in as_completed(future_to_target):
            desc, target_ip, assoc_ip = future_to_target[future]
            try:
                is_up, err = future.result()
                if is_up:
                    up_hosts.add(target_ip)
                    if assoc_ip:
                        # Inference: The port forward succeeded, so the internal device is up!
                        up_hosts.add(assoc_ip)
                    results.append((desc, True, ""))
                else:
                    results.append((desc, False, err))
            except Exception as e:
                results.append((desc, False, str(e)))

    # --- Display Results cleanly ---
    results.sort(key=lambda x: x[0])
    for desc, is_up, _err in results:
        if is_up:
            console.print(f"  [green]✔ {desc:<40} is UP[/green]")
        else:
            console.print(f"  [red]✖ {desc:<40} is DOWN[/red]")
            all_passed = False

    if all_passed:
        console.print("[green]All network targets reachable.[/green]\n")
    return all_passed

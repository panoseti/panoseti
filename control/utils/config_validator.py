from __future__ import annotations

import socket
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

console = Console()

# Pydantic Validation

## Validation graph
def print_compact_config(config_name: str, config_dict: dict):
    """Prints a configuration dictionary but collapses massive lists like module_ids."""
    import copy
    compact_dict = copy.deepcopy(config_dict)

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


def perform_network_ping_sweep(validated_configs: dict) -> bool:
    console.print("[bold cyan]Running Parallel Network Ping Sweep...[/bold cyan]")

    # targets: tuple of (Description, Target_IP, Port, Associated_IP_to_Mark_Up)
    targets = set()

    # --- 1. Head Node ---
    head_ip = validated_configs['daq'].get('head_node_ip_addr')
    if head_ip and not validated_configs['daq'].get('head_node_container', False):
        targets.add(("Head Node", head_ip, 22, None))

    # --- 2. WPS Power Strips ---
    for dome in validated_configs['obs'].get('domes', []):
        for mod in dome.get('modules', []):
            wps_name = mod.get('wps')
            if wps_name and wps_name in validated_configs['obs']:
                wps_url = validated_configs['obs'][wps_name].get('url', '')
                parsed = urllib.parse.urlparse(wps_url)
                if parsed.hostname:
                    # HTTP standard port is 80
                    targets.add((f"WPS ({wps_name})", parsed.hostname, 80, None))

    # --- 3. DAQ Nodes ---
    pf_daq_map = {d.get('ip_addr'): d.get('port_forwarding', {}) for d in
                  validated_configs['network'].get('daq_nodes', [])}
    for daq in validated_configs['daq'].get('daq_nodes', []):
        ip = daq.get('ip_addr')
        pf = pf_daq_map.get(ip, {})

        if pf.get('status') and pf.get('gw_ip'):
            # It's behind a gateway. Check the gateway port specifically forwarded for this DAQ.
            # If the gateway responds on this port, mark BOTH the gateway and the internal DAQ IP as UP.
            gw_ip = pf.get('gw_ip')
            forwarded_port = pf.get('port', 22)
            targets.add((f"DAQ Node ({ip}) via GW", gw_ip, forwarded_port, ip))
        else:
            # Direct connection
            targets.add((f"DAQ Node ({ip})", ip, 22, None))

    # --- 4. Modules/Quabos ---
    pf_mod_map = {m.get('ip_addr'): m.get('port_forwarding', {}) for m in
                  validated_configs['network'].get('modules', [])}
    ip_to_dome = {m.get('ip_addr'): d.get('name', 'Unknown') for d in validated_configs['obs'].get('domes', []) for m in
                  d.get('modules', [])}

    for ip, dome_name in ip_to_dome.items():
        pf = pf_mod_map.get(ip, {})

        if pf.get('status') and pf.get('gw_ip'):
            # Module is behind a gateway. Check the first CMD port on the gateway.
            gw_ip = pf.get('gw_ip')
            cmd_ports = pf.get('cmd_port', [60000])
            first_cmd_port = cmd_ports[0] if cmd_ports else 60000
            targets.add((f"Module ({dome_name}: {ip}) via GW", gw_ip, first_cmd_port, ip))
        else:
            # Direct connection to the module's Quabo 0 CMD port
            targets.add((f"Module ({dome_name}: {ip})", ip, 60000, None))

    # --- Execute Parallel Sweep ---
    up_hosts = set()
    all_passed = True
    results = []

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




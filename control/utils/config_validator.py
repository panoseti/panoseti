
import pydantic
import subprocess
import platform

import socket
import urllib.parse
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import ValidationError
from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel
from rich.tree import Tree

# from .config_file import




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

def _expand_module_ids(id_input) -> set:
    """Parses a string like '0-255' or '10,11,15-20', or a list of ints, into a set of integers."""
    ids = set()

    # If it was already parsed into a list by previous steps, handle it directly
    if isinstance(id_input, list):
        return set(int(x) for x in id_input)

    # Otherwise, parse the string
    for part in str(id_input).split(','):
        # strip() removes whitespace, strip('[]') removes brackets just in case it was stringified
        part = part.strip().strip('[]')
        if not part:
            continue

        if '-' in part:
            start, end = part.split('-')
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))

    return ids




def _check_reachability(ip: str, tcp_port: int = None, check_ssh: bool = False) -> Tuple[bool, str]:
    """Returns (is_up, status_string). Robust cross-platform ping and TCP/SSH fallbacks."""

    # 1. SSH Gateway Check
    if check_ssh and tcp_port:
        # Pinging DAQ node behind gateway using specific port
        cmd = ['ssh', '-p', str(tcp_port), '-o', 'ConnectTimeout=2', '-o', 'BatchMode=yes', f'panoseti@{ip}', 'echo',
               '1']
        res = subprocess.run(cmd, capture_output=True)
        return res.returncode == 0, f"Gateway DAQ Access ({ip}:{tcp_port})"

    # 2. Cross-platform ICMP Ping Check
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    try:
        # Use Python's timeout instead of ping's OS-specific -W parameter
        res = subprocess.run(['ping', param, '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             timeout=1.5)
        if res.returncode == 0:
            return True, ""
    except subprocess.TimeoutExpired:
        pass

    # 3. TCP Port Fallback (Crucial for routers/gateways that block ICMP)
    if tcp_port is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((ip, tcp_port))
            return True, ""
        except:
            pass

    return False, ""


def perform_network_ping_sweep(validated_configs: dict) -> bool:
    console.print("[bold cyan]Running Parallel Network Ping Sweep...[/bold cyan]")
    targets = set()

    # Head Node
    head_ip = validated_configs['daq'].get('head_node_ip_addr')
    if head_ip: targets.add(('Head Node', head_ip, 22, False))

    # WPS Power Strips
    for dome in validated_configs['obs'].get('domes', []):
        for mod in dome.get('modules', []):
            wps_name = mod.get('wps')
            if wps_name and wps_name in validated_configs['obs']:
                wps_url = validated_configs['obs'][wps_name].get('url', '')
                parsed = urllib.parse.urlparse(wps_url)
                if parsed.hostname:
                    targets.add((f"WPS ({wps_name})", parsed.hostname, 80, False))

    # DAQ Nodes
    pf_daq_map = {d.get('ip_addr'): d.get('port_forwarding', {}) for d in
                  validated_configs['network'].get('daq_nodes', [])}
    for daq in validated_configs['daq'].get('daq_nodes', []):
        ip = daq.get('ip_addr')
        pf = pf_daq_map.get(ip, {})
        if pf.get('status'):
            # Check gateway, AND attempt SSH through the gateway to the DAQ node
            targets.add((f"DAQ Node ({ip})", pf.get('gw_ip'), pf.get('port', 22), True))
        else:
            targets.add(("DAQ Node", ip, 22, False))

    # Modules
    pf_mod_map = {m.get('ip_addr'): m.get('port_forwarding', {}) for m in
                  validated_configs['network'].get('modules', [])}
    # Create mapping of IP to Dome Name for better labels
    ip_to_dome = {m.get('ip_addr'): d.get('name', 'Unknown') for d in validated_configs['obs'].get('domes', []) for m in
                  d.get('modules', [])}

    for ip, dome_name in ip_to_dome.items():
        pf = pf_mod_map.get(ip, {})
        if pf.get('status'):
            targets.add((f"Module Gateway ({dome_name})", pf.get('gw_ip'), 80, False))
        else:
            targets.add((f"Module ({dome_name})", ip, None, False))

    all_passed = True
    results = []
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_target = {executor.submit(_check_reachability, ip, port, ssh): (desc, ip) for desc, ip, port, ssh in
                            targets}
        for future in as_completed(future_to_target):
            desc, ip = future_to_target[future]
            try:
                is_up, extra = future.result()
                results.append((desc, ip, is_up, extra))
            except Exception:
                results.append((desc, ip, False, ""))

    results.sort(key=lambda x: (x[0], x[1]))
    failures = 0
    for desc, ip, is_up, extra in results:
        disp = f"{desc} [{extra}]" if extra else desc
        if is_up:
            console.print(f"  [green]✔ {disp:<30} ({ip}) is UP[/green]")
        else:
            console.print(f"  [red]✖ {disp:<30} ({ip}) is DOWN[/red]")
            failures += 1
            all_passed = False

    if failures == 0: console.print("[green]All network targets reachable.[/green]\n")
    return all_passed




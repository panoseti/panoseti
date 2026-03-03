import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel

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


def _check_tcp_port(ip: str, port: int, timeout: float = 1.0) -> bool:
    """Fast TCP check to see if a port is accepting connections (No SSH keys or ICMP needed)."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def perform_network_ping_sweep(validated_configs: dict) -> bool:
    console.print("[bold cyan]Running Parallel Network Ping Sweep...[/bold cyan]")

    # targets: tuple of (Description, IP, Port, Gateway_IP_if_applicable)
    targets = set()

    # Add DAQ nodes (Port 22)
    for daq in validated_configs['daq'].get('daq_nodes', []):
        ip = daq.get('ip_addr')
        gw_ip = daq.get('port_forwarding', {}).get('gw_ip')
        targets.add((f"DAQ Node ({ip})", ip, 22, gw_ip))

    # Add Modules/Quabos (Port 60000 for CMD)
    for dome in validated_configs['obs'].get('domes', []):
        for mod in dome.get('modules', []):
            ip = mod.get('ip_addr')
            targets.add((f"Module ({ip})", ip, 60000, None))

    up_hosts = set()
    all_passed = True

    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_target = {
            executor.submit(_check_tcp_port, ip, port): (desc, ip, gw)
            for desc, ip, port, gw in targets
        }

        for future in as_completed(future_to_target):
            desc, ip, gw = future_to_target[future]
            is_up = future.result()

            if is_up:
                up_hosts.add(ip)
                if gw:
                    up_hosts.add(gw)  # INFERENCE: If DAQ is up, its Gateway must be up!
                console.print(f"  [green]✔ {desc:<30} is UP[/green]")
            else:
                console.print(f"  [red]✖ {desc:<30} is DOWN[/red]")
                all_passed = False

    if all_passed:
        console.print("[green]All network targets reachable.[/green]\n")
    return all_passed




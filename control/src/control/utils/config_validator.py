from __future__ import annotations

import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from ipaddress import ip_address
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

from control.utils import util
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    ObsConfig,
)

console = Console()

# Pydantic Validation

## Validation graph
def print_compact_config(config_name: str, config_obj: Any) -> None:
    """Prints a configuration object but collapses massive lists like module_ids."""
    import copy

    from pydantic import BaseModel

    if isinstance(config_obj, BaseModel):
        compact_dict = copy.deepcopy(config_obj.model_dump())
    elif isinstance(config_obj, dict):
        compact_dict = copy.deepcopy(config_obj)
    else:
        compact_dict = {"error": f"Unsupported config type: {type(config_obj)}"}

    if config_name.lower() == 'daq':
        for node in compact_dict.get('daq_nodes', []):
            if isinstance(node.get('module_ids'), list):
                # Compress [0, 1, 2... 255] into a readable string
                ids = node['module_ids']
                if len(ids) > 5:
                    node['module_ids'] = f"[{ids[0]}, {ids[1]} ... {len(ids)} total IDs ... {ids[-1]}]"

    console.print(Panel(f"[bold green]Parsed {config_name} Config[/bold green]"))
    pprint(compact_dict, expand_all=True)


def _check_reachability(target_ip: str, port: int, target_type: str = "tcp", timeout: float = 2.0) -> tuple[bool, str]:
    """Check if a target is reachable using either TCP or Quabo-specific UDP ping."""
    if target_type == "quabo":
        # Quabo check uses UDP command exchange
        try:
            res = util.ping(ip_address(target_ip), port)
            return res, "" if res else "Quabo ping failed (UDP timeout)"
        except Exception as e:
            return False, str(e)
    else:
        # Fast TCP check for other services
        try:
            with socket.create_connection((target_ip, port), timeout=timeout):
                return True, ""
        except OSError as e:
            return False, str(e)


def perform_network_ping_sweep(validated_configs: dict[str, Any]) -> bool:
    console.print("[bold cyan]Running Parallel Network Reachability Sweep...[/bold cyan]")

    # targets: tuple of (Description, Target_IP, Port, Associated_IP_to_Mark_Up, Target_Type)
    targets: set[tuple[str, str, int, str | None, str]] = set()

    from pydantic import BaseModel
    # Strictly enforce that incoming data perfectly matches the schema
    def _strict_validate(key: str, model_type: type[BaseModel]) -> Any:
        cfg = validated_configs.get(key)
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            # This will raise ValidationError if the dictionary is incomplete or invalid
            return model_type.model_validate(cfg)
        # If it's already a model, it must be an instance of the expected class
        if not isinstance(cfg, model_type):
            raise TypeError(f"Expected {model_type.__name__} for '{key}', got {type(cfg)}")
        return cfg

    obs_cfg: ObsConfig | None = _strict_validate('obs', ObsConfig)
    daq_cfg: DaqConfig | None = _strict_validate('daq', DaqConfig)
    net_cfg: NetworkConfig | None = _strict_validate('network', NetworkConfig)

    # --- 1. Head Node ---
    if daq_cfg and hasattr(daq_cfg, 'head_node_ip_addr'):
        head_ip = str(daq_cfg.head_node_ip_addr)
        if head_ip and not getattr(daq_cfg, 'head_node_container', False):
            targets.add(("Head Node", head_ip, 22, None, "tcp"))

    # --- 2. WPS Power Strips ---
    # Access wps config through model_extra
    if obs_cfg and hasattr(obs_cfg, 'domes'):
        extra_obs = obs_cfg.model_extra or {}
        for dome in obs_cfg.domes:
            for mod in dome.modules:
                wps_name = getattr(mod, 'wps', None) or 'wps'
                if wps_name in extra_obs:
                    wps_data = extra_obs[wps_name]
                    wps_url = wps_data.get('url', '')
                    parsed = urllib.parse.urlparse(wps_url)
                    if parsed.hostname:
                        # HTTP standard port is 80
                        targets.add((f"WPS ({wps_name})", parsed.hostname, 80, None, "tcp"))

    # --- 3. DAQ Nodes ---
    if net_cfg and hasattr(net_cfg, 'daq_nodes') and daq_cfg and hasattr(daq_cfg, 'daq_nodes'):
        pf_daq_map = {str(d.ip_addr): d.port_forwarding for d in net_cfg.daq_nodes if hasattr(d, 'ip_addr')}
        for daq in daq_cfg.daq_nodes:
            if not hasattr(daq, 'ip_addr'):
                continue
            ip = str(daq.ip_addr)
            pf = pf_daq_map.get(ip)

            if pf and getattr(pf, 'status', False) and getattr(pf, 'gw_ip', None):
                # It's behind a gateway. Check the gateway port specifically forwarded for this DAQ.
                gw_ip = str(pf.gw_ip)
                forwarded_port = getattr(pf, 'port', 22) or 22
                targets.add((f"DAQ Node ({ip}) via GW {gw_ip}", gw_ip, forwarded_port, ip, "tcp"))
            else:
                # Direct connection
                targets.add((f"DAQ Node ({ip})", ip, 22, None, "tcp"))

    # --- 4. Modules/Quabos ---
    if obs_cfg and hasattr(obs_cfg, 'domes') and net_cfg and hasattr(net_cfg, 'modules'):
        from control.utils import config_file
        
        for dome in obs_cfg.domes:
            dome_name = getattr(dome, 'name', 'Unknown')
            for mod in dome.modules:
                for i in range(4):
                    ip_ports = util.get_quabo_ip_port(mod.ip_addr, i, net_cfg)
                    real_ip = str(ip_ports.ip_addr)
                    cmd_port = ip_ports.cmd_port
                    quabo_ip = config_file.quabo_ip_addr(str(mod.ip_addr), i)
                    
                    desc = f"Quabo {i} ({dome_name}: {quabo_ip})"
                    if real_ip != quabo_ip:
                         desc += f" via GW {real_ip}"
                    
                    targets.add((desc, real_ip, cmd_port, quabo_ip, "quabo"))

    # --- Execute Parallel Sweep ---
    up_hosts = set()
    all_passed = True
    results: list[tuple[str, bool, str]] = []

    if not targets:
        return True

    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_target = {
            executor.submit(_check_reachability, target_ip, port, target_type): (desc, target_ip, assoc_ip)
            for desc, target_ip, port, assoc_ip, target_type in targets
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
            console.print(f"  [green]✔ {desc:<50} is UP[/green]")
        else:
            console.print(f"  [red]✖ {desc:<50} is DOWN[/red]")
            all_passed = False

    if all_passed:
        console.print("[green]All network targets reachable.[/green]\n")
    return all_passed

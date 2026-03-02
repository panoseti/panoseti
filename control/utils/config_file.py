#! /usr/bin/env python3

# functions to read and parse config files

import os,sys,json
import subprocess
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import ValidationError
from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel
from rich.tree import Tree

# import Pydantic validation models
from .pydantic_config_models import (
    DataConfigValidator, ObsConfigValidator, DaqConfigValidator,
    NetworkConfigValidator, DaemonConfigValidator, FirmwareConfigValidator
)

from .validation_report import ValidationReport
from .global_validator import GlobalConfigValidator

# Globals to control console verbosity
IS_CLI_VALIDATION = False
DEBUG_VALIDATION = False
RAISE_VALIDATION_ERRORS = False

console = Console()

import logging
# TODO: we need to improve the file path
# configs file
obs_config_filename = 'configs/obs_config.json'
daq_config_filename = 'configs/daq_config.json'
data_config_filename = 'configs/data_config.json'
network_config_filename = 'configs/network_config.json'
daemons_config_filename = 'configs/daemons.json'
firmware_config_filename = 'configs/firmware.json'
# quabo realted files
quabo_info_filename = 'quabos/quabo_info.json'
detector_info_filename = 'quabos/detector_info.json'
quabo_calib_filename = 'quabos/detovervol_%dv/%s/quabo_calib_%s.json'
# These files are creatd during the run,
# and will be copied to the final data dir
quabo_uids_filename = 'tmp/quabo_uids.json'
quabo_ph_baseline_filename = 'tmp/quabo_ph_baseline.json'
sw_info_filename = 'tmp/sw_info.json'
quabo_config_filename = 'tmp/quabo_config_*.json'

# list of config files copied to data dir
config_file_names = [
    obs_config_filename, daq_config_filename, data_config_filename,
    quabo_uids_filename, quabo_ph_baseline_filename, sw_info_filename,
    quabo_config_filename, network_config_filename
]

# compute a 'module ID', given its base quabo IP addr: bits 2..9 of IP addr
#
def ip_addr_to_module_id(ip_addr_str):
    pieces = ip_addr_str.split('.')
    n = int(pieces[3]) + 256*int(pieces[2])
    return (n>>2)&255

# given module base IP address, return IP addr of quabo i
#
def quabo_ip_addr(base, i):
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)


def get_boardloc(module_ip_addr, quabo_index):
    """Given a module ip address and a quabo index, returns the BOARDLOC of
    the corresponding quabo."""
    pieces = module_ip_addr.split('.')
    boardloc = int(pieces[2]) * 256 + int(pieces[3]) + quabo_index
    return boardloc


# assign sequential numbers to domes,
# and IDs to modules
#
def assign_numbers(c):
    ndome = 0
    for dome in c['domes']:
        dome['num'] = ndome
        ndome += 1
        for module in dome['modules']:
            module['id'] = ip_addr_to_module_id(module['ip_addr'])

# input: a string of the form "0-2, 5-6"
# output: a list of integers, e.g. 0,1,2,5,6
#
def string_to_list(s):
    out = []
    parts = s.split(',')
    for part in parts:
        nums = part.split('-')
        if len(nums) > 1:
            a = int(nums[0])
            b = int(nums[1])
            for i in range(a,b+1):
                out.append(i)
        else:
            out.append(int(nums[0]))
    return out

# in DAQ node objects, expand module range strings
# to list of module numbers
#
def expand_ranges(daq_config):
    for node in daq_config['daq_nodes']:
        node['module_ids'] = string_to_list(node['module_ids'])

# given a module ID, find the DAQ node that's handling it
#
def module_id_to_daq_node(daq_config, module_id):
    for node in daq_config['daq_nodes']:
        if module_id in node['module_ids']:
            return node
    raise Exception("no DAQ node is handling module %d"%module_id)

def check_config_file(name, dir='.'):
    path = os.path.join(dir, name)
    if not os.path.isfile(path):
    # if not os.path.exists('%s/%s'%(dir, name)):
        print("The config file '%s' doesn't exist."%name)
        print("Create a symbolic link from %s to a specific config file, e.g.:"%name)
        print("   ln -s %s_lick.json %s"%(name.split('.')[0], name))

        sys.exit()


def get_obs_config(dir='.'):
    # pass assign_numbers so it injects `id` and `num` before validation
    return load_and_validate(ObsConfigValidator, obs_config_filename, dir, "Obs Config", assign_numbers)

def get_daq_config(dir='.'):
    # pass expand_ranges so it parses module string ranges before validation
    return load_and_validate(DaqConfigValidator, daq_config_filename, dir, "DAQ Config", expand_ranges)

def get_data_config(dir='.'):
    return load_and_validate(DataConfigValidator, data_config_filename, dir, "Data Config")

def get_network_config(dir='.'):
    check_config_file(network_config_filename, dir)
    path = '%s/%s'%(dir, network_config_filename)
    # as the network config file is not designed to the users,
    # we check it manually, instead of using check_config_file.
    try:
        with open(path) as f:
            s = f.read()
        net_conf = json.loads(s)
    except:
        print("***********Warning: No network config file! **************")
        print("******All the devices should be in the same subnet *******")
        net_conf = {}
        return net_conf

    return load_and_validate(NetworkConfigValidator, network_config_filename, dir, "Network Config")


def get_firmware_config(dir='.'):
    return load_and_validate(FirmwareConfigValidator, firmware_config_filename, dir, "Firmware Config")

def get_daemons_config(dir='.'):
    return load_and_validate(DaemonConfigValidator, daemons_config_filename, dir, "Daemons Config")

def get_quabo_uids():
    if not os.path.exists(quabo_uids_filename):
        print("%s is missing.  Run get_uids.py"%quabo_uids_filename)
        sys.exit()
    with open(quabo_uids_filename) as f:
        s = f.read()
    quabo_uids_conf = json.loads(s)
    assign_numbers(quabo_uids_conf)
    return quabo_uids_conf

# get detector info as an array indexed by serialno
#
def get_detector_info():
    check_config_file(detector_info_filename)
    with open(detector_info_filename) as f:
        s = f.read()
    c = json.loads(s)
    d = {}
    with open(obs_config_filename) as f:
        s = f.read()
    obs_config = json.loads(s)
    with open(data_config_filename) as f:
        s = f.read()
    data_config = json.loads(s)
    for det in c:
        try:
            d[str(det['serialno'])] = float(det['operating_voltage'])
        except:
            try:
                d[str(det['serialno'])] = float(det['breakdown_voltage']) + data_config['detector_overvoltage']
            except:
                d[str(det['serialno'])] = float(det['breakdown_voltage']) + 3
    if 'detector_overvoltage' not in data_config:
        print('**************************************************************************')
        print('detector_overvoltage is not set in data_config.json')
        print('Use the default overvoltage: 3V')
        print('**************************************************************************')
    return d;

# get quabo info as an array indexed by uid
#
def get_quabo_info():
    check_config_file(quabo_info_filename)
    with open(quabo_info_filename) as f:
        s = f.read()
    c = json.loads(s)
    d = {}
    for q in c:
        d[q['uid']] = q
    return d;

def get_quabo_ph_baselines():
    check_config_file(quabo_ph_baseline_filename)
    with open(quabo_ph_baseline_filename) as f:
        s = f.read()
    c = json.loads(s)
    return c

# get quabo calibration info
#
def get_quabo_calib(serialno, detovervol, mode):
    #print('reading calib file %s'%serialno)
    path = quabo_calib_filename%(detovervol, mode, serialno)
    with open(path) as f:
        s = f.read()
    return json.loads(s)

# return list of modules from obs_config
#
def get_modules(c):
    modules = []
    for dome in c['domes']:
        for module in dome['modules']:
            modules.append(module)
    return modules

# link modules to DAQ nodes:
# - in the daq_config data structure, add a list "modules"
#   to each daq node object, of the module objects
#   in the quabo_uids data structure;
# - in the quabo_uids data structure, in each module object,
#   add a link "daq_node" to the DAQ node that's handling it.
#
def associate(daq_config, quabo_uids):
    for node in daq_config['daq_nodes']:
        node['modules'] = []
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            daq_node = module_id_to_daq_node(daq_config, module['id'])
            daq_node['modules'].append(module)
            module['daq_node'] = daq_node

# show which module is going to which data recorder
#
def show_daq_assignments(quabo_uids):
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            ip_addr = module['ip_addr']
            daq_node = module['daq_node']
            for i in range(4):
                q = module['quabos'][i];
                print("data from quabo %s (%s) -> DAQ node %s"
                    %(q['uid'], quabo_ip_addr(ip_addr, i), daq_node['ip_addr'])
                )

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


def print_topology_graph(obs_conf, daq_conf, net_conf):
    """Generates an ASCII/Unicode visual tree of the data routing and hardware topology."""
    if not obs_conf: return

    console.print(Panel("[bold cyan]Observatory Topology & Routing Graph[/bold cyan]"))

    obs_name = obs_conf.get('name', 'Unknown Observatory')
    root = Tree(f"[bold magenta]Observatory: {obs_name}[/bold magenta]")

    # Map network configs for fast lookup
    net_module_map = {m.get('ip_addr'): m.get('port_forwarding', {}) for m in net_conf.get('modules', [])}

    # Pre-parse DAQ ranges using our updated _expand_module_ids function
    daq_map = {}
    for daq in daq_conf.get('daq_nodes', []):
        daq_ids = _expand_module_ids(daq.get('module_ids', ''))
        for mod_id in daq_ids:
            daq_map[mod_id] = daq

    for dome in obs_conf.get('domes', []):
        d_name = dome.get('name', 'Unknown Dome')
        dome_node = root.add(f"[bold yellow]Dome: {d_name}[/bold yellow]")

        for mod in dome.get('modules', []):
            m_ip = mod.get('ip_addr')
            m_hw = mod.get('quabo_version', 'unknown')
            m_timing = mod.get('timing_mode', 'wr')

            # Use the existing codebase utility to get the official Module ID
            try:
                mod_id = ip_addr_to_module_id(m_ip)
            except Exception as e:
                mod_id = -1

            # Lookup the destination DAQ based on the official ID
            dest_daq = daq_map.get(mod_id)
            if dest_daq:
                daq_str = f"{dest_daq.get('ip_addr')} (Interface: {dest_daq.get('bindhost', 'eth0')})"
            else:
                daq_str = "[red]UNMAPPED - No matching DAQ module_ids[/red]"

            # Check network routing
            pf = net_module_map.get(m_ip, {})
            is_forwarded = pf.get('status') is True
            gw_ip = pf.get('gw_ip', 'Local Direct')
            cmd_ports = pf.get('cmd_port', [60000, 60000, 60000, 60000])  # Fallback to default UDP port

            # Display the resolved Module ID so the observer can trace the DAQ routing
            mod_node = dome_node.add(
                f"Module {m_ip} [ID: {mod_id}] [HW: {m_hw}] [Timing: {m_timing}] -> Stream DAQ: [cyan]{daq_str}[/cyan]")
            for q in range(4):
                base_ip_parts = m_ip.split('.')
                if len(base_ip_parts) == 4:
                    q_ip = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{base_ip_parts[2]}.{int(base_ip_parts[3]) + q}"
                else:
                    q_ip = "Invalid IP Format"

                real_ip = gw_ip if is_forwarded else q_ip
                real_port = cmd_ports[q] if len(cmd_ports) > q else 60000

                route_str = f"──[Tunnel: [green]{gw_ip}[/green]]──> [magenta]{real_ip}:{real_port}[/magenta]" if is_forwarded else f"──[Direct]──> [magenta]{real_ip}:{real_port}[/magenta]"

                mod_node.add(f"Quabo {q} ({q_ip}) {route_str}")

    console.print(root)
    print("\n")

## Test network connectivity



def _check_reachability(ip: str, tcp_port: int = None) -> bool:
    """Check if an IP is reachable via ICMP ping, or optional TCP port check."""
    # 1. Try standard ICMP Ping first
    res = subprocess.run(['ping', '-c', '1', '-W', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0:
        return True

    # 2. If ICMP fails (e.g., router blocks it), try TCP port if provided
    if tcp_port is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((ip, tcp_port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

    # 3. As a final fallback for gateways (often running web servers), try TCP 80
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((ip, 80))
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

    return False


def perform_network_ping_sweep(validated_configs: dict) -> bool:
    """Parses required target IPs from configs and pings them concurrently."""
    console.print("[bold cyan]Running Parallel Network Ping Sweep...[/bold cyan]")

    # Now storing tuples of (Description, IP, TCP_Fallback_Port)
    targets_to_ping = set()

    head_ip = validated_configs['daq'].get('head_node_ip_addr')
    if head_ip:
        targets_to_ping.add(('Head Node', head_ip, 22))

    pf_daq_map = {d.get('ip_addr'): d.get('port_forwarding', {}) for d in
                  validated_configs['network'].get('daq_nodes', [])}
    for daq in validated_configs['daq'].get('daq_nodes', []):
        daq_ip = daq.get('ip_addr')
        pf = pf_daq_map.get(daq_ip, {})
        if pf.get('status') is True:
            targets_to_ping.add((f"DAQ Gateway ({daq_ip})", pf.get('gw_ip'), pf.get('port', 22)))
        else:
            targets_to_ping.add(("DAQ Node", daq_ip, 22))

    pf_mod_map = {m.get('ip_addr'): m.get('port_forwarding', {}) for m in
                  validated_configs['network'].get('modules', [])}
    for dome in validated_configs['obs'].get('domes', []):
        for mod in dome.get('modules', []):
            m_ip = mod.get('ip_addr')
            pf = pf_mod_map.get(m_ip, {})
            if pf.get('status') is True:
                targets_to_ping.add((f"Module Gateway ({m_ip})", pf.get('gw_ip'), 80))
            else:
                targets_to_ping.add(("Module Base IP", m_ip, None))

    all_passed = True
    ping_failures = 0
    results = []

    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_target = {
            executor.submit(_check_reachability, ip, port): (desc, ip)
            for desc, ip, port in targets_to_ping if ip
        }

        for future in as_completed(future_to_target):
            desc, ip = future_to_target[future]
            try:
                is_up = future.result()
                results.append((desc, ip, is_up))
            except Exception:
                results.append((desc, ip, False))

    results.sort(key=lambda x: (x[0], x[1]))

    for desc, ip, is_up in results:
        if is_up:
            console.print(f"  [green]✔ {desc:<25} ({ip}) is UP[/green]")
        else:
            console.print(f"  [red]✖ {desc:<25} ({ip}) is DOWN / UNREACHABLE[/red]")
            ping_failures += 1
            all_passed = False

    if ping_failures == 0:
        console.print("[green]All network targets reachable.[/green]\n")
    else:
        console.print(f"[red]{ping_failures} network target(s) failed ping sweep.[/red]\n")

    return all_passed


## Apply global validation

def validate_all(check_network: bool = False, debug: bool = False, graph: bool = False) -> bool:
    """
    Master validation orchestrator.
    Now supports Global Tier-2 validation and topology graphing.
    """
    global IS_CLI_VALIDATION, DEBUG_VALIDATION
    IS_CLI_VALIDATION = True
    DEBUG_VALIDATION = debug

    all_passed = True
    ips_to_ping = set()
    validated_configs = {}

    console.print(Panel.fit("[bold cyan]Starting PanoSETI Pre-Flight Validation[/bold cyan]"))

    # 1. Tier 1: Strict File Validation (Existing Logic)
    try:
        validated_configs['firmware'] = get_firmware_config()
        validated_configs['daemons'] = get_daemons_config()
        validated_configs['obs'] = get_obs_config()
        validated_configs['network'] = get_network_config()
        validated_configs['daq'] = get_daq_config()
        validated_configs['data'] = get_data_config()
        console.print("[green]✔ Tier-1 File Syntax & Schema Validation Passed.[/green]")
    except Exception as e:
        console.print(f"[red]✖ Tier-1 Validation Failed: {e}[/red]")
        all_passed = False

    # 2. Tier 2: Global Cross-Configuration Validation
    if all_passed:
        console.print("\n[bold cyan]Running Tier-2 Global Cross-Checks...[/bold cyan]")
        global_validator = GlobalConfigValidator(validated_configs)
        if not global_validator.validate_all_rules():
            all_passed = False

    # 3. Visual Topology Graph
    if all_passed and graph:
        print_topology_graph(validated_configs['obs'], validated_configs['daq'], validated_configs['network'])

    # 4. Network Ping Checks
    if check_network:
        if not perform_network_ping_sweep(validated_configs):
            all_passed = False

    return all_passed


def load_and_validate(validator_class, filename, dir, config_name, preprocessor=None):
    """
    Unified loader: reads JSON, applies runtime preprocessing, validates against Pydantic models.
    Only prints UI elements if the explicit CLI validator is running.
    """
    path = os.path.join(dir, filename)

    if IS_CLI_VALIDATION:
        console.print(f"\n[bold yellow]Target:[/bold yellow] {config_name}")

    if not os.path.exists(path):
        if IS_CLI_VALIDATION:
            console.print(f"[bold red][FAIL][/bold red] {filename} not found.")
        if RAISE_VALIDATION_ERRORS:
            raise FileNotFoundError(f"{path} not found.")
        else:
            check_config_file(filename, dir)

    # Symlink printing logic
    if IS_CLI_VALIDATION:
        if os.path.islink(path):
            real_path = os.path.realpath(path)
            console.print(f"[dim]Symlink detected:[/dim] {filename} -> {os.path.abspath(real_path)}")
        else:
            console.print(f"[dim]File path:[/dim] {os.path.abspath(path)}")

    # 1. Load Data
    with open(path, 'r') as f:
        raw_data = json.load(f)

    # 2. Preprocess (e.g. assign_numbers, expand_ranges)
    if preprocessor:
        preprocessor(raw_data)

    # 3. Validate
    try:
        validated = validator_class(**raw_data)

        if IS_CLI_VALIDATION:
            console.print("[bold green][OK][/bold green] Passed validation.")

        if IS_CLI_VALIDATION and DEBUG_VALIDATION:
            console.print("\n[dim]Validated Configuration Structure:[/dim]")
            pprint(validated.model_dump(exclude_unset=True), expand_all=True)

        return validated.model_dump(mode='json', exclude_unset=True)

    except ValidationError as e:
        console.print(f"\n[bold red][FAIL] Schema Validation Error in {config_name} ({filename}):[/bold red]")
        for err in e.errors():
            loc = " -> ".join([str(l) for l in err["loc"]])
            msg = err["msg"]
            console.print(f"  [bold red]Field:[/bold red] {loc}")
            console.print(f"  [bold red]Error:[/bold red] {msg}\n")

        if IS_CLI_VALIDATION and DEBUG_VALIDATION:
            console.print("[dim]Raw Config Dictionary (for debugging):[/dim]")
            pprint(raw_data, expand_all=True)

        if RAISE_VALIDATION_ERRORS:
            raise e
        sys.exit(1)

    except json.JSONDecodeError as e:
        console.print(f"\n[bold red][FAIL] JSON Parsing Error in {config_name} ({filename}):[/bold red] {e}")
        if RAISE_VALIDATION_ERRORS:
            raise e
        sys.exit(1)




if __name__ == "__main__":
    c = get_detector_info()
    print(c)

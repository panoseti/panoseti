#! /usr/bin/env python3
from __future__ import annotations

import json

# functions to read and parse config files
# CWD CONTRACT: all relative paths in this file are relative to the control/ directory.
# Scripts must be launched from control/ (e.g. `cd control && python start.py`).
import os
import pathlib
import sys

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich.tree import Tree

from .config_validator import perform_network_ping_sweep
from .global_validator import GlobalConfigValidator

# import Pydantic validation models
from .pydantic_config_models import (
    DaemonConfigValidator,
    DaqConfigValidator,
    DataConfigValidator,
    FirmwareConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
)

console = Console()

# Resolved root of the control/ package (useful for tests and tooling).
# Do NOT use this for runtime hardware paths — use CWD-relative strings instead.
_CONTROL_BASE = pathlib.Path(__file__).parent.parent.resolve()

# Globals to control console verbosity
IS_CLI_VALIDATION = False
DEBUG_VALIDATION = False
RAISE_VALIDATION_ERRORS = False

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
        module_ids = node['module_ids']
        # If it was already parsed into a list by previous steps, handle it directly
        if isinstance(module_ids, list):
            unique_module_ids = set(int(x) for x in module_ids)
            module_ids_list = list(unique_module_ids)
        elif isinstance(module_ids, str):
            module_ids_list = string_to_list(module_ids)
        else:
            raise ValueError(f"Expected 'module_ids' to be a list or str, not {type(module_ids)=}")
        # print(module_ids_list)
        node['module_ids'] = module_ids_list

# given a module ID, find the DAQ node that's handling it
#
def module_id_to_daq_node(daq_config, module_id):
    for node in daq_config['daq_nodes']:
        if module_id in node['module_ids']:
            return node
    raise Exception(f"no DAQ node is handling module {module_id}")

def check_config_file(name, dir='.'):
    path = os.path.join(dir, name)
    if not os.path.isfile(path):
    # if not os.path.exists('%s/%s'%(dir, name)):
        print(f"The config file '{name}' doesn't exist.")
        print(f"Create a symbolic link from {name} to a specific config file, e.g.:")
        print("   ln -s {}_lick.json {}".format(name.split('.')[0], name))

        sys.exit(1)


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
    path = f'{dir}/{network_config_filename}'
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
        print(f"{quabo_uids_filename} is missing.  Run get_uids.py")
        sys.exit(1)
    with open(quabo_uids_filename) as f:
        s = f.read()
    quabo_uids_conf = json.loads(s)
    assign_numbers(quabo_uids_conf)
    # return load_and_validate(QuaboUidsValidator, quabo_uids_filename, dir, "UID Config", assign_numbers)
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
    json.loads(s)
    with open(data_config_filename) as f:
        s = f.read()
    data_config = json.loads(s)
    for det in c:
        try:
            d[str(det['serialno'])] = float(det['operating_voltage'])
        except (KeyError, ValueError, TypeError):
            try:
                d[str(det['serialno'])] = float(det['breakdown_voltage']) + data_config['detector_overvoltage']
            except (KeyError, ValueError, TypeError):
                d[str(det['serialno'])] = float(det['breakdown_voltage']) + 3
    if 'detector_overvoltage' not in data_config:
        print('**************************************************************************')
        print('detector_overvoltage is not set in data_config.json')
        print('Use the default overvoltage: 3V')
        print('**************************************************************************')
    return d

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
    return d

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
                q = module['quabos'][i]
                print("data from quabo {} ({}) -> DAQ node {}".format(q['uid'], quabo_ip_addr(ip_addr, i), daq_node['ip_addr'])
                )

## Apply global validation


def print_topology_graph(obs_conf, daq_conf, net_conf):
    console.print(Panel("[bold cyan]Observatory Topology & Routing Graph[/bold cyan]"))

    obs_name = obs_conf.get('name', 'Unknown Observatory')
    root = Tree(f"[bold magenta]Observatory: {obs_name}[/bold magenta]")

    net_module_map = {m.get('ip_addr'): m.get('port_forwarding', {}) for m in net_conf.get('modules', [])}
    daq_map = {}
    expand_ranges(daq_conf)
    for daq in daq_conf.get('daq_nodes', []):
        for mod_id in daq.get('module_ids', ''):
            daq_map[mod_id] = daq

    # Group by Gateway
    gw_tree_map = {}
    local_tree = root.add("[bold green] Local Direct Network [/bold green]")

    for dome in obs_conf.get('domes', []):
        d_name = dome.get('name', 'Unknown Dome')

        for mod in dome.get('modules', []):
            m_ip = mod.get('ip_addr')
            m_hw = mod.get('quabo_version', 'unknown')
            m_timing = mod.get('timing_mode', 'wr')

            try:
                mod_id = ip_addr_to_module_id(m_ip)
            except (ValueError, IndexError, AttributeError):
                mod_id = -1

            dest_daq = daq_map.get(mod_id)
            daq_str = f"{dest_daq.get('ip_addr')} ({dest_daq.get('bindhost', 'eth0')})" if dest_daq else "UNMAPPED"

            pf = net_module_map.get(m_ip, {})
            gw_ip = pf.get('gw_ip') if pf.get('status') else None

            # Decide which tree branch to add this to
            if gw_ip:
                if gw_ip not in gw_tree_map:
                    gw_tree_map[gw_ip] = root.add(f"[bold green] Gateway: {gw_ip}[/bold green]")
                target_node = gw_tree_map[gw_ip].add(f"[bold blue]Dome: {d_name}[/bold blue]")
            else:
                target_node = local_tree.add(f"[bold blue]Dome: {d_name}[/bold blue]")

            cmd_ports = pf.get('cmd_port', [60000] * 4)
            mod_node = target_node.add(
                f"[bold gold3] Module {mod_id} [/bold gold3][IP: {m_ip}] [HW: {m_hw}] [Timing: {m_timing}]  -> [bold dark_orange] DAQ Node: {daq_str}[/bold dark_orange]")

            for q in range(4):
                base_ip_parts = m_ip.split('.')
                q_ip = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{base_ip_parts[2]}.{int(base_ip_parts[3]) + q}" if len(
                    base_ip_parts) == 4 else "Invalid"
                real_ip = gw_ip if gw_ip else q_ip
                real_port = cmd_ports[q] if len(cmd_ports) > q else 60000
                mod_node.add(f"[bold yellow] Q{q} [/bold yellow]({q_ip}) -> {real_ip}:{real_port}")

    console.print(root)
    print("\n")



def validate_all(check_network: bool = True, debug: bool = False, graph: bool = False) -> bool:
    """
    Master validation orchestrator.
    Batches Tier-1 errors, supports Global Tier-2 validation, and topology graphing.
    """
    global IS_CLI_VALIDATION, DEBUG_VALIDATION
    IS_CLI_VALIDATION = True
    DEBUG_VALIDATION = debug

    all_passed = True
    validated_configs = {}

    console.print(Panel.fit("[bold cyan]Starting PANOSETI Configuration Validation[/bold cyan]"))

    # 1. Tier 1: Strict File Validation (Batched)
    console.print("\n[bold cyan]Running Tier-1 File Syntax & Schema Checks...[/bold cyan]")
    t1_errors = 0
    loaders = [
        ('firmware', get_firmware_config),
        ('daemons', get_daemons_config),
        ('obs', get_obs_config),
        ('network', get_network_config),
        ('daq', get_daq_config),
        ('data', get_data_config)
    ]

    for key, loader in loaders:
        try:
            validated_configs[key] = loader()
        except ValueError:
            # The specific error details are printed by load_and_validate.
            # We just catch it here so we don't crash, allowing the loop to continue.
            t1_errors += 1
            all_passed = False
            pass
        except Exception:
            console.print_exception()

    if t1_errors == 0:
        console.print("\n[green]✔ Tier-1 File Syntax & Schema Validation Passed.[/green]")
    else:
        # If Tier-1 fails, we cannot proceed to Tier-2 because the data structures are missing/corrupt.
        console.print(
            f"\n[bold red]✖ Tier-1 Validation Failed: {t1_errors} configuration file(s) contained errors.[/bold red]")
        console.print("[red]Please fix the above schema errors before proceeding to Tier-2 checks.[/red]")
        return False
    # 2. Tier 2: Global Cross-Configuration Validation
    console.print("\n[bold cyan]Running Tier-2 Global Cross-Config Checks...[/bold cyan]")
    global_validator = GlobalConfigValidator(validated_configs)
    if not global_validator.validate_all_rules():
        all_passed = False
    else:
        console.print("\n[green]✔ Tier-2 Global Cross-Config Validation Passed.[/green]")

    # 3. Visual Topology Graph
    if graph:
        print_topology_graph(
            validated_configs.get('obs'),
            validated_configs.get('daq'),
            validated_configs.get('network')
        )

    # 4. Network Ping Checks
    if check_network:
        if not perform_network_ping_sweep(validated_configs):
            all_passed = False


    if all_passed:
        console.print("\n[bold green]✅ ALL VALIDATION CHECKS PASSED.[/bold green] The observatory is ready.")
    else:
        console.print(
            "\n[bold red]❌ VALIDATION FAILED.[/bold red] Please review the errors above before observing.")

    return all_passed

def load_and_validate(validator_class, filename, dir, config_name, preprocessor=None):
    """
    Unified loader: reads JSON, applies runtime preprocessing, validates against Pydantic models.
    Batches errors by raising Exceptions instead of immediately exiting the program.
    """
    path = os.path.join(dir, filename)

    if IS_CLI_VALIDATION:
        console.print(f"\n[bold yellow]Target:[/bold yellow] {config_name}")

    if not os.path.exists(path):
        if IS_CLI_VALIDATION:
            console.print(f"[bold red][FAIL][/bold red] {filename} not found.")
            console.print(f"Target: {config_name} - [red]1 Error(s), 0 Warning(s)[/red]")
        if RAISE_VALIDATION_ERRORS:
            raise FileNotFoundError(f"{path} not found.")
        raise ValueError(f"Missing file: {filename}")

    # Symlink printing logic
    if IS_CLI_VALIDATION:
        if os.path.islink(path):
            real_path = os.path.realpath(path)
            console.print(f"[dim]Symlink detected:[/dim] {filename} -> {os.path.abspath(real_path)}")
        else:
            console.print(f"[dim]File path:[/dim] {os.path.abspath(path)}")

    # 1. Load Data
    try:
        with open(path) as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"\n[bold red][FAIL] JSON Parsing Error in {config_name} ({filename}):[/bold red] {e}")
        if IS_CLI_VALIDATION:
            console.print(f"Target: {config_name} - [red]1 Error(s), 0 Warning(s)[/red]")
        if RAISE_VALIDATION_ERRORS:
            raise e
        raise ValueError(f"JSON Parse Error in {filename}") from None

    # 2. Preprocess (e.g. assign_numbers, expand_ranges)
    if preprocessor:
        preprocessor(raw_data)

    # 3. Validate
    try:
        validated = validator_class(**raw_data)

        if IS_CLI_VALIDATION:
            console.print("[bold green][OK][/bold green] Passed validation (0 Errors, 0 Warnings).")

        if IS_CLI_VALIDATION and DEBUG_VALIDATION:
            console.print("\n[dim]Validated Configuration Structure:[/dim]")
            pprint(validated.model_dump(exclude_unset=True), expand_all=False)
        return validated.model_dump(mode='json', exclude_unset=True)

    except ValidationError as e:
        err_count = len(e.errors())
        console.print(f"\n[bold red][FAIL] Schema Validation Error in {config_name} ({filename}):[/bold red]")
        for err in e.errors():
            loc = " -> ".join([str(l) for l in err["loc"]])
            msg = err["msg"]
            console.print(f"  [bold red]Field:[/bold red] {loc}")
            console.print(f"  [bold red]Error:[/bold red] {msg}\n")
        if IS_CLI_VALIDATION and DEBUG_VALIDATION:
            console.print("[dim]Raw Config Dictionary (for debugging):[/dim]")
            pprint(raw_data, expand_all=True, max_length=5)
        if IS_CLI_VALIDATION:
            console.print(f"Target: {config_name} - [red]{err_count} Error(s), 0 Warning(s)[/red]")
            # 'from None' suppresses the double traceback in Python
            raise ValueError(f"Pydantic Validation failed for {config_name}") from None
        else:
            if RAISE_VALIDATION_ERRORS:
                raise e
            sys.exit(1)  # Clean exit for normal runs, no traceback
    except json.JSONDecodeError as e:
        console.print(f"\n[bold red][FAIL] JSON Parsing Error in {config_name} ({filename}):[/bold red] {e}")
        if IS_CLI_VALIDATION:
            console.print(f"Target: {config_name} - [red]1 Error(s), 0 Warning(s)[/red]")
            raise ValueError(f"JSON Parse Error in {filename}") from None
        else:
            if RAISE_VALIDATION_ERRORS:
                console.print_exception()
                raise e
            sys.exit(1)



if __name__ == "__main__":
    c = get_detector_info()
    print(c)

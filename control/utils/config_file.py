#! /usr/bin/env python3

# functions to read and parse config files

import os,sys,json
import subprocess
from pydantic import ValidationError
from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel

from .pydantic_config_models import (
    DataConfigValidator, ObsConfigValidator, DaqConfigValidator,
    NetworkConfigValidator, DaemonConfigValidator, FirmwareConfigValidator
)

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


# --- Orchestrated Validation Methods ---

def ping_host(ip_addr: str) -> bool:
    cmd = ['ping', '-c', '1', '-W', '1', str(ip_addr)]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception:
        return False


def validate_all(check_network: bool = False, debug: bool = False) -> bool:
    """
    The orchestrator function. Sets global CLI flags so the loaders output
    pretty formatted terminal text, loops over configs, and pings the network
    (resolving port-forwarded gateway IPs using util.py natively).
    """
    global IS_CLI_VALIDATION, DEBUG_VALIDATION, RAISE_VALIDATION_ERRORS
    IS_CLI_VALIDATION = True
    DEBUG_VALIDATION = debug
    RAISE_VALIDATION_ERRORS = True

    all_passed = True
    ips_to_ping = set()
    validated_configs = {}

    console.print(Panel.fit("[bold cyan]PANOSETI Configuration Validator[/bold cyan]"))

    configs_to_check = [
        ("Data Config", get_data_config),
        ("Obs Config", get_obs_config),
        ("DAQ Config", get_daq_config),
        ("Network Config", get_network_config),
        ("Daemons Config", get_daemons_config),
        ("Firmware Config", get_firmware_config),
    ]

    # 1. Validate all files and store the dictionaries
    for name, getter in configs_to_check:
        try:
            validated_configs[name] = getter()
        except FileNotFoundError:
            all_passed = False
        except Exception:
            all_passed = False

    # 2. Network Connectivity Check
    if check_network:
        # Import dynamically to avoid circular dependency on load
        from utils import util

        net_conf = validated_configs.get("Network Config", {})

        # Gather Obs Config IPs using util.get_quabo_ip_port
        obs_conf = validated_configs.get("Obs Config", {})
        if obs_conf.get('wr_ip_addr'):
            ips_to_ping.add(str(obs_conf['wr_ip_addr']))
        if obs_conf.get('dome_controller_ip_addr'):
            ips_to_ping.add(str(obs_conf['dome_controller_ip_addr']))

        for dome in obs_conf.get('domes', []):
            for mod in dome.get('modules', []):
                ip = mod.get('ip_addr')
                if not ip:
                    continue
                # util.get_quabo_ip_port seamlessly returns the Gateway IP if port forwarded,
                # or safely falls back to the local Quabo IP if not.
                try:
                    # We only need to ping the first Quabo representation (index 0)
                    ip_ports = util.get_quabo_ip_port(ip, 0, net_conf)
                    ips_to_ping.add(str(ip_ports['ip_addr']))
                except Exception as e:
                    console.print(f"[bold red]Error resolving IP for module {ip}: {e}[/bold red]")
                    all_passed = False

        # Gather DAQ Config IPs using util.attach_daq_config
        daq_conf = validated_configs.get("DAQ Config", {})
        if daq_conf:
            # Let util securely mutate the dictionary
            util.attach_daq_config(daq_conf, net_conf)

            if daq_conf.get('head_node_ip_addr'):
                ips_to_ping.add(str(daq_conf['head_node_ip_addr']))
            for node in daq_conf.get('daq_nodes', []):
                if 'port_forwarding' in node and node['port_forwarding'].get('status'):
                    ips_to_ping.add(str(node['port_forwarding']['gw_ip']))
                elif node.get('ip_addr'):
                    ips_to_ping.add(str(node['ip_addr']))

        # Ping the extracted IPs
        if ips_to_ping:
            console.print("\n" + "=" * 50)
            console.print(Panel.fit("[bold cyan]Network Connectivity Check[/bold cyan]"))

            for ip in sorted(list(ips_to_ping)):
                console.print(f"Pinging [bold cyan]{ip}[/bold cyan]...", end=" ")
                if ping_host(ip):
                    console.print("[bold green]ALIVE[/bold green]")
                else:
                    console.print("[bold red]UNREACHABLE[/bold red]")
                    all_passed = False

    console.print("\n" + "=" * 50)
    if all_passed:
        console.print("[bold green]SUCCESS: Validation Complete![/bold green]")
    else:
        console.print("[bold red]FAILURE: Validation found errors. See above.[/bold red]")

    # Reset globals in case standard script execution continues
    IS_CLI_VALIDATION = False
    DEBUG_VALIDATION = False
    RAISE_VALIDATION_ERRORS = False

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

if __name__ == "__main__":
    c = get_detector_info()
    print(c)

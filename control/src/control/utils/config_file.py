#! /usr/bin/env python3
from __future__ import annotations

import json

# functions to read and parse config files
# CWD CONTRACT: all relative paths in this file are relative to the control/ directory.
# Scripts must be launched from control/ (e.g. `cd control && python start.py`).
import os
import pathlib
import sys
from typing import Any

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich.tree import Tree

from control.utils.config_validator import perform_network_ping_sweep
from control.utils.global_validator import GlobalConfigValidator
from control.utils.paths import PanoPaths

# import Pydantic validation models
from control.utils.pydantic_config_models import (
    DaemonConfigValidator,
    DaqConfigValidator,
    DaqNodeValidator,
    DataConfigValidator,
    FirmwareConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
    ObsModuleConfig,
    QuaboUidsValidator,
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
obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'
network_config_filename = 'network_config.json'
daemons_config_filename = 'daemons.json'
firmware_config_filename = 'firmware.json'
# quabo realted files
quabo_info_filename = 'quabo_info.json'
detector_info_filename = 'detector_info.json'
quabo_calib_filename = 'detovervol_%dv/%s/quabo_calib_%s.json'
# These files are creatd during the run,
# and will be copied to the final data dir
quabo_uids_filename = 'quabo_uids.json'
quabo_ph_baseline_filename = 'quabo_ph_baseline.json'
sw_info_filename = 'sw_info.json'
quabo_config_filename = 'quabo_config_*.json'

# list of config files copied to data dir
config_file_names = [
    obs_config_filename, daq_config_filename, data_config_filename,
    quabo_uids_filename, quabo_ph_baseline_filename, sw_info_filename,
    quabo_config_filename, network_config_filename
]

def ip_addr_to_module_id(ip_addr_str: str) -> int:
    """Compute a 'module ID' from a base quabo IP address.
    
    The module ID is derived from bits 2..9 of the IP address 
    (formed by the 3rd and 4th octets).

    Args:
        ip_addr_str: The base IP address string (e.g., '192.168.3.200').

    Returns:
        The calculated module ID as an integer (0-255).
    """
    pieces = ip_addr_str.split('.')
    n = int(pieces[3]) + 256*int(pieces[2])
    return (n>>2)&255

def quabo_ip_addr(base: str, i: int) -> str:
    """Return the IP address of the i-th quabo in a module.

    Args:
        base: The base IP address of the module (quabo 0).
        i: The index of the quabo (0-3).

    Returns:
        The IP address string of the specified quabo.
    """
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)


def get_boardloc(module_ip_addr: str, quabo_index: int) -> int:
    """Calculate the BOARDLOC for a quabo given its module base IP and index.

    The BOARDLOC formula is: (octet3 * 256) + octet4 + quabo_index.

    Args:
        module_ip_addr: The base IP address string of the module.
        quabo_index: The index of the quabo within the module (0-3).

    Returns:
        The BOARDLOC identifier as an integer.
    """
    pieces = module_ip_addr.split('.')
    boardloc = int(pieces[2]) * 256 + int(pieces[3]) + quabo_index
    return boardloc


def assign_numbers(c: dict[str, Any]) -> None:
    """Assign sequential numbers to domes and IDs to modules within a config.
    
    This function injects 'num' into each dome object and 'id' (derived from 
    IP address) into each module object. This is a pre-validation step.

    Args:
        c: The configuration dictionary to modify in-place.
    """
    for ndome, dome in enumerate(c['domes']):
        dome['num'] = ndome
        for module in dome['modules']:
            module['id'] = ip_addr_to_module_id(module['ip_addr'])

def string_to_list(s: str) -> list[int]:
    """Parse a range string like '0-2, 5-6' into a list of integers.

    Args:
        s: A comma-separated string of ranges or individual numbers.

    Returns:
        A list of integers represented by the input string.
    """
    out: list[int] = []
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

def expand_ranges(daq_config: DaqConfigValidator | dict[str, Any]) -> None:
    """Expand module range strings to lists of module numbers in DAQ node objects.
    
    Mutates the input dictionary or model by converting 'module_ids' 
    string ranges into actual lists of integers.

    Args:
        daq_config: The DAQ configuration to process, either as a model or a dict.

    Raises:
        TypeError: If module_ids is not a string or a list.
    """
    if hasattr(daq_config, "daq_nodes"):
        # Implementation for model
        # MyPy needs help knowing daq_config is a model here if isinstance check fails it
        for node in daq_config.daq_nodes: 
            if isinstance(node.module_ids, str):
                node.module_ids = string_to_list(node.module_ids)
            elif isinstance(node.module_ids, list):
                node.module_ids = list(set(int(x) for x in node.module_ids))
            elif node.module_ids is not None:
                raise TypeError(f"module_ids must be str or list, not {type(node.module_ids)}")
    else:
        # Implementation for dict
        for node in daq_config.get('daq_nodes', []): 
            module_ids = node.get('module_ids')
            if isinstance(module_ids, str):
                node['module_ids'] = string_to_list(module_ids)
            elif isinstance(module_ids, list):
                node['module_ids'] = list(set(int(x) for x in module_ids))
            elif module_ids is not None:
                raise TypeError(f"module_ids must be str or list, not {type(module_ids)}")

def module_id_to_daq_node(daq_config: DaqConfigValidator, module_id: int) -> DaqNodeValidator:
    """Find the DAQ node responsible for handling a specific module ID.

    Args:
        daq_config: The validated DAQ configuration to search.
        module_id: The ID of the module to locate.

    Returns:
        The DaqNodeValidator object handling the module.

    Raises:
        Exception: If no DAQ node is found for the given module ID.
    """
    for node in daq_config.daq_nodes:
        # After validation/preprocessing, module_ids is list[int]
        if module_id in node.module_ids:
            return node
    raise Exception(f"no DAQ node is handling module {module_id}")

def check_config_file(name: str, dir: str = '.') -> None:
    """Verify that a configuration file exists. Exits the program if missing.

    Args:
        name: The filename to check.
        dir: The directory containing the file.
    """
    path = os.path.join(dir, name)
    if not os.path.isfile(path):
    # if not os.path.exists('%s/%s'%(dir, name)):
        print(f"The config file '{name}' doesn't exist.")
        print(f"Create a symbolic link from {name} to a specific config file, e.g.:")
        print("   ln -s {}_lick.json {}".format(name.split('.')[0], name))

        sys.exit(1)


def get_obs_config(dir: str | None = None) -> ObsConfigValidator:
    """Load and validate the observatory configuration.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated ObsConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    # pass assign_numbers so it injects `id` and `num` before validation
    return load_and_validate(ObsConfigValidator, obs_config_filename, config_dir, "Obs Config", assign_numbers)

def get_daq_config(dir: str | None = None) -> DaqConfigValidator:
    """Load and validate the DAQ configuration.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated DaqConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    # pass expand_ranges so it parses module string ranges before validation
    return load_and_validate(DaqConfigValidator, daq_config_filename, config_dir, "DAQ Config", expand_ranges)

def get_data_config(dir: str | None = None) -> DataConfigValidator:
    """Load and validate the data (science/engineering) configuration.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated DataConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    return load_and_validate(DataConfigValidator, data_config_filename, config_dir, "Data Config")

def get_network_config(dir: str | None = None) -> NetworkConfigValidator:
    """Load and validate the network configuration.

    Falls back to an empty NetworkConfigValidator (no port forwarding) with a
    warning if the file is missing or invalid, assuming a flat local network.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated NetworkConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    path = os.path.join(config_dir, network_config_filename)
    try:
        with open(path) as f:
            s = f.read()
        json.loads(s)
    except Exception:
        print("***********Warning: No network config file! **************")
        print("******All the devices should be in the same subnet *******")
        return NetworkConfigValidator()

    return load_and_validate(NetworkConfigValidator, network_config_filename, config_dir, "Network Config")


def get_firmware_config(dir: str | None = None) -> FirmwareConfigValidator:
    """Load and validate the firmware configuration.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated FirmwareConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    return load_and_validate(FirmwareConfigValidator, firmware_config_filename, config_dir, "Firmware Config")

def get_daemons_config(dir: str | None = None) -> DaemonConfigValidator:
    """Load and validate the daemons configuration.

    Args:
        dir: The directory containing the config file. Defaults to PanoPaths.config_dir().

    Returns:
        A validated DaemonConfigValidator model.
    """
    config_dir = dir if dir is not None else str(PanoPaths.config_dir())
    return load_and_validate(DaemonConfigValidator, daemons_config_filename, config_dir, "Daemons Config")

def get_quabo_uids() -> QuaboUidsValidator:
    """Load and validate the Quabo UIDs from the local cache file.

    Returns:
        A validated QuaboUidsValidator model.
    """
    path = PanoPaths.tmp_dir() / quabo_uids_filename
    if not path.exists():
        print(f"{path} is missing.  Run get_uids.py")
        sys.exit(1)
    with open(path) as f:
        s = f.read()
    quabo_uids_conf: dict[str, Any] = json.loads(s)
    assign_numbers(quabo_uids_conf)
    return QuaboUidsValidator(**quabo_uids_conf)

def get_module_quabo_uids_from_dict(uids_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Validate a raw dictionary of UIDs and return a simplified mapping.

    Args:
        uids_dict: Raw dictionary containing Quabo UID information.

    Returns:
        A dictionary mapping module IP addresses to lists of Quabo UIDs.
    """
    validated = QuaboUidsValidator(**uids_dict)
    res = {}
    for dome in validated.domes:
        for module in dome.modules:
            res[str(module.ip_addr)] = [q.uid for q in module.quabos]
    return res

def get_detector_info() -> dict[str, float]:
    """Retrieve detector operating voltages indexed by serial number.
    
    Voltages are derived from detector_info.json and may be adjusted based 
    on the detector_overvoltage setting in data_config.json.

    Returns:
        A dictionary mapping detector serial numbers to operating voltages.
    """
    check_config_file(detector_info_filename, str(PanoPaths.quabos_dir()))
    path = PanoPaths.quabos_dir() / detector_info_filename
    with open(path) as f:
        s = f.read()
    c: list[dict[str, Any]] = json.loads(s)
    d: dict[str, float] = {}

    obs_config_path = PanoPaths.config_dir() / obs_config_filename
    with open(obs_config_path) as f:
        s = f.read()
    json.loads(s)

    data_config_path = PanoPaths.config_dir() / data_config_filename
    with open(data_config_path) as f:
        s = f.read()
    data_config: dict[str, Any] = json.loads(s)
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

def get_quabo_info() -> dict[str, Any]:
    """Retrieve Quabo metadata indexed by UID.

    Returns:
        A dictionary mapping Quabo UIDs to their metadata objects.
    """
    check_config_file(quabo_info_filename, str(PanoPaths.quabos_dir()))
    path = PanoPaths.quabos_dir() / quabo_info_filename
    with open(path) as f:
        s = f.read()
    c: list[dict[str, Any]] = json.loads(s)
    d: dict[str, Any] = {}
    for q in c:
        d[q['uid']] = q
    return d

def get_quabo_ph_baselines() -> dict[str, Any]:
    """Retrieve Quabo pulse-height baselines from the local cache file.

    Returns:
        A dictionary containing the Quabo pulse-height baselines.
    """
    check_config_file(quabo_ph_baseline_filename, str(PanoPaths.tmp_dir()))
    path = PanoPaths.tmp_dir() / quabo_ph_baseline_filename
    with open(path) as f:
        s = f.read()
    c: dict[str, Any] = json.loads(s)
    return c

def get_quabo_calib(serialno: str, detovervol: int, mode: str) -> dict[str, Any]:
    """Load Quabo calibration data for a specific detector and mode.

    Args:
        serialno: The serial number of the detector.
        detovervol: The detector overvoltage (e.g., 2 or 3).
        mode: The calibration mode.

    Returns:
        A dictionary containing the calibration data.
    """
    #print('reading calib file %s'%serialno)
    path = PanoPaths.quabos_dir() / (quabo_calib_filename % (detovervol, mode, serialno))
    with open(path) as f:
        s = f.read()
    return json.loads(s)

def get_modules(c: ObsConfigValidator | dict[str, Any]) -> list[ObsModuleConfig]:
    """Extract a flat list of modules from an observatory configuration.

    Args:
        c: The observatory configuration to process.

    Returns:
        A list of ObsModuleConfig objects.
    """
    if isinstance(c, dict):
        c = ObsConfigValidator(**c)
    modules: list[ObsModuleConfig] = []
    for dome in c.domes:
        for module in dome.modules:
            modules.append(module)
    return modules

def associate(daq_config: DaqConfigValidator | dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any]) -> None:
    """Link modules to their corresponding DAQ nodes.
    
    Injects back-references between DAQ nodes and the modules they handle.
    - Adds a 'modules' list to each DAQ node object.
    - Adds a 'daq_node' link to each module object.

    Args:
        daq_config: The DAQ configuration model or dict.
        quabo_uids: The Quabo UIDs configuration model or dict.
    """
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)

    for node in daq_config.daq_nodes:
        node.modules = []

    for dome in quabo_uids.domes:
        for module in dome.modules:
            daq_node = module_id_to_daq_node(daq_config, module.id) # type: ignore
            daq_node.modules.append(module)
            module.daq_node = daq_node

def show_daq_assignments(quabo_uids: QuaboUidsValidator | dict[str, Any]) -> None:
    """Print the assignment of Quabos to DAQ nodes to the console.

    Args:
        quabo_uids: The Quabo UIDs configuration model or dict.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)

    for dome in quabo_uids.domes:
        for module in dome.modules:
            ip_addr = str(module.ip_addr)
            daq_node = module.daq_node
            daq_ip = str(daq_node.ip_addr) if daq_node else "unknown"
            for i in range(4):
                q = module.quabos[i]
                print(f"data from quabo {q.uid} ({quabo_ip_addr(ip_addr, i)}) -> DAQ node {daq_ip}")


## Apply global validation


def print_topology_graph(obs_conf: ObsConfigValidator | dict[str, Any], daq_conf: DaqConfigValidator | dict[str, Any], net_conf: NetworkConfigValidator | dict[str, Any]) -> None:
    if isinstance(obs_conf, dict):
        obs_conf = ObsConfigValidator(**obs_conf)
    if isinstance(daq_conf, dict):
        daq_conf = DaqConfigValidator(**daq_conf)
    if isinstance(net_conf, dict):
        net_conf = NetworkConfigValidator(**net_conf)

    console.print(Panel("[bold cyan]Observatory Topology & Routing Graph[/bold cyan]"))

    obs_name = obs_conf.name
    root = Tree(f"[bold magenta]Observatory: {obs_name}[/bold magenta]")

    net_module_map = {str(m.ip_addr): m.port_forwarding for m in net_conf.modules}
    daq_map = {}
    expand_ranges(daq_conf)
    
    for daq in daq_conf.daq_nodes:
        for mod_id in daq.module_ids:
            daq_map[mod_id] = daq

    # Group by Gateway
    gw_tree_map = {}
    local_tree = root.add("[bold green] Local Direct Network [/bold green]")

    for dome in obs_conf.domes:
        d_name = dome.name

        for mod in dome.modules:
            m_ip = str(mod.ip_addr)
            m_hw = mod.quabo_version
            m_timing = mod.timing_mode

            try:
                mod_id = ip_addr_to_module_id(m_ip)
            except (ValueError, IndexError, AttributeError):
                mod_id = -1

            dest_daq = daq_map.get(mod_id)
            if dest_daq:
                daq_str = f"{dest_daq.ip_addr} ({dest_daq.bindhost or 'eth0'})"
            else:
                daq_str = "UNMAPPED"

            pf = net_module_map.get(m_ip)
            gw_ip = str(pf.gw_ip) if pf and pf.status else None

            # Decide which tree branch to add this to
            if gw_ip:
                if gw_ip not in gw_tree_map:
                    gw_tree_map[gw_ip] = root.add(f"[bold green] Gateway: {gw_ip}[/bold green]")
                target_node = gw_tree_map[gw_ip].add(f"[bold blue]Dome: {d_name}[/bold blue]")
            else:
                target_node = local_tree.add(f"[bold blue]Dome: {d_name}[/bold blue]")

            cmd_ports = pf.cmd_port if pf else [60000] * 4
            mod_node = target_node.add(
                f"[bold gold3] Module {mod_id} [/bold gold3][IP: {m_ip}] [HW: {m_hw}] [Timing: {m_timing}]  -> [bold dark_orange] DAQ Node: {daq_str}[/bold dark_orange]")

            for q in range(4):
                base_ip_parts = m_ip.split('.')
                q_ip = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{base_ip_parts[2]}.{int(base_ip_parts[3]) + q}" if len(
                    base_ip_parts) == 4 else "Invalid"
                real_ip = gw_ip if gw_ip else q_ip
                real_port = cmd_ports[q] if cmd_ports and len(cmd_ports) > q else 60000
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
    validated_configs: dict[str, Any] = {}

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
    
    # Cast to ensure MyPy is happy with the cross-config validation
    global_validator = GlobalConfigValidator(validated_configs)
    if not global_validator.validate_all_rules():
        all_passed = False
    else:
        console.print("\n[green]✔ Tier-2 Global Cross-Config Validation Passed.[/green]")

    # 3. Visual Topology Graph
    if graph:
        obs_cfg: ObsConfigValidator = validated_configs['obs']
        daq_cfg: DaqConfigValidator = validated_configs['daq']
        net_cfg: NetworkConfigValidator = validated_configs['network']
        print_topology_graph(obs_cfg, daq_cfg, net_cfg)

    # 4. Network Ping Checks
    if check_network and not perform_network_ping_sweep(validated_configs):
        all_passed = False


    if all_passed:
        console.print("\n[bold green]✅ ALL VALIDATION CHECKS PASSED.[/bold green] The observatory is ready.")
    else:
        console.print(
            "\n[bold red]❌ VALIDATION FAILED.[/bold red] Please review the errors above before observing.")

    return all_passed
def load_and_validate[T: BaseModel](
    validator_class: type[T],
    filename: str,
    dir: str,
    config_name: str,
    preprocessor: Any = None
) -> T:
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
        raise ValueError(f"Missing file: {config_name} config file not found: {path}")

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
        return validated

    except ValidationError as e:
        err_count = len(e.errors())
        console.print(f"\n[bold red][FAIL] Schema Validation Error in {config_name} ({filename}):[/bold red]")
        for err in e.errors():
            loc = " -> ".join([str(part) for part in err["loc"]])
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
            # If we're not in explicit CLI validation mode, we still want to raise
            # so that orchestration (start.py) can run its rollback ladder.
            if RAISE_VALIDATION_ERRORS:
                raise e
            raise ValueError(f"Pydantic Validation failed for {config_name}: {e}") from None
    except json.JSONDecodeError as e:
        console.print(f"\n[bold red][FAIL] JSON Parsing Error in {config_name} ({filename}):[/bold red] {e}")
        if IS_CLI_VALIDATION:
            console.print(f"Target: {config_name} - [red]1 Error(s), 0 Warning(s)[/red]")
            raise ValueError(f"JSON Parse Error in {filename}") from None
        else:
            if RAISE_VALIDATION_ERRORS:
                console.print_exception()
                raise e
            raise ValueError(f"JSON Parse Error in {filename}: {e}") from None



if __name__ == "__main__":
    c = get_detector_info()
    print(c)

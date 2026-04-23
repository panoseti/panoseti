from __future__ import annotations

import contextlib
import datetime
import json

# control script utilities
# CWD CONTRACT: relative paths in this module are relative to the control/ directory.
# Scripts must be launched from control/ (e.g. `cd control && python start.py`).
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import psutil

import __main__

# this script will be copied to daq nodes,
# but the quabo_driver and config_file won't be copied to daq nodes
# TODO: we may need to improve this
try:
    from control.driver import quabo_driver
    from control.utils import config_file
    from control.utils.paths import PanoPaths
    from control.utils.pydantic_config_models import (
        DaemonConfigValidator,
        DaqConfigValidator,
        DaqNodeValidator,
        DataConfigValidator,
        NetworkConfigValidator,
        ObsConfigValidator,
        QuaboUidsValidator,
    )
except ImportError:
    # Fallback for DAQ nodes or environments without the full package installed
    import pathlib
    class PanoPaths: # type: ignore
        @classmethod
        def base_dir(cls): return pathlib.Path('.')
        @classmethod
        def tmp_dir(cls): return pathlib.Path('./tmp')
        @classmethod
        def tools_dir(cls): return pathlib.Path('./tools')
        @classmethod
        def daemons_dir(cls): return pathlib.Path('./daemons')
        @classmethod
        def logs_dir(cls): return pathlib.Path('./logs')

import logging

#-------------- DEFAULTS ---------------

default_max_file_size_mb = 0        # no limit

#-------------- FILE NAMES ---------------

run_name_file = str(PanoPaths.tmp_dir() / 'current_run')
    # stores the name of the current run
run_symlink = str(PanoPaths.base_dir() / 'run')
    # name of symlink to current run
img_symlink = str(PanoPaths.base_dir() / 'img')
ph_symlink= str(PanoPaths.base_dir() / 'ph')
hk_symlink= str(PanoPaths.base_dir() / 'hk')
    # names of symlinks to first img and ph file in current run

hk_file_name = 'hk.pff'
    # housekeeping file in run dir

# files written by stop.py
recording_ended_filename = 'recording_ended'
collect_complete_filename = 'collect_complete'
run_complete_filename = 'run_complete'

hk_recorder_name = str(PanoPaths.tools_dir() / 'store_redis_data.py')

hv_updater_name = str(PanoPaths.tools_dir() / 'hv_updater.py')

module_temp_monitor_name = str(PanoPaths.tools_dir() / 'module_temp_monitor.py')

hashpipe_name = 'hashpipe'

daq_hashpipe_pid_filename = 'daq_hashpipe_pid'
    # stores PID of hashpipe process
daq_run_name_filename = 'daq_run_name'
    # stores name of current run
hp_stdout_prefix = 'hp_stdout'
    # hashpipe stdout file is prefix_ipaddr
pss_prefix = 'pss_'
    # process snapshot file is pss_prefix_ipaddr

# Base daemons (always included in the "capture"/redis daemon set)
redis_daemons = [
    str(PanoPaths.daemons_dir() / 'storeInfluxDB.py'),
    # 'daemons/storeLoki.py'
]
#capture_power.py

#-------------- TIME ---------------

def now_str() -> str:
    t = int(time.time())
    dt = datetime.datetime.fromtimestamp(t)
    return dt.isoformat()

#-------------- NETWORK ---------------

# quabos send HK packets here at first.
# so (currently) you can only reboot quabos from this host
#
default_hk_dest = '192.168.1.100'

def daq_grpc_endpoint(node: DaqNodeValidator) -> tuple[str, int]:
    """Return (host, port) for the gRPC DAQ-control server on this node.

    Reads port_forwarding from the node model (attached by attach_daq_config).
    Falls back to direct connection on port 50051.
    """
    if node.port_forwarding and node.port_forwarding.status:
        return str(node.port_forwarding.gw_ip), node.port_forwarding.grpc_port or 50051
    return str(node.ip_addr), 50051


def local_ip() -> list[str]:
    """our IP address on local network (192.x.x.x)"""
    ips: list[str] = []
    # psutil.net_if_addrs() returns a dictionary of interfaces and their addresses
    for _interface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            # Check if it's an IPv4 address to match your previous logic
            if snic.family == socket.AF_INET:
                ips.append(str(snic.address))
                
    if not ips:
        raise Exception("can't get local IP")
    
    return ips

def ip_addr_str_to_bytes(ip_addr_str: str) -> bytearray:
    pieces = ip_addr_str.strip().split('.')
    if len(pieces) != 4:
        raise Exception(f'bad IP addr {ip_addr_str}')
    b = bytearray(4)
    for i in range(4):
        x = int(pieces[i])
        if x<0 or x>255:
            raise Exception(f'bad IP addr {ip_addr_str}')
        b[i] = x
    return b


# return true if can ping IP addr
#
def ping(ip_addr: str, cmd_port: int) -> bool:
    logging.getLogger('PANOSETI.Config.util.ping')
    #return not subprocess.run(['ping', '-c', '1', '-w', '1', '-q', ip_addr], capture_output=True).returncode
    # TODO: implement the qping cmd in the firmware
    # For now, we just use the data_packet_destination to see if we can talk to Quabo
    s = subprocess.run(['ping', '-c', '1', '-w', '1', '-q', ip_addr], capture_output=True).returncode
    if not s:
        return True
    else:
        quabo = quabo_driver.QUABO(ip_addr, cmd_port)
        return quabo.data_packet_destination('192.168.1.1')


def mac_addr_str(b: bytes) -> str:
    s: list[str] = ['']*6
    for i in range(6):
        s[i] = hex(b[i])[2:]
    return ':'.join(s)

#-------------- BINARY DATA ---------------

def print_binary(data: bytes) -> None:
    n = len(data)
    print(f'got {n} bytes')
    for i in range(n):
        print(f"{i}: {data[i]}")

#-------------- QUABO OPS ---------------

# get the UID of quabo i in a given module
#
def quabo_uid(module: dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any], i: int) -> str:
    """Retrieve the hardware UID for a specific Quabo.

    Args:
        module: A dictionary describing the module (must contain 'ip_addr').
        quabo_uids: The validated Quabo UID configuration or its dict representation.
        i: The index of the Quabo within the module (0-3).

    Returns:
        The UID string.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)

    for dome in quabo_uids.domes:
        for m in dome.modules:
            if str(m.ip_addr) == module['ip_addr']:
                q = m.quabos[i]
                return q.uid
    raise Exception("no module {} found; run get_uids.py".format(module['ip_addr']))


# see if quabo is alive by seeing if we got its UID
#
def is_quabo_alive(module: dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any], i: int) -> bool:
    return quabo_uid(module, quabo_uids, i) != ''


# is quabo new or old hardware version, as specified in obs_config?
# can be specified as either string or array of 4 strings
#
'''
def is_quabo_old_version(module, i):
    v = module['quabo_version']
    if isinstance(v, list):
        v = v[i]
    return v == 'qfp'
'''
def is_quabo_old_version(module: dict[str, Any], i: int, quabo_uids: QuaboUidsValidator | dict[str, Any], quabo_info: dict[str, Any]) -> bool | None:
    """Check if a Quabo is an older hardware version (qfp)."""
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)

    uid = ""
    for dome in quabo_uids.domes:
        for m in dome.modules:
            if str(m.ip_addr) == module['ip_addr']:
                uid = m.quabos[i].uid

    try:
        v = quabo_info[uid]['board_version']
    except (KeyError, TypeError):
        print(f'uid: {uid} can\'t be found in quabo_info.json')
        return None
    return v == 'qfp'

#-------------- RECORDING ---------------

def start_daemon(prog: str) -> None:
    if is_script_running(prog):
        print(f'{prog} is already running')
        return
    try:
        subprocess.Popen(
            [sys.executable, prog], start_new_session=True,
            close_fds=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError:
        print(f"can't launch {prog}")
        return
    print(f'started {prog}')


def _stop_daemon(prog: str, sig: int = signal.SIGKILL) -> None:
    """
    Stop a daemon started via start_daemon(prog).
    """
    for p in psutil.process_iter():
        try:
            c = p.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if prog in c:
            with contextlib.suppress(ProcessLookupError):
                os.kill(p.pid, sig)
            print(f'stopped {prog}')


def _show_daemon(prog: str) -> None:
    if is_script_running(prog):
        print(f'{prog} is running')
    else:
        print(f'{prog} is not running')


def _are_daemons_running(progs: list[str]) -> bool:
    return all(is_script_running(prog) for prog in progs)


def _safe_get_daemons_config() -> DaemonConfigValidator | dict[str, Any]:
    # Handle "util.py copied to daq nodes" case (config_file may not exist).
    try:
        return config_file.get_daemons_config()
    except Exception:
        return {}


def get_daemons() -> list[str]:
    """
    Return the capture daemon list built from daemons.json, without mutating globals.

    - Always includes daemons/storeInfluxDB.py (base list).
    - Adds daemons/capture_<k>.py for enabled items in daemons_config['daemons'].
    """
    daemons_config = _safe_get_daemons_config()
    if isinstance(daemons_config, DaemonConfigValidator):
        enabled = daemons_config.daemons.model_dump()
    else:
        enabled = daemons_config.get('daemons', {})

    lst: list[str] = list(redis_daemons)  # copy base list; do NOT mutate global
    for k, v in enabled.items():
        if v:
            lst.append(str(PanoPaths.daemons_dir() / f'capture_{k}.py'))
    return lst


# start daemons that write HK/GPS/WR data to Redis
#
def start_redis_daemons() -> None:
    for daemon in get_daemons():
        start_daemon(daemon)


def stop_redis_daemons() -> None:
    for d in get_daemons():
        _stop_daemon(d, sig=signal.SIGKILL)


def show_redis_daemons() -> None:
    for daemon in get_daemons():
        _show_daemon(daemon)


def are_redis_daemons_running() -> bool:
    return _are_daemons_running(get_daemons())


# ---- Permanent daemons (new set) ----
# Convention: daemons/permanent_<name>.py enabled by daemons.json key "permanent_daemons"
# Also includes daemons/storeInfluxDB.py by default (so it can run with permanent services).

def get_permanent_daemons() -> list[str]:
    daemons_config = _safe_get_daemons_config()
    if isinstance(daemons_config, DaemonConfigValidator):
        enabled = daemons_config.permanent_daemons.model_dump()
    else:
        enabled = daemons_config.get('permanent_daemons', {})

    lst: list[str] = [str(PanoPaths.daemons_dir() / 'storeInfluxDB.py')]
    for k, v in enabled.items():
        if v:
            lst.append(str(PanoPaths.daemons_dir() / f'permanent_{k}.py'))
    return lst


def start_permanent_daemons() -> None:
    for d in get_permanent_daemons():
        start_daemon(d)


def stop_permanent_daemons() -> None:
    for d in get_permanent_daemons():
        _stop_daemon(d, sig=signal.SIGKILL)


def show_permanent_daemons() -> None:
    for d in get_permanent_daemons():
        _show_daemon(d)


def are_permanent_daemons_running() -> bool:
    return _are_daemons_running(get_permanent_daemons())


def start_hk_recorder(daq_config: DaqConfigValidator, run_name: str) -> None:
    path = f'{daq_config.head_node_data_dir}/{run_name}/{hk_file_name}'
    try:
        subprocess.Popen([sys.executable, hk_recorder_name, path])
    except OSError:
        print("can't launch HK recorder")
        raise


# Start high-voltage updater daemon
def start_hv_updater() -> None:
    if is_hv_updater_running():
        print('hv_updater.py is already running')
        return
    try:
        subprocess.Popen([sys.executable, hv_updater_name])
    except OSError:
        print("can't launch HV updater")
        raise


# Start module temperature monitor daemon.
def start_module_temp_monitor() -> None:
    if is_module_temp_monitor_running():
        print('module_temp_monitor.py is already running')
        return
    try:
        subprocess.Popen([sys.executable, module_temp_monitor_name])
    except OSError:
        print("can't launch module temperature monitor")
        raise


# write run name to a file, and symlink 'run' to the run dir
def write_run_name(daq_config: DaqConfigValidator, run_name: str) -> None:
    with open(run_name_file, 'w') as f:
        f.write(run_name)
    if os.path.lexists(run_symlink):
        os.unlink(run_symlink)
    run_dir = f'{daq_config.head_node_data_dir}/{run_name}'
    os.symlink(run_dir, run_symlink, True)
    # record the run name in skymap_info_dir, which will be used by skymap_helper
    shutil.copy(run_name_file, 'tmp/skymap_info_dir')


def read_run_name() -> str | None:
    if not os.path.exists(run_name_file):
        return None
    with open(run_name_file) as f:
        return f.read()


def remove_run_name() -> None:
    if os.path.exists(run_name_file):
        os.unlink(run_name_file)


# if hashpipe is running, send it a SIGINT and wait for it to exit
#
def stop_hashpipe(pid: int) -> bool:
    for p in psutil.process_iter():
        if p.pid == pid and p.name() == hashpipe_name:
            os.kill(pid, signal.SIGINT)
            while True:
                try:
                    os.kill(pid, 0)
                except (OSError, ProcessLookupError):
                    return True
                time.sleep(0.1)
    return False


def is_script_running(script: str) -> bool:
    """
    Original behavior checked only for './<script>' in cmdline.
    This version also recognizes '<script>' (without './') to match
    scripts launched via subprocess.Popen([script]) elsewhere in this file.
    """
    s1 = f'./{script}'
    s2 = f'{script}'
    for p in psutil.process_iter():
        try:
            cmd = p.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if s1 in cmd or s2 in cmd:
            return True
    return False


def is_hashpipe_running() -> bool:
    return any(p.name() == hashpipe_name for p in psutil.process_iter())


def is_hk_recorder_running() -> bool:
    for p in psutil.process_iter():
        try:
            if hk_recorder_name in p.cmdline():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def is_hv_updater_running() -> bool:
    return is_script_running(hv_updater_name)


def is_module_temp_monitor_running() -> bool:
    return is_script_running(module_temp_monitor_name)


def kill_hashpipe() -> None:
    for p in psutil.process_iter():
        if p.name() == hashpipe_name:
            os.kill(p.pid, signal.SIGKILL)


def kill_hk_recorder() -> None:
    for p in psutil.process_iter():
        try:
            if hk_recorder_name in p.cmdline():
                os.kill(p.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def kill_hv_updater() -> None:
    for p in psutil.process_iter():
        try:
            if hv_updater_name in p.cmdline():
                os.kill(p.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def kill_module_temp_monitor() -> None:
    for p in psutil.process_iter():
        try:
            if module_temp_monitor_name in p.cmdline():
                os.kill(p.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


# write a message to per-run log file, and to stdout
#
def write_log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%B %d, %Y, %I:%M%p")
    log_line = f"{__main__.__file__}: {now}: {msg}"
    print(log_line)
    try:
        with open("run/log.txt", "a") as f:
            f.write(log_line)
    except OSError:
        with open("log.txt", "a") as f:
            f.write(log_line)


def disk_usage(dir: str) -> int:
    x = 0
    for f in os.listdir(dir):
        x += os.path.getsize(f'{dir}/{f}')
    return x


def free_space(path: str) -> int:
    _total, _used, free = shutil.disk_usage(os.path.realpath(path))
    return free


# estimate bytes per second per module for a given data config
def daq_bytes_per_sec_per_module(data_config: DataConfigValidator | dict[str, Any]) -> float:
    """Estimate the data generation rate (bytes per second) per module.
    
    This calculation includes overhead for housekeeping (hk.pff), 
    image mode data (if enabled), and pulse-height mode events.

    Args:
        data_config: The science/engineering configuration model or dict.

    Returns:
        Estimated data rate in bytes per second per module.
    """
    if isinstance(data_config, dict):
        data_config = DataConfigValidator(**data_config)

    img_json_header_size = 600
    ph_json_header_size = 150
    x = 0.0

    # hk.pff
    x += 2000 + 800*4

    if data_config.image:
        image = data_config.image
        fps = 1e6/image.integration_time_usec
        bpf = 1 if image.quabo_sample_size == 8 else 2
        x += fps*(1024*bpf + img_json_header_size)
    if data_config.pulse_height:
        # assume one PH event per sec per quabo
        ph_per_sec = 1
        x += ph_per_sec*(4*(256*2+ph_json_header_size))
    return x


def get_daq_node_status(node: DaqNodeValidator | dict[str, Any]) -> dict[str, Any]:
    """Retrieve the DAQ status from a remote node via SSH.

    Args:
        node: A dictionary or validator describing the DAQ node (username, ip_addr, data_dir).

    Returns:
        A dictionary containing the parsed JSON status from the remote node.

    Raises:
        Exception: If the remote node cannot be reached.
    """
    if isinstance(node, dict):
        node = DaqNodeValidator(**node)

    # TODO: add port forwarding code here
    x = subprocess.run(['ssh',
        f'{node.username}@{node.ip_addr}',
        f'cd {node.data_dir}; ./status_daq.py',
        ],
        stdout = subprocess.PIPE
    )
    if x.stdout == b'':
        raise Exception("can't talk to DAQ node")
    y = x.stdout.decode()
    return json.loads(y)

#-------------- functions only for DAQ nodes ---------------

def daq_get_run_name() -> str | None:
    """Extract the current PANOSETI run name from the local cache file.

    Returns:
        The run name string if found, otherwise None.
    """
    if os.path.exists(daq_run_name_filename):
        with open(daq_run_name_filename) as f:
            return f.read().strip()
    return None

#-------------- WR and GPS---------------

def get_wr_ip_addr(obs_config: ObsConfigValidator | dict[str, Any]) -> str:
    """Retrieve the White Rabbit switch IP address from the configuration.

    Args:
        obs_config: The observatory configuration model or dict.

    Returns:
        The White Rabbit IP address string (defaults to 192.168.1.254).
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)
    
    if obs_config.wr_ip_addr:
        return str(obs_config.wr_ip_addr)
    return '192.168.1.254'


def get_gps_port(obs_config: ObsConfigValidator | dict[str, Any]) -> str:
    """Retrieve the TTY device path for the GPS receiver.

    Args:
        obs_config: The observatory configuration model or dict.

    Returns:
        The GPS port string (defaults to /dev/ttyUSB0).
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)
    
    if obs_config.gps_port:
        return str(obs_config.gps_port)
    return '/dev/ttyUSB0'


# We may use port forwarding, so we need to get the real IP and ports.
# this is based on the network_config.
#
DEFAULT_CMD_PORT=60000
DEFAULT_REBOOT_PORT=69
def get_quabo_ip_port(ip_addr: str, i: int, network_config: NetworkConfigValidator | dict[str, Any]) -> dict[str, Any]:
    """Determine the effective IP and port for a specific Quabo.

    Accounts for network port forwarding if configured. If no mapping
    is found, it defaults to the standard local subnet layout.

    Args:
        ip_addr: The base IP address of the module.
        i: The index of the Quabo within the module (0-3).
        network_config: The network configuration model or dict.

    Returns:
        A dictionary containing 'ip_addr', 'reboot_port', and 'cmd_port'.
    """
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    ip_ports: dict[str, Any] = {}
    x = ip_addr.split('.')
    x[3] = str(int(x[3])+i)
    quabo_ip =  '.'.join(x)

    # these are the default config
    ip_ports['ip_addr'] = quabo_ip
    ip_ports['reboot_port'] = DEFAULT_REBOOT_PORT
    ip_ports['cmd_port'] = DEFAULT_CMD_PORT

    for m in network_config.modules:
        if ip_addr == str(m.ip_addr):
            p = m.port_forwarding
            if p and p.status:
                ip_ports['ip_addr'] = str(p.gw_ip)
                if p.reboot_port:
                    ip_ports['reboot_port'] = p.reboot_port[i]
                if p.cmd_port:
                    ip_ports['cmd_port'] = p.cmd_port[i]
            break
    return ip_ports


def stop_data_flow(
    quabo_uids: QuaboUidsValidator,
    network_config: NetworkConfigValidator,
) -> None:
    """Tells all Quabos to stop sending data. Used for rollback and clean shutdown.

    Sends a DAQ_PARAMS command with all acquisition modes disabled to
    every Quabo listed in the UID cache.

    Args:
        quabo_uids: The validated Quabo UID configuration.
        network_config: The validated network configuration model.
    """
    daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
    for dome in quabo_uids.domes:
        for module in dome.modules:
            base_ip_addr = str(module.ip_addr)
            for i in range(4):
                if module.quabos[i].uid == '':
                    continue
                ip_ports = get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                quabo = quabo_driver.QUABO(real_ip, cmd_port)
                quabo.send_daq_params(daq_params)
                quabo.close()


def attach_daq_config(
    daq_config: DaqConfigValidator,
    network_config: NetworkConfigValidator | None,
) -> None:
    """Merge port forwarding metadata into the DAQ configuration.

    Iterates through DAQ nodes and attaches corresponding port forwarding
    details from the network configuration if the IP addresses match.

    Args:
        daq_config: The validated DAQ configuration model to modify in-place.
        network_config: The validated network configuration model.
    """
    if network_config is not None:
        for daq in daq_config.daq_nodes:
            for pdaq in network_config.daq_nodes:
                if str(daq.ip_addr) == str(pdaq.ip_addr) and pdaq.port_forwarding.status:
                    daq.port_forwarding = pdaq.port_forwarding



def get_valid_ip(obs_config: ObsConfigValidator) -> list[str]:
    """Extract all valid Quabo IP addresses from the observatory configuration.

    Args:
        obs_config: The validated observatory configuration model.

    Returns:
        A list of IP address strings for all quabos in all modules.
    """
    ips: list[str] = []
    for dome in obs_config.domes:
        for m in dome.modules:
            ip = str(m.ip_addr)
            ip_str = ip.split('.')
            for i in range(4):
                val = int(ip_str[3]) + i
                quabo_ip = f"{ip_str[0]}.{ip_str[1]}.{ip_str[2]}.{val}"
                ips.append(quabo_ip)
    return ips


def convert_ip(ip: str) -> tuple[str, int]:
    """Convert an IP address or module ID into its base IP and Quabo index.

    Args:
        ip: Either an IP address string or an integer module ID.

    Returns:
        A tuple of (base_ip_string, quabo_index).
    """
    try:
        qid = int(ip)
        return f"192.168.{qid>>8}.{qid&0xfc}", qid&0x3
    except Exception:
        ipstr = ip.split('.')
        last = int(ipstr[3])
        blast = 4*(last//4)
        index = last - blast
        return f"{ipstr[0]}.{ipstr[1]}.{ipstr[2]}.{blast!s}", index



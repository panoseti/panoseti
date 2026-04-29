from __future__ import annotations

import contextlib
import datetime
import json
import logging

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
from ipaddress import ip_address
from typing import Any

import psutil
from pydantic import IPvAnyAddress

import __main__

# this script will be copied to daq nodes,
# but the quabo_driver and config_file won't be copied to daq nodes
# TODO: we may need to improve this
from control.driver import quabo_driver
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaemonConfig,
    DaqConfig,
    DaqNode,
    DataConfig,
    NetworkConfig,
    ObsConfig,
    ObsModuleConfig,
    QuaboIpPorts,
    QuaboUids,
)

#-------------- DEFAULTS ---------------

default_max_file_size_mb = 0        # no limit

# SSH options for automation (BatchMode skips passwords, No checking avoids prompt)
ssh_options = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=5"
]

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
# We store filenames and resolve them lazily to absolute paths via PanoPaths
# to ensure they respect environment overrides (e.g. in Tier 2 isolated tests).
redis_daemons = [
    'storeInfluxDB.py',
    # 'storeLoki.py'
]

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

def daq_grpc_endpoint(node: DaqNode, daq_config: DaqConfig | None = None) -> tuple[str, int]:
    """Return (host, port) for the gRPC DAQ-control server on this node.

    Reads port_forwarding from the node model (attached by attach_daq_config).
    If we are not local to the node, it uses the gateway IP and forwarded port.
    Falls back to direct connection on port 50051.
    """
    # 1. If we have a daq_config, check if we are actually ON the node or in its local network.
    # If so, bypass the gateway entirely.
    if daq_config and is_local(node.ip_addr, daq_config):
        return str(node.ip_addr), 50051

    # 2. If port forwarding is enabled and we are not local, use the gateway.
    if node.port_forwarding and node.port_forwarding.status:
        # If grpc_port is None, assume the default 50051 is forwarded.
        host = str(node.port_forwarding.gw_ip)
        port = node.port_forwarding.grpc_port or 50051
        return host, port

    # 3. Default direct connection
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


def is_local(ip_addr: str | IPvAnyAddress, daq_config: DaqConfig) -> bool:
    """Return True if the IP address refers to the local machine.
    
    In CI/containerized environments (head_node_container=True), the 
    specified head_node_ip_addr is always treated as local even if 
    the container's dynamic IP doesn't match it.
    """
    ip_str = str(ip_addr)
    try:
        if ip_str in local_ip():
            return True
    except OSError:
        pass
    return bool(daq_config.head_node_container and ip_str == str(daq_config.head_node_ip_addr))

def ip_addr_str_to_bytes(ip_addr: IPvAnyAddress) -> bytearray:
    ip_addr_str = str(ip_addr)
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
def ping(ip_addr: IPvAnyAddress, cmd_port: int) -> bool:
    ip_addr_str = str(ip_addr)
    logging.getLogger('PANOSETI.Config.util.ping')
    #return not subprocess.run(['ping', '-c', '1', '-w', '1', '-q', ip_addr], capture_output=True).returncode
    # TODO: implement the qping cmd in the firmware
    # For now, we just use the data_packet_destination to see if we can talk to Quabo
    s = subprocess.run(['ping', '-c', '1', '-w', '1', '-q', ip_addr_str], capture_output=True).returncode
    if not s:
        return True
    else:
        quabo = quabo_driver.QUABO(ip_addr, cmd_port)
        return quabo.data_packet_destination(ip_address('192.168.1.1'))


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
def quabo_uid(module: ObsModuleConfig, quabo_uids: QuaboUids, i: int) -> str:
    """Retrieve the hardware UID for a specific Quabo.

    Args:
        module: A configuration model describing the module.
        quabo_uids: The validated Quabo UID configuration.
        i: The index of the Quabo within the module (0-3).

    Returns:
        The UID string.
    """
    m_ip_str = str(module.ip_addr)
    for dome in quabo_uids.domes:
        for m in dome.modules:
            if str(m.ip_addr) == m_ip_str:
                q = m.quabos[i]
                return q.uid
    raise Exception(f"no module {m_ip_str} found; run get_uids.py")


# see if quabo is alive by seeing if we got its UID
#
def is_quabo_alive(module: ObsModuleConfig, quabo_uids: QuaboUids, i: int) -> bool:
    return quabo_uid(module, quabo_uids, i) != ''


# is quabo new or old hardware version, as specified in obs_config?
# can be specified as either string or array of 4 strings
#
def is_quabo_old_version(module: ObsModuleConfig, i: int, quabo_uids: QuaboUids, quabo_info: dict[str, Any]) -> bool | None:
    """Check if a Quabo is an older hardware version (qfp)."""
    uid = quabo_uid(module, quabo_uids, i)

    try:
        v = quabo_info[uid]['board_version']
    except (KeyError, TypeError):
        print(f'uid: {uid} can\'t be found in quabo_info.json')
        return None
    return v == 'qfp'

#-------------- RECORDING ---------------

def start_daemon(prog: str | list[str], name: str | None = None) -> None:
    """Launch a daemon process in a new session, detached from the caller.

    stdout and stderr are redirected to ``state/logs/<name>/stdout.log`` and
    ``state/logs/<name>/stderr.log`` so that crashes are always recoverable
    even before the daemon's structured logger is initialised.

    Args:
        prog: Either a path to a Python script (str) or a full command list
            such as ``["python", "-m", "control.transfer"]``.  When a str is
            given the daemon is launched as ``[sys.executable, prog]``.  When a
            list is given it is used verbatim, enabling module-style invocation.
        name: Human-readable daemon name used as the log directory stem.  When
            ``None``, derived from the last path component of *prog*.
    """
    if isinstance(prog, list):
        cmd = prog
        prog_label = " ".join(prog)
        _name = name or cmd[-1].replace("-", "_").replace(".", "_")
    else:
        cmd = [sys.executable, prog]
        prog_label = prog
        _name = name or os.path.splitext(os.path.basename(prog))[0]

    if is_script_running(prog_label):
        print(f'{prog_label} is already running')
        return

    log_dir = PanoPaths.daemon_logs_dir(_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    try:
        stdout_fd = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        stderr_fd = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        subprocess.Popen(
            cmd, start_new_session=True,
            close_fds=True, stdin=subprocess.DEVNULL,
            stdout=stdout_fd, stderr=stderr_fd,
        )
        os.close(stdout_fd)
        os.close(stderr_fd)
    except OSError:
        print(f"can't launch {prog_label}")
        return
    print(f'started {prog_label}')


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


def _safe_get_daemons_config() -> DaemonConfig | None:
    # Handle "util.py copied to daq nodes" case (config_file may not exist).
    try:
        return config_file.get_daemons_config()
    except Exception:
        return None


def get_daemons() -> list[str]:
    """
    Return the capture daemon list built from daemons.json, without mutating globals.

    - Always includes daemons/storeInfluxDB.py (base list).
    - Adds daemons/capture_<k>.py for enabled items in daemons_config['daemons'].
    """
    daemons_config = _safe_get_daemons_config()
    enabled = daemons_config.daemons.model_dump() if daemons_config else {}

    # Copy base list and resolve to absolute paths
    lst: list[str] = [
        str(PanoPaths.daemons_dir() / d) for d in redis_daemons
    ]
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
    enabled = daemons_config.permanent_daemons.model_dump() if daemons_config else {}

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


def start_hk_recorder(daq_config: DaqConfig, run_name: str) -> None:
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
def write_run_name(daq_config: DaqConfig, run_name: str) -> None:
    with open(run_name_file, 'w') as f:
        f.write(run_name)
    if os.path.lexists(run_symlink):
        os.unlink(run_symlink)
    run_dir = f'{daq_config.head_node_data_dir}/{run_name}'
    os.symlink(run_dir, run_symlink, True)
    # record the run name in skymap_info_dir, which will be used by skymap_helper
    # skymap_info_dir = 'tmp/skymap_info_dir'
    # os.makedirs(skymap_info_dir, exist_ok=True)
    # shutil.copy(run_name_file, skymap_info_dir)


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
def daq_bytes_per_sec_per_module(data_config: DataConfig) -> float:
    """Estimate the data generation rate (bytes per second) per module.
    
    This calculation includes overhead for housekeeping (hk.pff), 
    image mode data (if enabled), and pulse-height mode events.

    Args:
        data_config: The science/engineering configuration model.

    Returns:
        Estimated data rate in bytes per second per module.
    """
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


def get_daq_node_status(node: DaqNode) -> dict[str, Any]:
    """Retrieve the DAQ status from a remote node via SSH.

    Args:
        node: A validator describing the DAQ node (username, ip_addr, data_dir).

    Returns:
        A dictionary containing the parsed JSON status from the remote node.

    Raises:
        Exception: If the remote node cannot be reached.
    """
    # TODO: add port forwarding code here
    x = subprocess.run(['ssh', *ssh_options,
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

def get_wr_ip_addr(obs_config: ObsConfig) -> str:
    """Retrieve the White Rabbit switch IP address from the configuration.

    Args:
        obs_config: The observatory configuration model.

    Returns:
        The White Rabbit IP address string (defaults to 192.168.1.254).
    """
    if obs_config.wr_ip_addr:
        return str(obs_config.wr_ip_addr)
    return '192.168.1.254'


def get_gps_port(obs_config: ObsConfig) -> str:
    """Retrieve the TTY device path for the GPS receiver.

    Args:
        obs_config: The observatory configuration model.

    Returns:
        The GPS port string (defaults to /dev/ttyUSB0).
    """
    if obs_config.gps_port:
        return str(obs_config.gps_port)
    return '/dev/ttyUSB0'


# We may use port forwarding, so we need to get the real IP and ports.
# this is based on the network_config.
#
DEFAULT_CMD_PORT=60000
DEFAULT_REBOOT_PORT=69
def get_quabo_ip_port(ip_addr: IPvAnyAddress, i: int, network_config: NetworkConfig) -> QuaboIpPorts:
    """Determine the effective IP and port for a specific Quabo.

    Accounts for network port forwarding if configured. If no mapping
    is found, it defaults to the standard local subnet layout.

    Args:
        ip_addr: The base IP address of the module.
        i: The index of the Quabo within the module (0-3).
        network_config: The network configuration model.

    Returns:
        A QuaboIpPorts model containing 'ip_addr', 'reboot_port', and 'cmd_port'.
    """
    ip_addr_str = str(ip_addr)
    x = ip_addr_str.split('.')
    x[3] = str(int(x[3])+i)
    quabo_ip =  '.'.join(x)

    # these are the default config
    real_ip = quabo_ip
    reboot_port = DEFAULT_REBOOT_PORT
    cmd_port = DEFAULT_CMD_PORT

    for m in network_config.modules:
        if ip_addr_str == str(m.ip_addr):
            p = m.port_forwarding
            if p and p.status:
                real_ip = str(p.gw_ip)
                if p.reboot_port:
                    from typing import cast
                    reboot_port = cast(int, p.reboot_port[i])
                if p.cmd_port:
                    from typing import cast
                    cmd_port = cast(int, p.cmd_port[i])
            break
    return QuaboIpPorts(
        ip_addr=ip_address(real_ip),
        reboot_port=reboot_port,
        cmd_port=cmd_port
    )


def stop_data_flow(
    quabo_uids: QuaboUids,
    network_config: NetworkConfig,
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
            for i in range(4):
                if module.quabos[i].uid == '':
                    continue
                ip_ports = get_quabo_ip_port(module.ip_addr, i, network_config)
                real_ip = ip_ports.ip_addr
                cmd_port = ip_ports.cmd_port
                quabo = quabo_driver.QUABO(real_ip, cmd_port)
                quabo.send_daq_params(daq_params)
                quabo.close()


def attach_daq_config(
    daq_config: DaqConfig,
    network_config: NetworkConfig | None,
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



def get_valid_ip(obs_config: ObsConfig) -> list[str]:
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



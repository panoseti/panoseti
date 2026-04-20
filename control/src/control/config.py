#! /usr/bin/env python3

# Initialize for (one or more) observing runs
# See usage() for options.
# see matlab/initq.m, startq*.py

# ---- PRINT WRAPPER: prefix UTC timestamp + prepend to UT-day logfile ----
import builtins as _builtins
import contextlib
import copy
import datetime
import json
import logging
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import typer
from panoseti_grpc.telemetry.logger import get_logger

from control.driver import quabo_driver
from control.driver.quabo_tftp import tftpw
from control.utils import config_file, file_xfer, pixel_coords, util
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DataConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
    ObsModuleConfig,
    QuaboUidsValidator,
)

app = typer.Typer(
    help="config.py alias",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"allow_extra_args": True, "help_option_names": ["-h", "--help"]},
)

firmware_silver_qfp = 'quabo_0206_2846D1AE.bin'
firmware_silver_bga = 'quabo_0207_28514055.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

_builtin_print = _builtins.print

def _utc_ts() -> str:
    # Human-readable UTC timestamp
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S UT")

def _ut_yyyymmdd() -> str:
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).strftime("%Y%m%d")

def _datarec_log_path() -> str:
    yyyymmdd = _ut_yyyymmdd()
    d = f"/mnt/data11/data/palomar/L0/{yyyymmdd}/obslogs"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"datarec_{yyyymmdd}.log")

def _prepend_to_file(path: str, text: str) -> None:
    # Prepend text to the beginning of the file (newest entries on top)
    tmp_path = path + ".tmp"
    old = ""
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                old = f.read()
    except Exception:
        old = ""
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        if old:
            f.write(old)
    os.replace(tmp_path, path)

def print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)

    msg = sep.join(str(a) for a in args) + end
    ts = _utc_ts()

    parts = msg.splitlines(True)  # keep line endings
    out_parts = []
    for p in parts:
        out_parts.append(f"{ts} {p}")

    out_text = "".join(out_parts)

    # Console output
    _builtin_print(out_text, end="", flush=flush)

    # Prepend to logfile
    with contextlib.suppress(Exception):
        _prepend_to_file(_datarec_log_path(), out_text)
# -----------------------------------------------------------------------


def ask_use_default_calibration(ip_addr: str) -> bool:
    while True:
        choice = input(f"Use default calibration file for {ip_addr}? (Y/N): ").strip().upper()
        if choice == "Y":
            return True
        elif choice == "N":
            return False
        else:
            print("Invalid input. Please enter Y or N.")

# print summary of obs and daq config files
#
def show_config(obs_config: ObsConfigValidator, quabo_uids: QuaboUidsValidator) -> None:
    """Print a human-readable summary of the observatory and hardware configuration.
    
    Args:
        obs_config: Validated observatory configuration model.
        quabo_uids: Validated Quabo UID registry model.
    """
    logger = logging.getLogger('PANOSETI.Config.show_config')
    logger.info('Show config')
    for dome in obs_config.domes:
        print(f'dome {dome.name}')
        for module in dome.modules:
            module_id = module.id
            ip_addr = str(module.ip_addr)
            print(f'   module ID {module_id}')
            print(f'      Mobo serial#: {module.mobo_serialno}')
            for i in range(4):
                quabo_ip = config_file.quabo_ip_addr(ip_addr, i)
                print(f'      quabo {i}')
                print(f'         IP addr: {quabo_ip}')
    #print(f"This node's IP addr: {util.local_ip()}")
    config_file.show_daq_assignments(quabo_uids.model_dump())

def do_reboot_single_quabo(ip: str, obs_config: ObsConfigValidator, network_config: NetworkConfigValidator | dict[str, Any], timeout: int = 60) -> None:
    """Reboot a specific Quabo identified by its IP address or module ID.

    Args:
        ip: Target IP address or module ID string.
        obs_config: Physical observatory configuration.
        network_config: Network routing configuration.
        timeout: Maximum seconds to wait for reboot completion (default 60).
    """
    logger = logging.getLogger('PANOSETI.Config.do_reboot_single_quabo')
    logger.info(f"The Quabo IP address/ID is {ip}.")
    ips = util.get_valid_ip(obs_config)
    ip_base, index = util.convert_ip(ip)
    ip_addr = config_file.quabo_ip_addr(ip_base, index)
    if ip_addr not in ips:
        logger.error(f"{ip} is not a valid IP address or Quabo ID.")
        return
    else:
        print(f'Rebooting {ip_addr}...')
        ip_ports = util.get_quabo_ip_port(ip_base, index, network_config)
        real_ip = ip_ports['ip_addr']
        cmd_port = ip_ports['cmd_port']
        reboot_port = ip_ports['reboot_port']
        logger.info(f'Quabo IP: {ip_addr}')
        logger.info(f'Real IP: {real_ip}')
        logger.info(f'Reboot port: {reboot_port}')
        x = tftpw(real_ip, reboot_port)
        x.reboot()
        # wait for the board to reboot
        timeout_remaining = timeout
        time.sleep(30)
        timeout_remaining -= 30
        while timeout_remaining > 0:
            logger.info(f'Pinging {ip_addr}; Timeout Remaining {timeout_remaining}s... ')
            if util.ping(real_ip, cmd_port):
                print(f'pinged {ip_addr}; reboot done')
                logger.info(f'Quabo ({ip_addr}) is rebooted successfully.')
                break
            time.sleep(5)
            timeout_remaining -= 5
        if timeout_remaining <= 0:
            print(f'reboot {ip_addr} failed; timeout ({timeout}s)')
            logger.error(f'Quabo ({ip_addr}) is failed to rebooted.')

# Reboot one module
#
def reboot_module(module: ObsModuleConfig | dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any], timeout: int = 60) -> list[dict[str, bool]]:
    """Reboot all four Quabos within a specific module sequentially.
    
    Configures timing mode (WR/GNSS) on Quabo 0 before rebooting.

    Args:
        module: Configuration for the target module.
        quabo_uids: Quabo hardware UID registry.
        network_config: Network routing rules.
        timeout: Seconds to wait for each Quabo to reboot (default 60).

    Returns:
        A list of status dictionaries for each Quabo reboot.
    """
    # Reboot the four quabos one by one
    if isinstance(module, dict):
        module = ObsModuleConfig(**module)
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.reboot_module')
    reboot_status: list[dict[str, bool]] = []
    
    m_ip = str(module.ip_addr)
    
    for i in range(4):
        if not util.is_quabo_alive(module.model_dump(), quabo_uids.model_dump(), i):
            continue
        ip_addr = config_file.quabo_ip_addr(m_ip, i)
        if i == 0:
            if module.timing_mode == 'gnss':
                print('*******************************************************')
                print(f'Timing Mode for Quabo ({ip_addr}): GNSS')
                print('*******************************************************')
            else:
                print('*******************************************************')
                print(f'Timing Mode for Quabo ({ip_addr}): WR')
                print('*******************************************************')
        
        print(f'rebooting quabo at {ip_addr}')
        ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
        real_ip = ip_ports['ip_addr']
        cmd_port = ip_ports['cmd_port']
        reboot_port = ip_ports['reboot_port']
        logger.info(f'Quabo IP: {ip_addr}')
        logger.info(f'Real IP: {real_ip}')
        logger.info(f'Reboot port: {reboot_port}')
        x = tftpw(real_ip, reboot_port)
        # check timing mode, and only use it on Quabo0
        if i == 0:
            if module.timing_mode == 'gnss':
                x.put_wrpc_filesys('wr/wrpc_filesys_gnss')
                logger.info(f'Set Timing Mode to GNSS on Quabo {ip_addr}')
            else:
                x.put_wrpc_filesys('wr/wrpc_filesys')
                logger.info(f'Set Timing Mode to WR on Quabo {ip_addr}')
        
        x.reboot()
        # wait for a while to let the quabo get rebooted successfully
        timeout_remaining = timeout
        time.sleep(30)
        timeout_remaining -= 30
        # check if the quabo is back online
        while timeout_remaining > 0:
            logger.info(f'Pinging {ip_addr}; Timeout Remaining {timeout_remaining}s... ')
            if util.ping(real_ip, cmd_port):
                reboot_status.append({f"{ip_addr}" : True})
                print(f'pinged {ip_addr}; reboot done')
                logger.info(f'Quabo ({ip_addr}) is rebooted successfully.')
                break
            else:
                time.sleep(5)
                timeout_remaining -= 5
        if timeout_remaining <=0:
            reboot_status.append({f"{ip_addr}" : False})
            print(f'reboot {ip_addr} failed; timeout ({timeout}s)')
            logger.error(f'Quabo ({ip_addr}) is failed to rebooted.')
    return reboot_status

def do_reboot(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any]) -> None:
    """Reboot multiple modules in parallel across the observatory.
    
    Reboots occur in lockstep (all Q0s, then all Q1s, etc.) to optimize 
    wait times while ensuring sequential ordering within modules.

    Args:
        modules: List of module configuration models or dicts.
        quabo_uids: Quabo hardware UID registry.
        network_config: Network routing rules.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)
    
    logger = logging.getLogger('PANOSETI.Config.do_reboot')
    logger.info("Rebooting all of the modules in parallel...")
    start_time = time.time()
    start_dt = datetime.datetime.fromtimestamp(start_time)
    nmodules = len(modules)
    with ThreadPoolExecutor(max_workers=nmodules) as pool:
        futures = {
            pool.submit(reboot_module, module, quabo_uids, network_config): module
            for module in modules
        }
    logger.info('Checking the reboot status...')
    print('*******************************************************')
    print("Reboot Status Summary:")
    print('*******************************************************')
    for f in as_completed(futures):
        status = f.result()
        for s in status:
            for k, v in s.items():
                if v:
                    print(f'Reboot {k} successfully.')
                else:
                    print(f'Reboot {k} failed.')
                logger.info(f"Rebooting {k} status is {v}.")
    print('*******************************************************')
    end_time = time.time()
    end_dt = datetime.datetime.fromtimestamp(end_time)
    elapsed = int(end_time - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    print("Reboot Start Time :", start_dt.strftime("%Y-%m-%d %H:%M:%S"))
    print("Reboot Stop  Time :", end_dt.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Reboot Process Time: {minutes} minutes {seconds} seconds")
    print('*******************************************************')

def do_loads(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], quabo_info: dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any]) -> None:
    """Load firmware binaries onto multiple modules via TFTP.
    
    Automatically selects the correct binary (BGA/QFP) based on Quabo hardware version.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        quabo_info: Detailed Quabo metadata.
        network_config: Network routing rules.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_loads')
    # TODO The hard-coded path may not be good
    firmware = config_file.get_firmware_config()
    # firmware is FirmwareConfigValidator, which allows extra fields
    # We access them via .model_extra
    extra = firmware.model_extra or {}
    firmware_silver_qfp = 'firmware/' + extra.get('qfp', '')
    firmware_silver_bga = 'firmware/' + extra.get('bga', '')
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            if not util.is_quabo_alive(module.model_dump(), quabo_uids.model_dump(), i):
                continue
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            port = ip_ports['reboot_port']
            logger.info(f'Real IP: {real_ip}')
            logger.info('Reboot Port: %d', port)
            if util.is_quabo_old_version(module.model_dump(), i, quabo_uids.model_dump(), quabo_info):
                fw = firmware_silver_qfp
                logger.info(f'Loading firmware: {firmware_silver_qfp}')
            else:
                fw = firmware_silver_bga
                logger.info(f'Loading firmware: {firmware_silver_bga}')
            x = tftpw(real_ip, port)
            print(f'loading {fw} into {ip_addr}')
            x.put_bin_file(fw)

def do_loadg(modules: list[dict[str, Any]]) -> None:
    print("not supported")
    #x.put_bin_file(firmware_gold, 0x0)

def do_ping(modules: list[ObsModuleConfig] | list[dict[str, Any]], network_config: NetworkConfigValidator | dict[str, Any], verbose: bool = False) -> dict[str, list[str]]:
    """Check network reachability for all Quabos in the specified modules.

    Args:
        modules: List of target module configurations.
        network_config: Network routing rules.
        verbose: If True, prints ping results to console.

    Returns:
        A dictionary containing lists of successful ('ping_true') and failed ('ping_false') IPs.
    """
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_ping')
    ping_record: dict[str, list[str]] = {
        "ping_true": [],
        "ping_false": []
    }
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            real_ip = ip_ports['ip_addr']
            port = ip_ports['cmd_port']
            logger.info(f'Real IP: {real_ip}')
            logger.info('Cmd Port: %d', port)
            if util.ping(real_ip, port):
                ping_record["ping_true"].append(ip_addr)
            else:
                ping_record["ping_false"].append(ip_addr)
    if verbose:
        for ip in ping_record["ping_true"]:
            print(f"pinged {ip}")
        for ip in ping_record["ping_false"]:
            print(f"can't ping {ip}")
    return ping_record

def do_hk_dest(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], daq_config: DaqConfigValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any]) -> None:
    """Configure Housekeeping (HK) packet destination for multiple modules.
    
    Points all Quabos to the head node IP address for telemetry reporting.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        daq_config: DAQ node configuration (contains head node IP).
        network_config: Network routing rules.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_hk_dest')
    headnode_ip_addr = str(daq_config.head_node_ip_addr)
    logger.info(f'Head node IP: {headnode_ip_addr}')
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            if uid == '':
                continue
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info(f'Quabo IP: {ip_addr}')
            logger.info(f'Real IP: {real_ip}')
            logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hk_packet_destination(headnode_ip_addr)
            quabo.close()

def do_hv_on(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], quabo_info: dict[str, Any], detector_info: dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any], verbose: bool = False) -> None:
    """Enable high voltage (HV) for all detectors in multiple modules.
    
    Calculates DAC values based on per-detector operating voltages.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        quabo_info: Detailed Quabo metadata.
        detector_info: Dictionary of per-detector operating voltages.
        network_config: Network routing rules.
        verbose: If True, prints HV settings for each Quabo.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_hv_on')
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            if uid == '':
                continue
            qi = quabo_info[uid]
            v = [0]*4
            for j in range(4):
                det_ser = qi['detector_serialno'][j]
                op_voltage = detector_info[str(det_ser)]
                # DAC LSB is 0.0011324717, instead of 0.00114
                v[j] = int(op_voltage/0.0011324717)
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info(f'Quabo IP: {ip_addr}')
            logger.info(f'Real IP: {real_ip}')
            logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hv_set(v)
            quabo.close()
            if verbose:
                print(f'{ip_addr}: set HV to [{v[0]} {v[1]} {v[2]} {v[3]}]')

def do_hv_off(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any]) -> None:
    """Disable high voltage (HV) for all detectors in multiple modules.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        network_config: Network routing rules.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_hv_off')
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            if uid == '':
                continue
            v = [0]*4
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info(f'Quabo IP: {ip_addr}')
            logger.info(f'Real IP: {real_ip}')
            logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.hv_set(v)
            quabo.close()
            print(f'{ip_addr}: set HV to zero')

# set the DAC1/DA2/GAIN* params for MAROC chips
#
MAROC_CONFIG_QUABO_CONFIG = quabo_driver.parse_quabo_config_file(
    str(pathlib.Path(__file__).parent / 'driver/quabo_config.txt')
)
cal_cache: dict[tuple[Any, ...], Any] = {}
def do_maroc_config(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], quabo_info: dict[str, Any], data_config: DataConfigValidator | dict[str, Any], obs_config: ObsConfigValidator | dict[str, Any], daq_config: DaqConfigValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any], verbose: bool = False, write_config: bool = True, do_log: bool = True) -> None:
    """Configure MAROC ASIC registers for multiple modules.
    
    This includes setting gains and discriminator thresholds (DAC1/DAC2) 
    derived from calibration data for the target observing mode.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        quabo_info: Detailed Quabo metadata.
        data_config: Science/engineering acquisition parameters.
        obs_config: Physical observatory configuration.
        daq_config: DAQ node networking configuration.
        network_config: Network routing rules.
        verbose: If True, prints detailed register settings.
        write_config: If True, writes JSON configuration snapshots.
        do_log: If True, enables logging for this operation.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)
    if isinstance(data_config, dict):
        data_config = DataConfigValidator(**data_config)
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)

    logger = None
    if do_log:
        logger = logging.getLogger('PANOSETI.Config.do_maroc_config')
    gain = float(data_config.gain) if data_config.gain is not None else 1.0
    do_img = data_config.image is not None
    do_ph = data_config.pulse_height is not None

    if do_img:
        pe_thresh1 = float(data_config.image.pe_threshold) if data_config.image else 1.0
    if do_ph:
        pe_thresh2 = float(data_config.pulse_height.pe_threshold) if data_config.pulse_height else 1.0
    if not do_img and not do_ph:
        raise Exception('data_config.json specifies no data products')

    stim_mask_quaboi = [1, 1, 1, 1]
    if data_config.stim_params:
        stim_mask_quaboi = [bool(m) for m in data_config.stim_params.mask]
        
    qc_dict_src = copy.deepcopy(MAROC_CONFIG_QUABO_CONFIG)
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            no_cali = False
            qc_dict = copy.deepcopy(qc_dict_src)
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            if uid == '':
                continue
            is_qfp = util.is_quabo_old_version(module.model_dump(), i, quabo_uids.model_dump(), quabo_info)
            try:
                qi = quabo_info[uid]
            except Exception:
                use_default_calib = ask_use_default_calibration(ip_addr)
                if use_default_calib:
                    qi = quabo_info['default']
                    is_qfp = False
                    no_cali = True
                else:
                    raise Exception(f'No calibration file is found for {ip_addr}') from None
            serialno = qi['serialno'][3:]
            # try to find the detector overvoltage in data_config.json
            # if we can't find it, we will use 3v by default.
            detovervol = data_config.detector_overvoltage or 3
            # We have different calibration files for different modes: image alone and image/ph together
            # so we have to specifiy the mode here.
            # TODO: If it's PH alone, what calibration file should we use?
            op_mode = 'img' if do_img and not do_ph else 'ph'

            # Cache calibration data to reduce I/O
            cal_cache_key = (serialno, detovervol, op_mode)
            if cal_cache_key not in cal_cache:
                quabo_calib = config_file.get_quabo_calib(serialno, detovervol, op_mode)
                cal_cache[cal_cache_key] = quabo_calib
            else:
                quabo_calib = cal_cache[cal_cache_key]

            # compute DAC1[] and possibly DAC2 based on calibration data
            dac1 = [0]*4
            dac2 = [0]*4
            for j in range(4):      # 4 detectors in a quabo
                quad = quabo_calib['quadrants'][j]
                a = quad['a']       # a and b are used for img mode
                b = quad['b']
                ah = quad['ah']      # ah and bh are used for ph mode
                bh = quad['bh']
                if do_img:
                    dac1[j] = int(a*gain*pe_thresh1 + b)
                if do_ph:
                    dac2[j] = int(ah*gain*pe_thresh2 + bh)
            if do_img:
                qc_dict['DAC1'] = f'{dac1[0]},{dac1[1]},{dac1[2]},{dac1[3]}'
                if verbose:
                    print('{}: DAC1 = {}'.format(ip_addr, qc_dict['DAC1'])) 
            if do_ph:
                qc_dict['DAC2'] = f'{dac2[0]},{dac2[1]},{dac2[2]},{dac2[3]}'
                if verbose:
                    print('{}: DAC2 = {}'.format(ip_addr, qc_dict['DAC2']))
            # compute GAIN0[]..GAIN63[] based on calibration data
            # TODO: fix indexing
            maroc_gain = [[0]*4 for i in range(64)]

            for j in range(4):
                for k in range(64):
                    [x, y] = pixel_coords.detector_to_quabo(k, j, bool(is_qfp))
                    delta = quabo_calib['pixel_gain'][x][y]
                    g = round(gain*(1+delta))
                    maroc_gain[k][j] = g
            for k in range(64):
                tag = f'GAIN{k}'
                qc_dict[tag] = f'{maroc_gain[k][0]},{maroc_gain[k][1]},{maroc_gain[k][2]},{maroc_gain[k][3]}'
                if verbose:
                    print(f'{ip_addr}: {tag} = {qc_dict[tag]}')
            # set D1_D2 based on the two_pixel_trigger and three_pixel_trigger in data_config.json
            do_two_pixel_trigger = False
            do_three_pixel_trigger = False
            if do_ph and data_config.pulse_height:
                do_two_pixel_trigger = bool(data_config.pulse_height.two_pixel_trigger)
                do_three_pixel_trigger = bool(data_config.pulse_height.three_pixel_trigger)
            # if using 2/3 pixel trigger, D1_D2 should be set to 1,1,1,1
            if do_two_pixel_trigger or do_three_pixel_trigger:
                qc_dict['D1_D2'] = '1,1,1,1'
            if verbose:
                print('{}: {} = {}'.format(ip_addr, 'D1_D2', qc_dict['D1_D2']))
            # send MAROC params to the quabo
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            if do_log and isinstance(logger, logging.Logger):
                logger.info(f'Quabo IP: {ip_addr}')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            # For ph mode, we seem to have a bug in firmware.
            # we need to set DAC2 to low, and make the quabos send out data first.
            if do_ph:
                tmp = [0] * 4
                # set the DAC2 value very low, 5.5 pe
                for j in range(4):      # 4 detectors in a quabo
                    quad = quabo_calib['quadrants'][j]
                    ah = quad['ah']
                    bh = quad['bh']
                    tmp[j] = int(ah*gain*5.5 + bh)
                qc_dict['DAC2'] = f'{tmp[0]},{tmp[1]},{tmp[2]},{tmp[3]}'
                quabo.send_maroc_params(qc_dict)
                # make the quabos send out some ph packets
                # set the DAC2 values back
                qc_dict['DAC2'] = f'{dac2[0]},{dac2[1]},{dac2[2]},{dac2[3]}'
            if no_cali and do_log and isinstance(logger, logging.Logger):
                logger.warning(f'No calibration data: UID -{uid}')
            # If the stim_mask is 0 for this quabo, set all CTEST values to 0
            if stim_mask_quaboi[i] == 0:
                for k in range(64):
                    ctest_key = f'CTEST_{k}'
                    assert ctest_key in qc_dict, f"{ctest_key=} not in qc_dict"
                    qc_dict[ctest_key] = "0,0,0,0"
            quabo.send_maroc_params(qc_dict)
            if write_config:
                quabo.write_maroc_config(qc_dict, '{}_{}.json'.format('tmp/quabo_config',ip_addr))
            quabo.close()

# set CHANMASK and GOEMASK for modules
#
MASK_CONFIG_QUABO_CONFIG = quabo_driver.parse_quabo_config_file(
    str(pathlib.Path(__file__).parent / 'driver/quabo_config.txt')
) # load once to avoid redundant I/O
def do_mask_config(modules: list[ObsModuleConfig] | list[dict[str, Any]], data_config: DataConfigValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any], quabo_uids: QuaboUidsValidator | dict[str, Any], verbose: bool = False, write_config: bool = True, do_flush_rx_buf: bool = False, do_log: bool = True) -> None:
    """Configure channel trigger masks and geometric coincidence masks.

    Args:
        modules: List of target module configurations.
        data_config: Science/engineering acquisition parameters.
        network_config: Network routing rules.
        quabo_uids: Quabo hardware UID registry.
        verbose: If True, prints mask values.
        write_config: If True, writes configuration snapshots.
        do_flush_rx_buf: If True, flushes Quabo RX buffer after update.
        do_log: If True, enables logging.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)
    if isinstance(data_config, dict):
        data_config = DataConfigValidator(**data_config)

    logger = None
    if do_log:
        logger = logging.getLogger('PANOSETI.Config.do_mask_config')
    qc_dict = copy.deepcopy(MASK_CONFIG_QUABO_CONFIG)
    qc_dict_int: dict[str, int] = {}
    do_ph = data_config.pulse_height is not None
    qc_dict_int['GOEMASK'] = int(qc_dict['GOEMASK'], 16)
    for i in range(9):
        qc_dict_int['CHANMASK_'+str(i)] = int(qc_dict['CHANMASK_'+str(i)], 16)
    if do_ph and data_config.pulse_height:
        # config CHANMASK_8 for any_trigger
        if data_config.pulse_height.any_trigger:
            qc_dict_int['CHANMASK_8'] = qc_dict_int['CHANMASK_8'] & 0x0ff
        else:
            qc_dict_int['CHANMASK_8'] = qc_dict_int['CHANMASK_8'] | (0x100)
        
        # config GOEMASK for 2/3 pixel_trigger
        if data_config.pulse_height.three_pixel_trigger:
            qc_dict_int['CHANMASK_8'] = qc_dict_int['CHANMASK_8'] | 0xff
            qc_dict_int['GOEMASK'] = qc_dict_int['GOEMASK'] & 0x1
        if data_config.pulse_height.two_pixel_trigger:
            qc_dict_int['CHANMASK_8'] = qc_dict_int['CHANMASK_8'] | 0xff
            qc_dict_int['GOEMASK'] = qc_dict_int['GOEMASK'] & 0x2

    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            if uid == '':
                continue
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            for tag in ['CHANMASK_8', 'GOEMASK']:
                if verbose:
                    print(f'{ip_addr}: {tag} = 0x{qc_dict_int[tag]:x}')
            # send MASK params to the quabo
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            if do_log and isinstance(logger, logging.Logger):
                logger.info(f'Quabo IP: {ip_addr}')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.send_trigger_mask(qc_dict_int, do_flush_rx_buf=do_flush_rx_buf)
            if write_config:
                quabo.write_trigger_mask_config(qc_dict_int, '{}_{}.json'.format('tmp/quabo_config',ip_addr))
            quabo.send_goe_mask(qc_dict_int, do_flush_rx_buf=do_flush_rx_buf)
            if write_config:
                quabo.write_goe_mask_config(qc_dict_int, '{}_{}.json'.format('tmp/quabo_config',ip_addr))
            quabo.close()

def do_calibrate_ph(modules: list[ObsModuleConfig] | list[dict[str, Any]], quabo_uids: QuaboUidsValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any]) -> None:
    """Trigger pulse-height (PH) baseline calibration on multiple modules.
    
    Results are saved to the local PH baseline cache file.

    Args:
        modules: List of target module configurations.
        quabo_uids: Quabo hardware UID registry.
        network_config: Network routing rules.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    logger = logging.getLogger('PANOSETI.Config.do_calibrate_ph')
    quabos: list[dict[str, Any]] = []
    for module in modules:
        if isinstance(module, dict):
            module = ObsModuleConfig(**module)
        m_ip = str(module.ip_addr)
        for i in range(4):
            uid = util.quabo_uid(module.model_dump(), quabo_uids.model_dump(), i)
            if uid == '':
                continue
            ip_addr = config_file.quabo_ip_addr(m_ip, i)
            ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            logger.info(f'Quabo IP: {ip_addr}')
            logger.info(f'Real IP: {real_ip}')
            logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            coefs = quabo.calibrate_ph_baseline()
            quabo.close()
            q: dict[str, Any] = {}
            q['uid'] = uid
            q['coefs'] = coefs
            quabos.append(q)
    x: dict[str, Any] = {}
    d = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    x['date'] = d.isoformat()
    x['quabos'] = quabos
    baseline_file = config_file.quabo_ph_baseline_filename
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
    with open(baseline_file, "w") as f:
        f.write(json.dumps(x, indent=4))


# show summary statistics for the PH baseline calibrations of each quabo
def do_show_ph_baselines(quabo_uids: QuaboUidsValidator | dict[str, Any]) -> None:
    """Print summary statistics for the cached PH baseline calibrations.

    Args:
        quabo_uids: Quabo hardware UID registry model or dict.
    """
    if isinstance(quabo_uids, dict):
        quabo_uids = QuaboUidsValidator(**quabo_uids)

    logger = logging.getLogger('PANOSETI.Config.do_show_ph_baselines')
    logger.info('Show PH baseline')
    quabo_ph_baselines = config_file.get_quabo_ph_baselines()
    msg = f"Creation date: {quabo_ph_baselines['date']}\n"
    for dome in quabo_uids.domes:
        for module in dome.modules:
            module_ip_addr = str(module.ip_addr)
            msg += f'module {module_ip_addr}:\n'
            for quabo_index in range(4):
                quabo_num = config_file.get_boardloc(module_ip_addr, quabo_index)
                quabo_uid = module.quabos[quabo_index].uid
                quabo_baselines: Any = None
                for q in quabo_ph_baselines['quabos']:
                    if q['uid'] == quabo_uid:
                        quabo_baselines = q
                if quabo_baselines is None:
                    msg += f'\tquabo {quabo_num}: found no ph baseline data\n'
                else:
                    coefs = quabo_baselines['coefs']
                    mean = statistics.mean(coefs)
                    median = statistics.median(coefs)
                    stdev = statistics.stdev(coefs)
                    msg += f'\tquabo {quabo_num: 5}: mean={round(mean, 2): 7}, ' \
                           f'median={round(median, 2): 7}, stdev={round(stdev, 2): 7},' \
                           f' min={min(coefs): 5}, max={max(coefs): 5}\n'
    print(msg)



# compute available recording time, given data config and free disk space.
# If verbose, show details
#
def do_disk_space(data_config: DataConfigValidator | dict[str, Any], daq_config: DaqConfigValidator | dict[str, Any], verbose: bool = False) -> float:
    """Estimate remaining recording time based on available disk space.

    Args:
        data_config: Science/engineering acquisition parameters.
        daq_config: DAQ node configuration with data paths.
        verbose: If True, prints per-node volume details.

    Returns:
        Estimated recording time in hours.
    """
    if isinstance(data_config, dict):
        data_config = DataConfigValidator(**data_config)
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)

    logger = logging.getLogger('PANOSETI.Config.do_disk_space')
    logger.info('Check disk space.')
    bps = util.daq_bytes_per_sec_per_module(data_config)
    if verbose:
        print(f'Data rate per module: {bps/1e6:.2f} MB/sec')
    nmod_total = 0
    available_hours = 1e9

    # loop over DAQ nodes
    #
    for node in daq_config.daq_nodes:
        # Check if this node has any modules assigned (node.modules is a list of Any)
        if not node.modules:
            continue
        nmod = len(node.modules)
        nmod_total += nmod
        ip_addr = str(node.ip_addr)
        if verbose:
            print(f'DAQ node {ip_addr}: {nmod} modules')

        # get list of volumes on the DAQ node
        #
        j = util.get_daq_node_status(node.model_dump())
        vols = j['vols']

        # initialize list of module IDs each vol will handle,
        # and find the default volume for this node
        #
        default_vol: Any = None
        for vol in vols.values():
            vol['mods_here'] = []
            if -1 in vol['modules']:
                default_vol = vol

        # loop over module IDs going to this DAQ node,
        # and add them to the mods_here list for the appropriate volume
        #
        for module in node.modules:
            mid = module.id if hasattr(module, 'id') else module.get('id')
            found = False
            for vol in vols.values():
                if mid in vol['modules']:
                    vol['mods_here'].append(mid)
                    found = True
                    break
            if not found and default_vol:
                default_vol['mods_here'].append(mid)

        for name in vols:
            vol = vols[name]
            free = vol['free']
            mods_here_list = vol.get('mods_here', [])
            nmods = len(mods_here_list)
            if verbose:
                print(f'   {name}:')
            if nmods:
                t = free/(3600.*bps*nmods)
                if verbose:
                    print('      modules: ', mods_here_list)
                    print(f'      space: {free/1e12:.2f}TB ({t:.2f} hours)')
                if t < available_hours:
                    available_hours = t
            else:
                if verbose:
                    print(f'      space: {free/1e12:.2f}TB')
    # TODO: this is hard-coded??
    with open("/home/panosetigraph/web/head_node_volumes.json") as f:
        head_node_vols = json.loads(f.read())
    hnd = str(daq_config.head_node_data_dir)
    hnd = os.path.realpath(hnd)
    print('head node:')
    for vol in head_node_vols:
        path = f'/home/panosetigraph/web/{vol}/data'
        path = os.path.realpath(path)
        hfree = util.free_space(path)
        if verbose:
            print(f'   {path} ({vol})')
        t = hfree/(3600*bps*nmod_total)
        if hnd == path:
            if t < available_hours:
                available_hours = t
            if verbose:
                print('      selected for write')
        print(f'      space: {hfree/1e12:.2f}TB ({t:.2f} hours)')

    if verbose:
        print(f'---------------\nAvailable recording time: {available_hours:.2f} hours')
    return available_hours


def do_shutter(action: str) -> None:
    if action == "open":
        os.system("tools/shutter.py --open")
    elif action == "close":
        os.system("tools/shutter.py --close")

def do_start_interleave() -> None:
    """Starts the interleaver process in the background. (SC-034b: Prevents duplicate daemons)"""
    pid_file = "tmp/interleave.pid"
    if os.path.exists(pid_file):
        print("ERROR: Interleave daemon is already running (PID file exists). Stop it first.")
        sys.exit(1)

    if not os.path.exists("tmp/current_run") and not os.path.exists("tmp/run_state.toml"):
        print("ERROR: Cannot start interleaving. No active observation running. Run start.py first.")
        sys.exit(1)

    print("Starting interleave controller in the background...")
    # Start detached background process
    subprocess.Popen(['python3', 'tools/interleave.py'],
                     stdout=open('logs/interleave.log', 'a'), # noqa: SIM115
                     stderr=subprocess.STDOUT)
    print("Interleave process started. Check logs/interleave.log for details.")

def do_stop_interleave() -> None:
    """Gracefully stops the background interleaver if it is running."""
    pid_file = "tmp/interleave.pid"
    if not os.path.exists(pid_file):
        print("No active interleave process found (PID file missing).")
        return # Return instead of sys.exit(0) so other scripts can call this safely

    with open(pid_file) as f:
        try:
            pid = int(f.read().strip())
        except ValueError:
            print("Stale PID file found. Cleaning up.")
            os.remove(pid_file)
            return

    print(f"Sending shutdown signal to interleave process (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        print("Signal sent. Waiting for hardware default restoration to complete...")
        # Simple wait loop to ensure process dies and deletes its pid file
        for _ in range(20):
            if not os.path.exists(pid_file):
                break
            time.sleep(0.5)
        print("Interleave process successfully stopped.")
    except OSError:
        print("Process was already dead. Cleaning up stale PID file.")
        os.remove(pid_file)


def do_dry_run_interleave() -> None:
    """Runs the interleaver in the foreground for 2 cycles without hardware commands."""
    print("Starting interleave DRY RUN (2 cycles) in the foreground...")

    # We use subprocess.run to block and stream output directly to the console for CI tools
    result = subprocess.run(
        ['python3', 'tools/interleave.py', '--dry-run', '--max-cycles', '2']
    )

    if result.returncode == 0:
        print("\nDry run completed successfully.")
    else:
        print(f"\nDry run failed with return code {result.returncode}.")
        sys.exit(result.returncode)




def setup_logging(name: str = 'PANOSETI.Config'):
    # if not os.path.exists('logs'):
    #     os.makedirs('logs')
    # logfile = 'logs/config.log'
    # util.create_logger(logfile, name, 'a')
    # logger = logging.getLogger(name)
    logger = get_logger(
        service_name=name,
        log_dir='logs',
        grpc_enabled=True,
    )
    # logger.info('************************************')
    return logger

@app.command()
def show():
    """Show list of domes/modules/quabos."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    quabo_uids = config_file.get_quabo_uids()
    show_config(obs_config, quabo_uids)
    util.show_redis_daemons()

@app.command()
def ping():
    """Ping quabos."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    network_config = config_file.get_network_config()
    do_ping(modules, network_config, verbose=True)

@app.command()
def reboot():
    """Reboot quabos."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)
    do_reboot(modules, quabo_uids, network_config)
    do_hk_dest(modules, quabo_uids, daq_config, network_config)

@app.command()
def reboot_single(reboot_single: str = typer.Argument(..., help="Reboot a single quabo.")):
    """Reboot a single quabo."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    do_reboot_single_quabo(reboot_single, obs_config, network_config)

@app.command()
def loads():
    """Load silver firmware in quabos."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    quabo_info = config_file.get_quabo_info()
    network_config = config_file.get_network_config()
    do_loads(modules, quabo_uids, quabo_info, network_config)

@app.command()
def init_daq_nodes():
    """Copy software to daq nodes."""
    logger = setup_logging('PANOSETI.Config.init_daq_nodes')
    logger.info('Init daq nodes.')
    daq_config = config_file.get_daq_config()
    file_xfer.copy_daq_files(daq_config)

@app.command()
def hk_dest():
    """Set the dest IP for HK packet."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)
    do_hk_dest(modules, quabo_uids, daq_config, network_config)

@app.command()
def redis_daemons():
    """Start daemons to populate Redis with HK/GPS/WR data, and to copy data from Redis to InfluxDB."""
    logger = setup_logging('PANOSETI.Config.start_redis_daemons')
    logger.info('Start redis daemons.')
    util.start_redis_daemons()

@app.command()
def stop_redis_daemons():
    """Stop the above."""
    logger = setup_logging('PANOSETI.Config.stop_redis_daemons')
    logger.info('Stop redis daemons.')
    util.stop_redis_daemons()

@app.command()
def permanent_daemons():
    """Start permanent daemons (permanent_*.py) plus storeInfluxDB.py."""
    logger = setup_logging('PANOSETI.Config.start_permanent_daemons')
    logger.info('Start permanent daemons.')
    util.start_permanent_daemons()

@app.command()
def stop_permanent_daemons():
    """Stop the above."""
    logger = setup_logging('PANOSETI.Config.stop_permanent_daemons')
    logger.info('Stop permanent daemons.')
    util.stop_permanent_daemons()

@app.command()
def show_permanent_daemons():
    """Show permanent daemon status."""
    setup_logging()
    util.show_permanent_daemons()

@app.command()
def hv_on():
    """Enable detectors."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    quabo_info = config_file.get_quabo_info()
    network_config = config_file.get_network_config()
    detector_info = config_file.get_detector_info()
    do_hv_on(modules, quabo_uids, quabo_info, detector_info, network_config, True)

@app.command()
def hv_off():
    """Disable detectors."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    do_hv_off(modules, quabo_uids, network_config)

@app.command()
def maroc_config():
    """Configure MAROCs based on data_config.json and quabo_calib_*.json."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    daq_config = config_file.get_daq_config()
    quabo_info = config_file.get_quabo_info()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    config_file.associate(daq_config, quabo_uids)
    data_config = config_file.get_data_config()
    do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config, True)

@app.command()
def mask_config():
    """Configure masks based on data_config.json."""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    data_config = config_file.get_data_config()
    do_mask_config(modules, data_config, network_config, quabo_uids, True)

@app.command()
def calibrate_ph():
    """Run PH baseline calibration on quabos and write to file"""
    setup_logging()
    obs_config = config_file.get_obs_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    do_calibrate_ph(modules, quabo_uids, network_config)

@app.command()
def show_ph_baselines():
    """Show PH baseline calibration summary statistics"""
    setup_logging()
    quabo_uids = config_file.get_quabo_uids()
    do_show_ph_baselines(quabo_uids)

@app.command()
def shutter_open():
    """Open all module shutters"""
    setup_logging()
    do_shutter("open")

@app.command()
def shutter_close():
    """Close all module shutters"""
    setup_logging()
    do_shutter("close")

@app.command()
def disk_space():
    """Check the disk_space."""
    setup_logging()
    daq_config = config_file.get_daq_config()
    data_config = config_file.get_data_config()
    do_disk_space(data_config.model_dump(), daq_config.model_dump(), True)

@app.command()
def start_interleave():
    """Start background interleaver"""
    setup_logging()
    do_start_interleave()

@app.command()
def stop_interleave():
    """Stop background interleaver"""
    setup_logging()
    do_stop_interleave()

@app.command()
def dry_run_interleave():
    """Test the interleave schedule for 2 cycles without hardware commands."""
    setup_logging()
    do_dry_run_interleave()

@app.command()
def validate(modifiers: list[str] = typer.Argument(None, help='Validate configs. Modifiers: "graph" (print network routing), "network" (ping IPs), "debug" (print parsed config). Example: --validate graph network')):
    """Validate configs. Modifiers: "graph" (print network routing), "network" (ping IPs), "debug" (print parsed config). Example: --validate graph network"""
    setup_logging()
    if modifiers is not None:
        debug_mode = 'debug' in modifiers
        network_mode = 'network' in modifiers
        graph_mode = 'graph' in modifiers

        # Run the comprehensive check and exit gracefully!
        passed = config_file.validate_all(check_network=network_mode, debug=debug_mode, graph=graph_mode)
        if not passed:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)
    else:
        passed = config_file.validate_all(check_network=False, debug=False, graph=False)
        if not passed:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

if __name__ == "__main__":
    app()


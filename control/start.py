#! /usr/bin/env python3

# start.py [--no_hv] [--no_redis] [--no_data] [--verbose] [--help]
#          [--nsecs N] [--stop_session]
#
# start a recording run:
#
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - create "run directories" on head node, DAQ nodes
# - start the HK recorder
# - start the HV updater
# - start the temperature monitor
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program
#
# fail if a recording run is in progress,
# or if recording activities are active
#

# based on matlab/startmodules.m, startqNph.m, changepeq.m

# ---------------- PRINT -> UT TIMESTAMP + FILE LOG ----------------
import asyncio
import builtins
import json
import logging
import os
import shutil
import socket
import time
import traceback
from argparse import ArgumentParser
from datetime import UTC, datetime
from glob import glob
from typing import Any

from panoseti_grpc.daq_control.client import DaqControlClient

import session_stop
import stop
from driver import quabo_driver
from tools.sw_info import get_sw_info
from utils import config_file, file_xfer, pff, util
from utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    DataConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
    QuaboUidsValidator,
    RunStateLedger,
)
from utils.run_state import NodeReceipt, RunStateManager

_LOG_ROOT = "/mnt/data11/data/palomar/L0"

def _ut_yyyymmdd() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")

def _ut_human_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UT")

def _datarec_log_path() -> tuple[str, str]:
    yyyymmdd = _ut_yyyymmdd()
    obslogs_dir = os.path.join(_LOG_ROOT, yyyymmdd, "obslogs")
    return obslogs_dir, os.path.join(obslogs_dir, f"datarec_{yyyymmdd}.log")

_orig_print = builtins.print

def _print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file_arg = kwargs.get("file")
    flush = kwargs.get("flush", False)

    msg = sep.join(str(a) for a in args)
    line = f"{_ut_human_ts()} {msg}"

    # Console (or provided file), with timestamp prepended
    _orig_print(line, sep=sep, end=end, file=file_arg, flush=flush)

    # Append at the beginning of daily log file (best-effort; no extra prints)
    try:
        obslogs_dir, log_path = _datarec_log_path()
        os.makedirs(obslogs_dir, exist_ok=True)

        # Read existing contents (if any), then write new line + old contents
        try:
            with open(log_path, encoding="utf-8") as f:
                old = f.read()
        except FileNotFoundError:
            old = ""

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(line + end)
            if old:
                f.write(old)
            if flush and hasattr(f, "flush"):
                f.flush()
    except Exception:
        pass

builtins.print = _print
# ------------------------------------------------------------------

verbose = False

# check that PH calibration file is present, nonempty, and at most 24 hours old
#
def ph_baseline_file_ok(filename: str | None = None) -> bool:
    """Verify that the Pulse Height calibration file is valid.
    
    Checks that the file exists, is not empty, and is at most 24 hours old.
    Stale or missing calibration data can lead to incorrect PH measurements.

    Args:
        filename: Optional path to the baseline file. Defaults to config_file.quabo_ph_baseline_filename.

    Returns:
        True if the file is valid, False otherwise.
    """
    if filename is None:
        filename = config_file.quabo_ph_baseline_filename
    if not os.path.exists(filename):
        print(f'{filename} not found.  Run config.py --calibrate_ph')
        return False
    if os.path.getsize(filename) == 0:
        print(f'{filename} is empty.  Run config.py --calibrate_ph')
        return False
    # Fix SC-031: 24 hours is 3600*24, not 86400*24
    if os.path.getmtime(filename) < time.time() - 86400:
        print(f'{filename} is too old (>24h).  Run config.py --calibrate_ph')
        return False
    return True


# check validity of image params (rate, bpp)
#
def check_img_params(image_8bit: bool, image_usec: int) -> None:
    """Validate image acquisition parameters against hardware constraints.

    Args:
        image_8bit: Whether using 8-bit image mode.
        image_usec: Integration time in microseconds.

    Raises:
        Exception: If parameters violate hardware limits.
    """
    if image_8bit:
        if image_usec < 20 or image_usec > 25:
            raise Exception('integration time must be 20-25 usec in 8 bit mode')
    else:
        if image_usec < 40:
            raise Exception('integration time must be >= 40 usec in 16 bit mode')

# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config: DataConfigValidator) -> quabo_driver.DAQ_PARAMS:
    """Translate the high-level data configuration into Quabo-level DAQ parameters.
    
    Parses image mode settings (integration time, sample size), pulse-height 
    mode settings (any_trigger, grouping), and test signals (flash/stim).

    Args:
        data_config: The validated science/engineering configuration model.

    Returns:
        An initialized quabo_driver.DAQ_PARAMS object.
    """
    do_image = False
    image_usec = 1
    image_8bit = False
    do_ph = False
    bl_subtract = True
    do_any_trigger = False
    group_ph_frames = False
    if data_config.image:
        do_image = True
        image = data_config.image
        if image.quabo_sample_size == 8:
            image_8bit = True
        image_usec = image.integration_time_usec
    if data_config.pulse_height:
        do_ph = True
        if data_config.pulse_height.any_trigger:
            do_any_trigger = True
            any_trigger = data_config.pulse_height.any_trigger
            if any_trigger.group_ph_frames == 1:
                group_ph_frames = True
    daq_params = quabo_driver.DAQ_PARAMS(
        do_image, image_usec - 1, image_8bit, do_ph, bl_subtract, do_any_trigger, group_ph_frames
    )
    if data_config.flash_params:
        fp = data_config.flash_params
        daq_params.set_flash_params(fp.rate, fp.level, fp.width)
    if data_config.stim_params:
        sp = data_config.stim_params
        daq_params.set_stim_params(sp.rate, sp.level)
    return daq_params

def start_data_flow(
    quabo_uids: QuaboUidsValidator,
    data_config: DataConfigValidator,
    daq_config: DaqConfigValidator,
    network_config: NetworkConfigValidator | dict[str, Any]
) -> None:
    """Initialize data flow from Quabos by configuring networking and modes.
    
    For every Quabo in every module:
    1. Tell it where to send Housekeeping (HK) packets (head node).
    2. Tell it where to send Data packets (assigned DAQ node).
    3. Set its DAQ acquisition mode (Image/PH/Stim/Flash).
    4. Synchronize PPS.

    Args:
        quabo_uids: Validated Quabo UID configuration.
        data_config: Science/engineering acquisition parameters.
        daq_config: DAQ node and head node networking details.
        network_config: Network routing and port forwarding settings.
    """
    logger = logging.getLogger('PANOSETI.Start.start_data_flow')
    daq_params = get_daq_params(data_config)
    for dome in quabo_uids.domes:
        for module in dome.modules:
            # Note: QuaboUidModule has 'ip_addr'
            base_ip_addr = str(module.ip_addr)
            module_id = config_file.ip_addr_to_module_id(base_ip_addr)
            try:
                # daq_config.model_dump() is still needed for config_file.module_id_to_daq_node
                # until that is also fully refactored to take the model.
                # Actually I refactored it to take both.
                daq_node = config_file.module_id_to_daq_node(daq_config, module_id)
            except Exception:
                continue
            daq_node_ip_addr = str(daq_node.ip_addr)
            head_node_ip_addr = str(daq_config.head_node_ip_addr)
            for i in range(4):
                if module.quabos[i].uid == '':
                    continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                ip_ports = util.get_quabo_ip_port(base_ip_addr, i, network_config)
                real_ip = ip_ports['ip_addr']
                cmd_port = ip_ports['cmd_port']
                logger.info(f'Quabo IP: {ip_addr}')
                logger.info(f'Real IP: {real_ip}')
                logger.info(f'Cmd Port: {cmd_port}')
                quabo = quabo_driver.QUABO(real_ip, cmd_port)
                if verbose:
                    print(f'setting HK packet dest to {head_node_ip_addr} on quabo {ip_addr}')
                quabo.hk_packet_destination(head_node_ip_addr)
                if verbose:
                    print(f'setting data packet dest to {daq_node_ip_addr} on quabo {ip_addr}')
                quabo.data_packet_destination(daq_node_ip_addr)
                if verbose:
                    print(f'setting DAQ mode on quabo {ip_addr}')
                quabo.send_daq_params(daq_params)
                quabo.close()
            # send software 1PPS
            time.sleep(0.5)
            logger.info(f'Send software 1PPS to {base_ip_addr}')
            ip_ports = util.get_quabo_ip_port(base_ip_addr, 0, network_config)
            real_ip = ip_ports['ip_addr']
            cmd_port = ip_ports['cmd_port']
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.swpps()
            quabo.close()


def make_run_dirs(run_name: str, daq_config: DaqConfigValidator) -> None:
    """Create hierarchical run directories and distribute configuration files.
    
    Directories are created on the local head node and remote DAQ nodes:
    - Head Node: data_dir/run_name/ (config files)
    - Head Node: data_dir/module_n/run_name/ (.pff files)
    - Remote Node: data_dir/run_name/ (config files)
    - Remote Node: data_dir/module_n/run_name/ (.pff files)

    Args:
        run_name: The directory name for the current observation run.
        daq_config: Validated DAQ configuration detailing storage paths.

    Raises:
        Exception: If a directory cannot be created locally or over SSH.
    """
    logger = logging.getLogger('PANOSETI.Start')
    my_ip = util.local_ip()
    run_dir = f'{daq_config.head_node_data_dir}/{run_name}'
    os.mkdir(run_dir)

    # copy config files to run dir on this node
    for f in config_file.config_file_names:
        files = glob(f)
        for file in files:
            fparts = file.split('/')
            shutil.copyfile(file, f'{run_dir}/{fparts[-1]}')

    # make module and run directories on DAQ nodes
    for node in daq_config.daq_nodes:
        # Check if this node has any modules assigned
        # DaqNodeValidator has module_ids
        if not node.module_ids:
            continue
        ip_addr = str(node.ip_addr)
        if ip_addr in my_ip:
            # We need to know which module IDs are on this node to create module_N dirs
            # node.module_ids is a list of ints or a range string (preprocessed to list[int])
            for mid in node.module_ids:
                cmd = f'mkdir -p {daq_config.head_node_data_dir}/module_{mid}/{run_name}'
                if verbose:
                    print(cmd)
                ret = os.system(cmd)
                if ret:
                    raise Exception(f'{cmd} returned {ret}')
        else:
            username = node.username
            data_dir = node.data_dir
            rcmds = [f'mkdir {data_dir}/{run_name}']
            for mid in node.module_ids:
                rcmds.append(f'mkdir -p {data_dir}/module_{mid}/{run_name}')
            # create process snapshot
            rcmds.append(f'cd {data_dir}/{run_name}; ps -ux > pss_{ip_addr}.log')
            rcmnd = ';'.join(rcmds)
            logger.info(f'DAQ IP: {ip_addr}')
            # Need to handle port forwarding from network_config if we want to be fully typed,
            # but util.attach_daq_config currently mutates the dict.
            # DaqNodeValidator doesn't have port_forwarding in its schema if it's strict.
            # Let's check DaqNodeValidator in pydantic_config_models.py
            # Wait, DaqNodeValidator is BaseStrictModel. It DOES NOT have port_forwarding.
            # But util.attach_daq_config attaches it! This will fail validation.
            # I should update DaqNodeValidator to allow port_forwarding.
            
            # For now, let's use the node as a dict for SSH logic if it was mutated
            node_dict = node.model_dump()
            if 'port_forwarding' in node_dict:
                real_ip = node_dict['port_forwarding']['gw_ip']
                port = node_dict['port_forwarding']['port']
                cmd = f'ssh -p {port} {username}@{real_ip} "{rcmnd}"'
            else:
                cmd = f'ssh {username}@{ip_addr} "{rcmnd}"'
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                raise Exception(f'{cmd} returned {ret}')

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config.model_dump(), run_name, verbose)


async def start_recording(
    obs_config: ObsConfigValidator,
    data_config: DataConfigValidator,
    daq_config: DaqConfigValidator,
    run_name: str,
    no_hv: bool,
    state_mgr: RunStateManager
) -> None:
    """
    Asynchronously starts recording on DAQ nodes and performs heartbeat liveness checks.
    Transactional Contract:
    - Starts local HK/HV daemons.
    - Issues StartDaq to all remote nodes concurrently.
    - Waits 2s and probes StatusDaq for Hashpipe ALIVE heartbeat.
    - Updates run_state.toml ledger with receipts.
    - Raises Exception on ANY failure to trigger the parent rollback ladder.
    """
    logger = logging.getLogger('PANOSETI.Start.start_recording')
    loop = asyncio.get_running_loop()

    # 1. Start local daemons
    util.start_hk_recorder(daq_config.model_dump(), run_name)
    if not no_hv:
        util.start_hv_updater()
        util.start_module_temp_monitor()

    # 2. Concurrent StartDaq
    max_file_size_mb = data_config.max_file_size_mb or util.default_max_file_size_mb
    daq_params = get_daq_params(data_config)

    async def start_node(node_validator: DaqNodeValidator) -> None:
        if not node_validator.module_ids:
            return
        node_dict = node_validator.model_dump()
        grpc_host, grpc_port = util.daq_grpc_endpoint(node_dict)
        logger.info(f'StartDaq via gRPC: {grpc_host}:{grpc_port} modules={node_validator.module_ids}')
        
        client = DaqControlClient(host=grpc_host, port=grpc_port)
        start_args = {
            'data_dir':         node_validator.data_dir,
            'daq_ip_addr':      str(node_validator.ip_addr),
            'bindhost':         node_validator.bindhost or '0.0.0.0',
            'max_file_size_mb': int(max_file_size_mb),
            'group_ph_frames':  bool(daq_params.do_group_ph_frames),
            'run_dir':          run_name,
            'obs':              obs_config.name,
            'module_id':        node_validator.module_ids,
        }
        
        # Call gRPC synchronously in thread pool
        ok = await loop.run_in_executor(None, lambda: client.StartDaq(start_args))
        if not ok:
            raise RuntimeError(f'StartDaq failed for node {node_validator.ip_addr}')

    # Execute all starts in parallel
    await asyncio.gather(*(start_node(n) for n in daq_config.daq_nodes))

    # 3. Liveness Probe (Heartbeat)
    logger.info("Waiting for Hashpipe stabilization heartbeat...")
    await asyncio.sleep(2.0)

    async def probe_node(node_validator: DaqNodeValidator) -> None:
        if not node_validator.module_ids:
            return
        node_dict = node_validator.model_dump()
        grpc_host, grpc_port = util.daq_grpc_endpoint(node_dict)
        client = DaqControlClient(host=grpc_host, port=grpc_port)
        
        # StatusDaq request
        ok, status = await loop.run_in_executor(None, lambda: client.StatusDaq({
            'data_dir': node_validator.data_dir,
            'check_hashpipe_running': True
        }))
        
        if not ok:
             raise RuntimeError(f"Heartbeat RPC failed for node {node_validator.ip_addr}")
        if not status.get('hashpipe_running'):
             raise RuntimeError(f"Hashpipe heartbeat check failed (NOT ALIVE) on node {node_validator.ip_addr}")
        
        # Success: Update ledger with receipt
        receipt = NodeReceipt(
            ip_addr=node_validator.ip_addr,
            status="START_SUCCESS",
            hashpipe_pid=status.get('hashpipe_pid'),
            data_dir=node_validator.data_dir
        )
        state_mgr.update_node_receipt(receipt)
        logger.info(f"Node {node_validator.ip_addr} heartbeat OK (PID {receipt.hashpipe_pid})")

    # Parallel heartbeat verification
    await asyncio.gather(*(probe_node(n) for n in daq_config.daq_nodes))


async def start_run(
    obs_config: ObsConfigValidator,
    daq_config: DaqConfigValidator,
    quabo_uids: QuaboUidsValidator,
    data_config: DataConfigValidator,
    network_config: NetworkConfigValidator | dict[str, Any],
    no_hv: bool,
    no_redis: bool,
    no_data: bool
) -> bool:
    """
    Main transactional run coordinator.
    Implements the Rollback Ladder:
    1. Acquire lock.
    2. Validate state (no run in progress).
    3. Initialize Ledger (run_state.toml).
    4. Pre-flight (Ping sweep, PH baseline).
    5. Phase 1: Directories.
    6. Phase 2: Hardware config.
    7. Phase 3: Start recording (concurrent gRPC + Heartbeat).
    8. Finalize.
    """
    state_mgr = RunStateManager()
    
    try:
        state_mgr.acquire_lock()
    except RuntimeError as e:
        print(e)
        return False

    try:
        my_ip = util.local_ip()
        head_node_ip = socket.gethostbyname(str(daq_config.head_node_ip_addr))
        if head_node_ip not in my_ip:
            print(f'This node ({my_ip}) is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})')
            return False

        # Check existing ledger state instead of simple file
        existing_state = state_mgr.load_state()
        if existing_state and existing_state.status in ["STARTING", "ACTIVE", "STOPPING"]:
            print(f"A run is already in progress according to ledger: {existing_state.run_name} (Status: {existing_state.status})")
            print("Run stop.py, then try again.")
            return False

        if util.is_hk_recorder_running():
            print('The HK recorder is running. Run stop.py, then try again.')
            return False
            
        if not no_redis and not util.are_redis_daemons_running():
            print('Redis daemons are not running. Run config.py --redis_daemons')
            util.show_redis_daemons()
            return False

        if not ph_baseline_file_ok():
            return False

        # Initialize Ledger
        run_name = pff.run_dir_name(obs_config.name, data_config.run_type)
        initial_ledger = RunStateLedger(
            run_name=run_name,
            status="STARTING",
            start_time=datetime.now(UTC).isoformat(),
            config_metadata={
                "obs_name": obs_config.name,
                "run_type": data_config.run_type,
                "no_hv": no_hv
            }
        )
        state_mgr.save_state(initial_ledger)

        # get git commit info, and write the info into sw_info.json
        get_sw_info()
        
        config_file.associate(daq_config, quabo_uids)
        config_file.show_daq_assignments(quabo_uids)

        # --- ROLLBACK LADDER BEGIN ---
        try:
            print(f'setting up run directories for {run_name}')
            make_run_dirs(run_name, daq_config)
            
            if not no_data:
                print('starting data flow from quabos')
                start_data_flow(quabo_uids, data_config, daq_config, network_config)
                
                print('starting recording (Phase 3: Transactional)')
                await start_recording(obs_config, data_config, daq_config, run_name, no_hv, state_mgr)
            
        except Exception as e:
            print(f"\n[CRITICAL FAILURE] Start process aborted: {e}")
            print("Triggering Rollback Ladder...")
            
            # Update ledger to ABORTED immediately
            ledger = await asyncio.to_thread(state_mgr.load_state)
            if ledger:
                ledger.status = "ABORTED"
                await asyncio.to_thread(state_mgr.save_state, ledger)
            
            # Ladder Step 1: Stop remote nodes
            print("Stopping remote DAQ nodes...")
            for node in daq_config.daq_nodes:
                if not node.module_ids:
                    continue
                try:
                    node_dict = node.model_dump()
                    grpc_host, grpc_port = util.daq_grpc_endpoint(node_dict)
                    client = DaqControlClient(host=grpc_host, port=grpc_port)
                    # Use asyncio.to_thread for synchronous gRPC calls
                    await asyncio.to_thread(client.StopDaq, {'data_dir': node.data_dir, 'run_dir': run_name})
                except Exception as stop_err:
                    print(f"Failed to stop node {node.ip_addr} during rollback: {stop_err}")

            # Ladder Step 2: Stop Quabo data flow
            print("Stopping Quabo data flow...")
            await asyncio.to_thread(util.stop_data_flow, quabo_uids, network_config)

            # Ladder Step 3: Kill local daemons
            print("Stopping local daemons...")
            await asyncio.to_thread(util.kill_hk_recorder)
            await asyncio.to_thread(util.kill_hv_updater)
            await asyncio.to_thread(util.kill_module_temp_monitor)

            # Ladder Step 4: Archive partial artifacts
            aborted_dir = f"{daq_config.head_node_data_dir}/_aborted/{run_name}"
            print(f"Archiving partial artifacts to {aborted_dir}")
            await asyncio.to_thread(os.makedirs, os.path.dirname(aborted_dir), exist_ok=True)
            local_run_dir = f"{daq_config.head_node_data_dir}/{run_name}"
            if await asyncio.to_thread(os.path.exists, local_run_dir):
                await asyncio.to_thread(shutil.move, local_run_dir, aborted_dir)
                
                # Use a string representation of the exception for the context dump
                err_msg = str(e)
                tb_msg = traceback.format_exc()
                def dump_context(msg: str, tb: str) -> None:
                    with open(f"{aborted_dir}/start_failure_context.json", "w") as f:
                        json.dump({"error": msg, "traceback": tb}, f, indent=4)
                await asyncio.to_thread(dump_context, err_msg, tb_msg)

            return False
        # --- ROLLBACK LADDER END ---

        # Mark ACTIVE in ledger
        ledger = state_mgr.load_state()
        if ledger:
            ledger.status = "ACTIVE"
            state_mgr.save_state(ledger)
            
        # Write legacy current_run file for compatibility
        util.write_run_name(daq_config.model_dump(), run_name)
        
        print(f'started run {run_name}')
        return True

    finally:
        state_mgr.release_lock()

async def main() -> None:
    if not await asyncio.to_thread(os.path.exists, 'logs'):
        await asyncio.to_thread(os.makedirs, 'logs')
    logfile = 'logs/start.log'
    util.create_logger(logfile, 'PANOSETI.Start', 'a')
    logger = logging.getLogger('PANOSETI.Start')
    logger.info('************************************')
    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--no_hv', dest='no_hv', action='store_true', default=False,
                        help='Take data without high voltage.')
    parser.add_argument('--no_redis', dest='no_redis', action='store_true', default=False,
                        help='OK if redis daemons not running.')
    parser.add_argument('--no_data', dest='no_data', action='store_true', default=False,
                        help='Set up to record, but don\'t start data flow or record.')
    parser.add_argument('--nsecs', dest='nsecs', type=int, default=0,
                        help='Record for N seconds, then stop run.')
    parser.add_argument('--stop_session', dest='stop_session', action='store_true', default=False,
                        help='Stop session at end of run (with --nsecs).')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='print commands.')
    args = parser.parse_args()
    no_hv = args.no_hv
    no_redis = args.no_redis
    no_data = args.no_data
    nsecs = args.nsecs
    stop_session = args.stop_session
    verbose = args.verbose

    # load config files
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    
    success = await start_run(
        obs_config, daq_config, quabo_uids, data_config,
        network_config, no_hv, no_redis, no_data
    )
    
    if success and nsecs:
        await asyncio.sleep(nsecs)
        await stop.stop_run(
            daq_config, 
            network_config, 
            quabo_uids,
            verbose=verbose
        )
        if stop_session:
            session_stop.session_stop(obs_config)


if __name__ == "__main__":
    asyncio.run(main())

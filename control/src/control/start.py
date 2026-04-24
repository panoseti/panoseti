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

# ---------------- TELEMETRY LOGGING ----------------
import asyncio
import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import traceback
from datetime import UTC, datetime
from typing import Any

import grpc
import typer

# ---------------------------------------------------
# panoseti-grpc imports
from panoseti_grpc.daq_control.client import AsyncDaqControlClient
from panoseti_grpc.telemetry.logger import get_logger

import control.session_stop as session_stop
import control.stop as stop
from control.driver import quabo_driver
from control.tools.sw_info import get_sw_info
from control.utils import config_file, file_xfer, pff, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    DataConfig,
    NetworkConfig,
    ObsConfig,
    QuaboUids,
    RunStateLedger,
)
from control.utils.run_state import LockError, NodeReceipt, RunStateManager, ValidationError

# ---------------------------------------------------

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(
    "PSETI.Start",
    log_dir=log_dir,
    grpc_enabled=True,
    reset=True
)

# ---------------------------------------------------

verbose = False

class StartTransaction:
    """
    Context manager for a transactional observing run startup.
    Implements a robust rollback ladder and lock management.
    """
    def __init__(
        self, 
        state_mgr: RunStateManager, 
        run_name: str, 
        daq_config: DaqConfig, 
        quabo_uids: QuaboUids, 
        network_config: NetworkConfig
    ) -> None:
        self.state_mgr = state_mgr
        self.run_name = run_name
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.network_config = network_config
        self.success = False

    async def __aenter__(self) -> StartTransaction:
        await asyncio.to_thread(self.state_mgr.acquire_lock)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        try:
            if exc_type is not None:
                # Ladder Step 0: Identify the failure
                if exc_type is ValidationError:
                    logger.warning(f"Aborting start due to validation failure: {exc_val}")
                else:
                    logger.error(f"[CRITICAL FAILURE] Start process aborted: {exc_val}")
                
                logger.info("Triggering Rollback Ladder...")

                # Wait briefly for cancelled tasks to finish their synchronous I/O
                await asyncio.sleep(0.2)

                # Ladder Step 1: Update ledger to ABORTED immediately (WAL pattern)
                # We re-load to ensure we have any node receipts written just before cancellation
                ledger = await asyncio.to_thread(self.state_mgr.load_state)
                if ledger:
                    ledger.status = "ABORTED"
                    await asyncio.to_thread(self.state_mgr.save_state, ledger)

                # Ladder Step 2: Stop remote DAQ nodes (Any that were attempted)
                logger.info("Stopping remote DAQ nodes...")
                # Re-load again to be absolutely sure we have all concurrent updates
                ledger = await asyncio.to_thread(self.state_mgr.load_state)
                
                async def rollback_node(node: DaqNode) -> None:
                    receipt = next((n for n in ledger.nodes if str(n.ip_addr) == str(node.ip_addr)), None) if ledger else None
                    if not receipt:
                        return

                    logger.info(f"Rolling back node {node.ip_addr} (Status: {receipt.status})...")
                    try:
                        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
                        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                            await client.StopDaq({'data_dir': node.data_dir, 'run_dir': self.run_name}, timeout=15.0)
                    except Exception as stop_err:
                        logger.warning(f"StopDaq RPC failed for {node.ip_addr} during rollback ({stop_err}). Escalating to SSH pkill...")
                        try:
                            ssh_args = ["ssh", "-o", "BatchMode=yes"]
                            if node.port_forwarding and node.port_forwarding.status:
                                real_ip = str(node.port_forwarding.gw_ip)
                                port = str(node.port_forwarding.port)
                                ssh_args.extend(["-p", port, f"{node.username}@{real_ip}"])
                            else:
                                ssh_args.append(f"{node.username}@{node.ip_addr}")

                            ssh_args.append("pkill -9 hashpipe")
                            res = await asyncio.create_subprocess_exec(*ssh_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            await res.wait()
                            if res.returncode in [0, 1]:
                                logger.info(f"Hard-kill escalation succeeded for node {node.ip_addr}")
                            else:
                                logger.error(f"Hard-kill escalation failed for node {node.ip_addr} (rc={res.returncode})")
                        except Exception as ssh_err:
                            logger.error(f"Failed to stop node {node.ip_addr} even with SSH escalation: {ssh_err}")

                async with asyncio.TaskGroup() as tg:
                    for node in self.daq_config.daq_nodes:
                        if node.module_ids:
                            tg.create_task(rollback_node(node))

                # Ladder Step 3: Stop Quabo data flow
                logger.info("Stopping Quabo data flow...")
                try:
                    await asyncio.to_thread(util.stop_data_flow, self.quabo_uids, self.network_config)
                except Exception as e2:
                    logger.error(f"Failed to stop Quabo data flow: {e2}")

                # Ladder Step 4: Kill local daemons
                logger.info("Stopping local daemons...")
                try:
                    await asyncio.to_thread(util.kill_hk_recorder)
                    await asyncio.to_thread(util.kill_hv_updater)
                    await asyncio.to_thread(util.kill_module_temp_monitor)
                except Exception as e3:
                    logger.error(f"Failed to kill local daemons: {e3}")

                # Ladder Step 5: Archive partial artifacts
                try:
                    aborted_base = f"{self.daq_config.head_node_data_dir}/_aborted/{self.run_name}"
                    suffix = 1
                    aborted_dir = aborted_base
                    while await asyncio.to_thread(os.path.exists, aborted_dir):
                        aborted_dir = f"{aborted_base}_{suffix}"
                        suffix += 1

                    logger.info(f"Archiving partial artifacts to {aborted_dir}")
                    await asyncio.to_thread(os.makedirs, aborted_dir, exist_ok=True)

                    # Write failure context
                    err_msg = str(exc_val)
                    tb_msg = "".join(traceback.format_tb(exc_tb)) if exc_tb else ""
                    def dump_context(msg: str, tb: str) -> None:
                        with open(f"{aborted_dir}/start_failure_context.json", "w") as f:
                            json.dump({"error": msg, "traceback": tb}, f, indent=4)
                    await asyncio.to_thread(dump_context, err_msg, tb_msg)

                    local_run_dir = f"{self.daq_config.head_node_data_dir}/{self.run_name}"
                    if await asyncio.to_thread(os.path.exists, local_run_dir):
                        items = os.listdir(local_run_dir)
                        logger.info(f"Found {len(items)} items in {local_run_dir} to archive: {items}")
                        for item in items:
                            s = os.path.join(local_run_dir, item)
                            d = os.path.join(aborted_dir, item)
                            await asyncio.to_thread(shutil.move, s, d)
                        await asyncio.to_thread(os.rmdir, local_run_dir)
                    else:
                        logger.warning(f"local_run_dir {local_run_dir} does not exist; nothing to archive.")
                except Exception as e4:
                    logger.error(f"Failed to archive partial artifacts: {e4}")

            if exc_type is ValidationError:
                return True # Suppress validation errors for a clean exit
            
            if exc_type is not None:
                return not issubclass(exc_type, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))
            
            return False

        finally:
            await asyncio.to_thread(self.state_mgr.release_lock)

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
def get_daq_params(data_config: DataConfig) -> quabo_driver.DAQ_PARAMS:
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
    quabo_uids: QuaboUids,
    data_config: DataConfig,
    daq_config: DaqConfig,
    network_config: NetworkConfig
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
    # logger = logging.getLogger('PSETI.Start.start_data_flow')
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
                ip_ports = util.get_quabo_ip_port(module.ip_addr, i, network_config)
                real_ip = ip_ports.ip_addr
                cmd_port = ip_ports.cmd_port
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
            ip_ports = util.get_quabo_ip_port(module.ip_addr, 0, network_config)
            real_ip = ip_ports.ip_addr
            cmd_port = ip_ports.cmd_port
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            quabo.swpps()
            quabo.close()


def make_run_dirs(
    run_name: str,
    obs_config: ObsConfig,
    daq_config: DaqConfig,
    quabo_uids: QuaboUids,
    data_config: DataConfig,
    network_config: NetworkConfig
) -> None:
    """Create hierarchical run directories and snapshot configuration files.
    
    Snapshotting Contract:
    - Instead of copying from disk (which can mutate), this method writes the 
      in-memory Pydantic models back to JSON files in the run directory.
    - Ensures the run directory is a faithful record of the actual run parameters.
    """
    # logger = logging.getLogger('PSETI.Start')
    my_ip = util.local_ip()
    run_dir = f'{daq_config.head_node_data_dir}/{run_name}'
    os.makedirs(run_dir, exist_ok=True)

    # 1. Snapshot in-memory config models to head node run dir
    # This prevents mid-flight disk mutations from leaking into the run records.
    # Note: we exclude bidirectional links ('modules', 'daq_node') to avoid circular serialization.
    config_snapshots = {
        config_file.obs_config_filename: obs_config,
        config_file.daq_config_filename: daq_config,
        config_file.data_config_filename: data_config,
        config_file.network_config_filename: network_config,
        config_file.quabo_uids_filename: quabo_uids,
    }
    
    for filename, model in config_snapshots.items():
        base_name = os.path.basename(filename)
        dest_path = f"{run_dir}/{base_name}"
        with open(dest_path, "w") as f:
            # We use model_dump then json.dump to allow fine-grained exclusion
            data = model.model_dump(exclude={'modules', 'daq_node'})
            json.dump(data, f, indent=4, default=str)
            
    # Copy other transient artifacts (sw_info.json, ph_baseline.json) from their respective locations
    # to the head node run dir.
    artifact_map = {
        config_file.quabo_ph_baseline_filename: PanoPaths.tmp_dir() / config_file.quabo_ph_baseline_filename,
        config_file.sw_info_filename: PanoPaths.software_root_dir() / config_file.sw_info_filename,
    }
    for base_name, src_path in artifact_map.items():
        if src_path.exists():
             shutil.copyfile(src_path, f'{run_dir}/{base_name}')
        else:
             logger.debug(f"Artifact {src_path} not found; skipping snapshot.")

    # 2. make module and run directories on DAQ nodes
    for node in daq_config.daq_nodes:
        # Check if this node has any modules assigned
        # DaqNode has module_ids
        if not node.module_ids:
            continue
        ip_addr = str(node.ip_addr)
        if ip_addr in my_ip:
            # We need to know which module IDs are on this node to create module_N dirs
            # node.module_ids is a list of ints or a range string (preprocessed to list[int])
            for mid in node.module_ids:
                path = f'{daq_config.head_node_data_dir}/module_{mid}/{run_name}'
                if verbose:
                    print(f"mkdir -p {path}")
                os.makedirs(path, exist_ok=True)
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
            ssh_args = ["ssh"]
            if node.port_forwarding and node.port_forwarding.status:
                real_ip = str(node.port_forwarding.gw_ip)
                port = str(node.port_forwarding.port)
                ssh_args.extend(["-p", port, f"{username}@{real_ip}"])
            else:
                ssh_args.append(f"{username}@{ip_addr}")
            ssh_args.append(rcmnd)
            
            if verbose:
                print(" ".join(ssh_args))
            res = subprocess.run(ssh_args, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Failed to create run dirs on {ip_addr}: {res.stderr}")

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name, verbose)


async def start_recording(
    obs_config: ObsConfig,
    data_config: DataConfig,
    daq_config: DaqConfig,
    run_name: str,
    no_hv: bool,
    state_mgr: RunStateManager,
    cancel_event: asyncio.Event,
    startdaq_timeout: float = 10.0,
    startdaq_retries: int = 3
) -> None:
    """
    Asynchronously starts recording on DAQ nodes and performs heartbeat liveness checks.
    Transactional Contract:
    - Starts local HK/HV daemons.
    - Issues StartDaq to all remote nodes concurrently.
    - Updates run_state.toml with STARTING receipts.
    - Performs retry heartbeat probe loop (≤ 5 attempts x 1 s back-off).
    - Upgrades to START_SUCCESS after heartbeat.
    - Raises Exception on ANY failure or cancellation to trigger the parent rollback ladder.
    """
    # logger = logging.getLogger('PSETI.Start.start_recording')
    # loop = asyncio.get_running_loop()

    # 1. Start local daemons
    util.start_hk_recorder(daq_config, run_name)
    if not no_hv:
        util.start_hv_updater()
        util.start_module_temp_monitor()

    # 2. Concurrent StartDaq
    max_file_size_mb = data_config.max_file_size_mb or util.default_max_file_size_mb
    daq_params = get_daq_params(data_config)

    # Pre-write STARTING receipts to ensure rollback ladder catches them if TaskGroup is cancelled early
    for node_validator in daq_config.daq_nodes:
        if node_validator.module_ids:
            await state_mgr.update_node_receipt(NodeReceipt(
                ip_addr=node_validator.ip_addr,
                status="STARTING",
                data_dir=node_validator.data_dir
            ))

    async def start_node(node_validator: DaqNode) -> None:
        if not node_validator.module_ids:
            return
        
        grpc_host, grpc_port = util.daq_grpc_endpoint(node_validator)
        logger.info(f'StartDaq via gRPC: {grpc_host}:{grpc_port} modules={node_validator.module_ids}')
        
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
        
        last_err = ""
        
        for attempt in range(1, startdaq_retries + 1):
            try:
                async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                    ok = await asyncio.wait_for(
                        client.StartDaq(start_args, timeout=startdaq_timeout),
                        timeout=startdaq_timeout + 5
                    )
                if ok:
                    return # Success
                else:
                    last_err = "StartDaq RPC returned False"
                    break # Hard failure, don't retry
            except TimeoutError:
                last_err = f"StartDaq TIMEOUT ({startdaq_timeout})"
                break # Timeout usually means non-transient or black hole
            except (grpc.RpcError, ConnectionError) as e:
                # We need to reach into .__cause__ to get the original RpcError
                original_e = e.__cause__ if isinstance(e, ConnectionError) else e
                if isinstance(original_e, grpc.RpcError):
                    last_err = f"gRPC {original_e.code()}: {original_e.details()}"
                    if original_e.code() == grpc.StatusCode.UNAVAILABLE and attempt < startdaq_retries:
                        logger.warning(f"Node {node_validator.ip_addr} transiently unavailable. Retrying ({attempt}/{startdaq_retries})...")
                        await asyncio.sleep(1.0)
                        continue
                else:
                    last_err = f"StartDaq Error: {e}"
                break
        
        # If we reach here, it's a hard failure or we ran out of retries
        await state_mgr.update_node_receipt(NodeReceipt(
            ip_addr=node_validator.ip_addr,
            status="START_FAILED",
            message=last_err
        ))
        raise RuntimeError(f'StartDaq failed for node {node_validator.ip_addr}: {last_err}')

    # Execute all starts in parallel with TaskGroup for fail-fast behavior
    async with asyncio.TaskGroup() as tg:
        for n in daq_config.daq_nodes:
            tg.create_task(start_node(n))

    if cancel_event.is_set():
        raise asyncio.CancelledError("Start process cancelled by user")

    # 3. Liveness Probe (Heartbeat) with retry loop
    logger.info("Waiting for Hashpipe stabilization heartbeat...")
    
    async def probe_node(node_validator: DaqNode) -> None:
        if not node_validator.module_ids:
            return
        
        grpc_host, grpc_port = util.daq_grpc_endpoint(node_validator)
        
        # Retry loop: 5 attempts, 1s backoff
        last_err = ""
        for attempt in range(1, 6):
            if cancel_event.is_set():
                raise asyncio.CancelledError("Heartbeat check cancelled by user")
            
            await asyncio.sleep(1.0) # 1s between attempts
            
            try:
                async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                    ok, status = await client.StatusDaq({
                        'data_dir': node_validator.data_dir,
                        'check_hashpipe_running': True,
                        'check_disk_usage': False,
                        'check_run_dirs': False
                    })
                
                if ok:
                    if status.get('hashpipe_running'):
                        # Success: Update ledger with START_SUCCESS
                        receipt = NodeReceipt(
                            ip_addr=node_validator.ip_addr,
                            status="START_SUCCESS",
                            hashpipe_pid=status.get('hashpipe_pid'),
                            data_dir=node_validator.data_dir
                        )
                        await state_mgr.update_node_receipt(receipt)
                        logger.info(f"Node {node_validator.ip_addr} heartbeat OK on attempt {attempt} (PID {receipt.hashpipe_pid})")
                        return
                    else:
                        last_err = "hashpipe not running"
                else:
                    last_err = "StatusDaq RPC returned False"
            except Exception as e:
                last_err = str(e)
            
            logger.warning(f"Heartbeat attempt {attempt} failed for {node_validator.ip_addr}: {last_err}")

        # If we reach here, all retries failed
        await state_mgr.update_node_receipt(NodeReceipt(
            ip_addr=node_validator.ip_addr,
            status="START_FAILED",
            message=f"Heartbeat failed after 5 attempts: {last_err}"
        ))
        raise RuntimeError(f"Hashpipe heartbeat check failed on node {node_validator.ip_addr}: {last_err}")

    # Parallel heartbeat verification with TaskGroup
    async with asyncio.TaskGroup() as tg:
        for n in daq_config.daq_nodes:
            tg.create_task(probe_node(n))

    # Phase 5: Post-stabilization Liveness Probe (Early Exit Guard)
    logger.info("Phase 5: Performing 2s stabilization liveness probe...")
    await asyncio.sleep(2.0)
    
    async with asyncio.TaskGroup() as tg:
        for n in daq_config.daq_nodes:
            if not n.module_ids:
                continue
            
            async def verify_liveness(node_validator: DaqNode) -> None:
                grpc_host, grpc_port = util.daq_grpc_endpoint(node_validator)
                
                try:
                    async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                        ok, status = await client.StatusDaq({
                            'data_dir': node_validator.data_dir,
                            'check_hashpipe_running': True,
                            'check_disk_usage': False,
                            'check_run_dirs': False
                        })
                    
                    if not ok or not status.get('hashpipe_running'):
                        err = status.get('message', 'hashpipe exited during stabilization')
                        raise RuntimeError(f"Node {node_validator.ip_addr} liveness check failed: {err}")
                    
                    logger.info(f"Node {node_validator.ip_addr} Phase 5 Liveness OK")
                except Exception as e:
                    await state_mgr.update_node_receipt(NodeReceipt(
                        ip_addr=node_validator.ip_addr,
                        status="START_FAILED",
                        message=f"Liveness Check Failed: {e}"
                    ))
                    raise RuntimeError(f"Node {node_validator.ip_addr} liveness check failed: {e}") from e

            tg.create_task(verify_liveness(n))


async def _check_quabo_reachability(
    quabo_uids: QuaboUids,
    network_config: NetworkConfig,
    lenient: bool = False
) -> None:
    """Verify that all configured Quabos are reachable on the network."""
    logger.info("Performing Quabo reachability sweep...")
    
    tasks = []
    
    async def check_one(base_ip: str, index: int) -> None:
        ip_ports = util.get_quabo_ip_port(module.ip_addr, index, network_config)
        real_ip = ip_ports.ip_addr
        cmd_port = ip_ports.cmd_port
        
        from control.utils.config_validator import _check_reachability
        
        loop = asyncio.get_running_loop()
        ok, err = await loop.run_in_executor(None, lambda: _check_reachability(str(real_ip), cmd_port, target_type="quabo", timeout=2.0))
        if not ok:
            msg = f"Quabo at {real_ip}:{cmd_port} is UNREACHABLE: {err}"
            if lenient:
                 logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                 return
            raise ValidationError(msg)

    for dome in quabo_uids.domes:
        for module in dome.modules:
            base_ip = str(module.ip_addr)
            for i in range(4):
                if module.quabos[i].uid != '':
                    tasks.append(check_one(base_ip, i))
    
    if tasks:
        await asyncio.gather(*tasks)
    logger.info("All configured Quabos are reachable.")


async def start_run(
    obs_config: ObsConfig,
    daq_config: DaqConfig,
    quabo_uids: QuaboUids,
    data_config: DataConfig,
    network_config: NetworkConfig,
    no_hv: bool,
    no_redis: bool,
    no_data: bool,
    force_reset: bool = False,
    run_name: str | None = None
) -> str | None:
    """Main transactional run coordinator.

    Args:
        obs_config (ObsConfig): _description_
        daq_config (DaqConfig): _description_
        quabo_uids (QuaboUids): _description_
        data_config (DataConfig): _description_
        network_config (NetworkConfig): _description_
        no_hv (bool): _description_
        no_redis (bool): _description_
        no_data (bool): _description_
        force_reset (bool, optional): _description_. Defaults to False.
        run_name (str | None, optional): _description_. Defaults to None.

    Raises:
        ValidationError: _description_
        ValidationError: _description_
        ValidationError: _description_
        ValidationError: _description_
        ValidationError: _description_
        ValidationError: _description_
        asyncio.CancelledError: _description_
        asyncio.CancelledError: _description_

    Returns:
        str | None: _description_
    """
    
    # --- Pre-flight: DAQ gRPC reachability sweep ---
    logger.info("Performing DAQ node gRPC reachability sweep...")
    async def check_node_grpc(node: DaqNode) -> None:
        if not node.module_ids:
            return
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        try:
            async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                await client.StatusDaq({"data_dir": node.data_dir}, timeout=5.0)
        except Exception as e:
            raise ValidationError(f"DAQ node {node.ip_addr} gRPC is unreachable at {grpc_host}:{grpc_port}: {e}") from e

    try:
        async with asyncio.TaskGroup() as tg:
            for node in daq_config.daq_nodes:
                tg.create_task(check_node_grpc(node))
        logger.info("All configured DAQ nodes are reachable via gRPC.")
    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            logger.error(str(exc))
        raise ValidationError("One or more DAQ nodes are unreachable via gRPC.") from eg

    state_mgr = RunStateManager()
    cancel_event = asyncio.Event()
    
    def signal_handler(signal: signal.Signals):
        logger.critical(f"start.py received the signal {signal!r}")
        cancel_event.set()
         

    # Install signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(signal_handler, sig))
    
    if run_name is None:
        run_name = pff.run_dir_name(obs_config.name, data_config.run_type)

    tx = None
    try:
        async with StartTransaction(state_mgr, run_name, daq_config, quabo_uids, network_config) as tx:
            # Pre-flight Validation
            if not config_file.validate_all(check_network=False):
                 msg = "Pre-flight configuration validation failed."
                 if daq_config.head_node_container:
                     logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                 else:
                     raise ValidationError(msg)

            my_ip = util.local_ip()
            head_node_ip = socket.gethostbyname(str(daq_config.head_node_ip_addr))
            if head_node_ip not in my_ip:
                msg = f'This node ({my_ip}) is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})'
                if daq_config.head_node_container:
                    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                else:
                    raise ValidationError(msg)

            # Stale ledger self-heal
            existing_state = state_mgr.load_state()
            if existing_state and existing_state.status in ["STARTING", "ACTIVE", "STOPPING"]:
                stale = False
                if force_reset:
                    logger.info("Force reset requested. Archiving existing ledger.")
                    stale = True
                elif existing_state.host == socket.gethostname() and existing_state.pid:
                    # Check if PID is alive
                    try:
                        os.kill(existing_state.pid, 0)
                    except OSError:
                        logger.info(f"Detected stale ledger from dead PID {existing_state.pid} on this host.")
                        stale = True
                
                if stale:
                    aborted_base = f"{daq_config.head_node_data_dir}/_aborted/{existing_state.run_name}"
                    suffix = 1
                    aborted_dir = aborted_base
                    while await asyncio.to_thread(os.path.exists, aborted_dir):
                        aborted_dir = f"{aborted_base}_{suffix}"
                        suffix += 1
                    
                    logger.info(f"Archiving stale ledger to {aborted_dir}")
                    os.makedirs(aborted_dir, exist_ok=True)
                    shutil.move(str(state_mgr.state_path), f"{aborted_dir}/stale_run_state.toml")
                else:
                    raise ValidationError(f"A run is already in progress according to ledger: {existing_state.run_name} (Status: {existing_state.status}). Run stop.py, then try again, or use --force-reset.")

            if await asyncio.to_thread(util.is_hk_recorder_running):
                msg = 'The HK recorder is running. Run stop.py, then try again.'
                if daq_config.head_node_container:
                    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                else:
                    raise ValidationError(msg)
                
            if not no_redis and not await asyncio.to_thread(util.are_redis_daemons_running):
                await asyncio.to_thread(util.show_redis_daemons)
                msg = 'Redis daemons are not running. Run config.py --redis_daemons'
                if daq_config.head_node_container:
                    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                else:
                    raise ValidationError(msg)

            if not await asyncio.to_thread(ph_baseline_file_ok):
                msg = 'PH baseline file check failed.'
                if daq_config.head_node_container:
                    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                else:
                    raise ValidationError(msg)
            # Initialize Ledger
            initial_ledger = RunStateLedger(
                run_name=run_name,
                status="STARTING",
                start_time=datetime.now(UTC).isoformat(),
                pid=os.getpid(),
                host=socket.gethostname(),
                config_metadata={
                    "obs_name": obs_config.name,
                    "run_type": data_config.run_type,
                    "no_hv": no_hv
                }
            )
            await asyncio.to_thread(state_mgr.save_state, initial_ledger)

            # Snapshot configs into the directory immediately (before validation/delays)
            # This ensures we archive the ORIGINAL files even if they change on disk mid-setup
            if not no_data:
                logger.info(f'setting up run directories for {run_name}')
                await asyncio.to_thread(
                    make_run_dirs, run_name, obs_config, daq_config, quabo_uids, data_config, network_config
                )

            # Validation checks
            my_ip = util.local_ip()
            get_sw_info()
            
            config_file.associate(daq_config, quabo_uids)
            config_file.show_daq_assignments(quabo_uids)

            if cancel_event.is_set():
                raise asyncio.CancelledError()

            if not no_data:
                await _check_quabo_reachability(
                    quabo_uids, network_config, lenient=bool(daq_config.head_node_container)
                )

            if not no_data:
                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                logger.info('starting data flow from quabos')
                start_data_flow(quabo_uids, data_config, daq_config, network_config)
                
                logger.info('starting recording (Phase 3: Transactional)')
                await start_recording(
                    obs_config, data_config, daq_config, run_name, no_hv, state_mgr, cancel_event,
                    startdaq_timeout=10.0, startdaq_retries=3
                )
            
            # Mark ACTIVE in ledger
            ledger = state_mgr.load_state()
            if ledger:
                ledger.status = "ACTIVE"
                state_mgr.save_state(ledger)
                
            # Write legacy current_run file for compatibility
            util.write_run_name(daq_config, run_name)
            
            logger.info(f'started run {run_name}')
            tx.success = True
            return run_name

    except LockError as e:
        logger.error(f"FATAL: {e}")
        return None
    except Exception as e:
        # Unexpected errors should already be handled by tx.__aexit__ rollback,
        # but we catch here to ensure we don't crash the loop.
        logger.debug(f"start_run caught exception: {e}")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    
    return run_name if getattr(tx, 'success', False) else None


app = typer.Typer(help="Start a PSETI recording run.", no_args_is_help=False)

@app.command()
def main(
    no_hv: bool = typer.Option(False, "--no_hv", help="Take data without high voltage."),
    no_redis: bool = typer.Option(False, "--no_redis", help="OK if redis daemons not running."),
    no_data: bool = typer.Option(False, "--no_data", help="Set up to record, but don't start data flow or record."),
    nsecs: int = typer.Option(0, "--nsecs", help="Record for N seconds, then stop run."),
    stop_session: bool = typer.Option(False, "--stop_session", help="Stop session at end of run (with --nsecs)."),
    verbose_opt: bool = typer.Option(False, "--verbose", help="print commands."),
    force_reset: bool = typer.Option(False, "--force-reset", help="Force reset the state ledger if stale."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the action without prompting."),
):
    """
    start a recording run:

    - figure out association of quabos and DAQ nodes,
      based on config files
    - create \"run directories\" on head node, DAQ nodes
    - start the HK recorder
    - start the HV updater
    - start the temperature monitor
    - start the flow of data: set DAQ mode and dest IP addr of quabos
    - send commands to DAQ nodes to start hashpipe program

    fail if a recording run is in progress,
    or if recording activities are active
    """
    global verbose
    verbose = verbose_opt
    
    if not yes:
        typer.confirm("Are you sure you want to start a new recording run?", abort=True)
        
    success = asyncio.run(async_main_logic(
        no_hv, no_redis, no_data, nsecs, stop_session, verbose, force_reset
    ))
    if not success:
        raise typer.Exit(code=1)

async def async_main_logic(
    no_hv: bool,
    no_redis: bool,
    no_data: bool,
    nsecs: int,
    stop_session: bool,
    verbose: bool,
    force_reset: bool,
) -> bool:

    # load config files
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    
    success_run_name = await start_run(
        obs_config, daq_config, quabo_uids, data_config,
        network_config, no_hv, no_redis, no_data, force_reset
    )
    
    if not success_run_name:
        return False

    if success_run_name and nsecs:
        await asyncio.sleep(nsecs)
        await stop.stop_run(
            daq_config, 
            network_config, 
            quabo_uids,
            verbose=verbose
        )
        if stop_session:
            session_stop.session_stop(obs_config)
    
    return True


if __name__ == "__main__":
    app()

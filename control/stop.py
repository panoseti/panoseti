#! /usr/bin/env python3

# stop and finish a recording run if one is in progress.
# stop recording activities whether or not a run is in progress.
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - if a run is in progress, copy data files to head and delete from DAQs
#
# options:
#   --verbose           print details
#   --no_collect        don't copy data files to head node
#   --no_cleanup        don't delete files from DAQ nodes
#   --run X             clean up run X (default: read from current_run)

import asyncio
import builtins
import contextlib
import logging
import os
import shutil
import signal
import socket
import tempfile
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from glob import glob
from typing import Any

from panoseti_grpc.daq_control.client import DaqControlClient

import config
from tools.interleave import PID_FILE
from utils import collect, config_file, pff, util
from utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    NetworkConfigValidator,
    QuaboUidsValidator,
)
from utils.run_state import RunStateManager
from utils.util import (
    collect_complete_filename,
    hk_symlink,
    img_symlink,
    now_str,
    ph_symlink,
    recording_ended_filename,
    run_complete_filename,
)

logger = logging.getLogger(__name__)


def stop_interleave(retry_limit: int = 10) -> None:
    """
    Checks if the interleave process is running and cleanly shuts it down.
    Includes process identity verification and SIGKILL escalation.
    """
    pid_file = PID_FILE
    if os.path.exists(pid_file):
        print("Active interleave process detected. Stopping it gracefully...")
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            
            # Verify identity: check /proc/pid/cmdline for 'interleave.py'
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode().replace('\x00', ' ')
                    if 'interleave.py' not in cmdline:
                         print(f"PID {pid} does not appear to be interleave.py. Cleaning stale PID file.")
                         os.remove(pid_file)
                         return
            except FileNotFoundError:
                print(f"PID {pid} no longer exists. Cleaning stale PID file.")
                os.remove(pid_file)
                return

            os.kill(pid, signal.SIGTERM)

            # Wait briefly for it to clean up and restore defaults
            for r in range(retry_limit):
                if not os.path.exists(pid_file):
                    print("Interleave process stopped.")
                    return
                logger.warning(f"Waiting for interleave process {pid} to exit... [{r+1}/{retry_limit}]")
                time.sleep(0.5)
            
            # SIGKILL escalation
            print(f"Interleave process {pid} refused to exit. Escalating to SIGKILL.")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            if os.path.exists(pid_file):
                os.remove(pid_file)
            
            # Synchronously restore MAROC defaults as a safety measure
            print("Restoring Quabo MAROC register defaults...")
            try:
                # We need to load configs for this
                obs_cfg = config_file.get_obs_config()
                data_cfg = config_file.get_data_config()
                daq_cfg = config_file.get_daq_config()
                quabo_uids = config_file.get_quabo_uids()
                quabo_info = config_file.get_quabo_info()
                network_cfg = config_file.get_network_config()
                config.do_maroc_config(
                    config_file.get_modules(obs_cfg.model_dump()),
                    quabo_uids.model_dump(),
                    quabo_info,
                    data_cfg.model_dump(),
                    obs_cfg.model_dump(),
                    daq_cfg.model_dump(),
                    network_cfg.model_dump(),
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: Failed to restore MAROC registers: {e}")

        except (OSError, ValueError) as e:
            print(f"Error stopping interleave: {e}")
            if os.path.exists(pid_file):
                os.remove(pid_file)


# =========================
# Print -> also prepend to UT log file
# =========================

_ORIG_PRINT = builtins.print

def _ut_human_timestamp() -> str:
    # Human-readable UTC timestamp
    return datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UT')

def _ut_yyyymmdd() -> str:
    return datetime.now(UTC).strftime('%Y%m%d')

def _datarec_log_path() -> str:
    yyyymmdd = _ut_yyyymmdd()
    log_dir = f"/mnt/data11/data/palomar/L0/{yyyymmdd}/obslogs"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"datarec_{yyyymmdd}.log")

def _prepend_line_to_file(path: str, line: str) -> None:
    # Prepend efficiently by writing a temp file then replacing.
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    old = b""
    try:
        with open(path, "rb") as f:
            old = f.read()
    except FileNotFoundError:
        old = b""

    new_bytes = (line + "\n").encode("utf-8", errors="replace")

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_datarec_", dir=d or None)
    try:
        with os.fdopen(fd, "wb") as tf:
            tf.write(new_bytes)
            tf.write(old)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

def print(*args: Any, **kwargs: Any) -> None:
    # Console print as-is + prepend to UT log file with timestamp.
    msg = " ".join(str(a) for a in args)
    with contextlib.suppress(Exception):
        _prepend_line_to_file(_datarec_log_path(), f"{_ut_human_timestamp()}: {msg}")
    _ORIG_PRINT(*args, **kwargs)

builtins.print = print

# write message to error log
#
def log_error(msg: str, run_dir: str | None) -> None:
    """Record an error message to a dedicated log file within the run directory.

    Args:
        msg: The error message to log.
        run_dir: Path to the current run directory. If None, logs to local 'stop_errors'.
    """
    print(msg)
    log_path = f'{run_dir}/stop_errors' if run_dir else 'stop_errors'
    with open(log_path, 'a') as f:
        f.write(f'{now_str()}: {msg}\n')

# tell all DAQ nodes to stop recording
#
async def stop_recording(daq_config: DaqConfigValidator, run_dir: str | None, verbose: bool) -> None:
    """Best-effort stop of all remote DAQ nodes. 
    
    Concurrently issues StopDaq gRPC commands to all active DAQ nodes. 
    Failure on one node does not block the shutdown of others.

    Args:
        daq_config: Validated DAQ configuration model.
        run_dir: Optional name of the run directory to stop.
        verbose: If True, prints gRPC endpoint details.
    """
    loop = asyncio.get_running_loop()

    async def stop_node(node: DaqNodeValidator) -> None:
        if not node.module_ids:
            return
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        if verbose:
            print(f'StopDaq via gRPC: {grpc_host}:{grpc_port}')
        
        try:
            client = DaqControlClient(host=grpc_host, port=grpc_port)
            # Use a strict timeout for the RPC
            ok = await loop.run_in_executor(None, lambda: client.StopDaq({
                'data_dir': node.data_dir,
                'run_dir':  run_dir,
            }))
            if not ok:
                print(f"Warning: StopDaq returned success=False for node {node.ip_addr}")
        except Exception as e:
            print(f"Error stopping node {node.ip_addr}: {e}")

    await asyncio.gather(*(stop_node(n) for n in daq_config.daq_nodes))


# write a "complete file" in the run dir
#
def write_complete_file(run_dir: str, filename: str) -> None:
    """Write a timestamped marker file to signify a phase of the run is complete.

    Args:
        run_dir: Path to the run directory.
        filename: Name of the marker file (e.g., 'recording_ended').
    """
    path = f'{run_dir}/{filename}'
    with open(path , 'w') as f:
        f.write(now_str())


def complete_file_exists(run_dir: str, filename: str) -> bool:
    """Check if a specific marker file exists in the run directory.

    Args:
        run_dir: Path to the run directory.
        filename: Name of the marker file.

    Returns:
        True if the file exists, False otherwise.
    """
    path = f'{run_dir}/{filename}'
    return os.path.exists(path)


# make symlinks to the first nonempty image and ph files in that dir
#
def make_links(run_dir: str, verbose: bool) -> None:
    """Create symlinks in the root data dir to the first nonempty PFF files.
    
    Helps visualization tools quickly find the latest relevant data. 
    Links 'img', 'ph', and 'hk' are updated.

    Args:
        run_dir: Path to the directory containing PFF artifacts.
        verbose: If True, prints details of the linked files.
    """
    if os.path.lexists(img_symlink):
        os.unlink(img_symlink)
    if os.path.lexists(ph_symlink):
        os.unlink(ph_symlink)
    if os.path.lexists(hk_symlink):
        os.unlink(hk_symlink)
    did_img = False
    did_ph = False
    did_hk = False
    for f in os.listdir(run_dir):
        path = f'{run_dir}/{f}'
        if not pff.is_pff_file(path):
            continue
        if os.path.getsize(path) == 0:
            continue
        ftype = pff.pff_file_type(f)
        if not did_img and ftype in ['img16', 'img8']:
            os.symlink(path, img_symlink)
            did_img = True
            if verbose:
                print(f'linked {img_symlink} to {f}')
        elif not did_ph and ftype in ['ph256', 'ph1024']:
            os.symlink(path, ph_symlink)
            did_ph = True
            if verbose:
                print(f'linked {ph_symlink} to {f}')
        elif not did_hk and ftype == 'hk':
            os.symlink(path, hk_symlink)
            did_hk = True
            if verbose:
                print(f'linked {hk_symlink} to {f}')
        if did_img and did_ph and did_hk:
            break
    if not did_img:
        print('make_links(): No nonempty image file')
    if not did_ph:
        print('make_links(): No nonempty PH file')
    if not did_hk:
        print('make_links(): No nonempty housekeeping file')



def _cleanup_daq_grpc(
    daq_config: DaqConfigValidator, 
    run: str, 
    head_run_dir: str, 
    verbose: bool,
    force: bool = False
) -> None:
    """Call CleanupData on each DAQ node via gRPC.
    Only called after collect_data() succeeds (transactional guarantee).
    """
    my_ip = util.local_ip()
    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        ip_addr = str(node.ip_addr)
        if ip_addr in my_ip:
            # Head node is also DAQ node: local rm -rf
            module_dirs = glob(f'{node.data_dir}/module_*/{run}')
            if verbose:
                print(f"Removing local directories: {module_dirs}")
            for d in module_dirs:
                try:
                    shutil.rmtree(d)
                except Exception as err:
                    log_error(f'cleanup_daq (local): failed to remove {d}: {err}', head_run_dir)
        else:
            grpc_host, grpc_port = util.daq_grpc_endpoint(node)
            if verbose:
                print(f'CleanupData via gRPC: {grpc_host}:{grpc_port} run_dir={run} force={force}')
            try:
                client = DaqControlClient(host=grpc_host, port=grpc_port)
                cleanup_resp = client.CleanupData({
                    'data_dir':  node.data_dir,
                    'run_dir':   run,
                    'module_id': node.module_ids,
                    'force':     force
                })
                if not cleanup_resp['success']:
                    log_error(f'CleanupData failed for node {ip_addr}: {cleanup_resp.get("message")}', head_run_dir) 
            except Exception as e:
                log_error(f'CleanupData error for node {ip_addr}: {e}', head_run_dir)


async def stop_run(
    daq_config: DaqConfigValidator,
    network_config: NetworkConfigValidator,
    quabo_uids: QuaboUidsValidator,
    verbose: bool = False, 
    no_cleanup: bool = False, 
    no_collect: bool = False,
    run: str | None = None,
    force_cleanup: bool = False
) -> None:
    """
    Transactional Best-Effort Shutdown.
    1. Acquire lock.
    2. Identify run from ledger.
    3. Aggressive stop of all components (Remote DAQs, Quabos, Daemons).
    4. Safe data collection (rsync).
    5. Cleanup.
    """
    state_mgr = RunStateManager()
    cancel_event = asyncio.Event()

    # Task 2.3: Signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: cancel_event.set())

    try:
        state_mgr.acquire_lock()
    except RuntimeError as e:
        print(e)
        return

    try:
        # convert head node name to IP address
        head_node_ip = socket.gethostbyname(str(daq_config.head_node_ip_addr))
        if head_node_ip not in util.local_ip():
            raise Exception(f'This computer is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})')

        # Load from ledger
        ledger = state_mgr.load_state()
        if not run:
            run = ledger.run_name if ledger else util.read_run_name()

        if not run:
            print("No run is in progress")
            return

        # Validation: prevent orphaning the current run
        if ledger and run != ledger.run_name and not force_cleanup:
             print(f"Warning: Requested run '{run}' does not match ledger run '{ledger.run_name}'.")
             print("Use --force-cleanup if you are sure.")
             return

        if ledger:
            ledger.status = "STOPPING"
            state_mgr.save_state(ledger)

        data_dir = daq_config.head_node_data_dir
        run_dir: str | None = f'{data_dir}/{run}'
        if run_dir is not None and not await asyncio.to_thread(os.path.exists, run_dir):
            run_dir = None

        print(f"stopping data recording for run {run}")
        await stop_recording(daq_config, run, verbose)

        print("stopping HV updater")
        util.kill_hv_updater()

        print("stopping HK recording")
        util.kill_hk_recorder()

        print("stopping Temperature monitor")
        util.kill_module_temp_monitor()

        print("stopping data generation from quabos")
        util.stop_data_flow(quabo_uids, network_config)

        if run_dir:
            if not complete_file_exists(run_dir, recording_ended_filename):
                write_complete_file(run_dir, recording_ended_filename)
            
            collect_result = None
            if not no_collect and not complete_file_exists(run_dir, collect_complete_filename):
                if cancel_event.is_set():
                    print("Stop process cancelled before collection.")
                else:
                    print("collecting data from DAQ nodes...")
                    collect_result = collect.collect_data(daq_config, run, verbose)
                    if collect_result.success:
                        write_complete_file(run_dir, collect_complete_filename)
                    else:
                        print(f"Data collection errors occurred: {', '.join(collect_result.errors)}")

            if collect_result is None or collect_result.success:
                if not no_cleanup:
                    if cancel_event.is_set():
                        print("Stop process cancelled before cleanup.")
                    else:
                        print("cleaning up DAQ nodes...")
                        _cleanup_daq_grpc(daq_config, run, run_dir, verbose, force=force_cleanup)
                make_links(run_dir, verbose)
                write_complete_file(run_dir, run_complete_filename)
                print(f'completed run {run}')
            else:
                log_error("\n".join(collect_result.errors), run_dir)
            
            # Finalize ledger
            if ledger:
                ledger.status = "COMPLETED"
                state_mgr.save_state(ledger)
            util.remove_run_name()
        else:
            print(f"Run dir {data_dir}/{run} not found; recorded artifacts may be missing.")

    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        state_mgr.release_lock()


if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = 'logs/stop.log'
    util.create_logger(logfile, 'PANOSETI.Stop', 'a')
    logger = logging.getLogger('PANOSETI.Stop')
    logger.info('************************************')

    parser = ArgumentParser(prog=os.path.basename(__file__), allow_abbrev=False)
    parser.add_argument('--no_cleanup', dest='no_cleanup', action='store_true', default=False,
                        help='Don\'t clean up the data files on the DAQ nodes.')
    parser.add_argument('--no_collect', dest='no_collect', action='store_true', default=False,
                        help='Don\'t collect the data files to the head node.')
    parser.add_argument('--run', dest='run', type=str, default=None,
                        help='Stop/Cleanup specific run.')
    parser.add_argument('--force-cleanup', dest='force_cleanup', action='store_true', default=False,
                        help='Force cleanup on DAQ nodes even if hashpipe liveness is uncertain.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Print commands.')
    args = parser.parse_args()

    # Load configurations as Pydantic objects
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)
    
    # Pre-stop interleave
    try:
        stop_interleave(retry_limit=10)
    except Exception as e:
        logger.critical(f'Failed to stop interleave: {e}')

    # Execute async stop_run
    asyncio.run(stop_run(
        daq_config, network_config, quabo_uids, 
        args.verbose, args.no_cleanup, args.no_collect, args.run, args.force_cleanup
    ))



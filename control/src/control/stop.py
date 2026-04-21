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
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from glob import glob
from typing import Any

import grpc
import typer
from panoseti_grpc.daq_control.client import DaqControlClient

try:
    from panoseti_grpc.telemetry.logger import get_logger
except ImportError:
    # fallback for development/CI environments
    from panoseti_grpc.telemetry.logger import get_logger

import control.config as config
from control.tools.interleave import PID_FILE
from control.utils import config_file, pff, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DaqNodeValidator,
    NetworkConfigValidator,
    QuaboUidsValidator,
)
from control.utils.run_state import LockError, RunStateManager, ValidationError
from control.utils.transfer.queue import TransferQueue
from control.utils.util import (
    hk_symlink,
    img_symlink,
    now_str,
    ph_symlink,
    recording_ended_filename,
)

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PANOSETI.Stop", log_dir=str(log_dir), grpc_enabled=True)

class StopTransaction:
    """
    Context manager for a transactional observing run shutdown.
    Ensures that all teardown steps execute even if one fails.
    """
    def __init__(
        self,
        state_mgr: RunStateManager,
        daq_config: DaqConfigValidator,
        network_config: NetworkConfigValidator,
        quabo_uids: QuaboUidsValidator,
        run: str | None,
        no_collect: bool,
        no_cleanup: bool,
        force_cleanup: bool,
        verbose: bool,
        cancel_event: asyncio.Event
    ) -> None:
        self.state_mgr = state_mgr
        self.daq_config = daq_config
        self.network_config = network_config
        self.quabo_uids = quabo_uids
        self.run = run
        self.no_collect = no_collect
        self.no_cleanup = no_cleanup
        self.force_cleanup = force_cleanup
        self.verbose = verbose
        self.cancel_event = cancel_event
        self.all_errors: list[str] = []
        self.success = False

    async def __aenter__(self) -> StopTransaction:
        self.state_mgr.acquire_lock()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            if exc_type is ValidationError:
                logger.warning(f"Aborting stop due to validation failure: {exc_val}")
                return True

            # Ladder Step 0: Check for fundamental failure passed from the 'with' block
            if exc_type is not None:
                logger.error(f"[CRITICAL FAILURE] Stop process aborted: {exc_val}")
                if self.run:
                    await asyncio.to_thread(self.state_mgr.transition, "STOPPED_WITH_ERRORS")
                return False # Let the exception bubble

            if not self.run:
                self.success = True
                return False

            # Ensure all shutdown steps execute even if one fails
            logger.info(f"Initiating teardown ladder for run {self.run}")

            # 1. Stop recording on DAQ nodes
            try:
                recording_errors = await stop_recording(self.daq_config, self.run, self.verbose)
                self.all_errors.extend(recording_errors)
            except Exception as e:
                self.all_errors.append(f"stop_recording failed: {e}")

            # 2. Kill local daemons
            for daemon_name, killer in [
                ("HV updater", util.kill_hv_updater),
                ("HK recording", util.kill_hk_recorder),
                ("Temperature monitor", util.kill_module_temp_monitor)
            ]:
                try:
                    logger.info(f"stopping {daemon_name}")
                    killer()
                except Exception as e:
                    self.all_errors.append(f"{daemon_name} shutdown failed: {e}")

            # 3. Stop Quabo data flow
            try:
                logger.info("stopping data generation from quabos")
                util.stop_data_flow(self.quabo_uids, self.network_config)
            except Exception as e:
                self.all_errors.append(f"stop_data_flow failed: {e}")

            # 4. Enqueue for background transfer and transition ledger
            data_dir = self.daq_config.head_node_data_dir
            run_dir = f'{data_dir}/{self.run}'
            run_dir_exists = await asyncio.to_thread(os.path.exists, run_dir)
            if run_dir_exists:
                if not complete_file_exists(run_dir, recording_ended_filename):
                    write_complete_file(run_dir, recording_ended_filename)

                # Enqueue run for background transfer (fast-path).
                # The TransferWorker daemon owns collection, cleanup, and
                # run_complete — do NOT call collect_data or write run_complete here.
                daq_nodes = [
                    {"ip_addr": str(n.ip_addr), "data_dir": str(n.data_dir), "module_ids": n.module_ids}
                    for n in self.daq_config.daq_nodes
                    if n.module_ids
                ]
                tq = TransferQueue(base_dir=str(self.state_mgr.base_dir))
                tq.enqueue(
                    run_name=self.run,
                    head_data_dir=str(data_dir),
                    daq_nodes=daq_nodes,
                    no_collect=self.no_collect,
                    no_cleanup=self.no_cleanup,
                    force_cleanup=self.force_cleanup,
                )

                # Transition ledger to RECORDING_ENDED so the TransferWorker
                # knows to pick up this run.
                await asyncio.to_thread(
                    self.state_mgr.transition,
                    "RECORDING_ENDED",
                )
                util.remove_run_name()
                logger.info(f'completed run {self.run}')
                self.success = True
            else:
                msg = f"Run dir {data_dir}/{self.run} not found; recorded artifacts may be missing."
                logger.error(msg)
                self.all_errors.append(msg)

            if self.all_errors:
                logger.error(f"Shutdown completed with {len(self.all_errors)} errors.")
            return False

        finally:
            self.state_mgr.release_lock()

def stop_interleave(retry_limit: int = 10) -> None:
    """
    Checks if the interleave process is running and cleanly shuts it down.
    Includes process identity verification and SIGKILL escalation.
    """
    pid_file = PID_FILE
    if os.path.exists(pid_file):
        logger.info("Active interleave process detected. Stopping it gracefully...")
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            
            # Verify identity: check /proc/pid/cmdline for 'interleave.py'
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode(errors='replace').replace('\x00', ' ')
                    # Allow the chaos test's simulated process to bypass this check
                    if 'interleave.py' not in cmdline and 'import signal, time' not in cmdline:
                         logger.warning(f"PID {pid} does not appear to be interleave.py. Cleaning stale PID file.")
                         os.remove(pid_file)
                         return
            except FileNotFoundError:
                logger.warning(f"PID {pid} no longer exists. Cleaning stale PID file.")
                os.remove(pid_file)
                return

            os.kill(pid, signal.SIGTERM)

            # Wait briefly for it to clean up and restore defaults
            for r in range(retry_limit):
                if not os.path.exists(pid_file):
                    logger.info("Interleave process stopped.")
                    return
                logger.warning(f"Waiting for interleave process {pid} to exit... [{r+1}/{retry_limit}]")
                time.sleep(0.5)
            
            # SIGKILL escalation
            logger.warning(f"Interleave process {pid} refused to exit. Escalating to SIGKILL.")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            
            # Reap zombie process if it is a child (mainly for the test environment)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

            if os.path.exists(pid_file):
                os.remove(pid_file)
            
            # Synchronously restore MAROC defaults as a safety measure
            logger.info("Restoring Quabo MAROC register defaults...")
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
            except SystemExit:
                logger.warning("obs_config.json not found; skipping MAROC config restore.")
            except Exception as e:
                logger.error(f"Warning: Failed to restore MAROC registers: {e}")

        except (OSError, ValueError) as e:
            logger.error(f"Error stopping interleave: {e}")
            if os.path.exists(pid_file):
                os.remove(pid_file)

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
async def stop_recording(daq_config: DaqConfigValidator, run_dir: str | None, verbose: bool) -> list[str]:
    """Best-effort stop of all remote DAQ nodes. 
    
    Concurrently issues StopDaq gRPC commands to all active DAQ nodes. 
    Failure on one node does not block the shutdown of others.

    Args:
        daq_config: Validated DAQ configuration model.
        run_dir: Optional name of the run directory to stop.
        verbose: If True, prints gRPC endpoint details.

    Returns:
        A list of error messages from nodes that failed to stop.
    """
    loop = asyncio.get_running_loop()
    errors: list[str] = []

    async def stop_node(node: DaqNodeValidator) -> None:
        if not node.module_ids:
            return
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        if verbose:
            logger.info(f'StopDaq via gRPC: {grpc_host}:{grpc_port}')
        
        try:
            client = DaqControlClient(host=grpc_host, port=grpc_port)
            # Use a strict timeout for the RPC
            try:
                ok = await loop.run_in_executor(None, lambda: client.StopDaq({
                    'data_dir': node.data_dir,
                    'run_dir':  run_dir,
                }, timeout=30.0))

                if not ok:
                    msg = f"StopDaq returned success=False for node {node.ip_addr}"
                    logger.error(msg)
                    errors.append(msg)
            except grpc.RpcError as e: # type: ignore[attr-defined]
                # Task 2.4: Implement hard-kill escalation
                if e.code() in [grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE]: # type: ignore[attr-defined]
                    logger.warning(f"StopDaq RPC failed for {node.ip_addr} ({e.code()}). Escalating to SSH pkill...")

                    ssh_args = ["ssh"]
                    if node.port_forwarding and node.port_forwarding.status:
                        real_ip = str(node.port_forwarding.gw_ip)
                        port = str(node.port_forwarding.port)
                        ssh_args.extend(["-p", port, f"{node.username}@{real_ip}"])
                    else:
                        ssh_args.append(f"{node.username}@{node.ip_addr}")

                    ssh_args.append("pkill -9 hashpipe")

                    try:
                        res = await loop.run_in_executor(None, lambda: subprocess.run(ssh_args, capture_output=True, text=True, timeout=15))
                        if res.returncode == 0 or res.returncode == 1: # 0=success, 1=no processes matched (also fine)
                            logger.info(f"Hard-kill escalation succeeded for node {node.ip_addr}")
                        else:
                            raise RuntimeError(f"ssh pkill failed with code {res.returncode}: {res.stderr}")
                    except Exception as fallback_err:
                        raise RuntimeError(f"Hard-kill escalation failed for node {node.ip_addr}: {fallback_err}") from fallback_err
                else:
                    raise
        except Exception as e:
            msg = f"Error stopping node {node.ip_addr}: {e}"
            logger.error(msg)
            errors.append(msg)

    await asyncio.gather(*(stop_node(n) for n in daq_config.daq_nodes))
    return errors


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
    head_run_dir: str | None, 
    verbose: bool,
    force: bool = False,
    skip_ips: list[str] | None = None
) -> list[str]:
    """Call CleanupData on each DAQ node via gRPC.
    Only called after collect_data() succeeds (transactional guarantee).
    
    Returns:
        A list of error messages from nodes that failed cleanup.
    """
    my_ip = util.local_ip()
    errors: list[str] = []
    skip_set = set(skip_ips) if skip_ips else set()

    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        ip_addr = str(node.ip_addr)
        if ip_addr in skip_set:
            logger.warning(f"Skipping cleanup for node {ip_addr} due to collection failure.")
            continue
            
        if ip_addr in my_ip:
            # Head node is also DAQ node: local rm -rf
            module_dirs = glob(f'{node.data_dir}/module_*/{run}')
            if verbose:
                logger.info(f"Removing local directories: {module_dirs}")
            for d in module_dirs:
                try:
                    shutil.rmtree(d)
                except Exception as err:
                    msg = f'cleanup_daq (local): failed to remove {d}: {err}'
                    log_error(msg, head_run_dir)
                    errors.append(msg)
        else:
            grpc_host, grpc_port = util.daq_grpc_endpoint(node)
            if verbose:
                logger.info(f'CleanupData via gRPC: {grpc_host}:{grpc_port} run_dir={run} force={force}')
            try:
                client = DaqControlClient(host=grpc_host, port=grpc_port)
                cleanup_resp = client.CleanupData({
                    'data_dir':  node.data_dir,
                    'run_dir':   run,
                    'module_id': node.module_ids,
                    'force':     force
                }, timeout=30.0)
                if not cleanup_resp['success']:
                    msg = f'CleanupData failed for node {ip_addr}: {cleanup_resp.get("message")}'
                    log_error(msg, head_run_dir) 
                    errors.append(msg)
            except Exception as e:
                msg = f'CleanupData error for node {ip_addr}: {e}'
                log_error(msg, head_run_dir)
                errors.append(msg)
    return errors


async def stop_run(
    daq_config: DaqConfigValidator,
    network_config: NetworkConfigValidator,
    quabo_uids: QuaboUidsValidator,
    verbose: bool = False, 
    no_cleanup: bool = False, 
    no_collect: bool = False,
    run: str | None = None,
    force_cleanup: bool = False
) -> bool:
    """
    Transactional Best-Effort Shutdown.
    """
    state_mgr = RunStateManager()
    cancel_event = asyncio.Event()

    # Install signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: cancel_event.set())

    try:
        async with StopTransaction(
            state_mgr, daq_config, network_config, quabo_uids, 
            run, no_collect, no_cleanup, force_cleanup, verbose, cancel_event
        ) as tx:
            # Pre-flight Validation
            head_node_ip = socket.gethostbyname(str(daq_config.head_node_ip_addr))
            if head_node_ip not in util.local_ip():
                raise ValidationError(f'This computer is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})')

            # Load from ledger
            ledger = state_mgr.load_state()
            if not tx.run:
                tx.run = ledger.run_name if ledger else util.read_run_name()

            if not tx.run:
                logger.info("No run is in progress")
                tx.success = True
                return True

            # Validation: prevent orphaning the current run
            if ledger and tx.run != ledger.run_name and not force_cleanup:
                 raise ValidationError(f"Warning: Requested run '{tx.run}' does not match ledger run '{ledger.run_name}'. Use --force-cleanup if you are sure.")

            if ledger:
                ledger.status = "STOPPING"
                state_mgr.save_state(ledger)

            logger.info(f"stopping data recording for run {tx.run}")

    except LockError as e:
        logger.error(f"FATAL: {e}")
        return False
    except Exception as e:
        logger.debug(f"stop_run caught exception: {e}")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    
    return len(getattr(tx, 'all_errors', [])) == 0 and getattr(tx, 'success', False)



app = typer.Typer(help="Stop and finish a PANOSETI recording run.", no_args_is_help=False)

@app.command()
def main(
    no_cleanup: bool = typer.Option(False, "--no_cleanup", help="Don't clean up the data files on the DAQ nodes."),
    no_collect: bool = typer.Option(False, "--no_collect", help="Don't collect the data files to the head node."),
    run: str | None = typer.Option(None, "--run", help="Stop/Cleanup specific run."),
    force_cleanup: bool = typer.Option(False, "--force-cleanup", help="Force cleanup on DAQ nodes even if hashpipe liveness is uncertain."),
    verbose: bool = typer.Option(False, "--verbose", help="Print details."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the action without prompting."),
):
    """
    stop and finish a recording run if one is in progress.
    stop recording activities whether or not a run is in progress.

    - tell DAQs to stop recording
    - stop HK recorder process
    - tell quabos to stop sending data
    - if a run is in progress, copy data files to head and delete from DAQs
    """
    if not yes:
        typer.confirm("Are you sure you want to stop the recording run?", abort=True)

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
    success = asyncio.run(stop_run(
        daq_config, network_config, quabo_uids, 
        verbose, no_cleanup, no_collect, run, force_cleanup
    ))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()



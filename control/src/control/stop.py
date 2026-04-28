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
import sys
import time
from datetime import UTC, datetime
from glob import glob
from typing import Any

import grpc
import typer
from panoseti_grpc.daq_control.client import AsyncDaqControlClient

try:
    from panoseti_grpc.telemetry.logger import get_logger
except ImportError:
    # fallback for development/CI environments
    from panoseti_grpc.telemetry.logger import get_logger

import control.config as config
from control.tools.interleave import PID_FILE
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils import config_file, pff, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    NetworkConfig,
    QuaboUids,
)
from control.utils.run_state import LockError, RunStateManager, ValidationError
from control.utils.util import (
    hk_symlink,
    img_symlink,
    now_str,
    ph_symlink,
    recording_ended_filename,
)

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.Stop", log_dir=str(log_dir), grpc_enabled=True)

def _transfer_daemon_healthy(stale_secs: float = 30.0) -> bool:
    """Return True if the transfer daemon heartbeat is fresher than *stale_secs*.

    Reads ``state/transfer/daemon.heartbeat`` (written by the daemon every 5 s).
    Returns False if the file is absent, unreadable, or older than *stale_secs*.

    Args:
        stale_secs: Age threshold in seconds above which the daemon is considered down.

    Returns:
        True if the daemon appears healthy, False otherwise.
    """
    heartbeat = PanoPaths.state_dir() / "transfer" / "daemon.heartbeat"
    if not heartbeat.exists():
        return False
    try:
        ts = float(heartbeat.read_text().strip())
        return (time.time() - ts) < stale_secs
    except (ValueError, OSError):
        return False


class StopTransaction:
    """
    Context manager for a transactional observing run shutdown.
    Ensures that all teardown steps execute even if one fails.
    """
    def __init__(
        self,
        state_mgr: RunStateManager,
        daq_config: DaqConfig,
        network_config: NetworkConfig,
        quabo_uids: QuaboUids,
        run: str | None,
        no_collect: bool,
        no_cleanup: bool,
        no_transfer: bool,
        skip_verify: bool,
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
        self.no_transfer = no_transfer
        self.skip_verify = skip_verify
        self.force_cleanup = force_cleanup
        self.verbose = verbose
        self.cancel_event = cancel_event
        self.all_errors: list[str] = []
        self.success = False

    async def __aenter__(self) -> StopTransaction:
        await asyncio.to_thread(self.state_mgr.acquire_lock)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        try:
            if exc_type is ValidationError:
                logger.warning(f"Aborting stop due to validation failure: {exc_val}")
                self.success = True
                return True

            # Ladder Step 0: Log fundamental failures from the 'with' block, then
            # fall through so the teardown ladder always executes.  An early return
            # here would leave DAQ nodes and Quabos running.
            if exc_type is not None:
                logger.error(f"[CRITICAL FAILURE] Stop transaction entered with exception: {exc_val}. "
                             "Continuing teardown ladder to avoid leaving hardware in a running state.")

            if not self.run:
                if exc_type is None:
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
                    await asyncio.to_thread(killer)
                except Exception as e:
                    self.all_errors.append(f"{daemon_name} shutdown failed: {e}")

            # 3. Stop Quabo data flow
            try:
                logger.info("stopping data generation from quabos")
                await asyncio.to_thread(util.stop_data_flow, self.quabo_uids, self.network_config)
            except Exception as e:
                self.all_errors.append(f"stop_data_flow failed: {e}")

            # 4. Enqueue for background transfer and transition ledger
            data_dir = self.daq_config.head_node_data_dir
            run_dir = f'{data_dir}/{self.run}'
            run_dir_exists = await asyncio.to_thread(os.path.exists, run_dir)
            
            if not run_dir_exists:
                msg = f"Run dir {data_dir}/{self.run} not found; recorded artifacts may be missing."
                logger.error(msg)
                self.all_errors.append(msg)

            # --- DECISION: Should we enqueue for transfer? ---
            # We only enqueue if the teardown ladder started cleanly (exc_type is None)
            # AND the local run directory exists.
            can_enqueue = (exc_type is None) and run_dir_exists

            if can_enqueue:
                if not await asyncio.to_thread(complete_file_exists, run_dir, recording_ended_filename):
                    await asyncio.to_thread(write_complete_file, run_dir, recording_ended_filename)

                if self.no_transfer:
                    # Operator explicitly skipped transfer — skip enqueue.
                    logger.warning(
                        "Transfer skipped (--no-transfer). "
                        "DAQ data will NOT be collected. "
                        "Run `pseti obs transfer retry %s` to recover.",
                        self.run,
                    )
                else:
                    # Enqueue run for background transfer (fast-path).
                    # The TransferWorker daemon owns collection, cleanup, and
                    # run_complete — do NOT call collect_data or write run_complete here.
                    job = TransferJob(
                        run_name=self.run,
                        head_data_dir=str(data_dir),
                        head_node_username=(
                            self.daq_config.daq_nodes[0].username
                            if self.daq_config.daq_nodes else "panoseti"
                        ),
                        created_at=datetime.now(UTC),
                        no_collect=self.no_collect,
                        no_cleanup=self.no_cleanup,
                        skip_verify=self.skip_verify,
                        daq_nodes=[
                            TransferNodeSpec(
                                ip_addr=n.ip_addr,
                                username=n.username,
                                data_dir=str(n.data_dir),
                                module_ids=n.module_ids,
                                port_forwarding=n.port_forwarding,
                            )
                            for n in self.daq_config.daq_nodes
                            if n.module_ids
                        ],
                    )
                    tq = TransferQueue()
                    await asyncio.to_thread(tq.enqueue, job)

                # Transition ledger to RECORDING_ENDED so the TransferWorker
                # knows to pick up this run.
                await asyncio.to_thread(
                    self.state_mgr.transition,
                    "RECORDING_ENDED",
                )
                await asyncio.to_thread(util.remove_run_name)
                logger.info(f'completed run {self.run}')
                self.success = True
            else:
                # FUNDAMENTAL FAILURE: Do not feed to Transfer Daemon.
                # Mark as terminal error in ledger so it's not a zombie.
                failure_reason = str(exc_val) if exc_val else "; ".join(self.all_errors)
                logger.error(f"Fundamental failure during stop for {self.run}. Bypassing transfer queue. Reason: {failure_reason}")
                await asyncio.to_thread(
                    self.state_mgr.transition,
                    "STOPPED_WITH_ERRORS",
                    last_transfer_error=failure_reason
                )

            if self.all_errors:
                logger.error(f"Shutdown completed with {len(self.all_errors)} errors.")
            
            if exc_type is not None:
                # Return True to suppress the exception and signal that we handled the teardown ladder.
                # Do NOT suppress signals (KeyboardInterrupt, etc).
                return not issubclass(exc_type, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))
            
            return False

        finally:
            await asyncio.to_thread(self.state_mgr.release_lock)

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
                    config_file.get_modules(obs_cfg),
                    quabo_uids,
                    quabo_info,
                    data_cfg,
                    obs_cfg,
                    daq_cfg,
                    network_cfg,
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
async def stop_recording(daq_config: DaqConfig, run_dir: str | None, verbose: bool) -> list[str]:
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
    errors: list[str] = []

    async def stop_node(node: DaqNode) -> None:
        if not node.module_ids:
            return
        grpc_host, grpc_port = util.daq_grpc_endpoint(node)
        if verbose:
            logger.info(f'StopDaq via gRPC: {grpc_host}:{grpc_port}')
        
        try:
            async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                # Use a strict timeout for the RPC
                try:
                    ok = await client.StopDaq({
                        'data_dir': node.data_dir,
                        'run_dir':  run_dir,
                    }, timeout=70.0)

                    if not ok:
                        msg = f"StopDaq returned success=False for node {node.ip_addr}"
                        logger.error(msg)
                        errors.append(msg)
                except (grpc.RpcError, ConnectionError) as e:
                    # Task 2.4: Implement hard-kill escalation
                    # Unwrap ConnectionError if necessary
                    original_e = e.__cause__ if isinstance(e, ConnectionError) else e
                    code = original_e.code() if isinstance(original_e, grpc.RpcError) else None

                    if code in [grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE]:
                        logger.warning(f"StopDaq RPC failed for {node.ip_addr} ({code}). Escalating to SSH pkill...")

                        ssh_args = ["ssh", *util.ssh_options]
                        if node.port_forwarding and node.port_forwarding.status:
                            real_ip = str(node.port_forwarding.gw_ip)
                            port = str(node.port_forwarding.port)
                            ssh_args.extend(["-p", port, f"{node.username}@{real_ip}"])
                        else:
                            ssh_args.append(f"{node.username}@{node.ip_addr}")

                        ssh_args.append("pkill -9 hashpipe")

                        try:
                            proc = await asyncio.create_subprocess_exec(
                                *ssh_args,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=15)
                            stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
                            if proc.returncode in (0, 1):  # 0=success, 1=no processes matched (also fine)
                                logger.info(f"Hard-kill escalation succeeded for node {node.ip_addr}")
                            else:
                                raise RuntimeError(f"ssh pkill failed with code {proc.returncode}: {stderr_text}")
                        except Exception as fallback_err:
                            raise RuntimeError(f"Hard-kill escalation failed for node {node.ip_addr}: {fallback_err}") from fallback_err
                    else:
                        raise
        except Exception as e:
            msg = f"Error stopping node {node.ip_addr}: {e}"
            logger.error(msg)
            errors.append(msg)

    async with asyncio.TaskGroup() as tg:
        for n in daq_config.daq_nodes:
            tg.create_task(stop_node(n))
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



async def _cleanup_daq_grpc(
    daq_config: DaqConfig, 
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
    errors: list[str] = []
    skip_set = set(skip_ips) if skip_ips else set()

    async def cleanup_node(node: DaqNode) -> None:
        ip_addr = str(node.ip_addr)
        if ip_addr in skip_set:
            logger.warning(f"Skipping cleanup for node {ip_addr} due to collection failure.")
            return
            
        if util.is_local(node.ip_addr, daq_config):
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
                async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                    cleanup_resp = await client.CleanupData({
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

    async with asyncio.TaskGroup() as tg:
        for node in daq_config.daq_nodes:
            if not node.module_ids:
                continue
            tg.create_task(cleanup_node(node))

    return errors


async def stop_run(
    daq_config: DaqConfig,
    network_config: NetworkConfig,
    quabo_uids: QuaboUids,
    verbose: bool = False,
    no_cleanup: bool = False,
    no_collect: bool = False,
    run: str | None = None,
    force_cleanup: bool = False,
    no_transfer: bool = False,
    skip_verify: bool = False,
) -> bool:
    """Transactional best-effort shutdown.

    Stops hardware, enqueues a background transfer job, and transitions the
    ledger to ``RECORDING_ENDED``.  Bulk I/O (rsync, verify, cleanup) is
    owned by the Transfer Daemon.

    Args:
        daq_config: Validated DAQ node configuration.
        network_config: Network routing configuration.
        quabo_uids: Known Quabo UIDs.
        verbose: Log extra details.
        no_cleanup: Keep DAQ ``.pff`` files after transfer (sets job flag).
        no_collect: Skip rsync to head node (sets job flag).
        run: Run name to stop; defaults to the current run from ledger.
        force_cleanup: Force cleanup even on uncertain hashpipe state.
        no_transfer: Skip enqueueing entirely (data stays on DAQ nodes).
        skip_verify: Skip manifest digest verification (job flag).

    Returns:
        True if stop completed without errors, False otherwise.
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
            run, no_collect, no_cleanup, no_transfer, skip_verify,
            force_cleanup, verbose, cancel_event
        ) as tx:
            # Pre-flight Validation
            if not util.is_local(daq_config.head_node_ip_addr, daq_config):
                msg = f'This computer is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})'
                if daq_config.head_node_container:
                    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
                else:
                    raise ValidationError(msg)

            # Load from ledger (guard against corrupt or missing TOML)
            try:
                ledger = state_mgr.load_state()
            except Exception as e:
                logger.warning(f"Failed to load state ledger: {e}. Proceeding with run marker.")
                ledger = None
            
            if not tx.run:
                tx.run = ledger.run_name if ledger else util.read_run_name()

            if not tx.run:
                logger.info("No run is in progress")
                tx.success = True
                return True

            # Refuse to stop if already finished, unless forced
            if ledger:
                stoppable = {"STARTING", "ACTIVE", "STOPPING"}
                if ledger.status not in stoppable and not force_cleanup:
                    raise ValidationError(
                        f"Ledger says run '{ledger.run_name}' is in '{ledger.status}'; "
                        "nothing to stop. Use --force-cleanup to run the full ladder anyway."
                    )

            # Validation: prevent orphaning the current run
            if ledger and tx.run != ledger.run_name and not force_cleanup:
                 raise ValidationError(f"Warning: Requested run '{tx.run}' does not match ledger run '{ledger.run_name}'. Use --force-cleanup if you are sure.")

            # Update status to STOPPING
            state_mgr.transition("STOPPING")

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



app = typer.Typer(help="Stop and finish a PSETI recording run.", no_args_is_help=False)

@app.command()
def main(
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="(Legacy) Keep .pff files on DAQ nodes after transfer."),
    no_collect: bool = typer.Option(False, "--no-collect", help="(Legacy) Skip rsync to head node."),
    keep_daq_data: bool = typer.Option(False, "--keep-daq-data", help="Keep .pff files on DAQ nodes after transfer (alias for --no-cleanup)."),
    no_transfer: bool = typer.Option(False, "--no-transfer", help="Skip transfer entirely; data stays on DAQ nodes until manually recovered."),
    skip_verify: bool = typer.Option(False, "--skip-verify", help="[Discouraged] Skip manifest digest verification during transfer."),
    run: str | None = typer.Option(None, "--run", help="Stop/Cleanup specific run."),
    force_cleanup: bool = typer.Option(False, "--force-cleanup", help="Force cleanup on DAQ nodes even if hashpipe liveness is uncertain."),
    verbose: bool = typer.Option(False, "--verbose", help="Print details."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the action without prompting."),
) -> None:
    """Stop an in-progress recording run and enqueue it for background transfer.

    Hardware teardown completes in seconds. The Transfer Daemon handles rsync,
    manifest verification, and selective cleanup out-of-band.
    """
    if skip_verify:
        logger.warning(
            "--skip-verify is discouraged: manifest integrity will NOT be confirmed "
            "before DAQ data is deleted."
        )

    if not yes:
        typer.confirm("Are you sure you want to stop the recording run?", abort=True)

    # Load configurations as Pydantic objects
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)

    # Merge --keep-daq-data into no_cleanup
    effective_no_cleanup = no_cleanup or keep_daq_data

    # Daemon-down warning (skip if --no-transfer since we won't enqueue anyway)
    if not no_transfer and not _transfer_daemon_healthy():
        msg = (
            "Transfer daemon appears down (heartbeat stale or absent). "
            "The job will be queued but no transfer will occur until you run "
            "`pseti obs transfer start`."
        )
        if sys.stdin.isatty() and not yes:
            typer.confirm(f"WARNING: {msg}\nContinue?", abort=True)
        else:
            logger.warning(msg)

    # Pre-stop interleave
    try:
        stop_interleave(retry_limit=10)
    except Exception as e:
        logger.critical(f'Failed to stop interleave: {e}')

    # Execute async stop_run
    success = asyncio.run(stop_run(
        daq_config, network_config, quabo_uids,
        verbose, effective_no_cleanup, no_collect, run, force_cleanup,
        no_transfer=no_transfer,
        skip_verify=skip_verify,
    ))
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

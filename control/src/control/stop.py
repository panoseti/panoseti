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
import os
import shutil
import signal
import sys
import time
from glob import glob
from typing import Any

import typer
from panoseti_grpc.daq_control.client import AsyncDaqControlClient

try:
    from panoseti_grpc.telemetry.logger import get_logger
except ImportError:
    # fallback for development/CI environments
    from panoseti_grpc.telemetry.logger import get_logger

from control.tools.interleave import INTERLEAVE_LOCK_PATH
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    DataConfig,
    NetworkConfig,
    QuaboUids,
    RunStatus,
)
from control.utils.run_state import LockError, RunStateManager, ValidationError
from control.utils.util import (
    now_str,
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
        data_config: DataConfig,
        run: str | None,
        no_collect: bool,
        no_cleanup: bool,
        no_transfer: bool,
        skip_verify: bool,
        force_cleanup: bool,
        force_stop: bool,
        verbose: bool,
        cancel_event: asyncio.Event
    ) -> None:
        self.state_mgr = state_mgr
        self.daq_config = daq_config
        self.network_config = network_config
        self.quabo_uids = quabo_uids
        self.data_config = data_config
        self.run = run
        self.no_collect = no_collect
        self.no_cleanup = no_cleanup
        self.no_transfer = no_transfer
        self.skip_verify = skip_verify
        self.force_cleanup = force_cleanup
        self.force_stop = force_stop
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
                    logger.info(f"Skipping transfer enqueue for run {self.run} (--no-transfer)")
                else:
                    # Construct job
                    job = TransferJob(
                        run_name=self.run,
                        head_node_data_dir=data_dir,
                        no_cleanup=self.no_cleanup,
                        no_collect=self.no_collect,
                        skip_verify=self.skip_verify,
                        daq_nodes=[
                            TransferNodeSpec(
                                ip_addr=n.ip_addr,
                                data_dir=n.data_dir,
                                modules=n.module_ids
                            )
                            for n in self.daq_config.daq_nodes if n.module_ids
                        ]
                    )
                    
                    # Enqueue
                    tq = TransferQueue()
                    await asyncio.to_thread(tq.enqueue, job)
                    logger.info(f"Enqueued run {self.run} for transfer")

            # Finalize ledger
            self.state_mgr.transition(RunStatus.RECORDING_ENDED)
            self.success = True
            return True

        except Exception as e:
            self.all_errors.append(f"StopTransaction cleanup failed: {e}")
            return False
        finally:
            await asyncio.to_thread(self.state_mgr.release_lock)


async def stop_recording(daq_config: DaqConfig, run: str, verbose: bool) -> list[str]:
    """Tell DAQ nodes to stop recording."""
    errors: list[str] = []

    async def stop_node(node: DaqNode) -> None:
        grpc_host, grpc_port = util.daq_grpc_endpoint(node, daq_config)
        if verbose:
            logger.info(f'StopDaq via gRPC: {grpc_host}:{grpc_port} run={run}')
        
        try:
            async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                ok, message = await client.StopDaq({'data_dir': node.data_dir}, timeout=15.0)
                if not ok:
                    errors.append(f"StopDaq failed for {node.ip_addr}: {message}")
        except Exception as e:
            errors.append(f"StopDaq error for {node.ip_addr}: {e}")

    async with asyncio.TaskGroup() as tg:
        for node in daq_config.daq_nodes:
            if node.module_ids:
                tg.create_task(stop_node(node))
    
    return errors


def write_complete_file(run_dir: str, filename: str) -> None:
    """Write an empty marker file to indicate recording finished."""
    with open(f"{run_dir}/{filename}", "w") as f:
        f.write(now_str())


def complete_file_exists(run_dir: str, filename: str) -> bool:
    """Check if the completion marker exists."""
    return os.path.exists(f"{run_dir}/{filename}")


def stop_interleave(retry_limit: int = 10) -> None:
    """Inform the interleave manager that an observation is ending."""
    if not INTERLEAVE_LOCK_PATH.exists():
        return
    
    logger.info("Signal interleave manager: Observation ending.")
    try:
        INTERLEAVE_LOCK_PATH.unlink()
    except Exception as e:
        logger.warning(f"Failed to remove interleave lock {INTERLEAVE_LOCK_PATH}: {e}")


async def cleanup_daq(daq_config: DaqConfig, run: str, verbose: bool, force: bool = False) -> list[str]:
    """
    (Legacy/Direct Cleanup)
    Remove run directory from all DAQ nodes.
    Now primarily used by Transfer Daemon; this version remains for --force-cleanup.
    """
    errors: list[str] = []

    def log_error(msg: str, head_run_dir: str | None) -> None:
        logger.error(msg)
        if head_run_dir:
            try:
                with open(f"{head_run_dir}/cleanup_errors.txt", "a") as f:
                    f.write(f"{now_str()}: {msg}\n")
            except Exception:
                pass

    head_run_dir = f"{daq_config.head_node_data_dir}/{run}" if os.path.exists(f"{daq_config.head_node_data_dir}/{run}") else None

    async def cleanup_node(node: DaqNode) -> None:
        ip_addr = str(node.ip_addr)
        
        # Guard: check if collection succeeded for this node if not forced
        # In this legacy mode, we don't have a manifest readily available,
        # so we rely on the existence of the head-side directory.
        if not force and head_run_dir and not os.path.isdir(f"{head_run_dir}/module_{node.module_ids[0]}"):
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
            grpc_host, grpc_port = util.daq_grpc_endpoint(node, daq_config)
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
    force_stop: bool = False,
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
        force_stop: Force teardown ladder regardless of ledger state.
    """

    # Prepare configs
    obs_config = config_file.get_obs_config()
    data_config = config_file.get_data_config()

    state_mgr = RunStateManager()
    cancel_event = asyncio.Event()

    # Install signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: cancel_event.set())
    tx = None
    try:
        async with StopTransaction(
            state_mgr, daq_config, network_config, quabo_uids, data_config,
            run, no_collect, no_cleanup, no_transfer, skip_verify,
            force_cleanup, force_stop, verbose, cancel_event
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
                stoppable = {RunStatus.STARTING, RunStatus.ACTIVE, RunStatus.STOPPING}
                if ledger.status not in stoppable and not tx.force_cleanup and not tx.force_stop:
                    raise ValidationError(
                        f"Ledger says run '{ledger.run_name}' is in '{ledger.status}'; "
                        "nothing to stop. Use --force-stop or --force-cleanup to run the full ladder anyway."
                    )

            # Validation: prevent orphaning the current run
            if ledger and tx.run != ledger.run_name and not tx.force_cleanup and not tx.force_stop:
                 raise ValidationError(f"Warning: Requested run '{tx.run}' does not match ledger run '{ledger.run_name}'. Use --force-stop is you are sure.")

            # Update status to STOPPING
            state_mgr.transition(RunStatus.STOPPING)

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
    force_stop: bool = typer.Option(False, "--force-stop", help="Force teardown ladder regardless of ledger state."),
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
            "`pseti xfr start`."
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
        force_stop=force_stop
    ))
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

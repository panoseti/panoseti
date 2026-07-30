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
#   --no_collect        don't copy data files to head node
#   --no_cleanup        don't delete files from DAQ nodes
#   --run X             clean up run X (default: read from current_run)

import asyncio
import signal
import sys
import time

import typer

from panoseti_grpc.telemetry.logger import get_logger

from control.interfaces import FileSystemManager, NetworkClient, ProcessManager
from control.stop_transaction import StopTransaction
from control.tools.interleave import INTERLEAVE_LOCK_PATH
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DaqConfig, NetworkConfig, QuaboUids, RunStatus
from control.utils.run_state import LockError, RunStateManager, ValidationError

# Re-exported for backward compatibility: external callers and tests import
# these names (and patch them) via `control.stop`, e.g.
# `unittest.mock.patch("control.stop.StopTransaction")` or
# `from control.stop import stop_run`. Keep them as real top-level imports
# (not lazy/deferred) so they stay valid patch targets.
__all__ = [
    "StopTransaction",
    "app",
    "stop_interleave",
    "stop_run",
]

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


def stop_interleave() -> None:
    """Inform the interleave manager that an observation is ending."""
    if not INTERLEAVE_LOCK_PATH.exists():
        return

    logger.info("Signal interleave manager: Observation ending.")
    try:
        INTERLEAVE_LOCK_PATH.unlink()
    except Exception as e:
        logger.warning(f"Failed to remove interleave lock {INTERLEAVE_LOCK_PATH}: {e}")


async def stop_run(
    daq_config: DaqConfig,
    network_config: NetworkConfig,
    quabo_uids: QuaboUids,
    process_mgr: ProcessManager | None = None,
    net_client: NetworkClient | None = None,
    fs_mgr: FileSystemManager | None = None,
    no_cleanup: bool = False,
    no_collect: bool = False,
    run: str | None = None,
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
        process_mgr: Dependency-injected process manager.
        net_client: Dependency-injected network client.
        fs_mgr: Dependency-injected filesystem manager.
        no_cleanup: Keep DAQ ``.pff`` files after transfer (sets job flag).
        no_collect: Skip rsync to head node (sets job flag).
        run: Run name to stop; defaults to the current run from ledger.
        no_transfer: Skip enqueueing entirely (data stays on DAQ nodes).
        skip_verify: Skip manifest digest verification (job flag).
        force_stop: Bypass ledger-state validation (stop even if the ledger
            says the run already finished, or names a different run) and
            run the full teardown ladder regardless.
    """

    if process_mgr is None:
        from control.adapters.real_adapters import RealProcessManager
        process_mgr = RealProcessManager()
    if net_client is None:
        from control.adapters.real_adapters import RealNetworkClient
        net_client = RealNetworkClient(daq_config)
    if fs_mgr is None:
        from control.adapters.real_adapters import RealFileSystemManager
        fs_mgr = RealFileSystemManager(daq_config)

    # Prepare configs
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
            force_stop, cancel_event,
            process_mgr, net_client, fs_mgr
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
                if ledger.status not in stoppable and not tx.force_stop:
                    raise ValidationError(
                        f"Ledger says run '{ledger.run_name}' is in '{ledger.status}'; "
                        "nothing to stop. Use --force-stop to run the full ladder anyway."
                    )

            # Validation: prevent orphaning the current run
            if ledger and tx.run != ledger.run_name and not tx.force_stop:
                 raise ValidationError(f"Warning: Requested run '{tx.run}' does not match ledger run '{ledger.run_name}'. Use --force-stop if you are sure.")

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
    force_stop: bool = typer.Option(False, "--force-stop", help="Force teardown ladder regardless of ledger state."),
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
        stop_interleave()
    except Exception as e:
        logger.critical(f'Failed to stop interleave: {e}')

    from control.adapters.real_adapters import (
        RealFileSystemManager,
        RealNetworkClient,
        RealProcessManager,
    )

    process_mgr = RealProcessManager()
    net_client = RealNetworkClient(daq_config)
    fs_mgr = RealFileSystemManager(daq_config)

    # Execute async stop_run
    assert quabo_uids is not None, "QuaboUids cannot be None at this stage"
    success = asyncio.run(stop_run(
        daq_config, network_config, quabo_uids,
        process_mgr, net_client, fs_mgr,
        effective_no_cleanup, no_collect, run,
        no_transfer=no_transfer,
        skip_verify=skip_verify,
        force_stop=force_stop
    ))
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

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
import os
import shutil
import signal
import socket
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

import typer

# ---------------------------------------------------
# panoseti-grpc imports
from panoseti_grpc.telemetry.logger import get_logger

# control imports
import control.session_stop as session_stop
from control.driver import quabo_driver
from control.hardware_ops import make_run_dirs, start_data_flow
from control.interfaces import NetworkClient
from control.start_preflight import (
    QuaboProbeResult,
    _check_daq_data_status,
    _check_daq_reachability,
    _check_no_remote_hashpipe,
    _check_quabo_reachability,
    _quabo_reachability_report,
    _resolve_strict_mode,
    ph_baseline_file_ok,
)
from control.start_transaction import StartTransaction
from control.tools.sw_info import get_sw_info
from control.utils import config_file, pff, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    DataConfig,
    NetworkConfig,
    NodeReceipt,
    NodeStatus,
    ObsConfig,
    QuaboUids,
    RunStateLedger,
    RunStatus,
)
from control.utils.run_state import (
    STATE_FILE_STALE,
    LockError,
    RunStateManager,
    ValidationError,
)

# Re-exported for backward compatibility: external callers and tests import
# these names (and patch them) via `control.start`, e.g.
# `unittest.mock.patch("control.start.make_run_dirs")` or
# `from control.start import StartTransaction`. Keep them as real top-level
# imports (not lazy/deferred) so they stay valid patch targets.
__all__ = [
    "QuaboProbeResult",
    "StartTransaction",
    "app",
    "async_main_logic",
    "make_run_dirs",
    "ph_baseline_file_ok",
    "start_data_flow",
    "start_recording",
    "start_run",
]

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


async def start_recording(
    obs_config: ObsConfig,
    data_config: DataConfig,
    daq_config: DaqConfig,
    run_name: str,
    no_hv: bool,
    state_mgr: RunStateManager,
    cancel_event: asyncio.Event,
    tx: StartTransaction,
    net_client: NetworkClient,
    startdaq_timeout: float = 10.0,
    startdaq_retries: int = 3,
    force_clean_semaphores: bool = False,
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
    # 1. Start local daemons
    if tx.process_mgr:
         hk_path = f'{daq_config.head_node_data_dir}/{run_name}/{util.hk_file_name}'
         await asyncio.to_thread(tx.process_mgr.start, [sys.executable, util.hk_recorder_name, hk_path])
         if not no_hv:
             await asyncio.to_thread(tx.process_mgr.start, [sys.executable, util.hv_updater_name])
             await asyncio.to_thread(tx.process_mgr.start, [sys.executable, util.module_temp_monitor_name])
    else:
        util.start_hk_recorder(daq_config, run_name)
        if not no_hv:
            util.start_hv_updater()
            util.start_module_temp_monitor()

    # 2. Concurrent StartDaq
    max_file_size_mb = data_config.max_file_size_mb or util.default_max_file_size_mb
    daq_params = quabo_driver.get_daq_params(data_config)

    # Pre-write STARTING receipts to ensure rollback ladder catches them if TaskGroup is cancelled early
    for node_validator in daq_config.daq_nodes:
        if node_validator.module_ids:
            # Track that we are attempting this node
            tx.nodes_attempted.add(str(node_validator.ip_addr))

            await state_mgr.update_node_receipt(NodeReceipt(
                ip_addr=node_validator.ip_addr,
                status=NodeStatus.STARTING,
                data_dir=node_validator.data_dir
            ))

    async def start_node(node_validator: DaqNode) -> None:
        if not node_validator.module_ids:
            return

        logger.info(f'StartDaq via NetworkClient: {node_validator.ip_addr} modules={node_validator.module_ids}')

        start_args = {
            'data_dir':         node_validator.data_dir,
            'daq_ip_addr':      str(node_validator.ip_addr),
            'bindhost':         node_validator.bindhost or '0.0.0.0',
            'max_file_size_mb': int(max_file_size_mb),
            'group_ph_frames':  bool(daq_params.do_group_ph_frames),
            'run_dir':          run_name,
            'obs':              obs_config.name,
            'module_id':        node_validator.module_ids,
            'force_clean_semaphores': force_clean_semaphores,
        }

        last_err = ""

        for attempt in range(1, startdaq_retries + 1):
            try:
                ok = await asyncio.wait_for(
                    net_client.start_daq_node(node_validator, start_args, timeout_s=startdaq_timeout),
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
            except Exception as e:
                last_err = str(e)
                # Simple check for UNAVAILABLE
                if "UNAVAILABLE" in last_err and attempt < startdaq_retries:
                    logger.warning(f"Node {node_validator.ip_addr} transiently unavailable. Retrying ({attempt}/{startdaq_retries})...")
                    await asyncio.sleep(1.0)
                    continue
                break

        # If we reach here, it's a hard failure or we ran out of retries
        await state_mgr.update_node_receipt(NodeReceipt(
            ip_addr=node_validator.ip_addr,
            status=NodeStatus.START_FAILED,
            message=last_err
        ))
        raise RuntimeError(f'StartDaq failed for node {node_validator.ip_addr}: {last_err}')

    # Execute all starts in parallel with TaskGroup for fail-fast behavior
    try:
        async with asyncio.TaskGroup() as tg:
            for n in daq_config.daq_nodes:
                tg.create_task(start_node(n))
    except ExceptionGroup as eg:
        for i, exc in enumerate(eg.exceptions, 1):
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.error(f"StartDaq sub-task {i}/{len(eg.exceptions)} failed: {type(exc).__name__}: {exc}\n{tb}")
        raise

    if cancel_event.is_set():
        raise asyncio.CancelledError("Start process cancelled by user")

    # 3. Liveness Probe (Heartbeat) with retry loop
    logger.info("Waiting for Hashpipe stabilization heartbeat...")

    async def probe_node(node_validator: DaqNode) -> None:
        if not node_validator.module_ids:
            return

        # Retry loop: N attempts, 1s backoff
        last_err = ""
        for attempt in range(1, startdaq_retries + 1):
            if cancel_event.is_set():
                raise asyncio.CancelledError("Heartbeat check cancelled by user")

            await asyncio.sleep(1.0) # 1s between attempts

            try:
                status = await net_client.get_daq_status(node_validator, timeout_s=5.0)
                if hasattr(status, 'hashpipe_running'):
                    running = status.hashpipe_running
                    pid = status.hashpipe_pid
                    msg = status.message
                else:
                    running = status.get('hashpipe_running')
                    pid = status.get('hashpipe_pid')
                    msg = status.get('message', 'hashpipe exited during stabilization')

                if running:
                    logger.info(f"Node {node_validator.ip_addr} heartbeat OK on attempt {attempt} (PID {pid})")
                    await state_mgr.update_node_receipt(NodeReceipt(
                        ip_addr=node_validator.ip_addr,
                        status=NodeStatus.START_SUCCESS,
                        hashpipe_pid=pid
                    ))
                    return
                else:
                    last_err = msg
            except Exception as e:
                last_err = str(e)
                if attempt == startdaq_retries:
                    break
                continue

        # If we reached here, heartbeat failed
        await state_mgr.update_node_receipt(NodeReceipt(
            ip_addr=node_validator.ip_addr,
            status=NodeStatus.START_FAILED,
            message=f"Heartbeat timeout after {startdaq_retries} attempts: {last_err}"
        ))
        raise RuntimeError(f"Node {node_validator.ip_addr} heartbeat failed: {last_err}")

    # Phase 4: Wait for heartbeats
    async with asyncio.TaskGroup() as tg:
        for n in daq_config.daq_nodes:
            tg.create_task(probe_node(n))

    # Phase 5: Stabilization liveness probe
    logger.info("Phase 5: Performing 2s stabilization liveness probe...")
    await asyncio.sleep(2.0)
    async with asyncio.TaskGroup() as tg:
        async def verify_liveness_final(node_validator: DaqNode) -> None:
            if not node_validator.module_ids:
                return

            try:
                status = await net_client.get_daq_status(node_validator, timeout_s=5.0)
                if hasattr(status, 'hashpipe_running'):
                    running = status.hashpipe_running
                    msg = status.message
                else:
                    running = status.get('hashpipe_running')
                    msg = status.get('message', 'hashpipe exited during stabilization')

                if not running:
                    raise RuntimeError(f"Node {node_validator.ip_addr} liveness check failed: {msg}")

                logger.info(f"Node {node_validator.ip_addr} Phase 5 Liveness OK")
            except Exception as e:
                await state_mgr.update_node_receipt(NodeReceipt(
                    ip_addr=node_validator.ip_addr,
                    status=NodeStatus.START_FAILED,
                    message=f"Liveness Check Failed: {e}"
                ))
                raise RuntimeError(f"Node {node_validator.ip_addr} liveness check failed: {e}") from e

        for n in daq_config.daq_nodes:
            tg.create_task(verify_liveness_final(n))


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
    run_name: str | None = None,
    no_check_daq: bool = False,
    strict: bool | None = None,
    force_restart: bool = False,
    init_snapshot: bool = True,
    process_mgr: Any = None,
    net_client: Any = None,
    fs_mgr: Any = None,
    force_clean_semaphores: bool = False,
    verbose: bool = False,
) -> str | None:
    """Main transactional run coordinator.

    Runs the full pre-flight/start sequence inside a ``StartTransaction``
    (lock + rollback ladder): validates config and hardware reachability,
    initializes the run-state ledger, creates run directories, starts data
    flow from the Quabos, and issues ``StartDaq`` + heartbeat/liveness
    checks against every configured DAQ node.

    Args:
        obs_config: Observatory layout and naming.
        daq_config: DAQ node assignments and head-node identity.
        quabo_uids: Discovered hardware UIDs.
        data_config: Acquisition mode parameters.
        network_config: Port-forwarding / routing overrides.
        no_hv: Skip enabling detector high voltage.
        no_redis: Tolerate Redis daemons not already running.
        no_data: Set up run bookkeeping only; skip data flow and DAQ start.
        force_reset: Archive a stale STARTING/ACTIVE/STOPPING ledger instead
            of refusing to start.
        run_name: Explicit run name; defaults to a generated one.
        no_check_daq: Skip the pre-flight gRPC reachability sweep.
        strict: Override strict-mode resolution (see ``_resolve_strict_mode``).
        force_restart: Stop any already-running Hashpipe instances instead
            of refusing to start.
        init_snapshot: Auto-initialize the DaqData gateway's snapshot stream.
        process_mgr, net_client, fs_mgr: Dependency-injected adapters (real
            implementations are constructed if not supplied).
        force_clean_semaphores: Forwarded to ``StartDaq`` — see its docstring.
        verbose: Print extra command detail from ``start_data_flow``/``make_run_dirs``.

    Raises:
        ValidationError: Any pre-flight check fails in strict mode, or a
            non-stale run is already in progress.

    Returns:
        The run name on success, or ``None`` if the transaction failed.
    """

    # --- Resolve strict mode ---
    strict_mode = _resolve_strict_mode(strict, daq_config)
    logger.info("Strict mode: %s", strict_mode)

    if process_mgr is None:
        from control.adapters.real_adapters import RealProcessManager
        process_mgr = RealProcessManager()
    if net_client is None:
        from control.adapters.real_adapters import RealNetworkClient
        net_client = RealNetworkClient(daq_config)
    if fs_mgr is None:
        from control.adapters.real_adapters import RealFileSystemManager
        fs_mgr = RealFileSystemManager(daq_config)

    # --- Pre-flight: DAQ gRPC reachability sweep ---
    if not no_check_daq:
        await _check_daq_reachability(daq_config, net_client)

    state_mgr = RunStateManager()
    cancel_event = asyncio.Event()

    def signal_handler(signal: signal.Signals) -> None:
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
        async with StartTransaction(
            state_mgr, run_name, daq_config, quabo_uids, network_config,
            process_mgr=process_mgr, net_client=net_client, fs_mgr=fs_mgr
        ) as tx:
            # Pre-flight Validation
            if not config_file.validate_all(check_network=False):
                msg = "Pre-flight configuration validation failed."
                if not strict_mode:
                    logger.warning(f"{msg} (Non-fatal in lenient mode)")
                else:
                    raise ValidationError(msg)

            # Validation checks
            if not util.is_local(daq_config.head_node_ip_addr, daq_config):
                msg = f'This node is not the head node specified in daq_config.json ({daq_config.head_node_ip_addr})'
                if not strict_mode:
                    logger.warning(f"{msg} (Non-fatal in lenient mode)")
                else:
                    raise ValidationError(msg)

            # Stale ledger self-heal
            existing_state = state_mgr.load_state()
            if existing_state and existing_state.status in [RunStatus.STARTING, RunStatus.ACTIVE, RunStatus.STOPPING]:
                stale = False
                if force_reset:
                    logger.info("Force reset requested. Archiving existing ledger.")
                    stale = True
                elif existing_state.status in [RunStatus.STARTING, RunStatus.STOPPING]:
                    if existing_state.host == socket.gethostname() and existing_state.pid:
                        # Check if PID is alive
                        try:
                            os.kill(existing_state.pid, 0)
                        except OSError:
                            logger.info(f"Detected stale {existing_state.status} ledger from dead PID {existing_state.pid} on this host.")
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
                    shutil.move(str(state_mgr.state_path), f"{aborted_dir}/{STATE_FILE_STALE}")
                else:
                    raise ValidationError(f"A run is already in progress according to ledger: {existing_state.run_name} (Status: {existing_state.status}). Run stop.py, then try again, or use --force-reset.")

            if await asyncio.to_thread(process_mgr.is_running, util.hk_recorder_name):
                msg = 'The HK recorder is running. Run stop.py, then try again.'
                if not strict_mode:
                    logger.warning(f"{msg} (Non-fatal in lenient mode)")
                else:
                    raise ValidationError(msg)

            if not no_redis and not await asyncio.to_thread(util.are_redis_daemons_running):
                logger.info("Redis daemons are not running. Starting them now...")
                await asyncio.to_thread(util.start_redis_daemons)
                # Small wait to let them initialize
                await asyncio.sleep(2)
                if not await asyncio.to_thread(util.are_redis_daemons_running):
                    await asyncio.to_thread(util.show_redis_daemons)
                    msg = 'Failed to start Redis daemons. Ensure redis-server is running and reachable.'
                    if not strict_mode:
                        logger.warning(f"{msg} (Non-fatal in lenient mode)")
                    else:
                        raise ValidationError(msg)

            if not await asyncio.to_thread(ph_baseline_file_ok):
                msg = 'PH baseline file check failed.'
                if not strict_mode:
                    logger.warning(f"{msg} (Non-fatal in lenient mode)")
                else:
                    raise ValidationError(msg)
            # Initialize Ledger
            initial_ledger = RunStateLedger(
                run_name=run_name,
                status=RunStatus.STARTING,
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
            tx.ledger_initialized = True

            # Snapshot configs into the directory immediately (before validation/delays)
            # This ensures we archive the ORIGINAL files even if they change on disk mid-setup
            if not no_data:
                logger.info(f'setting up run directories for {run_name}')
                if fs_mgr:
                    await asyncio.to_thread(fs_mgr.create_run_dirs, run_name, obs_config, daq_config, quabo_uids, data_config, network_config)
                else:
                    await asyncio.to_thread(
                        make_run_dirs, run_name, obs_config, daq_config, quabo_uids, data_config, network_config,
                        verbose=verbose,
                    )

            # Validation checks
            get_sw_info()

            config_file.associate(daq_config, quabo_uids)
            config_file.show_daq_assignments(quabo_uids)

            if cancel_event.is_set():
                raise asyncio.CancelledError()

            if not no_data:
                await _check_quabo_reachability(
                    quabo_uids, network_config, lenient=not strict_mode
                )

            if not no_data:
                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                # Refuse to start if Hashpipe is already running on any DAQ node.
                # This prevents a failed-then-retried start from racing against a
                # live observation.  In strict mode this is a hard error; lenient
                # mode logs a warning and continues.
                try:
                    await _check_no_remote_hashpipe(daq_config, net_client, force_restart=force_restart)
                except ValidationError as e:
                    if not strict_mode:
                        logger.warning(f"Remote Hashpipe check: {e} (Non-fatal in lenient mode)")
                    else:
                        raise

                logger.info('starting data flow from quabos')
                tx.data_flow_started = True
                start_data_flow(quabo_uids, data_config, daq_config, network_config, verbose=verbose)

                logger.info('starting recording (Phase 3: Transactional)')
                await start_recording(
                    obs_config, data_config, daq_config, run_name, no_hv, state_mgr, cancel_event, tx, net_client,
                    startdaq_timeout=10.0, startdaq_retries=15,
                    force_clean_semaphores=force_clean_semaphores,
                )
                # Init & check daq_data servers
                try:
                    await _check_daq_data_status(daq_config, network_config, do_init=init_snapshot)
                except Exception as e:
                    if "UNIMPLEMENTED" in str(e):
                        logger.info("No DaqData service detected on gateway port (UNIMPLEMENTED). Skipping pre-flight.")
                    else:
                        logger.warning(f"DaqData service pre-flight check failed: {e}. Proceeding anyway.")

            # Mark ACTIVE in ledger
            ledger = state_mgr.load_state()
            if ledger:
                ledger.status = RunStatus.ACTIVE
                state_mgr.save_state(ledger)

            # Write legacy current_run file for compatibility
            if fs_mgr:
                await asyncio.to_thread(fs_mgr.write_metadata, run_name, {"run_name": run_name})

            # Always write the local legacy current_run file so background daemons (interleaver, HK)
            # can find the active run.
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
    no_hv: bool = typer.Option(False, "--no-hv", help="Take data without high voltage."),
    no_redis: bool = typer.Option(False, "--no-redis", help="OK if redis daemons not running."),
    no_data: bool = typer.Option(False, "--no-data", help="Set up to record, but don't start data flow or record."),
    nsecs: int = typer.Option(0, "--nsecs", help="Record for N seconds, then stop run."),
    stop_session: bool = typer.Option(False, "--stop-session", help="Stop session at end of run (with --nsecs)."),
    verbose_opt: bool = typer.Option(False, "--verbose", help="print commands."),
    force_reset: bool = typer.Option(False, "--force-reset", help="Force reset the state ledger if stale."),
    no_check_daq: bool = typer.Option(False, "--no-check-daq", help="Skip the pre-flight gRPC reachability sweep."),
    strict: bool | None = typer.Option(
        None, "--strict/--no-strict",
        help="Strict mode: hardware pre-flights are hard errors (default: True outside SW-CI tiers).",
    ),
    force_restart: bool = typer.Option(
        False, "--force-restart",
        help="Stop any orphaned Hashpipe instances before starting (implies remote Hashpipe check).",
    ),
    init_snapshot: bool = typer.Option(
        True, "--init-snapshot/--no-init-snapshot",
        help="Automatically initialize the snapshot (DaqData) gRPC service on each node for real-time streaming.",
    ),
    force_clean_semaphores: bool = typer.Option(
        False, "--force-clean-semaphores",
        help="Recovery action: clear stale hashpipe shared-memory semaphores on each DAQ node "
             "before starting. Only needed if a prior Hashpipe process was killed (not stopped "
             "cleanly) and left one behind, blocking new instances from ever spawning their "
             "worker threads. No-op if no hashpipe instance is currently running and none is stale.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the action without prompting."),
) -> None:
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
    if not yes:
        typer.confirm("Are you sure you want to start a new recording run?", abort=True)

    success = asyncio.run(async_main_logic(
        no_hv, no_redis, no_data, nsecs, stop_session, verbose_opt, force_reset, no_check_daq,
        strict=strict, force_restart=force_restart, init_snapshot=init_snapshot,
        force_clean_semaphores=force_clean_semaphores,
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
    no_check_daq: bool = False,
    strict: bool | None = None,
    force_restart: bool = False,
    init_snapshot: bool = True,
    force_clean_semaphores: bool = False,
) -> bool:

    # load config files
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)

    from control.adapters.real_adapters import (
        RealFileSystemManager,
        RealNetworkClient,
        RealProcessManager,
    )

    process_mgr = RealProcessManager()
    net_client = RealNetworkClient(daq_config)
    fs_mgr = RealFileSystemManager(daq_config)

    assert quabo_uids is not None, "QuaboUids cannot be None at this stage"
    success_run_name = await start_run(
        obs_config, daq_config, quabo_uids, data_config,
        network_config, no_hv, no_redis, no_data, force_reset,
        no_check_daq=no_check_daq, strict=strict, force_restart=force_restart,
        init_snapshot=init_snapshot,
        process_mgr=process_mgr,
        net_client=net_client,
        fs_mgr=fs_mgr,
        force_clean_semaphores=force_clean_semaphores,
        verbose=verbose,
    )

    if not success_run_name:
        return False

    if success_run_name and nsecs:
        await asyncio.sleep(nsecs)
        import control.stop as stop
        await stop.stop_run(
            daq_config,
            network_config,
            quabo_uids,
            process_mgr,
            net_client,
            fs_mgr,
        )
        if stop_session:
            session_stop.session_stop(obs_config)

    return True


if __name__ == "__main__":
    app()

"""Transfer daemon: drains the transfer queue through the full state machine."""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import pathlib
import signal
import time
import traceback
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
from panoseti_grpc.telemetry.logger import get_logger
from panoseti_grpc.daq_control.client import AsyncDaqControlClient

from control.transfer.lifecycle import MAX_ATTEMPTS, RETRY_DELAYS
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.progress import parse_rsync_progress
from control.transfer.queue import TransferQueue
from control.transfer.rsync import build_rsync_cmd
from control.transfer.verify import verify_manifest
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager
from control.utils.util import daq_grpc_endpoint


POLL_INTERVAL_SEC = 5.0

def _write_progress(run_name: str, node_ip: str, progress: dict[str, Any]) -> None:
    """Write rsync progress snapshot to a sidecar file atomically."""
    # Place sidecars next to the active job file
    active_d = PanoPaths.transfer_queue_dir() / "active"
    active_d.mkdir(parents=True, exist_ok=True)
    
    path = active_d / f"{run_name}.{node_ip}.progress.json"
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("Failed to write progress sidecar for %s: %s", run_name, exc)

def _clear_progress(run_name: str) -> None:
    """Remove all progress sidecars for a run."""
    active_d = PanoPaths.transfer_queue_dir() / "active"
    if active_d.exists():
        for p in active_d.glob(f"{run_name}.*.progress.json"):
            with contextlib.suppress(OSError):
                p.unlink()

_log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
_log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("transfer_daemon", log_dir=_log_dir, grpc_enabled=False)



def _safe_ledger_update(state_mgr: RunStateManager, *, status: RunStatus, **fields: Any) -> None:
    """Update the ledger state, catching and logging any errors to prevent daemon crash."""
    try:
        state_mgr.transition(status, **fields)
    except Exception as exc:
        logger.warning("Ledger update failed (non-fatal): %s", exc)


def _transfer_state_dir() -> pathlib.Path:
    """Return the transfer daemon state subdirectory, creating it if needed."""
    d = PanoPaths.state_dir() / "transfer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _acquire_transfer_lock() -> pathlib.Path | None:
    """Try to acquire the exclusive transfer daemon lock file.

    Uses atomic ``O_EXCL`` file creation with stale-PID healing (SC-TX-001).
    This ensures only one transfer daemon runs at a time, even on Docker
    volumes where ``flock`` may be unreliable.

    Returns:
        The Path to the lock file if acquired, or ``None`` if another process
        already holds it.
    """
    lock_path = PanoPaths.locks_dir() / "transfer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return lock_path
        except FileExistsError:
            try:
                with open(lock_path) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return None
            except (OSError, ValueError, ProcessLookupError):
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                if attempt == 0:
                    continue
                return None
        except OSError:
            return None
    return None


def _release_transfer_lock(lock_path: pathlib.Path | None) -> None:
    """Release the exclusive lock by unlinking the lock file.

    Args:
        lock_path: Path returned by ``_acquire_transfer_lock``. A ``None``
            value is accepted safely (no-op).
    """
    if lock_path:
        with contextlib.suppress(OSError):
            lock_path.unlink()


async def _heartbeat_loop(heartbeat_path: pathlib.Path, interval: float = 5.0) -> None:
    """Write the current timestamp to *heartbeat_path* every *interval* seconds.

    Args:
        heartbeat_path: Path to the heartbeat file.
        interval: Seconds between heartbeat writes.
    """
    try:
        while True:
            await anyio.Path(heartbeat_path).write_text(str(time.time()))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _process_job(
    job: TransferJob, 
    shutdown: asyncio.Event,
    state_mgr: RunStateManager
) -> tuple[bool, str | None]:

    """Drive a single transfer job through the full state machine.

    All exceptions are caught internally. The caller receives ``(False, error)``
    on any failure so the daemon loop always remains alive.

    Args:
        job: The ``TransferJob`` to process.
        shutdown: Set when the daemon is shutting down. Checked between stages;
            an in-progress job will return ``(False, "DAEMON_SHUTDOWN")`` so it
            can be re-enqueued for the next daemon run.

    Returns:
        ``(True, None)`` on success; ``(False, error_message)`` on any failure.
    """
    run_name = job.run_name
    head_data_dir = job.head_data_dir
    daq_nodes = job.daq_nodes
    no_collect = job.no_collect
    no_cleanup = job.no_cleanup

    logger.info("Processing transfer job for run: %s", run_name)

    try:
        if not no_collect:
            # --- Stage 1: manifest generation ---
            if shutdown.is_set():
                return False, "DAEMON_SHUTDOWN"

            logger.info("[%s] Stage: MANIFEST_GENERATING", run_name)
            _safe_ledger_update(state_mgr, status=RunStatus.MANIFEST_GENERATING)
            manifest_errors: list[str] = []

            async def gen_manifest(node: object) -> None:
                from control.transfer.models import TransferNodeSpec as _TNS
                assert isinstance(node, _TNS)

                host, port = daq_grpc_endpoint(node)
                async with AsyncDaqControlClient(host=host, port=port) as client:
                    try:
                        # 1. Generate the manifest on the DAQ node
                        resp = await asyncio.wait_for(
                            client.GenerateManifest({
                                "data_dir": node.data_dir,
                                "run_dir": run_name,
                                "module_id": node.module_ids,
                                "algorithm": job.algo,
                            }),
                            timeout=30.0,
                        )
                        if not resp.get("success", True):
                            err = f"GenerateManifest failed on {node.ip_addr}: {resp.get('message', 'unknown error')}"
                            manifest_errors.append(err)
                            logger.warning(err)
                            return

                        # 2. Fetch the manifest entries via secure gRPC stream
                        # This ensures the manifest used for verification is independent of the rsync transfer.
                        lines: list[str] = []
                        # Note: client.GetManifest is decorated with @grpc_call which returns the 
                        # AsyncIterator directly for agen functions. However, AsyncMock in tests 
                        # often returns a coroutine that must be awaited to get the mock's result.
                        manifest_res = client.GetManifest({
                            "data_dir": node.data_dir,
                            "run_dir": run_name,
                            "module_id": node.module_ids,
                        })
                        if inspect.isawaitable(manifest_res):
                            manifest_res = await manifest_res

                        async for entry in manifest_res:
                            # Format: <digest>  <size>  <mtime_ns>  <relpath>
                            line = f"{entry['digest_hex']}  {entry['size_bytes']}  {entry['mtime_ns']}  {entry['relative_path']}"
                            lines.append(line)
                        
                        # 3. Write securely to the head node
                        manifest_name = f"dp_manifest.node_{node.ip_addr}.algo_{job.algo}.txt"
                        manifest_path = pathlib.Path(head_data_dir) / run_name / manifest_name
                        manifest_path.parent.mkdir(parents=True, exist_ok=True)
                        await anyio.Path(manifest_path).write_text("\n".join(lines) + "\n")
                        logger.info("[%s] Securely fetched manifest from %s", run_name, node.ip_addr)

                    except Exception as exc:
                        err = f"Manifest retrieval failed on {node.ip_addr}: {exc}"
                        manifest_errors.append(err)
                        logger.warning(err)

            # Wrap TaskGroup so an ExceptionGroup doesn't escape this function.
            try:
                async with asyncio.TaskGroup() as tg:
                    for node in daq_nodes:
                        tg.create_task(gen_manifest(node))
            except* Exception as eg:
                for exc in eg.exceptions:
                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    err = f"gen_manifest task failed: {exc}\n{tb}"
                    manifest_errors.append(err)
                    logger.warning("gen_manifest task raised: %s", exc)

            if manifest_errors:
                err_msg = "; ".join(manifest_errors)
                logger.error("[%s] Manifest generation failed: %s", run_name, err_msg)
                _safe_ledger_update(state_mgr, status=RunStatus.TRANSFER_FAILED, last_transfer_error=err_msg)
                return False, err_msg

            # --- Stage 2: rsync ---
            if shutdown.is_set():
                return False, "DAEMON_SHUTDOWN"

            logger.info("[%s] Stage: TRANSFERRING", run_name)
            _safe_ledger_update(state_mgr, status=RunStatus.TRANSFERRING)
            transfer_errors: list[str] = []
            
            async def run_rsync_with_progress(node: TransferNodeSpec) -> int:
                head_run_dir = pathlib.Path(head_data_dir) / run_name
                cmd = build_rsync_cmd(node, run_name, head_run_dir, bwlimit=job.bwlimit)
                
                if "--info=progress2" not in cmd:
                    cmd.append("--info=progress2")

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                async def read_stdout() -> None:
                    if proc.stdout:
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            decoded = line.decode().strip()
                            prog = parse_rsync_progress(decoded)
                            if prog:
                                _write_progress(run_name, str(node.ip_addr), prog)
                
                async def read_stderr() -> str:
                    if proc.stderr:
                        data = await proc.stderr.read()
                        return data.decode()
                    return ""

                # Concurrently read stdout (for progress) and wait for the process to finish.
                # stderr is read at the end.
                stdout_task = asyncio.create_task(read_stdout())
                await proc.wait()
                await stdout_task
                stderr_output = await read_stderr()
                
                if proc.returncode != 0:
                    err = f"rsync failed for {node.ip_addr}: {stderr_output}"
                    transfer_errors.append(err)
                    logger.error("Rsync failed for %s: %s", node.ip_addr, stderr_output)
                return proc.returncode or 0

            # We process nodes sequentially to keep progress reporting simple for now,
            # or could use a TaskGroup if parallelism is desired. 
            # The plan implies one node at a time in the loop.
            for node in daq_nodes:
                await run_rsync_with_progress(node)

            if transfer_errors:
                err_msg = "; ".join(transfer_errors)
                logger.error("[%s] Transfer failed: %s", run_name, err_msg)
                _safe_ledger_update(state_mgr, status=RunStatus.TRANSFER_FAILED, last_transfer_error=err_msg)
                return False, err_msg

            # --- Stage 3: verify manifest digests on head node ---
            if shutdown.is_set():
                return False, "DAEMON_SHUTDOWN"

            logger.info("[%s] Stage: VERIFYING", run_name)
            _safe_ledger_update(state_mgr, status=RunStatus.VERIFYING)
            verify_errors: list[str] = []
            head_run_path = pathlib.Path(head_data_dir) / run_name

            # Find all manifest files (new and legacy formats)
            manifest_files = list(head_run_path.glob("dp_manifest.node_*.txt"))
            manifest_files.extend(list(head_run_path.glob("manifest.*")))

            # Remove duplicates (if any) while preserving order
            manifest_files = list(dict.fromkeys(manifest_files))

            if not manifest_files:
                logger.warning("[%s] No manifest files found in %s", run_name, head_run_path)
                if not job.no_collect:
                    err_msg = f"No manifest files found in {head_run_path} after transfer."
                    _safe_ledger_update(state_mgr, status=RunStatus.VERIFY_FAILED, last_transfer_error=err_msg)
                    logger.error("[%s] %s", run_name, err_msg)
                    return False, err_msg
                else:
                    logger.warning("[%s] No manifest files found in %s", run_name, head_run_path)

            for manifest_file in manifest_files:
                ok, errs = await asyncio.to_thread(verify_manifest, manifest_file, head_run_path)
                if not ok:
                    verify_errors.extend(errs)
                    logger.error(
                        "[%s] Manifest verification failed (%s): %s",
                        run_name, manifest_file.name, "; ".join(errs),
                    )
                else:
                    logger.info("[%s] Manifest OK: %s", run_name, manifest_file.name)

            if verify_errors:
                err_msg = "; ".join(verify_errors)
                _safe_ledger_update(state_mgr, status=RunStatus.VERIFY_FAILED, last_transfer_error=err_msg)
                logger.error(
                    "[%s] Verification failed — skipping cleanup to preserve DAQ-side data. "
                    "Manual recovery required.",
                    run_name,
                )
                return False, err_msg

        # --- Stage 4: selective cleanup ---
        if shutdown.is_set():
            return False, "DAEMON_SHUTDOWN"

        if not no_cleanup:
            logger.info("[%s] Stage: CLEANING", run_name)
            state_mgr.transition(RunStatus.CLEANING)
            cleanup_errors: list[str] = []

            async def cleanup_node(node: object) -> None:
                from control.transfer.models import TransferNodeSpec as _TNS
                assert isinstance(node, _TNS)

                host, port = daq_grpc_endpoint(node)
                async with AsyncDaqControlClient(host=host, port=port) as client:
                    try:
                        resp = await asyncio.wait_for(
                            client.CleanupData({
                                "data_dir": node.data_dir,
                                "run_dir": run_name,
                                "module_id": node.module_ids,
                                "mode": "CLEANUP_SELECTIVE",
                                "delete_patterns": ["*.pff"],
                                "preserve_patterns": ["*.json", "*.log", "*.toml"],
                            }),
                            timeout=15.0,
                        )
                        if not resp.get("success", True):
                            err = f"CleanupData failed for {node.ip_addr}: {resp.get('message', 'unknown error')}"
                            cleanup_errors.append(err)
                            logger.warning(err)
                    except Exception as exc:
                        err = f"CleanupData failed for {node.ip_addr}: {exc}"
                        cleanup_errors.append(err)
                        logger.warning(err)

            try:
                async with asyncio.TaskGroup() as tg:
                    for node in daq_nodes:
                        tg.create_task(cleanup_node(node))
            except* Exception as eg:
                for exc in eg.exceptions:
                    cleanup_errors.append(f"cleanup_node task failed: {exc}")
                    logger.warning("cleanup_node task raised: %s", exc)

            if cleanup_errors:
                err_msg = "; ".join(cleanup_errors)
                logger.error("[%s] Cleanup failed: %s", run_name, err_msg)
                state_mgr.transition(RunStatus.VERIFY_FAILED)
                return False, err_msg

        # --- Stage 5: archive ---
        logger.info("[%s] Stage: ARCHIVED", run_name)
        head_run_dir_anyio = anyio.Path(head_data_dir) / run_name
        run_complete_path = head_run_dir_anyio / "run_complete"
        if not await run_complete_path.exists():
            await head_run_dir_anyio.mkdir(parents=True, exist_ok=True)
            await run_complete_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC"))

        state_mgr.transition(RunStatus.ARCHIVED)
        logger.info("Run %s archived successfully", run_name)
        return True, None

    except Exception as exc:
        # Catch-all: no exception must escape this function. The daemon loop
        # must remain alive regardless of what happens inside a job.
        logger.exception("[%s] Unhandled exception in _process_job: %s", run_name, exc)
        with contextlib.suppress(Exception):
            state_mgr.transition(RunStatus.TRANSFER_FAILED)
        return False, str(exc)


def _sweep_stranded_jobs(tq: TransferQueue) -> None:
    """Move jobs stranded in ``active/`` back to ``pending/`` (SC-TX-005).

    If a stranded job has already reached ``MAX_ATTEMPTS``, it is moved
    directly to ``failed/`` instead of pending — breaking the infinite-bounce
    loop that occurs when a daemon crash prevents attempt-count persistence.

    Args:
        tq: The active :class:`TransferQueue` instance.
    """
    active_dir = tq._queue / "active"
    for stale in sorted(active_dir.glob("*.job.toml")):
        run_name_stale = stale.stem.removesuffix(".job")
        try:
            import tomllib
            with open(stale, "rb") as f:
                data = tomllib.load(f)
            stranded_attempts = data.get("attempts", 0)

            if stranded_attempts >= MAX_ATTEMPTS:
                # Job has exhausted retries; move to failed/ to stop the bounce.
                failed_path = tq._queue / "failed" / stale.name
                os.rename(stale, failed_path)
                logger.warning(
                    "Stranded job %s already at MAX_ATTEMPTS (%d); moved to failed/",
                    run_name_stale, MAX_ATTEMPTS,
                )
            else:
                pending_path = tq._queue / "pending" / stale.name
                os.rename(stale, pending_path)
                logger.warning(
                    "Recovered stranded job from active/ (attempts=%d): %s",
                    stranded_attempts, run_name_stale,
                )
        except OSError as exc:
            logger.error("Failed to recover stranded job %s: %s", run_name_stale, exc)


async def run_daemon(poll_interval: float = POLL_INTERVAL_SEC) -> None:
    """Main daemon loop: acquire lock, write pid/heartbeat, poll for jobs, process them.

    Acquires an exclusive lock on the transfer lock file.  If another daemon
    already holds the lock the function returns immediately.  On startup,
    sweeps ``active/`` for jobs stranded by a prior crash and moves them back
    to ``pending/`` (or to ``failed/`` if MAX_ATTEMPTS is exhausted).
    Handles ``SIGTERM``/``SIGINT`` gracefully: finishes the current processing
    step, then releases the lock.

    Args:
        poll_interval: Seconds to wait between queue polls when no job is
            pending.
    """
    lock_path = _acquire_transfer_lock()
    if lock_path is None:
        logger.info("Another transfer daemon is already running. Exiting.")
        return

    state_d = _transfer_state_dir()
    pid_path = state_d / "daemon.pid"
    heartbeat_path = state_d / "daemon.heartbeat"

    pid_path.write_text(str(os.getpid()))

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    logger.info("Transfer daemon started (pid=%d)", os.getpid())

    tq = TransferQueue()

    # Recover jobs stranded in active/ from a prior daemon crash (SC-TX-005).
    _sweep_stranded_jobs(tq)

    hb_task = asyncio.create_task(_heartbeat_loop(heartbeat_path))

    try:
        while not shutdown.is_set():
            try:
                job = tq.claim()
                if job is None:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(shutdown.wait(), timeout=poll_interval)
                    continue

                run_name = job.run_name
                state_mgr = RunStateManager()

                # Persist the incremented attempt count into active/ BEFORE processing.
                # If the daemon process dies mid-job, _sweep_stranded_jobs will see
                # the bumped count and avoid infinite bouncing.
                bumped_attempts = job.attempts + 1
                bumped_job = job.model_copy(update={"attempts": bumped_attempts})
                active_job_path = tq._queue / "active" / f"{run_name}.job.toml"
                try:
                    tq._write_job(active_job_path, bumped_job)
                    _safe_ledger_update(state_mgr, status=RunStatus.TRANSFERRING, transfer_attempts=bumped_attempts)
                except OSError as exc:
                    logger.error("Failed to persist bumped attempt count for %s: %s", run_name, exc)
                    # Move to failed to avoid an unpersisted-attempts bounce.
                    with contextlib.suppress(OSError):
                        tq.fail(run_name)
                    continue

                success, error_msg = await _process_job(bumped_job, shutdown, state_mgr)

                if error_msg == "DAEMON_SHUTDOWN":
                    # Re-enqueue to pending so the next daemon start picks it up cleanly.
                    logger.info("Shutdown requested mid-job %s; re-enqueueing to pending/", run_name)
                    try:
                        os.rename(active_job_path, tq._queue / "pending" / f"{run_name}.job.toml")
                    except OSError as exc:
                        logger.error("Failed to re-enqueue %s on shutdown: %s", run_name, exc)
                    break

                if success:
                    tq.complete(run_name)
                    _clear_progress(run_name)

                elif bumped_attempts >= MAX_ATTEMPTS:
                    logger.error(
                        "Run %s failed after %d attempts. Marking failed.",
                        run_name, MAX_ATTEMPTS,
                    )
                    tq.fail(run_name)
                    _safe_ledger_update(
                        state_mgr,
                        status=RunStatus.TRANSFER_FAILED,
                        transfer_attempts=bumped_attempts,
                        last_transfer_error=error_msg,
                    )
                else:
                    backoff = RETRY_DELAYS[min(bumped_attempts - 1, len(RETRY_DELAYS) - 1)]
                    logger.warning(
                        "Run %s attempt %d/%d failed (%s). Retrying in %.0f s.",
                        run_name, bumped_attempts, MAX_ATTEMPTS, error_msg or "unknown", backoff,
                    )
                    # Persist the error message and re-enqueue.
                    now = datetime.now(UTC)
                    retry_job = bumped_job.model_copy(update={
                        "last_error": error_msg,
                        "last_error_at": now,
                    })
                    try:
                        tq._write_job(active_job_path, retry_job)
                        os.rename(active_job_path, tq._queue / "pending" / f"{run_name}.job.toml")
                        _safe_ledger_update(
                            state_mgr,
                            status=RunStatus.TRANSFERRING,  # still in flight (retrying)
                            transfer_attempts=bumped_attempts,
                            last_transfer_error=error_msg,
                            next_action_not_before=now + timedelta(seconds=RETRY_DELAYS[min(bumped_attempts - 1, len(RETRY_DELAYS) - 1)]),
                        )
                    except OSError as exc:
                        logger.error("Failed to re-enqueue %s for retry: %s; marking failed", run_name, exc)
                        with contextlib.suppress(OSError):
                            tq.fail(run_name)
                        continue
                    await asyncio.sleep(backoff)
            except Exception as e:
                logger.error("Unexpected error in daemon loop: %s", e, exc_info=True)
                await asyncio.sleep(poll_interval)

    finally:
        hb_task.cancel()
        await asyncio.gather(hb_task, return_exceptions=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        _release_transfer_lock(lock_path)
        with contextlib.suppress(OSError):
            pid_path.unlink()
        logger.info("Transfer daemon stopped")

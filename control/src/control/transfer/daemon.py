"""Transfer daemon: drains the transfer queue through the full state machine."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import pathlib
import signal
import subprocess
import time
from typing import IO

import anyio

from control.transfer.models import TransferJob
from control.transfer.queue import TransferQueue
from control.transfer.rsync import build_rsync_cmd
from control.transfer.verify import verify_manifest
from control.utils.paths import PanoPaths

logger = logging.getLogger("panoseti.transfer_daemon")

POLL_INTERVAL_SEC = 5.0
MAX_ATTEMPTS = 3
# Exponential backoff delays between transfer retries: attempt 1->2 waits 5 s,
# attempt 2->3 waits 30 s.  A 3rd failure is final (-> failed/).
_RETRY_BACKOFF_SEC = [5.0, 30.0]


def _transfer_state_dir() -> pathlib.Path:
    """Return the transfer daemon state subdirectory, creating it if needed."""
    d = PanoPaths.state_dir() / "transfer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _acquire_transfer_lock() -> IO[str] | None:
    """Try to acquire the exclusive transfer daemon lock file.

    Uses a non-blocking ``flock`` so that only one transfer daemon runs at a
    time.  The lock is automatically released when the process exits (the
    kernel drops it).

    Returns:
        An open file handle holding the lock, or ``None`` if another process
        already holds it.
    """
    lock_path = PanoPaths.locks_dir() / "transfer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except BlockingIOError:
        fh.close()
        return None


def _release_transfer_lock(fh: IO[str] | None) -> None:
    """Release the exclusive lock obtained by ``_acquire_transfer_lock``.

    Args:
        fh: File handle returned by ``_acquire_transfer_lock``.  A ``None``
            value is accepted safely (no-op).
    """
    if fh:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


async def _heartbeat_loop(heartbeat_path: pathlib.Path, interval: float = 5.0) -> None:
    """Write the current timestamp to *heartbeat_path* every *interval* seconds.

    Runs until cancelled.  Silently swallows ``asyncio.CancelledError`` on
    exit so the caller need not catch it.

    Args:
        heartbeat_path: Path to the heartbeat file.
        interval: Seconds between heartbeat writes.
    """
    try:
        while True:
            heartbeat_path.write_text(str(time.time()))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _process_job(job: TransferJob) -> bool:
    """Drive a single transfer job through the full state machine.

    The state machine advances through these stages in order:

    1. **MANIFEST_GENERATING** — ask each DAQ node's gRPC server to produce a
       content manifest for the run's PFF files.
    2. **TRANSFERRING** — rsync each DAQ node's run directory to the head node.
    3. **VERIFYING** — verify manifest digests on the head node.
    4. **CLEANING** — selectively delete PFF files from DAQ nodes via gRPC
       ``CleanupData``.
    5. **ARCHIVED** — write a ``run_complete`` marker and log success.

    ``no_collect`` skips stages 1-3; ``no_cleanup`` skips stage 4.

    Args:
        job: The ``TransferJob`` to process.

    Returns:
        ``True`` when the job reaches ARCHIVED; ``False`` on any failure.
    """
    run_name = job.run_name
    head_data_dir = job.head_data_dir
    daq_nodes = job.daq_nodes
    no_collect = job.no_collect
    no_cleanup = job.no_cleanup

    from control.utils.run_state import RunStateManager
    state_mgr = RunStateManager(base_dir=str(PanoPaths.base_dir()))

    logger.info("Processing transfer job for run: %s", run_name)

    if not no_collect:
        # --- Stage 1: manifest generation ---
        logger.info("[%s] Stage: MANIFEST_GENERATING", run_name)
        state_mgr.transition("MANIFEST_GENERATING")
        try:
            from panoseti_grpc.daq_control.client import AsyncDaqControlClient

            async def gen_manifest(node: object) -> None:
                from control.transfer.models import TransferNodeSpec as _TNS
                assert isinstance(node, _TNS)
                module_ids: list[int] = node.module_ids
                async with AsyncDaqControlClient(host=str(node.ip_addr), port=50051) as client:
                    for mid in module_ids:
                        try:
                            await client.GenerateManifest({
                                "data_dir": node.data_dir,
                                "run_dir": run_name,
                                "module_id": mid,
                                "algorithm": "blake3",
                                "include_patterns": ["*.pff"],
                            })
                        except Exception as exc:
                            logger.warning(
                                "GenerateManifest failed for module %s on %s: %s",
                                mid, node.ip_addr, exc
                            )

            async with asyncio.TaskGroup() as tg:
                for node in daq_nodes:
                    tg.create_task(gen_manifest(node))

        except ImportError:
            logger.warning("panoseti_grpc not available; skipping manifest generation")

        # --- Stage 2: rsync ---
        logger.info("[%s] Stage: TRANSFERRING", run_name)
        state_mgr.transition("TRANSFERRING")
        transfer_errors: list[str] = []
        for node in daq_nodes:
            head_run_dir = pathlib.Path(head_data_dir) / run_name
            cmd = build_rsync_cmd(node, run_name, head_run_dir)
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True
            )
            if result.returncode != 0:
                err = f"rsync failed for {node.ip_addr}: {result.stderr}"
                transfer_errors.append(err)
                logger.error("Rsync failed for %s: %s", node.ip_addr, result.stderr)

        if transfer_errors:
            logger.error("[%s] Transfer failed: %s", run_name, "; ".join(transfer_errors))
            state_mgr.transition("TRANSFER_FAILED")
            return False

        # --- Stage 3: verify manifest digests on head node ---
        logger.info("[%s] Stage: VERIFYING", run_name)
        state_mgr.transition("VERIFYING")
        verify_errors: list[str] = []
        head_run_path = pathlib.Path(head_data_dir) / run_name
        for algo_suffix in ("blake3", "xxh3_128", "sha256"):
            candidate = head_run_path / f"manifest.{algo_suffix}"
            if candidate.exists():
                ok, errs = await asyncio.to_thread(verify_manifest, candidate, head_run_path)
                if not ok:
                    verify_errors.extend(errs)
                    logger.error(
                        "[%s] Manifest verification failed (%s): %s",
                        run_name, candidate.name, "; ".join(errs),
                    )
                else:
                    logger.info("[%s] Manifest OK: %s", run_name, candidate.name)
        if verify_errors:
            state_mgr.transition("VERIFY_FAILED")
            logger.error(
                "[%s] Verification failed — skipping cleanup to preserve DAQ-side data. "
                "Manual recovery required.",
                run_name,
            )
            return False

    # --- Stage 4: selective cleanup ---
    if not no_cleanup:
        logger.info("[%s] Stage: CLEANING", run_name)
        state_mgr.transition("CLEANING")
        try:
            from panoseti_grpc.daq_control.client import AsyncDaqControlClient

            async def cleanup_node(node: object) -> None:
                from control.transfer.models import TransferNodeSpec as _TNS
                assert isinstance(node, _TNS)
                async with AsyncDaqControlClient(host=str(node.ip_addr), port=50051) as client:
                    try:
                        await client.CleanupData({
                            "data_dir": node.data_dir,
                            "run_dir": run_name,
                            "module_id": node.module_ids,
                            "mode": "CLEANUP_SELECTIVE",
                            "delete_patterns": ["*.pff"],
                            "preserve_patterns": ["*.json", "*.log", "*.toml"],
                        })
                    except Exception as exc:
                        logger.warning(
                            "CleanupData failed for %s: %s", node.ip_addr, exc
                        )

            async with asyncio.TaskGroup() as tg:
                for node in daq_nodes:
                    tg.create_task(cleanup_node(node))

        except ImportError:
            logger.warning("panoseti_grpc not available; skipping cleanup")

    # --- Stage 5: archive ---
    logger.info("[%s] Stage: ARCHIVED", run_name)
    head_run_dir_anyio = anyio.Path(head_data_dir) / run_name
    run_complete_path = head_run_dir_anyio / "run_complete"
    if not await run_complete_path.exists():
        await head_run_dir_anyio.mkdir(parents=True, exist_ok=True)
        await run_complete_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC"))

    state_mgr.transition("ARCHIVED")
    logger.info("Run %s archived successfully", run_name)
    return True


def _sweep_stranded_jobs(tq: TransferQueue) -> None:
    """Move jobs stranded in ``active/`` back to ``pending/`` (SC-TX-005).

    Called at daemon startup to recover jobs left behind by a prior crash.
    Each rename is POSIX-atomic; failures are logged but do not abort startup.

    Args:
        tq: The active :class:`TransferQueue` instance.
    """
    active_dir = tq._queue / "active"
    for stale in sorted(active_dir.glob("*.job.toml")):
        run_name_stale = stale.stem.removesuffix(".job")
        pending_path = tq._queue / "pending" / stale.name
        try:
            os.rename(stale, pending_path)
            logger.warning("Recovered stranded job from active/: %s", run_name_stale)
        except OSError as exc:
            logger.error("Failed to recover stranded job %s: %s", run_name_stale, exc)


async def run_daemon(poll_interval: float = POLL_INTERVAL_SEC) -> None:
    """Main daemon loop: acquire lock, write pid/heartbeat, poll for jobs, process them.

    Acquires an exclusive flock on the transfer lock file.  If another daemon
    already holds the lock the function returns immediately.  On startup,
    sweeps ``active/`` for jobs stranded by a prior crash and moves them back
    to ``pending/``.  Handles ``SIGTERM``/``SIGINT`` gracefully: finishes the
    current processing step, then releases the lock.

    Args:
        poll_interval: Seconds to wait between queue polls when no job is
            pending.
    """
    lock_fh = _acquire_transfer_lock()
    if lock_fh is None:
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
            job = tq.claim()
            if job is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(shutdown.wait(), timeout=poll_interval)
                continue

            run_name = job.run_name
            attempts: int = job.attempts + 1

            try:
                success = await _process_job(job)
                if success:
                    tq.complete(run_name)
                elif attempts >= MAX_ATTEMPTS:
                    logger.error(
                        "Run %s failed after %d attempts. Marking failed.",
                        run_name,
                        MAX_ATTEMPTS,
                    )
                    tq.fail(run_name)
                else:
                    backoff = _RETRY_BACKOFF_SEC[min(attempts - 1, len(_RETRY_BACKOFF_SEC) - 1)]
                    logger.warning(
                        "Run %s attempt %d/%d failed. Retrying in %.0f s.",
                        run_name,
                        attempts,
                        MAX_ATTEMPTS,
                        backoff,
                    )
                    # Re-enqueue with incremented attempt count by claiming and
                    # writing back to pending.
                    updated_job = job.model_copy(update={"attempts": attempts})
                    updated_job_path = tq._queue / "active" / f"{run_name}.job.toml"
                    if not updated_job_path.exists():
                        # Write back so retry logic can pick it up
                        tq._write_job(updated_job_path, updated_job)
                    os.rename(updated_job_path, tq._queue / "pending" / f"{run_name}.job.toml")
                    await asyncio.sleep(backoff)
            except Exception:
                logger.exception("Unhandled error processing %s", run_name)
                tq.fail(run_name)
    finally:
        hb_task.cancel()
        await asyncio.gather(hb_task, return_exceptions=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        _release_transfer_lock(lock_fh)
        with contextlib.suppress(OSError):
            pid_path.unlink()
        logger.info("Transfer daemon stopped")

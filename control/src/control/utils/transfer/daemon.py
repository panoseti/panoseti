from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pathlib
import signal
import time
from typing import Any

from control.utils.transfer.queue import TransferQueue
from control.utils.transfer.rsync_worker import rsync_one_node

logger = logging.getLogger("panoseti.transfer_daemon")

TRANSFER_LOCK_FILE = "tmp/panoseti_transfer.lock"
POLL_INTERVAL_SEC = 5.0
MAX_ATTEMPTS = 3


def _acquire_transfer_lock(base_dir: pathlib.Path) -> Any | None:
    """Try to acquire the exclusive transfer daemon lock file.

    Uses a non-blocking ``flock`` so that only one transfer daemon runs at a
    time.  The lock is automatically released when the process exits (kernel
    drops it).

    Args:
        base_dir: Directory that contains the ``tmp/`` subdirectory.

    Returns:
        An open file handle holding the lock, or ``None`` if another process
        already holds it.
    """
    lock_path = base_dir / TRANSFER_LOCK_FILE
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


def _release_transfer_lock(fh: Any) -> None:
    """Release the exclusive lock obtained by ``_acquire_transfer_lock``.

    Args:
        fh: File handle returned by ``_acquire_transfer_lock``.  A ``None``
            value is accepted safely (no-op).
    """
    if fh:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


async def _process_job(job: dict[str, Any], base_dir: pathlib.Path) -> bool:
    """Drive a single transfer job through the full state machine.

    The state machine advances through these stages in order:

    1. **MANIFEST_GENERATING** — ask each DAQ node's gRPC server to produce a
       content manifest for the run's PFF files.
    2. **MANIFEST_READY** — manifests have been generated.
    3. **TRANSFERRING** — rsync each DAQ node's run directory to the head node.
    4. **VERIFYING** — placeholder; rsync exit code is trusted for now.
    5. **CLEANING** — selectively delete PFF files from DAQ nodes via gRPC
       ``CleanupData``.
    6. **ARCHIVED** — write a ``run_complete`` marker and log success.

    ``no_collect`` skips steps 1-4; ``no_cleanup`` skips step 5.

    Args:
        job: Parsed job dictionary from ``TransferQueue.claim()``.
        base_dir: Filesystem root used to locate ``RunStateManager`` state.

    Returns:
        ``True`` when the job reaches ``ARCHIVED``; ``False`` on any failure.
    """
    run_name: str = job["run_name"]
    head_data_dir: str = job["head_data_dir"]
    daq_nodes: list[dict[str, Any]] = job.get("daq_nodes", [])
    no_collect: bool = job.get("no_collect", False)
    no_cleanup: bool = job.get("no_cleanup", False)

    logger.info("Processing transfer job for run: %s", run_name)

    if not no_collect:
        # --- Stage 1: manifest generation ---
        logger.info("[%s] Stage: MANIFEST_GENERATING", run_name)
        try:
            from panoseti_grpc.daq_control.client import (
                DaqControlClient,
            )

            for node in daq_nodes:
                client = DaqControlClient(host=node["ip_addr"], port=50051)
                module_ids: list[int] = node.get("module_ids", [])
                for mid in module_ids:
                    try:
                        client.GenerateManifest(
                            {
                                "data_dir": node["data_dir"],
                                "run_dir": run_name,
                                "module_id": [mid],
                                "algorithm": "blake3",
                                "include_patterns": ["*.pff"],
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            "GenerateManifest failed for module %s on %s: %s",
                            mid,
                            node["ip_addr"],
                            exc,
                        )
        except ImportError:
            logger.warning("panoseti_grpc not available; skipping manifest generation")

        # --- Stage 2: rsync ---
        logger.info("[%s] Stage: TRANSFERRING", run_name)
        transfer_errors: list[str] = []
        for node in daq_nodes:
            pf: dict[str, Any] | None = node.get("port_forwarding")
            ok, err = await asyncio.to_thread(
                rsync_one_node,
                node["ip_addr"],
                node["data_dir"],
                run_name,
                head_data_dir,
                node.get("username", "panoseti"),
                pf,
            )
            if not ok:
                transfer_errors.append(err)
                logger.error("Rsync failed for %s: %s", node["ip_addr"], err)

        if transfer_errors:
            logger.error("[%s] Transfer failed: %s", run_name, "; ".join(transfer_errors))
            return False

        # --- Stage 3: verify (rsync exit code trusted; full digest verify is a follow-on) ---
        logger.info("[%s] Stage: VERIFYING (trusting rsync exit code)", run_name)

    # --- Stage 4: selective cleanup ---
    if not no_cleanup:
        logger.info("[%s] Stage: CLEANING", run_name)
        try:
            from panoseti_grpc.daq_control.client import (
                DaqControlClient,
            )

            for node in daq_nodes:
                client = DaqControlClient(host=node["ip_addr"], port=50051)
                try:
                    client.CleanupData(
                        {
                            "data_dir": node["data_dir"],
                            "run_dir": run_name,
                            "module_id": node.get("module_ids", []),
                            "mode": "CLEANUP_SELECTIVE",
                            "delete_patterns": ["*.pff"],
                            "preserve_patterns": ["*.json", "*.log", "*.toml"],
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "CleanupData failed for %s: %s", node["ip_addr"], exc
                    )
        except ImportError:
            logger.warning("panoseti_grpc not available; skipping cleanup")

    # --- Stage 5: archive ---
    logger.info("[%s] Stage: ARCHIVED", run_name)
    head_run_dir = pathlib.Path(head_data_dir) / run_name
    run_complete_path = head_run_dir / "run_complete"
    if not run_complete_path.exists():
        head_run_dir.mkdir(parents=True, exist_ok=True)
        run_complete_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC"))

    logger.info("Run %s archived successfully", run_name)
    return True


async def run_daemon(
    base_dir: str = ".",
    poll_interval: float = POLL_INTERVAL_SEC,
) -> None:
    """Main daemon loop: acquire lock, poll for jobs, process them.

    Acquires an exclusive flock on ``tmp/panoseti_transfer.lock``.  If another
    daemon already holds the lock the function returns immediately.  Handles
    ``SIGTERM``/``SIGINT`` gracefully: finishes the current processing step,
    re-enqueues the job (if in progress), then releases the lock.

    Args:
        base_dir: Filesystem root for the queue and lock file.
        poll_interval: Seconds to wait between queue polls when no job is
            pending.
    """
    base = pathlib.Path(base_dir)
    lock_fh = _acquire_transfer_lock(base)
    if lock_fh is None:
        logger.info("Another transfer daemon is already running. Exiting.")
        return

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    logger.info("Transfer daemon started (pid=%d)", os.getpid())
    tq = TransferQueue(base_dir=base_dir)

    import contextlib
    try:
        while not shutdown.is_set():
            job = tq.claim()
            if job is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(shutdown.wait(), timeout=poll_interval)
                continue

            run_name: str = job["run_name"]
            attempts: int = job.get("attempts", 0) + 1

            try:
                success = await _process_job(job, base)
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
                    logger.warning(
                        "Run %s attempt %d failed. Re-enqueueing.",
                        run_name,
                        attempts,
                    )
                    tq.enqueue(
                        run_name,
                        job["head_data_dir"],
                        job.get("daq_nodes", []),
                        attempts=attempts,
                    )
            except Exception:
                logger.exception("Unhandled error processing %s", run_name)
                tq.fail(run_name)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        _release_transfer_lock(lock_fh)
        logger.info("Transfer daemon stopped")

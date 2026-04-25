"""Durable filesystem-backed transfer job queue."""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import tomllib

import tomli_w

from control.transfer.models import TransferJob
from control.utils.paths import PanoPaths


class TransferQueue:
    """Durable job queue for async data transfer using filesystem-atomic renames.

    Jobs move through: pending/ -> active/ -> completed/ (or failed/).
    All state transitions use ``os.rename`` for POSIX atomicity.

    The queue root directory defaults to ``PanoPaths.transfer_queue_dir()``
    but can be overridden by passing ``queue_dir`` or by setting the
    ``PSETI_TQ_DIR`` environment variable.
    """

    def __init__(self, queue_dir: pathlib.Path | None = None) -> None:
        """Initialize the TransferQueue, creating queue subdirectories as needed.

        Args:
            queue_dir: Explicit root directory for the queue.  When ``None``,
                ``PanoPaths.transfer_queue_dir()`` is used (which itself
                respects the ``PSETI_TQ_DIR`` environment variable).
        """
        self._queue: pathlib.Path = (
            queue_dir if queue_dir is not None else PanoPaths.transfer_queue_dir()
        )
        for sub in ("pending", "active", "completed", "failed"):
            (self._queue / sub).mkdir(parents=True, exist_ok=True)

    def _job_path(self, subdir: str, run_name: str) -> pathlib.Path:
        """Return the path for a job file in the given subdirectory.

        Args:
            subdir: One of ``pending``, ``active``, ``completed``, ``failed``.
            run_name: The run identifier (without the ``.job.toml`` suffix).

        Returns:
            Absolute path to the job TOML file.
        """
        return self._queue / subdir / f"{run_name}.job.toml"

    def _write_job(self, path: pathlib.Path, job: TransferJob) -> None:
        """Serialize *job* to TOML atomically via a temp file and ``os.replace``.

        Args:
            path: Destination path for the TOML file.
            job: The ``TransferJob`` Pydantic model to serialize.
        """
        tmp_dir = path.parent
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
        try:
            data = job.model_dump(mode="json")
            # Flatten IPvAnyAddress objects to plain strings for tomli_w
            for node in data.get("daq_nodes", []):
                if "ip_addr" in node:
                    node["ip_addr"] = str(node["ip_addr"])
                pf = node.get("port_forwarding")
                if pf and "gw_ip" in pf:
                    pf["gw_ip"] = str(pf["gw_ip"])
            # created_at: tomli_w handles datetime objects natively
            with os.fdopen(fd, "wb") as f:
                f.write(tomli_w.dumps(data).encode())
            os.replace(tmp_path, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def enqueue(self, job: TransferJob) -> bool:
        """Create ``pending/{run_name}.job.toml`` atomically. Idempotent across all subdirs.

        If a job for this run_name already exists in ``pending/``, ``active/``,
        ``completed/``, or ``failed/``, the call returns ``False`` without
        creating a duplicate.

        Args:
            job: The ``TransferJob`` to enqueue.

        Returns:
            ``True`` if the job was newly enqueued; ``False`` if it already
            existed in any bucket.
        """
        for subdir in ("pending", "active", "completed", "failed"):
            if self._job_path(subdir, job.run_name).exists():
                return False
        self._write_job(self._job_path("pending", job.run_name), job)
        return True

    def claim(self) -> TransferJob | None:
        """Move one job from pending/ to active/, return the parsed job or None.

        Returns:
            A ``TransferJob`` instance if a job was claimed, ``None`` if no
            pending jobs exist.
        """
        pending_dir = self._queue / "pending"
        for entry in sorted(pending_dir.iterdir()):
            if entry.suffix != ".toml":
                continue
            run_name = entry.stem.removesuffix(".job")
            active_path = self._job_path("active", run_name)
            try:
                os.rename(entry, active_path)
            except OSError:
                continue
            with open(active_path, "rb") as f:
                data = tomllib.load(f)
            return TransferJob.model_validate(data)
        return None

    def complete(self, run_name: str) -> None:
        """Move job from active/ to completed/.

        Args:
            run_name: The run identifier of the job to mark complete.

        Raises:
            FileNotFoundError: If no active job exists for *run_name*.
        """
        src = self._job_path("active", run_name)
        dst = self._job_path("completed", run_name)
        os.rename(src, dst)

    def fail(self, run_name: str) -> None:
        """Move job from active/ to failed/.

        Args:
            run_name: The run identifier of the job to mark failed.

        Raises:
            FileNotFoundError: If no active job exists for *run_name*.
        """
        src = self._job_path("active", run_name)
        dst = self._job_path("failed", run_name)
        os.rename(src, dst)

    def retry(self, run_name: str) -> bool:
        """Move job from failed/ back to pending/ and reset attempts to 0.

        Args:
            run_name: The run identifier of the job to retry.

        Returns:
            ``True`` if the job was moved; ``False`` if no failed job exists
            for *run_name*.
        """
        src = self._job_path("failed", run_name)
        if not src.exists():
            return False
        with open(src, "rb") as f:
            data = tomllib.load(f)
        data["attempts"] = 0
        job = TransferJob.model_validate(data)
        target = self._job_path("pending", run_name)
        self._write_job(target, job)
        os.unlink(src)
        return True

    def list_jobs(self, bucket: str) -> list[str]:
        """Return run names in a queue bucket.

        Args:
            bucket: One of ``"pending"``, ``"active"``, ``"completed"``,
                ``"failed"``.

        Returns:
            Sorted list of run names (without the ``.job.toml`` suffix).
        """
        bucket_dir = self._queue / bucket
        return [
            e.stem.removesuffix(".job")
            for e in sorted(bucket_dir.iterdir())
            if e.suffix == ".toml"
        ]

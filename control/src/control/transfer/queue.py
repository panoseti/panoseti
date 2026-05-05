"""Durable filesystem-backed transfer job queue."""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import tomllib
from typing import Any

from control.transfer.models import TransferJob, TransferStatus
from control.utils.paths import PanoPaths


class TransferQueue:
    """Durable job queue for async data transfer using filesystem-atomic renames.

    Jobs move through: pending/ -> active/ -> completed/ (or failed/).
    All state transitions use ``os.rename`` for POSIX atomicity.

    The queue root directory defaults to ``PanoPaths.transfer_queue_dir()``
    but can be overridden by passing ``queue_dir`` or by setting the
    ``PSETI_TQ_DIR`` environment variable.
    """

    def __init__(self, queue_dir: pathlib.Path | str | None = None) -> None:
        """Initialize the TransferQueue, creating queue subdirectories as needed.

        Args:
            queue_dir: Explicit root directory for the queue.  When ``None``,
                ``PanoPaths.transfer_queue_dir()`` is used (which itself
                respects the ``PSETI_TQ_DIR`` environment variable).
        """
        raw_queue = (
            queue_dir if queue_dir is not None else PanoPaths.transfer_queue_dir()
        )
        self._queue: pathlib.Path = pathlib.Path(raw_queue)
        for sub in TransferStatus:
            (self._queue / sub).mkdir(parents=True, exist_ok=True)

    def _job_path(self, subdir: TransferStatus, run_name: str) -> pathlib.Path:
        """Return the path for a job file in the given subdirectory.

        Args:
            subdir: One of TransferStatus: (``pending``, ``active``, ``completed``, ``failed``).
            run_name: The run identifier (without the ``.job.toml`` suffix).

        Returns:
            Absolute path to the job TOML file.
        """
        return self._queue / subdir / f"{run_name}.job.toml"

    @staticmethod
    def _escape_toml_str(s: str) -> str:
        """Escapes a string for TOML using JSON-style escaping (compatible with TOML)."""
        import json
        return json.dumps(s)

    def _write_job(self, path: pathlib.Path, job: TransferJob) -> None:
        """Serialize *job* to TOML atomically via a temp file and ``os.replace``.

        Uses manual TOML serialization (no third-party dependency) to keep the
        package lean.  The format mirrors the legacy ``utils/transfer/queue.py``
        style so existing tooling continues to work.

        Args:
            path: Destination path for the TOML file.
            job: The ``TransferJob`` Pydantic model to serialize.
        """
        # model_dump(mode="json") converts IPvAnyAddress -> str, datetime -> str
        data = job.model_dump(mode="json")
        tmp_dir = path.parent
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
        _skip_keys = {"daq_nodes"}
        try:
            with os.fdopen(fd, "w") as f:
                # Top-level scalar fields first; skip None values (optional fields)
                for k in ["schema_version", "run_name", "head_data_dir", "head_node_username", 
                          "created_at", "attempts", "no_cleanup", "no_collect", "skip_verify", 
                          "bwlimit", "algo", "last_error", "last_error_at"]:
                    v = data.get(k)
                    if v is None:
                        continue
                    if isinstance(v, bool):
                        val = "true" if v else "false"
                    elif isinstance(v, (int, float)):
                        val = str(v)
                    else:
                        val = self._escape_toml_str(str(v))
                    f.write(f"{k} = {val}\n")
                
                # [[daq_nodes]] array-of-tables
                daq_nodes = data.get("daq_nodes", [])
                if not daq_nodes:
                    f.write("\ndaq_nodes = []\n")
                for node in daq_nodes:
                    f.write("\n[[daq_nodes]]\n")
                    pf_data: dict[str, Any] | None = node.pop("port_forwarding", None)
                    for k, v in node.items():
                        if isinstance(v, list):
                            # module_ids is a list of ints
                            f.write(f"{k} = [{', '.join(str(i) for i in v)}]\n")
                        elif isinstance(v, bool):
                            f.write(f"{k} = {'true' if v else 'false'}\n")
                        elif isinstance(v, (int, float)):
                            f.write(f"{k} = {v}\n")
                        else:
                            f.write(f"{k} = {self._escape_toml_str(str(v))}\n")
                    
                    if pf_data is not None:
                        f.write("\n[daq_nodes.port_forwarding]\n")
                        for k, v in pf_data.items():
                            if v is None:
                                continue
                            if isinstance(v, list):
                                # reboot_port / cmd_port may be lists
                                non_null = [str(i) for i in v if i is not None]
                                f.write(f"{k} = [{', '.join(non_null)}]\n")
                            elif isinstance(v, bool):
                                f.write(f"{k} = {'true' if v else 'false'}\n")
                            elif isinstance(v, (int, float)):
                                f.write(f"{k} = {v}\n")
                            else:
                                f.write(f"{k} = {self._escape_toml_str(str(v))}\n")
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
        for subdir in TransferStatus:
            if self._job_path(subdir, job.run_name).exists():
                return False
        self._write_job(self._job_path(TransferStatus.PENDING, job.run_name), job)
        return True

    def claim(self) -> TransferJob | None:
        """Move one job from pending/ to active/, return the parsed job or None.

        Returns:
            A ``TransferJob`` instance if a job was claimed, ``None`` if no
            pending jobs exist.
        """
        pending_dir = self._queue / TransferStatus.PENDING
        for entry in sorted(pending_dir.iterdir()):
            if entry.suffix != ".toml":
                continue
            run_name = entry.stem.removesuffix(".job")
            active_path = self._job_path(TransferStatus.ACTIVE, run_name)
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
        src = self._job_path(TransferStatus.ACTIVE, run_name)
        dst = self._job_path(TransferStatus.COMPLETED, run_name)
        os.rename(src, dst)

    def fail(self, run_name: str) -> None:
        """Move job from active/ to failed/.

        Args:
            run_name: The run identifier of the job to mark failed.

        Raises:
            FileNotFoundError: If no active job exists for *run_name*.
        """
        src = self._job_path(TransferStatus.ACTIVE, run_name)
        dst = self._job_path(TransferStatus.FAILED, run_name)
        os.rename(src, dst)

    def retry(self, run_name: str) -> bool:
        """Move job from failed/ back to pending/ and reset attempts to 0.

        Args:
            run_name: The run identifier of the job to retry.

        Returns:
            ``True`` if the job was moved; ``False`` if no failed job exists
            for *run_name*.
        """
        src = self._job_path(TransferStatus.FAILED, run_name)
        if not src.exists():
            return False
        with open(src, "rb") as f:
            data = tomllib.load(f)
        data["attempts"] = 0
        job = TransferJob.model_validate(data)
        target = self._job_path(TransferStatus.PENDING, run_name)
        self._write_job(target, job)
        os.unlink(src)
        return True

    def list_jobs(self, bucket: TransferStatus) -> list[str]:
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

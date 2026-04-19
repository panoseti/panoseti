from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import tomllib
from datetime import UTC, datetime
from typing import Any


class TransferQueue:
    """Durable job queue for async data transfer using filesystem-atomic renames.

    Jobs move through: pending/ -> active/ -> completed/ (or failed/)
    All state transitions use os.rename for POSIX atomicity.
    """

    QUEUE_ROOT = pathlib.Path("tmp") / "transfer_queue"

    def __init__(self, base_dir: str = ".") -> None:
        """Initialize the TransferQueue, creating queue subdirectories as needed.

        Args:
            base_dir: Root directory under which the queue lives.
        """
        self._base = pathlib.Path(base_dir)
        self._queue = self._base / self.QUEUE_ROOT
        for sub in ("pending", "active", "completed", "failed"):
            (self._queue / sub).mkdir(parents=True, exist_ok=True)

    def _job_path(self, subdir: str, run_name: str) -> pathlib.Path:
        """Return the path for a job file in the given subdirectory.

        Args:
            subdir: One of pending, active, completed, failed.
            run_name: The run identifier (without .job.toml suffix).

        Returns:
            Absolute path to the job TOML file.
        """
        return self._queue / subdir / f"{run_name}.job.toml"

    def _write_job_toml(self, path: pathlib.Path, content: dict[str, Any]) -> None:
        """Write job dict to TOML atomically via a temp file and os.replace.

        Args:
            path: Destination path for the TOML file.
            content: Job dictionary to serialize.
        """
        tmp_dir = path.parent
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f'run_name = "{content["run_name"]}"\n')
                f.write(f'head_data_dir = "{content["head_data_dir"]}"\n')
                f.write(f'created_at = "{content["created_at"]}"\n')
                f.write(f'attempts = {content.get("attempts", 0)}\n')
                for node in content.get("daq_nodes", []):
                    f.write("\n[[daq_nodes]]\n")
                    for k, v in node.items():
                        if isinstance(v, str):
                            f.write(f'{k} = "{v}"\n')
                        else:
                            f.write(f'{k} = {v}\n')
            os.replace(tmp_path, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def enqueue(self, run_name: str, head_data_dir: str, daq_nodes: list[dict[str, Any]]) -> pathlib.Path:
        """Create pending/{run_name}.job.toml atomically. Idempotent if job exists.

        Args:
            run_name: Unique run identifier.
            head_data_dir: Path to the head node data directory for this run.
            daq_nodes: List of DAQ node descriptors (dicts with node metadata).

        Returns:
            Path to the pending job file.
        """
        target = self._job_path("pending", run_name)
        if target.exists():
            return target
        content: dict[str, Any] = {
            "run_name": run_name,
            "head_data_dir": head_data_dir,
            "created_at": datetime.now(UTC).isoformat(),
            "attempts": 0,
            "daq_nodes": daq_nodes,
        }
        self._write_job_toml(target, content)
        return target

    def claim(self) -> dict[str, Any] | None:
        """Move one job from pending/ to active/, return job dict or None.

        Returns:
            Parsed job dict if a job was claimed, None if no pending jobs exist.
        """
        pending_dir = self._queue / "pending"
        for entry in sorted(pending_dir.iterdir()):
            if entry.suffix != ".toml":
                continue
            active_path = self._job_path("active", entry.stem.removesuffix(".job"))
            try:
                os.rename(entry, active_path)
            except OSError:
                continue
            with open(active_path, "rb") as f:
                return tomllib.load(f)
        return None

    def complete(self, run_name: str) -> None:
        """Move job from active/ to completed/.

        Args:
            run_name: The run identifier of the job to mark complete.
        """
        src = self._job_path("active", run_name)
        dst = self._job_path("completed", run_name)
        os.rename(src, dst)

    def fail(self, run_name: str) -> None:
        """Move job from active/ to failed/.

        Args:
            run_name: The run identifier of the job to mark failed.
        """
        src = self._job_path("active", run_name)
        dst = self._job_path("failed", run_name)
        os.rename(src, dst)

    def list_pending(self) -> list[str]:
        """Return list of pending run_names (without .job.toml suffix).

        Returns:
            Sorted list of run names with pending jobs.
        """
        pending_dir = self._queue / "pending"
        return [
            e.stem.removesuffix(".job")
            for e in sorted(pending_dir.iterdir())
            if e.suffix == ".toml"
        ]

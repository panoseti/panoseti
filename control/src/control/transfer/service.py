"""Public API for transfer queue management, used by stop.py and CLI."""
from __future__ import annotations

from control.transfer.models import TransferJob
from control.transfer.queue import TransferQueue


def enqueue(job: TransferJob) -> bool:
    """Enqueue a transfer job.

    Args:
        job: The ``TransferJob`` to add to the queue.

    Returns:
        ``True`` if the job was newly enqueued; ``False`` if it already existed
        in any queue bucket.
    """
    q = TransferQueue()
    return q.enqueue(job)


def get_queue_summary() -> dict[str, list[str]]:
    """Return run names in each queue bucket.

    Returns:
        Dict with keys ``"pending"``, ``"active"``, ``"completed"``,
        ``"failed"``, each mapping to a sorted list of run names.
    """
    q = TransferQueue()
    return {
        "pending": q.list_jobs("pending"),
        "active": q.list_jobs("active"),
        "completed": q.list_jobs("completed"),
        "failed": q.list_jobs("failed"),
    }


def retry_job(run_name: str) -> bool:
    """Move a failed job back to pending.

    Args:
        run_name: Identifier of the run whose failed job should be retried.

    Returns:
        ``True`` if the job was successfully moved; ``False`` if no failed job
        exists for *run_name*.
    """
    q = TransferQueue()
    return q.retry(run_name)

"""Public API for transfer queue management, used by stop.py and CLI."""
from __future__ import annotations

from control.transfer.models import TransferJob, TransferStatus
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


def get_queue_summary() -> dict[TransferStatus, list[str]]:
    """Return run names in each queue bucket.

    Returns:
        Dict with keys ``"pending"``, ``"active"``, ``"completed"``,
        ``"failed"``, each mapping to a sorted list of run names.
    """
    q = TransferQueue()
    return {
        TransferStatus.PENDING: q.list_jobs(TransferStatus.PENDING),
        TransferStatus.ACTIVE: q.list_jobs(TransferStatus.ACTIVE),
        TransferStatus.COMPLETED: q.list_jobs(TransferStatus.COMPLETED),
        TransferStatus.FAILED: q.list_jobs(TransferStatus.FAILED),
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

"""
Transfer queue helpers.

Wraps TransferQueue.list_jobs() so tests can assert run placement in a
specific bucket without needing to know the underlying directory structure.
"""

from __future__ import annotations

import time

from control.transfer.queue import TransferQueue
from control.utils.pydantic_config_models import TransferStatus


def bucket_for(run_name: str) -> str | None:
    """Return the bucket ('pending', 'active', 'completed', 'failed') containing *run_name*.

    Returns None if the run_name is not in any bucket.
    """
    tq = TransferQueue()
    for bucket in TransferStatus:
        if run_name in tq.list_jobs(bucket):
            return str(bucket)
    return None


def assert_only_in_bucket(run_name: str, expected_bucket: str, timeout: float = 0.0) -> None:
    """Assert that *run_name* is present in *expected_bucket* and absent from all others.

    Args:
        run_name: The run identifier.
        expected_bucket: One of 'pending', 'active', 'completed', 'failed'.
        timeout: If > 0, poll until the condition is met or timeout expires.

    Raises:
        AssertionError: if the run is missing from the expected bucket, or
            present in another bucket at assertion time.
    """
    if timeout > 0:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            actual = bucket_for(run_name)
            if actual == expected_bucket:
                break
            time.sleep(2.0)
        else:
            actual = bucket_for(run_name)
            raise AssertionError(
                f"run {run_name!r} did not reach bucket {expected_bucket!r} within {timeout:.0f}s "
                f"(last observed: {actual!r})"
            )
    else:
        actual = bucket_for(run_name)

    assert actual == expected_bucket, (
        f"run {run_name!r} is in bucket {actual!r}, expected {expected_bucket!r}"
    )

    # Also assert the run is absent from all other buckets.
    tq = TransferQueue()
    conflicts = [
        str(b)
        for b in TransferStatus
        if str(b) != expected_bucket and run_name in tq.list_jobs(b)
    ]
    assert not conflicts, (
        f"run {run_name!r} appears in unexpected bucket(s): {conflicts}"
    )

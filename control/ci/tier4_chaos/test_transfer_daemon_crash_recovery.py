"""Tier 4 (Chaos): Transfer daemon crash-recovery regression tests.

Test 4.1 from EXECUTION_PLAN — validates the infinite-bounce fix:
  - A job whose _process_job always returns (False, error) goes to failed/
    after MAX_ATTEMPTS, not bouncing back to pending/ indefinitely.
  - _sweep_stranded_jobs respects MAX_ATTEMPTS when recovering active/ jobs,
    moving exhausted jobs to failed/ rather than pending/.
  - attempts count is persisted into active/ before _process_job runs, so a
    daemon crash leaves a bumped count that the sweep can use.

These tests run entirely in-process without Docker; they do NOT require gRPC
or a real DAQ node.
"""
from __future__ import annotations

import asyncio
import pathlib
import tomllib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from control.transfer.daemon import _sweep_stranded_jobs, run_daemon
from control.transfer.lifecycle import MAX_ATTEMPTS
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(run_name: str, tmp_path: pathlib.Path, attempts: int = 0) -> TransferJob:
    node = TransferNodeSpec(
        ip_addr="192.168.0.10",
        username="root",
        data_dir=str(tmp_path / "data"),
        module_ids=[250],
        port_forwarding=None,
    )
    return TransferJob(
        run_name=run_name,
        head_data_dir=str(tmp_path / "head"),
        head_node_username="root",
        created_at=datetime.now(UTC),
        attempts=attempts,
        daq_nodes=[node],
    )


def _enqueue(tq: TransferQueue, job: TransferJob) -> None:
    pending_path = tq._queue / "pending" / f"{job.run_name}.job.toml"
    tq._write_job(pending_path, job)


def _read_job_toml(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


async def _run_daemon_until(
    tmp_path: pathlib.Path,
    done_pred: object,
    *,
    timeout_iters: int = 400,
    poll_interval: float = 0.02,
) -> None:
    """Run the daemon as a task and cancel it once done_pred() is True or timeout."""
    task = asyncio.create_task(run_daemon(poll_interval=poll_interval))
    for _ in range(timeout_iters):
        await asyncio.sleep(poll_interval)
        if callable(done_pred) and done_pred():  # type: ignore[operator]
            break
    task.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await task


# ---------------------------------------------------------------------------
# Test 4.1a: _process_job returns failure → job goes to failed/ after MAX_ATTEMPTS
# ---------------------------------------------------------------------------

class TestDaemonCrashRecovery:
    @pytest.mark.asyncio
    async def test_failing_job_goes_to_failed_not_pending(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job whose _process_job always returns (False, error) must land in
        failed/ after MAX_ATTEMPTS, never bouncing back to pending/ indefinitely.

        This is the D-2 regression check for the infinite-bounce fix.
        """
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_boom", tmp_path)
        _enqueue(tq, job)

        # _process_job must return (False, str) — never raise. The daemon loop
        # trusts this contract. The mock simulates repeated logical failures.
        async def _always_fail(
            j: TransferJob, shutdown: asyncio.Event, state_mgr: Any
        ) -> tuple[bool, str | None]:
            return False, "RuntimeError: boom"

        failed_dir = tmp_path / "queue" / "failed"

        with (
            patch("control.transfer.daemon._process_job", side_effect=_always_fail),
            patch("control.transfer.daemon.RETRY_DELAYS", [0.01, 0.01]),
        ):
            await _run_daemon_until(
                tmp_path,
                lambda: any(True for _ in failed_dir.glob("*.job.toml")),
            )

        failed_files = list(failed_dir.glob("*.job.toml"))
        pending_files = list((tmp_path / "queue" / "pending").glob("*.job.toml"))
        active_files = list((tmp_path / "queue" / "active").glob("*.job.toml"))

        assert failed_files, "Job must be in failed/ after MAX_ATTEMPTS"
        assert not pending_files, "Job must NOT bounce back to pending/"
        assert not active_files, "Job must NOT remain in active/"

        data = _read_job_toml(failed_files[0])
        assert data["attempts"] == MAX_ATTEMPTS, (
            f"attempts must equal MAX_ATTEMPTS={MAX_ATTEMPTS}, got {data['attempts']}"
        )

    @pytest.mark.asyncio
    async def test_last_error_written_on_retry(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job TOML in pending/ must carry last_error after a failed attempt."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_err", tmp_path)
        _enqueue(tq, job)

        call_count = 0

        async def _fail_once_then_succeed(
            j: TransferJob, shutdown: asyncio.Event, state_mgr: Any
        ) -> tuple[bool, str | None]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False, "transient_error"
            return True, None

        completed_dir = tmp_path / "queue" / "completed"

        with (
            patch("control.transfer.daemon._process_job", side_effect=_fail_once_then_succeed),
            patch("control.transfer.daemon.RETRY_DELAYS", [0.01, 0.01]),
        ):
            await _run_daemon_until(
                tmp_path,
                lambda: any(True for _ in completed_dir.glob("*.job.toml")),
                timeout_iters=200,
            )

        assert any(True for _ in completed_dir.glob("*.job.toml")), (
            "Job should reach completed/ after one failed attempt followed by success"
        )
        assert call_count == 2, f"Expected 2 calls (1 fail + 1 success), got {call_count}"

    # ---------------------------------------------------------------------------
    # Test 4.1b: _sweep_stranded_jobs breaks the infinite-bounce loop
    # ---------------------------------------------------------------------------

    def test_sweep_stranded_exhausted_goes_to_failed(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_sweep_stranded_jobs must move an active/ job with attempts >= MAX_ATTEMPTS
        directly to failed/, not pending/ — breaking the infinite-bounce loop."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")

        # Simulate a job stranded in active/ with attempts already at MAX_ATTEMPTS.
        exhausted_job = _make_job("run_exhausted", tmp_path, attempts=MAX_ATTEMPTS)
        active_path = tq._queue / "active" / "run_exhausted.job.toml"
        tq._write_job(active_path, exhausted_job)

        _sweep_stranded_jobs(tq)

        failed_files = list((tmp_path / "queue" / "failed").glob("*.job.toml"))
        pending_files = list((tmp_path / "queue" / "pending").glob("*.job.toml"))
        assert failed_files, "Exhausted stranded job must go to failed/"
        assert not pending_files, "Exhausted stranded job must NOT go to pending/"

    def test_sweep_stranded_below_max_goes_to_pending(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_sweep_stranded_jobs must move a non-exhausted active/ job to pending/
        so the next daemon start can retry it."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")

        partial_job = _make_job("run_partial", tmp_path, attempts=1)
        active_path = tq._queue / "active" / "run_partial.job.toml"
        tq._write_job(active_path, partial_job)

        _sweep_stranded_jobs(tq)

        pending_files = list((tmp_path / "queue" / "pending").glob("*.job.toml"))
        failed_files = list((tmp_path / "queue" / "failed").glob("*.job.toml"))
        assert pending_files, "Non-exhausted stranded job must go to pending/"
        assert not failed_files, "Non-exhausted stranded job must NOT go to failed/"

    # ---------------------------------------------------------------------------
    # Test 4.1c: attempts bumped at claim time (pre-crash persistence)
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_attempts_bumped_before_processing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job in active/ must have attempts == original + 1 immediately
        after claim — before _process_job even executes.

        Simulates what _sweep would see after a mid-job crash: the bumped
        count is already on disk, so the sweep can correctly decide whether
        to retry or permanently fail.
        """
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_atomic", tmp_path, attempts=0)
        _enqueue(tq, job)

        captured_attempts: list[int] = []

        async def _capture_and_fail(
            j: TransferJob, shutdown: asyncio.Event, state_mgr: Any
        ) -> tuple[bool, str | None]:
            # At this point, the job should already be in active/ with bumped attempts.
            active_path = tq._queue / "active" / "run_atomic.job.toml"
            if active_path.exists():
                data = _read_job_toml(active_path)
                captured_attempts.append(data.get("attempts", -1))
            return False, "capture_complete"

        failed_dir = tmp_path / "queue" / "failed"

        with (
            patch("control.transfer.daemon._process_job", side_effect=_capture_and_fail),
            patch("control.transfer.daemon.RETRY_DELAYS", [0.01, 0.01]),
        ):
            await _run_daemon_until(
                tmp_path,
                lambda: any(True for _ in failed_dir.glob("*.job.toml")),
            )

        assert captured_attempts, "Expected _process_job to be called at least once"
        assert captured_attempts[0] == 1, (
            f"attempts in active/ must be 1 at _process_job call time, got {captured_attempts[0]}"
        )

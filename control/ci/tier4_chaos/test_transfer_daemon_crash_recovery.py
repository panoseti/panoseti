"""Tier 4 (Chaos): Transfer daemon crash-recovery regression tests.

Test 4.1 from EXECUTION_PLAN — validates the infinite-bounce fix:
  - A job whose _process_job always raises goes to failed/ (not pending/) after
    one attempt, because attempts are persisted at claim time.
  - The daemon log contains the traceback string.
  - _sweep_stranded_jobs respects MAX_ATTEMPTS when recovering active/ jobs,
    moving exhausted jobs to failed/ rather than pending/.

These tests run entirely in-process without Docker; they do NOT require gRPC
or a real DAQ node.
"""
from __future__ import annotations

import asyncio
import pathlib
import tomllib
from datetime import UTC, datetime
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


# ---------------------------------------------------------------------------
# Test 4.1a: process raises → job goes to failed/ with attempts == 1
# ---------------------------------------------------------------------------

class TestDaemonCrashRecovery:
    @pytest.mark.asyncio
    async def test_failing_job_goes_to_failed_not_pending(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that always raises must land in failed/ after MAX_ATTEMPTS,
        never bouncing back to pending/."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_boom", tmp_path)
        _enqueue(tq, job)

        call_count = 0

        async def _always_raise(j: TransferJob, shutdown: asyncio.Event) -> tuple[bool, str | None]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        with patch("control.transfer.daemon._process_job", side_effect=_always_raise):
            # Run daemon with a short poll interval; stop it after MAX_ATTEMPTS
            # by monkey-patching the queue to detect job completion.
            async def _run_with_timeout() -> None:
                task = asyncio.create_task(run_daemon(poll_interval=0.05))
                # Wait until the job is in failed/ or timeout.
                for _ in range(200):
                    await asyncio.sleep(0.05)
                    if list((tmp_path / "queue" / "failed").glob("*.job.toml")):
                        break
                task.cancel()
                with pytest.raises((asyncio.CancelledError, Exception)):
                    await task

            await _run_with_timeout()

        # Must be in failed/, not pending/ or active/.
        failed_files = list((tmp_path / "queue" / "failed").glob("*.job.toml"))
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
    async def test_daemon_log_contains_traceback(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The daemon log file must contain the traceback from an exception in _process_job."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_traceback", tmp_path)
        _enqueue(tq, job)

        async def _always_raise(j: TransferJob, shutdown: asyncio.Event) -> tuple[bool, str | None]:
            raise RuntimeError("boom_unique_string")

        with patch("control.transfer.daemon._process_job", side_effect=_always_raise):
            async def _run_with_timeout() -> None:
                task = asyncio.create_task(run_daemon(poll_interval=0.05))
                for _ in range(200):
                    await asyncio.sleep(0.05)
                    if list((tmp_path / "queue" / "failed").glob("*.job.toml")):
                        break
                task.cancel()
                with pytest.raises((asyncio.CancelledError, Exception)):
                    await task

            await _run_with_timeout()

        log_dir = pathlib.Path(str(tmp_path / "state")) / "logs" / "transfer_daemon"
        log_files = list(log_dir.glob("*.log"))
        assert log_files, f"Expected log file in {log_dir}, got none"
        log_text = "".join(p.read_text() for p in log_files)
        assert "boom_unique_string" in log_text, (
            f"Expected 'boom_unique_string' in daemon log. Log contents:\n{log_text[:2000]}"
        )

    # ---------------------------------------------------------------------------
    # Test 4.1b: attempts persisted at claim time (pre-crash persistence)
    # ---------------------------------------------------------------------------

    def test_sweep_stranded_exhausted_goes_to_failed(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_sweep_stranded_jobs must move an active/ job with attempts >= MAX_ATTEMPTS
        directly to failed/, not pending/ — breaking the infinite-bounce loop."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")

        # Simulate a job stranded in active/ with attempts already at MAX_ATTEMPTS
        # (as if the daemon persisted the bumped count before crashing).
        exhausted_job = _make_job("run_exhausted", tmp_path, attempts=MAX_ATTEMPTS)
        active_path = tq._queue / "active" / "run_exhausted.job.toml"
        tq._write_job(active_path, exhausted_job)

        _sweep_stranded_jobs(tq)

        # Must land in failed/, not pending/.
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

        # Stranded with attempts == 1 (below MAX_ATTEMPTS).
        partial_job = _make_job("run_partial", tmp_path, attempts=1)
        active_path = tq._queue / "active" / "run_partial.job.toml"
        tq._write_job(active_path, partial_job)

        _sweep_stranded_jobs(tq)

        pending_files = list((tmp_path / "queue" / "pending").glob("*.job.toml"))
        failed_files = list((tmp_path / "queue" / "failed").glob("*.job.toml"))
        assert pending_files, "Non-exhausted stranded job must go to pending/"
        assert not failed_files, "Non-exhausted stranded job must NOT go to failed/"

    def test_attempts_bumped_before_processing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job in active/ must have attempts == original + 1 immediately
        after claim — before _process_job even runs.  Simulates what _sweep
        would see after a mid-job crash."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
        monkeypatch.setenv("PSETI_TQ_DIR", str(tmp_path / "queue"))

        tq = TransferQueue(queue_dir=tmp_path / "queue")
        job = _make_job("run_atomic", tmp_path, attempts=0)
        _enqueue(tq, job)

        # Capture the contents of active/ *during* _process_job.
        captured: list[int] = []

        async def _capture_and_fail(
            j: TransferJob, shutdown: asyncio.Event
        ) -> tuple[bool, str | None]:
            active_path = tq._queue / "active" / "run_atomic.job.toml"
            if active_path.exists():
                data = _read_job_toml(active_path)
                captured.append(data.get("attempts", -1))
            raise RuntimeError("capture_complete")

        async def _run_once() -> None:
            task = asyncio.create_task(run_daemon(poll_interval=0.05))
            for _ in range(100):
                await asyncio.sleep(0.05)
                if captured or any(True for _ in (tmp_path / "queue" / "failed").glob("*.job.toml")):
                    break
            task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await task

        with patch("control.transfer.daemon._process_job", side_effect=_capture_and_fail):
            asyncio.run(_run_once())

        assert captured, "Expected _process_job to be called at least once"
        assert captured[0] == 1, (
            f"attempts in active/ must be 1 before _process_job runs (pre-commit), got {captured[0]}"
        )

# mypy: ignore-errors
"""
test_transfer_queue.py

Phase 2 RED tests for the TransferQueue class (control/utils/transfer/queue.py).

All tests in this file should FAIL on the current codebase with ImportError or
AttributeError, and pass only after Phase 2 is implemented.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest

try:
    from control.utils.transfer.queue import TransferQueue
except ImportError:
    TransferQueue = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

RUN_NAME = "myrun.pffd"
RUN_ALPHA = "run_alpha"
RUN_BETA = "run_beta"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _queue_subdir(base: pathlib.Path, subdir: str) -> pathlib.Path:
    return base / "tmp" / "transfer_queue" / subdir


def _tq(tmp_path: pathlib.Path) -> TransferQueue:
    """Return a TransferQueue rooted at tmp_path, or raise ImportError if not yet implemented."""
    if TransferQueue is None:
        raise ImportError("utils.transfer.queue.TransferQueue is not yet implemented")
    return TransferQueue(base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

class TestTransferQueueImport:
    """Basic import sanity test — fails RED until the module exists."""

    def test_import_transfer_queue(self) -> None:
        """Importing TransferQueue from utils.transfer.queue must succeed."""
        if TransferQueue is None:
            raise ImportError("utils.transfer.queue.TransferQueue is not yet implemented")
        assert TransferQueue is not None


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

class TestTransferQueueEnqueue:

    def test_enqueue_creates_pending_job(self, tmp_path) -> None:
        """enqueue() must create a TOML file in tmp/transfer_queue/pending/."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        pending_dir = _queue_subdir(tmp_path, "pending")
        assert (pending_dir / f"{RUN_NAME}.job.toml").exists()

    def test_enqueue_idempotent(self, tmp_path) -> None:
        """enqueue() called twice for the same run must produce exactly one file."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        pending_dir = _queue_subdir(tmp_path, "pending")
        toml_files = list(pending_dir.glob("*.job.toml"))
        assert len(toml_files) == 1

    def test_list_pending(self, tmp_path) -> None:
        """list_pending() must return all enqueued run_names."""
        q = _tq(tmp_path)
        q.enqueue(RUN_ALPHA, "/data", [{"ip": "1.2.3.4"}])
        q.enqueue(RUN_BETA, "/data", [{"ip": "5.6.7.8"}])
        pending = q.list_pending()
        assert sorted(pending) == [RUN_ALPHA, RUN_BETA]

    def test_job_toml_contains_required_fields(self, tmp_path) -> None:
        """The generated TOML file must contain run_name, head_data_dir, daq_nodes, created_at."""
        q = _tq(tmp_path)
        job_path = q.enqueue(RUN_NAME, "/data/head", [{"ip": "1.2.3.4"}])
        with open(job_path, "rb") as f:
            data = tomllib.load(f)

        assert "run_name" in data
        assert "head_data_dir" in data
        assert "daq_nodes" in data
        assert "created_at" in data
        assert data["run_name"] == RUN_NAME
        assert data["head_data_dir"] == "/data/head"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

class TestTransferQueueClaim:

    def test_claim_moves_to_active(self, tmp_path) -> None:
        """claim() must atomically move the job from pending/ to active/."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])

        job = q.claim()
        assert job is not None
        assert job["run_name"] == RUN_NAME

        pending_dir = _queue_subdir(tmp_path, "pending")
        active_dir = _queue_subdir(tmp_path, "active")
        assert not (pending_dir / f"{RUN_NAME}.job.toml").exists()
        assert (active_dir / f"{RUN_NAME}.job.toml").exists()

    def test_claim_returns_none_when_empty(self, tmp_path) -> None:
        """claim() on an empty queue must return None."""
        q = _tq(tmp_path)
        result = q.claim()
        assert result is None

    def test_double_claim_is_none(self, tmp_path) -> None:
        """After claiming the only pending job, a second claim must return None."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        first = q.claim()
        assert first is not None
        second = q.claim()
        assert second is None


# ---------------------------------------------------------------------------
# Complete / Fail
# ---------------------------------------------------------------------------

class TestTransferQueueComplete:

    def test_complete_moves_to_completed(self, tmp_path) -> None:
        """complete() must move the job from active/ to completed/."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        q.claim()
        q.complete(RUN_NAME)

        active_dir = _queue_subdir(tmp_path, "active")
        completed_dir = _queue_subdir(tmp_path, "completed")
        assert not (active_dir / f"{RUN_NAME}.job.toml").exists()
        assert (completed_dir / f"{RUN_NAME}.job.toml").exists()

    def test_fail_moves_to_failed(self, tmp_path) -> None:
        """fail() must move the job from active/ to failed/."""
        q = _tq(tmp_path)
        q.enqueue(RUN_NAME, "/data", [{"ip": "1.2.3.4"}])
        q.claim()
        q.fail(RUN_NAME)

        active_dir = _queue_subdir(tmp_path, "active")
        failed_dir = _queue_subdir(tmp_path, "failed")
        assert not (active_dir / f"{RUN_NAME}.job.toml").exists()
        assert (failed_dir / f"{RUN_NAME}.job.toml").exists()


# ---------------------------------------------------------------------------
# Edge cases: complete/fail on unclaimed jobs
# ---------------------------------------------------------------------------

class TestTransferQueueEdgeCases:

    def test_complete_unclaimed_job_raises_or_noop(self, tmp_path) -> None:
        """complete() on a job that was never claimed must either raise
        FileNotFoundError (preferred — makes the bug loud) or return
        silently without corrupting queue state.

        Contract: complete() MUST NOT silently move a non-existent active
        job to completed/ (i.e., it must not create a completed/ entry from
        thin air). Either behaviour is acceptable at this stage — document
        which one Phase 2 chooses here.

        This test will fail RED until TransferQueue is implemented.
        """
        q = _tq(tmp_path)
        # Do NOT enqueue or claim — call complete() on an unknown run_name.
        try:
            q.complete("nonexistent.pffd")
        except FileNotFoundError:
            # Preferred: loud failure makes the bug obvious.
            pass
        except Exception as exc:
            pytest.fail(f"complete() raised unexpected {type(exc).__name__}: {exc}")
        # If it returns silently, verify no spurious completed/ entry was created.
        completed_dir = _queue_subdir(tmp_path, "completed")
        assert not (completed_dir / "nonexistent.pffd.job.toml").exists(), (
            "complete() must not create a completed entry for an unclaimed job"
        )

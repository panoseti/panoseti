# mypy: ignore-errors
"""
test_transfer_queue.py

Phase 2 RED tests for the TransferQueue class (control/utils/transfer/queue.py).

All tests in this file should FAIL on the current codebase with ImportError or
AttributeError, and pass only after Phase 2 is implemented.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.skipif(
    False,  # Never skip — let tests fail naturally via ImportError propagation
    reason="TransferQueue not yet implemented"
)


class TestTransferQueueImport:
    """Basic import sanity test — fails RED until the module exists."""

    def test_import_transfer_queue(self):
        """Importing TransferQueue from utils.transfer.queue must succeed."""
        from utils.transfer.queue import TransferQueue  # noqa: F401


# ---------------------------------------------------------------------------
# Helper that builds a TransferQueue or skips if import failed
# ---------------------------------------------------------------------------

def _tq(tmp_path: pathlib.Path) -> "TransferQueue":
    """Return a TransferQueue rooted at tmp_path, or raise ImportError."""
    from utils.transfer.queue import TransferQueue as TQ  # may raise ImportError
    return TQ(base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

class TestTransferQueueEnqueue:

    def test_enqueue_creates_pending_job(self, tmp_path):
        """enqueue() must create a TOML file in tmp/transfer_queue/pending/."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        pending_dir = tmp_path / "tmp" / "transfer_queue" / "pending"
        assert (pending_dir / "myrun.pffd.job.toml").exists()

    def test_enqueue_idempotent(self, tmp_path):
        """enqueue() called twice for the same run must produce exactly one file."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        pending_dir = tmp_path / "tmp" / "transfer_queue" / "pending"
        toml_files = list(pending_dir.glob("*.job.toml"))
        assert len(toml_files) == 1

    def test_list_pending(self, tmp_path):
        """list_pending() must return all enqueued run_names."""
        q = _tq(tmp_path)
        q.enqueue("run_alpha", "/data", [{"ip": "1.2.3.4"}])
        q.enqueue("run_beta", "/data", [{"ip": "5.6.7.8"}])
        pending = q.list_pending()
        assert sorted(pending) == ["run_alpha", "run_beta"]

    def test_job_toml_contains_required_fields(self, tmp_path):
        """The generated TOML file must contain run_name, head_data_dir, daq_nodes, created_at."""
        import tomllib

        q = _tq(tmp_path)
        job_path = q.enqueue("myrun.pffd", "/data/head", [{"ip": "1.2.3.4"}])
        with open(job_path, "rb") as f:
            data = tomllib.load(f)

        assert "run_name" in data
        assert "head_data_dir" in data
        assert "daq_nodes" in data
        assert "created_at" in data
        assert data["run_name"] == "myrun.pffd"
        assert data["head_data_dir"] == "/data/head"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

class TestTransferQueueClaim:

    def test_claim_moves_to_active(self, tmp_path):
        """claim() must atomically move the job from pending/ to active/."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])

        job = q.claim()
        assert job is not None
        assert job["run_name"] == "myrun.pffd"

        pending_dir = tmp_path / "tmp" / "transfer_queue" / "pending"
        active_dir = tmp_path / "tmp" / "transfer_queue" / "active"
        assert not (pending_dir / "myrun.pffd.job.toml").exists()
        assert (active_dir / "myrun.pffd.job.toml").exists()

    def test_claim_returns_none_when_empty(self, tmp_path):
        """claim() on an empty queue must return None."""
        q = _tq(tmp_path)
        result = q.claim()
        assert result is None

    def test_double_claim_is_none(self, tmp_path):
        """After claiming the only pending job, a second claim must return None."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        first = q.claim()
        assert first is not None
        second = q.claim()
        assert second is None


# ---------------------------------------------------------------------------
# Complete / Fail
# ---------------------------------------------------------------------------

class TestTransferQueueComplete:

    def test_complete_moves_to_completed(self, tmp_path):
        """complete() must move the job from active/ to completed/."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        q.claim()
        q.complete("myrun.pffd")

        active_dir = tmp_path / "tmp" / "transfer_queue" / "active"
        completed_dir = tmp_path / "tmp" / "transfer_queue" / "completed"
        assert not (active_dir / "myrun.pffd.job.toml").exists()
        assert (completed_dir / "myrun.pffd.job.toml").exists()

    def test_fail_moves_to_failed(self, tmp_path):
        """fail() must move the job from active/ to failed/."""
        q = _tq(tmp_path)
        q.enqueue("myrun.pffd", "/data", [{"ip": "1.2.3.4"}])
        q.claim()
        q.fail("myrun.pffd")

        active_dir = tmp_path / "tmp" / "transfer_queue" / "active"
        failed_dir = tmp_path / "tmp" / "transfer_queue" / "failed"
        assert not (active_dir / "myrun.pffd.job.toml").exists()
        assert (failed_dir / "myrun.pffd.job.toml").exists()


# ---------------------------------------------------------------------------
# Edge cases: complete/fail on unclaimed jobs
# ---------------------------------------------------------------------------

class TestTransferQueueEdgeCases:

    def test_complete_unclaimed_job_raises_or_noop(self, tmp_path):
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
        completed_dir = tmp_path / "tmp" / "transfer_queue" / "completed"
        assert not (completed_dir / "nonexistent.pffd.job.toml").exists(), (
            "complete() must not create a completed entry for an unclaimed job"
        )

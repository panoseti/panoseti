# mypy: ignore-errors
"""
test_transfer_queue.py

Unit tests for control/transfer/queue.py (TransferQueue).

All tests use tmp_path and monkeypatch PSETI_TQ_DIR to isolate queue state.
"""
from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def _make_job(run_name: str = "test_run_001", **kwargs) -> TransferJob:
    """Construct a minimal valid TransferJob for testing."""
    defaults: dict = dict(
        run_name=run_name,
        head_data_dir="/data/head",
        head_node_username="panoseti",
        created_at=_NOW,
        attempts=0,
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="panoseti",
                data_dir="/data",
                module_ids=[0, 1],
            )
        ],
    )
    defaults.update(kwargs)
    return TransferJob(**defaults)


def _make_job_with_pf(run_name: str = "pf_run_001") -> TransferJob:
    """Construct a TransferJob with port-forwarding configured."""
    from control.utils.pydantic_config_models import PortForwarding

    return TransferJob(
        run_name=run_name,
        head_data_dir="/data/head",
        head_node_username="panoseti",
        created_at=_NOW,
        daq_nodes=[
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="panoseti",
                data_dir="/data",
                module_ids=[5],
                port_forwarding=PortForwarding(
                    status=True,
                    gw_ip="10.0.1.254",
                    ssh_port=2222,
                ),
            )
        ],
    )


@pytest.fixture()
def tq(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> TransferQueue:
    """Return a TransferQueue isolated to tmp_path via PSETI_TQ_DIR."""
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("PSETI_TQ_DIR", str(queue_dir))
    return TransferQueue()


# ---------------------------------------------------------------------------
# 1. enqueue() creates pending/{run_name}.job.toml
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_enqueue_creates_pending_file(self, tq: TransferQueue) -> None:
        """enqueue() must create a TOML file in pending/."""
        job = _make_job("run_001")
        result = tq.enqueue(job)
        assert result is True
        pending = tq._queue / "pending" / "run_001.job.toml"
        assert pending.exists()

    def test_enqueue_idempotent_returns_false_second_time(self, tq: TransferQueue) -> None:
        """Calling enqueue() twice for the same run must return False the second time."""
        job = _make_job("run_002")
        first = tq.enqueue(job)
        second = tq.enqueue(job)
        assert first is True
        assert second is False

    def test_enqueue_idempotent_only_one_file(self, tq: TransferQueue) -> None:
        """After two enqueue() calls, only one .job.toml should exist in pending/."""
        job = _make_job("run_003")
        tq.enqueue(job)
        tq.enqueue(job)
        pending_dir = tq._queue / "pending"
        toml_files = list(pending_dir.glob("*.job.toml"))
        assert len(toml_files) == 1


# ---------------------------------------------------------------------------
# 2. claim() moves pending->active, returns TransferJob
# ---------------------------------------------------------------------------

class TestClaim:
    def test_claim_moves_to_active(self, tq: TransferQueue) -> None:
        """claim() must atomically move the job from pending/ to active/."""
        job = _make_job("run_004")
        tq.enqueue(job)
        claimed = tq.claim()
        assert claimed is not None
        assert claimed.run_name == "run_004"
        assert not (tq._queue / "pending" / "run_004.job.toml").exists()
        assert (tq._queue / "active" / "run_004.job.toml").exists()

    def test_claim_returns_none_when_empty(self, tq: TransferQueue) -> None:
        """claim() on an empty queue must return None."""
        result = tq.claim()
        assert result is None

    def test_claim_returns_transferjob_instance(self, tq: TransferQueue) -> None:
        """claim() must return a TransferJob model instance."""
        tq.enqueue(_make_job("run_005"))
        claimed = tq.claim()
        assert isinstance(claimed, TransferJob)


# ---------------------------------------------------------------------------
# 3. complete() moves active->completed
# ---------------------------------------------------------------------------

class TestComplete:
    def test_complete_moves_to_completed(self, tq: TransferQueue) -> None:
        """complete() must move the job from active/ to completed/."""
        tq.enqueue(_make_job("run_006"))
        tq.claim()
        tq.complete("run_006")
        assert not (tq._queue / "active" / "run_006.job.toml").exists()
        assert (tq._queue / "completed" / "run_006.job.toml").exists()

    def test_complete_nonexistent_raises(self, tq: TransferQueue) -> None:
        """complete() on a non-existent active job must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            tq.complete("nonexistent_run")


# ---------------------------------------------------------------------------
# 4. fail() moves active->failed
# ---------------------------------------------------------------------------

class TestFail:
    def test_fail_moves_to_failed(self, tq: TransferQueue) -> None:
        """fail() must move the job from active/ to failed/."""
        tq.enqueue(_make_job("run_007"))
        tq.claim()
        tq.fail("run_007")
        assert not (tq._queue / "active" / "run_007.job.toml").exists()
        assert (tq._queue / "failed" / "run_007.job.toml").exists()

    def test_fail_nonexistent_raises(self, tq: TransferQueue) -> None:
        """fail() on a non-existent active job must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            tq.fail("nonexistent_run")


# ---------------------------------------------------------------------------
# 5. retry() moves failed->pending, resets attempts
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_moves_failed_to_pending(self, tq: TransferQueue) -> None:
        """retry() must move the job from failed/ to pending/."""
        job = _make_job("run_008", attempts=2)
        tq.enqueue(job)
        tq.claim()
        tq.fail("run_008")
        result = tq.retry("run_008")
        assert result is True
        assert not (tq._queue / "failed" / "run_008.job.toml").exists()
        assert (tq._queue / "pending" / "run_008.job.toml").exists()

    def test_retry_resets_attempts_to_zero(self, tq: TransferQueue) -> None:
        """retry() must reset the attempts counter to 0 in the job file."""
        job = _make_job("run_009", attempts=2)
        tq.enqueue(job)
        tq.claim()
        tq.fail("run_009")
        tq.retry("run_009")
        reclaimed = tq.claim()
        assert reclaimed is not None
        assert reclaimed.attempts == 0

    def test_retry_nonexistent_returns_false(self, tq: TransferQueue) -> None:
        """retry() on a non-existent failed job must return False."""
        result = tq.retry("no_such_run")
        assert result is False


# ---------------------------------------------------------------------------
# 6. list_jobs() returns correct run names
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_list_jobs_pending(self, tq: TransferQueue) -> None:
        """list_jobs('pending') must return all enqueued run names."""
        tq.enqueue(_make_job("alpha"))
        tq.enqueue(_make_job("beta"))
        names = tq.list_jobs("pending")
        assert sorted(names) == ["alpha", "beta"]

    def test_list_jobs_empty_bucket(self, tq: TransferQueue) -> None:
        """list_jobs() on an empty bucket must return an empty list."""
        assert tq.list_jobs("completed") == []

    def test_list_jobs_active_after_claim(self, tq: TransferQueue) -> None:
        """After claim(), the run name must appear in active, not pending."""
        tq.enqueue(_make_job("gamma"))
        tq.claim()
        assert tq.list_jobs("pending") == []
        assert tq.list_jobs("active") == ["gamma"]


# ---------------------------------------------------------------------------
# 7. TransferJob round-trip with port_forwarding
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_roundtrip_all_fields(self, tq: TransferQueue) -> None:
        """Enqueue a job, claim it, verify all fields match including port_forwarding."""
        original = _make_job_with_pf("pf_run_999")
        tq.enqueue(original)
        claimed = tq.claim()
        assert claimed is not None
        assert claimed.run_name == original.run_name
        assert claimed.head_data_dir == original.head_data_dir
        assert claimed.head_node_username == original.head_node_username
        assert len(claimed.daq_nodes) == 1
        node = claimed.daq_nodes[0]
        orig_node = original.daq_nodes[0]
        assert str(node.ip_addr) == str(orig_node.ip_addr)
        assert node.username == orig_node.username
        assert node.data_dir == orig_node.data_dir
        assert node.module_ids == orig_node.module_ids
        assert node.port_forwarding is not None
        assert node.port_forwarding.status is True
        assert str(node.port_forwarding.gw_ip) == str(orig_node.port_forwarding.gw_ip)  # type: ignore[union-attr]
        assert node.port_forwarding.ssh_port == 2222

    def test_roundtrip_no_portforwarding(self, tq: TransferQueue) -> None:
        """Jobs without port_forwarding survive a round-trip cleanly."""
        original = _make_job("plain_run")
        tq.enqueue(original)
        claimed = tq.claim()
        assert claimed is not None
        assert claimed.daq_nodes[0].port_forwarding is None

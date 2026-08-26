# mypy: ignore-errors
"""
test_transfer_queue.py

Unit tests for control/transfer/queue.py (TransferQueue).

All tests use tmp_path and monkeypatch PSETI_TQ_DIR to isolate queue state.
"""
from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from ipaddress import IPv4Address

import pytest

from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


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
                ip_addr=IPv4Address("192.168.0.10"),
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
                ip_addr=IPv4Address("192.168.0.10"),
                username="panoseti",
                data_dir="/data",
                module_ids=[5],
                port_forwarding=PortForwarding(
                    status=True,
                    gw_ip=IPv4Address("10.0.1.254"),
                    reboot_port=None,
                    cmd_port=None,
                    grpc_port=50051,
                    port=2222,
                ),
            )
        ],
    )


@pytest.fixture()
def tq(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> TransferQueue:
    """Return a TransferQueue isolated to tmp_path via PSETI_TQ_DIR."""
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    monkeypatch.setenv("PSETI_TQ_DIR", str(queue_dir))
    return TransferQueue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_enqueue_creates_pending_file(self, tq: TransferQueue) -> None:
        """enqueue() must create a TOML file in pending/."""
        job = _make_job("run_001")
        tq.enqueue(job)

        pending_file = tq._queue / "pending" / "run_001.job.toml"
        assert pending_file.exists()

    def test_enqueue_idempotent_returns_false_second_time(self, tq: TransferQueue) -> None:
        """Calling enqueue() twice for the same run must return False the second time."""
        job = _make_job("run_002")
        assert tq.enqueue(job) is True
        assert tq.enqueue(job) is False

    def test_enqueue_idempotent_only_one_file(self, tq: TransferQueue) -> None:
        """After two enqueue() calls, only one .job.toml should exist in pending/."""
        job = _make_job("run_003")
        tq.enqueue(job)
        tq.enqueue(job)

        files = list((tq._queue / "pending").glob("*.job.toml"))
        assert len(files) == 1


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
        """claim() returns None if no jobs are pending."""
        assert tq.claim() is None

    def test_claim_returns_transferjob_instance(self, tq: TransferQueue) -> None:
        """claim() must return a TransferJob model instance."""
        tq.enqueue(_make_job("run_005"))
        claimed = tq.claim()
        assert isinstance(claimed, TransferJob)


class TestComplete:
    def test_complete_moves_to_completed(self, tq: TransferQueue) -> None:
        """complete() must move the job from active/ to completed/."""
        tq.enqueue(_make_job("run_006"))
        job = tq.claim()
        assert job is not None

        tq.complete(job.run_name)
        assert not (tq._queue / "active" / "run_006.job.toml").exists()
        assert (tq._queue / "completed" / "run_006.job.toml").exists()


class TestFail:
    def test_fail_moves_to_failed(self, tq: TransferQueue) -> None:
        """fail() must move the job from active/ to failed/."""
        tq.enqueue(_make_job("run_007"))
        job = tq.claim()
        assert job is not None

        tq.fail(job.run_name)
        assert not (tq._queue / "active" / "run_007.job.toml").exists()
        assert (tq._queue / "failed" / "run_007.job.toml").exists()


class TestRetry:
    def test_retry_moves_failed_to_pending(self, tq: TransferQueue) -> None:
        """retry() must move the job from failed/ to pending/."""
        job = _make_job("run_008", attempts=2)
        tq.enqueue(job)
        job_active = tq.claim()
        assert job_active is not None
        tq.fail(job_active.run_name)

        assert tq.retry("run_008") is True
        assert not (tq._queue / "failed" / "run_008.job.toml").exists()
        assert (tq._queue / "pending" / "run_008.job.toml").exists()

    def test_retry_resets_attempts_to_zero(self, tq: TransferQueue) -> None:
        """retry() must reset the attempts counter to 0 in the job file."""
        job = _make_job("run_009", attempts=2)
        tq.enqueue(job)
        job_active = tq.claim()
        assert job_active is not None
        tq.fail(job_active.run_name)

        tq.retry("run_009")
        job_retried = tq.claim()
        assert job_retried is not None
        assert job_retried.attempts == 0

    def test_retry_nonexistent_returns_false(self, tq: TransferQueue) -> None:
        """retry() returns False if the run name is not in failed/."""
        assert tq.retry("no_such_run") is False


class TestClean:
    def test_clean_removes_pending(self, tq: TransferQueue) -> None:
        """clean() must remove the job file from pending/."""
        tq.enqueue(_make_job("run_010"))

        removed = tq.clean("run_010")
        assert removed is not None
        assert removed.run_name == "run_010"
        assert not (tq._queue / "pending" / "run_010.job.toml").exists()

    def test_clean_nonexistent_returns_none(self, tq: TransferQueue) -> None:
        """clean() returns None if the run name is not in pending/."""
        assert tq.clean("no_such_run") is None

    def test_clean_does_not_touch_active(self, tq: TransferQueue) -> None:
        """clean() must not remove a job that's already been claimed into active/."""
        tq.enqueue(_make_job("run_011"))
        job = tq.claim()
        assert job is not None

        assert tq.clean("run_011") is None
        assert (tq._queue / "active" / "run_011.job.toml").exists()


class TestListJobs:
    def test_list_jobs_pending(self, tq: TransferQueue) -> None:
        """list_jobs('pending') must return all enqueued run names."""
        tq.enqueue(_make_job("alpha"))
        tq.enqueue(_make_job("beta"))
        pending = tq.list_jobs("pending")
        assert set(pending) == {"alpha", "beta"}

    def test_list_jobs_active_after_claim(self, tq: TransferQueue) -> None:
        """After claim(), the run name must appear in active, not pending."""
        tq.enqueue(_make_job("gamma"))
        tq.claim()
        assert "gamma" not in tq.list_jobs("pending")
        assert "gamma" in tq.list_jobs("active")

    def test_list_jobs_invalid_status_raises(self, tq: TransferQueue) -> None:
        """list_jobs() must raise ValueError for unknown statuses."""
        with pytest.raises(ValueError):
            tq.list_jobs("not_a_status")  # type: ignore


class TestRoundTrip:
    def test_roundtrip_all_fields(self, tq: TransferQueue) -> None:
        """Enqueue a job, claim it, verify all fields match including port_forwarding."""
        original = _make_job_with_pf("pf_run_999")
        tq.enqueue(original)
        claimed = tq.claim()

        assert claimed is not None
        assert claimed.run_name == original.run_name
        assert claimed.head_node_username == original.head_node_username
        assert len(claimed.daq_nodes) == 1
        
        node = claimed.daq_nodes[0]
        assert node.ip_addr == original.daq_nodes[0].ip_addr
        assert node.port_forwarding is not None
        assert node.port_forwarding.status is True
        assert node.port_forwarding.gw_ip == original.daq_nodes[0].port_forwarding.gw_ip
        assert node.port_forwarding.port == 2222
        assert node.port_forwarding.grpc_port == 50051

    def test_roundtrip_no_portforwarding(self, tq: TransferQueue) -> None:
        """Jobs without port_forwarding survive a round-trip cleanly."""
        original = _make_job("plain_run")
        tq.enqueue(original)
        claimed = tq.claim()

        assert claimed is not None
        assert claimed.daq_nodes[0].port_forwarding is None
        assert claimed.daq_nodes[0].grpc_port is None

    def test_roundtrip_grpc_port_explicit(self, tq: TransferQueue) -> None:
        """A node's explicit grpc_port override survives serialize/deserialize."""
        original = _make_job(
            "grpc_port_run",
            daq_nodes=[
                TransferNodeSpec(
                    ip_addr=IPv4Address("192.168.0.10"),
                    username="panoseti",
                    data_dir="/data",
                    module_ids=[0, 1],
                    grpc_port=50099,
                )
            ],
        )
        tq.enqueue(original)
        claimed = tq.claim()

        assert claimed is not None
        assert claimed.daq_nodes[0].grpc_port == 50099

    def test_roundtrip_grpc_port_unset_stays_none(self, tq: TransferQueue) -> None:
        """An omitted grpc_port must round-trip as None, not the string 'None'."""
        original = _make_job("grpc_port_unset_run")
        tq.enqueue(original)
        claimed = tq.claim()

        assert claimed is not None
        assert claimed.daq_nodes[0].grpc_port is None

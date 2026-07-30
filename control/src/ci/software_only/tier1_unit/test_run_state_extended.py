# mypy: ignore-errors
"""
test_run_state_extended.py

Phase 2 RED tests for the extended RunStateLedger / NodeReceipt model
and the RunStateManager.transition() helper.

All tests in this file should FAIL on the current codebase and pass only
after Phase 2 is implemented.

Exception: test_legacy_status_still_loads is a backward-compat canary —
it must PASS on both the old and new codebase.
"""

from __future__ import annotations

from datetime import UTC, datetime
from ipaddress import IPv4Address

import pytest
from pydantic import ValidationError

from control.utils.pydantic_config_models import NodeReceipt, RunStateLedger, RunStatus
from control.utils.run_state import RunStateManager

VALID_RUN_STATUSES = [
    RunStatus.STARTING,
    RunStatus.ACTIVE,
    RunStatus.RECORDING_ENDED,
    RunStatus.MANIFEST_PENDING,
    RunStatus.MANIFEST_GENERATING,
    RunStatus.MANIFEST_READY,
    RunStatus.TRANSFER_PENDING,
    RunStatus.TRANSFERRING,
    RunStatus.TRANSFER_FAILED,
    RunStatus.VERIFYING,
    RunStatus.VERIFY_FAILED,
    RunStatus.CLEANUP_PENDING,
    RunStatus.CLEANING,
    RunStatus.ARCHIVED,
    RunStatus.COMPLETED,
    RunStatus.STOPPED_WITH_ERRORS,
]


# ===========================================================================
# Serialization Parity
# ===========================================================================

class TestRunStateLedgerExtendedStatuses:
    """Verifies that all new statuses are accepted by the ledger model."""

    @pytest.mark.parametrize("status", VALID_RUN_STATUSES)
    def test_new_status_construction(self, status: RunStatus) -> None:
        """Ledger must accept all Phase 2 lifecycle statuses."""
        ledger = RunStateLedger(
            run_name="test_run",
            status=status,
            start_time="2024-01-01T00:00:00Z",
        )
        assert ledger.status == status

    def test_legacy_status_still_loads(self) -> None:
        """Backward compatibility: statuses from Phase 1 (STARTING, ACTIVE) must still work."""
        # This ensures we didn't accidentally break existing logic during the refactor.
        l1 = RunStateLedger(run_name="r", status=RunStatus.STARTING, start_time="...")
        l2 = RunStateLedger(run_name="r", status=RunStatus.ACTIVE, start_time="...")
        assert l1.status == "STARTING"
        assert l2.status == "ACTIVE"


# ===========================================================================
# Extended NodeReceipt fields
# ===========================================================================

class TestNodeReceiptNewFields:
    """Tests for the new Phase 2 fields on NodeReceipt."""

    def test_node_receipt_new_fields(self) -> None:
        """All new NodeReceipt fields must be accepted in a single construction."""
        receipt = NodeReceipt(
            ip_addr=IPv4Address("1.2.3.4"),
            manifest_path="/data/module_1/run/manifest.blake3",
            manifest_bytes=1024,
            rsync_bytes_transferred=100_000,
            rsync_last_progress_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
            verify_ok=True,
            cleanup_ok=False,
        )
        assert receipt.manifest_path == "/data/module_1/run/manifest.blake3"
        assert receipt.manifest_bytes == 1024
        assert receipt.rsync_bytes_transferred == 100_000
        assert receipt.rsync_last_progress_at is not None
        assert receipt.verify_ok is True
        assert receipt.cleanup_ok is False

    def test_node_receipt_new_fields_default_none(self) -> None:
        """New fields must default to None."""
        receipt = NodeReceipt(ip_addr=IPv4Address("1.2.3.4"))
        assert receipt.manifest_path is None
        assert receipt.manifest_bytes is None
        assert receipt.rsync_bytes_transferred is None
        assert receipt.rsync_last_progress_at is None


# ===========================================================================
# State Transition Logic (RunStateManager)
# ===========================================================================

class TestRunStateTransitions:
    """Integration tests for the RunStateManager's transition logic."""

    @pytest.fixture
    def state_mgr(self, tmp_path) -> RunStateManager:
        mgr = RunStateManager()
        mgr.state_path = tmp_path / "ledger.toml"
        return mgr

    def test_transition_updates_node_receipt(self, state_mgr: RunStateManager) -> None:
        """transition() must be able to update specific node receipts by IP."""
        # 1. Start a run
        initial = RunStateLedger(
            run_name="r1",
            status=RunStatus.STARTING,
            start_time="...",
            nodes=[NodeReceipt(ip_addr=IPv4Address("10.0.0.1"))]
        )
        state_mgr.save_state(initial)

        # 2. Update manifest path for node 10.0.0.1
        state_mgr.transition(
            RunStatus.MANIFEST_READY,
            node_ip="10.0.0.1",
            manifest_path="/remote/path.txt",
            manifest_bytes=500
        )

        # 3. Verify
        updated = state_mgr.load_state()
        assert updated.status == RunStatus.MANIFEST_READY
        assert len(updated.nodes) == 1
        assert str(updated.nodes[0].ip_addr) == "10.0.0.1"
        assert updated.nodes[0].manifest_path == "/remote/path.txt"
        assert updated.nodes[0].manifest_bytes == 500

    def test_transition_to_verify_failed_preserves_errors(self, state_mgr: RunStateManager) -> None:
        """VERIFY_FAILED transition should record the error message."""
        initial = RunStateLedger(run_name="r1", status=RunStatus.VERIFYING, start_time="...")
        state_mgr.save_state(initial)

        state_mgr.transition(RunStatus.VERIFY_FAILED, last_transfer_error="Hash mismatch on file X")

        updated = state_mgr.load_state()
        assert updated.status == RunStatus.VERIFY_FAILED
        assert updated.last_transfer_error == "Hash mismatch on file X"

    def test_transition_to_archived_clears_active_state(self, state_mgr: RunStateManager) -> None:
        """ARCHIVED is the terminal success state."""
        initial = RunStateLedger(run_name="r1", status=RunStatus.CLEANING, start_time="...")
        state_mgr.save_state(initial)

        state_mgr.transition(RunStatus.ARCHIVED)

        updated = state_mgr.load_state()
        assert updated.status == RunStatus.ARCHIVED


# ===========================================================================
# Validation Guards
# ===========================================================================

def test_invalid_status_raises_error() -> None:
    """Ledger must reject undefined status strings."""
    with pytest.raises(ValidationError):
        RunStateLedger(run_name="r", status="NOT_A_STATUS", start_time="...")

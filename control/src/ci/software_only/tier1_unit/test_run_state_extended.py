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

import pytest
from pydantic import ValidationError

from control.utils.pydantic_config_models import NodeReceipt, RunStateLedger, RunStatus
from control.utils.run_state import RunStateManager

VALID_RUN_STATUSES = [
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
]


class TestRunStateLedgerExtendedStatuses:
    """Tests for the new Phase 2 status values on RunStateLedger."""

    def test_new_status_recording_ended(self) -> None:
        """RECORDING_ENDED must be accepted by the validator."""
        ledger = RunStateLedger(
            run_name="r",
            status=RunStatus.RECORDING_ENDED,
            start_time="2024-01-01T00:00:00",
        )
        assert ledger.status == RunStatus.RECORDING_ENDED

    @pytest.mark.parametrize("status", VALID_RUN_STATUSES)
    def test_new_statuses_accepted(self, status) -> None:
        """All new Phase 2 statuses must be accepted."""
        ledger = RunStateLedger(
            run_name="r",
            status=status,
            start_time="2024-01-01T00:00:00",
        )
        assert ledger.status == status

    def test_legacy_status_still_loads(self) -> None:
        """COMPLETED must still be accepted (backward compatibility canary)."""
        ledger = RunStateLedger(
            run_name="r",
            status=RunStatus.COMPLETED,
            start_time="2024-01-01T00:00:00",
        )
        assert ledger.status == RunStatus.COMPLETED

    def test_unknown_status_rejected(self) -> None:
        """Arbitrary strings must still be rejected."""
        with pytest.raises(ValidationError):
            RunStateLedger(
                run_name="r",
                status="MADE_UP_STATUS",  # type: ignore[arg-type]
                start_time="2024-01-01T00:00:00",
            )


# ---------------------------------------------------------------------------
# New RunStateLedger fields
# ---------------------------------------------------------------------------

class TestRunStateLedgerNewFields:
    """Tests for the four new Phase 2 fields on RunStateLedger."""

    def _base(self) -> dict:
        return {"run_name": "r", "start_time": "2024-01-01T00:00:00"}

    def test_new_ledger_fields_transfer_attempts(self) -> None:
        """transfer_attempts must be accepted and default to 0."""
        ledger = RunStateLedger(**self._base(), transfer_attempts=3)
        assert ledger.transfer_attempts == 3

    def test_transfer_attempts_default_is_zero(self) -> None:
        """Default value for transfer_attempts must be 0."""
        ledger = RunStateLedger(**self._base())
        assert ledger.transfer_attempts == 0

    def test_new_ledger_fields_last_transfer_error(self) -> None:
        """last_transfer_error must be accepted as a string."""
        ledger = RunStateLedger(**self._base(), last_transfer_error="rsync failed")
        assert ledger.last_transfer_error == "rsync failed"

    def test_last_transfer_error_default_is_none(self) -> None:
        """Default value for last_transfer_error must be None."""
        ledger = RunStateLedger(**self._base())
        assert ledger.last_transfer_error is None

    def test_new_ledger_fields_manifest_algorithm(self) -> None:
        """manifest_algorithm must be accepted as a string."""
        ledger = RunStateLedger(**self._base(), manifest_algorithm="blake3")
        assert ledger.manifest_algorithm == "blake3"

    def test_manifest_algorithm_default_is_none(self) -> None:
        """Default value for manifest_algorithm must be None."""
        ledger = RunStateLedger(**self._base())
        assert ledger.manifest_algorithm is None

    def test_new_ledger_fields_next_action_not_before(self) -> None:
        """next_action_not_before must be accepted as a UTC datetime."""
        dt = datetime.now(UTC)
        ledger = RunStateLedger(**self._base(), next_action_not_before=dt)
        assert ledger.next_action_not_before == dt


# ---------------------------------------------------------------------------
# Extended NodeReceipt fields
# ---------------------------------------------------------------------------

class TestNodeReceiptNewFields:
    """Tests for the new Phase 2 fields on NodeReceipt."""

    def test_node_receipt_new_fields(self) -> None:
        """All new NodeReceipt fields must be accepted in a single construction."""
        receipt = NodeReceipt(
            ip_addr="1.2.3.4",
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
        receipt = NodeReceipt(ip_addr="1.2.3.4")
        assert receipt.manifest_path is None
        assert receipt.manifest_bytes is None
        assert receipt.rsync_bytes_transferred is None
        assert receipt.rsync_last_progress_at is None
        assert receipt.verify_ok is None
        assert receipt.cleanup_ok is None


# ---------------------------------------------------------------------------
# Save/load round-trip with new fields
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTripNewFields:
    """Tests that RunStateManager serialises and deserialises the new fields."""

    def test_save_load_round_trip_new_fields(self, tmp_path) -> None:
        """New fields must survive a save/load round-trip through run_state.toml."""
        mgr = RunStateManager(base_dir=str(tmp_path))
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        ledger = RunStateLedger(
            run_name="myrun",
            start_time="2024-01-01T00:00:00",
            status=RunStatus.RECORDING_ENDED,
            transfer_attempts=2,
            last_transfer_error="timeout",
            manifest_algorithm="blake3",
            next_action_not_before=dt,
        )
        mgr.save_state(ledger)

        loaded = mgr.load_state()
        assert loaded is not None
        assert loaded.status == RunStatus.RECORDING_ENDED
        assert loaded.transfer_attempts == 2
        assert loaded.last_transfer_error == "timeout"
        assert loaded.manifest_algorithm == "blake3"
        assert loaded.next_action_not_before == dt  # exact datetime equality


# ---------------------------------------------------------------------------
# RunStateManager.transition() helper
# ---------------------------------------------------------------------------

class TestTransitionHelper:
    """Tests for the new RunStateManager.transition() method."""

    def test_transition_helper_changes_status(self, tmp_path) -> None:
        """transition() must update the ledger status and return the new state."""
        mgr = RunStateManager(base_dir=str(tmp_path))
        # Pre-populate the state file with an ACTIVE ledger
        initial = RunStateLedger(
            run_name="myrun",
            start_time="2024-01-01T00:00:00",
            status=RunStatus.ACTIVE,
        )
        mgr.save_state(initial)

        result = mgr.transition(RunStatus.RECORDING_ENDED)
        assert result is not None
        assert result.status == RunStatus.RECORDING_ENDED

    def test_transition_helper_persists_to_disk(self, tmp_path) -> None:
        """transition() must save the updated ledger so load_state reflects it."""
        mgr = RunStateManager(base_dir=str(tmp_path))
        initial = RunStateLedger(
            run_name="myrun",
            start_time="2024-01-01T00:00:00",
            status=RunStatus.STOPPING,
        )
        mgr.save_state(initial)

        mgr.transition(RunStatus.RECORDING_ENDED)
        loaded = mgr.load_state()
        assert loaded is not None
        assert loaded.status == RunStatus.RECORDING_ENDED

    def test_transition_helper_accepts_extra_fields(self, tmp_path) -> None:
        """transition() must accept additional keyword fields to update."""
        mgr = RunStateManager(base_dir=str(tmp_path))
        initial = RunStateLedger(
            run_name="myrun",
            start_time="2024-01-01T00:00:00",
            status=RunStatus.ACTIVE,
        )
        mgr.save_state(initial)

        result = mgr.transition(RunStatus.TRANSFER_FAILED, transfer_attempts=1, last_transfer_error="rsync timeout")
        assert result is not None
        assert result.status == RunStatus.TRANSFER_FAILED
        assert result.transfer_attempts == 1
        assert result.last_transfer_error == "rsync timeout"

    def test_transition_helper_returns_none_when_no_state(self, tmp_path) -> None:
        """transition() on a missing state file must return None."""
        mgr = RunStateManager(base_dir=str(tmp_path))
        result = mgr.transition(RunStatus.RECORDING_ENDED)
        assert result is None

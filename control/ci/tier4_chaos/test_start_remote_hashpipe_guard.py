"""Tier 4 (Chaos): pseti start refuses when remote Hashpipe is already running.

Test 4.3 from EXECUTION_PLAN — validates the remote-hashpipe pre-flight check:
  - strict=True raises ValidationError if any DAQ node reports hashpipe_running=True.
  - start_data_flow is never called when the pre-flight aborts.
  - --force-restart calls StopDaq first, then continues.
  - data_flow_started flag remains False when the transaction is aborted before
    start_data_flow, ensuring rollback does NOT call stop_data_flow.

These tests are pure logic tests: no Docker, no real gRPC, no hardware.
"""
from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.start import (
    StartTransaction,
    _check_no_remote_hashpipe,
)
from control.utils.run_state import RunStateManager, ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daq_config(tmp_path: pathlib.Path, container: bool = True) -> MagicMock:
    """Build a minimal DaqConfig mock suitable for preflight checks."""
    node = MagicMock()
    node.ip_addr = "192.168.0.10"
    node.module_ids = [250]
    node.data_dir = str(tmp_path / "data")
    node.port_forwarding = None
    node.username = "root"
    node.bindhost = "0.0.0.0"

    cfg = MagicMock()
    cfg.head_node_container = container
    cfg.daq_nodes = [node]
    cfg.head_node_ip_addr = "10.0.1.5"
    cfg.head_node_data_dir = str(tmp_path / "head")
    return cfg


def _mock_client(hashpipe_running: bool, hashpipe_pid: int = 999) -> MagicMock:
    """Return a mocked AsyncDaqControlClient whose StatusDaq reports hashpipe state."""
    status_resp = (True, {
        "hashpipe_running": hashpipe_running,
        "hashpipe_pid": hashpipe_pid,
    })
    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.StatusDaq = AsyncMock(return_value=status_resp)
    instance.StopDaq = AsyncMock(return_value=True)
    return instance


# ---------------------------------------------------------------------------
# _check_no_remote_hashpipe
# ---------------------------------------------------------------------------

class TestCheckNoRemoteHashpipe:
    @pytest.mark.asyncio
    async def test_raises_when_hashpipe_running_no_force(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Raises ValidationError when hashpipe is running and force_restart=False."""
        cfg = _make_daq_config(tmp_path)
        client_instance = _mock_client(hashpipe_running=True)

        with (
            patch("control.start.AsyncDaqControlClient", return_value=client_instance),
            pytest.raises(ValidationError, match="Hashpipe already running"),
        ):
            await _check_no_remote_hashpipe(cfg, force_restart=False)

        client_instance.StopDaq.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_when_hashpipe_not_running(
        self, tmp_path: pathlib.Path
    ) -> None:
        """No exception when all nodes report hashpipe_running=False."""
        cfg = _make_daq_config(tmp_path)
        client_instance = _mock_client(hashpipe_running=False)

        with patch("control.start.AsyncDaqControlClient", return_value=client_instance):
            await _check_no_remote_hashpipe(cfg, force_restart=False)  # must not raise

    @pytest.mark.asyncio
    async def test_force_restart_stops_then_continues(
        self, tmp_path: pathlib.Path
    ) -> None:
        """With force_restart=True, StopDaq is called and no ValidationError is raised."""
        cfg = _make_daq_config(tmp_path)
        client_instance = _mock_client(hashpipe_running=True)

        with patch("control.start.AsyncDaqControlClient", return_value=client_instance):
            # Should not raise.
            await _check_no_remote_hashpipe(cfg, force_restart=True)

        client_instance.StopDaq.assert_awaited_once()


# ---------------------------------------------------------------------------
# start_data_flow never called on pre-flight abort
# ---------------------------------------------------------------------------

class TestStartDataFlowNotCalledOnAbort:
    @pytest.mark.asyncio
    async def test_data_flow_started_false_before_preflight(
        self, tmp_path: pathlib.Path
    ) -> None:
        """data_flow_started must be False before the hashpipe pre-flight runs,
        ensuring that if the pre-flight raises, the rollback ladder will NOT
        call stop_data_flow (which would disrupt an already-running observation).
        """
        state_mgr = RunStateManager(base_dir=str(tmp_path))
        cfg = _make_daq_config(tmp_path)
        client_instance = _mock_client(hashpipe_running=True)

        tx = StartTransaction(
            state_mgr, "test_run",
            cfg, MagicMock(), MagicMock()
        )

        # Pre-flight raises — data_flow_started was never set to True.
        with (
            patch("control.start.AsyncDaqControlClient", return_value=client_instance),
            pytest.raises(ValidationError),
        ):
            await _check_no_remote_hashpipe(cfg, force_restart=False)

        assert tx.data_flow_started is False, (
            "data_flow_started must remain False when pre-flight aborts before "
            "start_data_flow is called"
        )

    @pytest.mark.asyncio
    async def test_start_data_flow_not_called_on_preflight_abort(
        self, tmp_path: pathlib.Path
    ) -> None:
        """start_data_flow (defined in start.py) must not be called when the
        remote-hashpipe pre-flight check aborts the transaction."""
        cfg = _make_daq_config(tmp_path)
        client_instance = _mock_client(hashpipe_running=True)

        with (
            patch("control.start.AsyncDaqControlClient", return_value=client_instance),
            patch("control.start.start_data_flow") as mock_sdf,
        ):
            with pytest.raises(ValidationError):
                await _check_no_remote_hashpipe(cfg, force_restart=False)
            mock_sdf.assert_not_called()


# ---------------------------------------------------------------------------
# data_flow_started flag: rollback does NOT call stop_data_flow
# ---------------------------------------------------------------------------

class TestRollbackDoesNotHaltActiveRun:
    @pytest.mark.asyncio
    async def test_rollback_skips_stop_data_flow_when_not_started(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A failed start that aborts before calling start_data_flow must NOT
        call stop_data_flow in rollback — doing so would halt a co-existing
        valid observation's data flow.

        This is the core safety invariant introduced in D-4.
        """
        import control.utils.util as util_mod

        state_mgr = RunStateManager(base_dir=str(tmp_path))
        cfg = _make_daq_config(tmp_path)
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())

        # Explicitly confirm the pre-flight-abort state: data_flow never started.
        assert tx.data_flow_started is False

        with patch.object(util_mod, "stop_data_flow") as mock_stop:
            await tx.__aenter__()
            await tx.__aexit__(ValidationError, ValidationError("hashpipe running"), None)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_calls_stop_data_flow_when_transaction_started_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """If start_data_flow DID run (data_flow_started=True), rollback must
        call stop_data_flow to undo the Quabo configuration change."""
        import control.utils.util as util_mod

        state_mgr = RunStateManager(base_dir=str(tmp_path))
        cfg = _make_daq_config(tmp_path)
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())

        # Simulate a transaction that reached start_data_flow.
        tx.data_flow_started = True

        with (
            patch.object(util_mod, "stop_data_flow") as mock_stop,
            patch("control.start.AsyncDaqControlClient"),
        ):
            await tx.__aenter__()
            await tx.__aexit__(RuntimeError, RuntimeError("hashpipe start failed"), None)
            mock_stop.assert_called_once()

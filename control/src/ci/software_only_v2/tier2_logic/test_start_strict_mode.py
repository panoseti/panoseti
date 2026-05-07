# mypy: ignore-errors
"""
test_start_strict_mode.py — Strict-mode resolution and rollback gating.

Ported from ci/software_only/tier2_logic/test_start_strict_mode.py.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.start import (
    StartTransaction,
    _check_no_remote_hashpipe,
    _resolve_strict_mode,
)
from control.utils.run_state import RunStateManager, ValidationError
from ci.software_only_v2.infra.workspace import Workspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_daq_config(tmp_path: pathlib.Path, container: bool = True) -> MagicMock:
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


# ---------------------------------------------------------------------------
# _resolve_strict_mode
# ---------------------------------------------------------------------------

class TestResolveStrictMode:
    def test_explicit_true_overrides_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(True, cfg) is True

    def test_explicit_false_overrides_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PSETI_STRICT", "1")
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(False, cfg) is False

    def test_env_var_takes_precedence_over_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PSETI_STRICT", "0")
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(None, cfg) is False

    def test_sw_tier_container_defaults_lenient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.setenv("PSETI_TEST_TIER", "tier4_chaos")
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(None, cfg) is False

    def test_hw_sw_container_defaults_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(None, cfg) is True

    def test_bare_metal_always_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(None, cfg) is True


# ---------------------------------------------------------------------------
# _check_no_remote_hashpipe
# ---------------------------------------------------------------------------

class TestCheckNoRemoteHashpipe:
    @pytest.mark.asyncio
    async def test_raises_if_hashpipe_running(self, tmp_path: pathlib.Path) -> None:
        cfg = _make_minimal_daq_config(tmp_path)

        ok_resp = (True, {"hashpipe_running": True, "hashpipe_pid": 999})

        with patch("control.start.AsyncDaqControlClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.StatusDaq = AsyncMock(return_value=ok_resp)
            MockClient.return_value = instance

            with pytest.raises(ValidationError, match="Hashpipe already running"):
                await _check_no_remote_hashpipe(cfg, force_restart=False)

    @pytest.mark.asyncio
    async def test_passes_if_hashpipe_not_running(self, tmp_path: pathlib.Path) -> None:
        cfg = _make_minimal_daq_config(tmp_path)
        ok_resp = (True, {"hashpipe_running": False})

        with patch("control.start.AsyncDaqControlClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.StatusDaq = AsyncMock(return_value=ok_resp)
            MockClient.return_value = instance

            await _check_no_remote_hashpipe(cfg, force_restart=False)  # must not raise

    @pytest.mark.asyncio
    async def test_force_restart_calls_stopdaq(self, tmp_path: pathlib.Path) -> None:
        cfg = _make_minimal_daq_config(tmp_path)
        status_resp = (True, {"hashpipe_running": True, "hashpipe_pid": 42})
        stop_resp = True

        with patch("control.start.AsyncDaqControlClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.StatusDaq = AsyncMock(return_value=status_resp)
            instance.StopDaq = AsyncMock(return_value=stop_resp)
            MockClient.return_value = instance

            await _check_no_remote_hashpipe(cfg, force_restart=True)
            instance.StopDaq.assert_awaited_once()


# ---------------------------------------------------------------------------
# data_flow_started flag correctness
# ---------------------------------------------------------------------------

class TestDataFlowStartedFlag:
    def test_flag_starts_false(self, tmp_path: pathlib.Path) -> None:
        state_mgr = RunStateManager(base_dir=str(tmp_path))
        tx = StartTransaction(
            state_mgr, "test_run",
            _make_minimal_daq_config(tmp_path), MagicMock(), MagicMock()
        )
        assert tx.data_flow_started is False

    @pytest.mark.asyncio
    async def test_rollback_does_not_call_stop_data_flow_when_flag_false(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Rollback must NOT call stop_data_flow if data_flow_started is False.

        This guards against a failed start (pre-flight abort) halting data flow
        for a pre-existing valid observation on the same Quabos.
        """
        import control.utils.util as util_mod
        state_mgr = RunStateManager(base_dir=str(tmp_path))
        cfg = _make_minimal_daq_config(tmp_path)
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())
        tx.data_flow_started = False  # explicitly: pre-flight never started data flow

        with patch.object(util_mod, "stop_data_flow") as mock_stop:
            # Simulate rollback by calling __aexit__ with a ValidationError
            await tx.__aenter__()
            await tx.__aexit__(ValidationError, ValidationError("fail"), None)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_calls_stop_data_flow_when_flag_true(
        self, tmp_path: pathlib.Path
    ) -> None:
        import control.utils.util as util_mod
        state_mgr = RunStateManager(base_dir=str(tmp_path))
        cfg = _make_minimal_daq_config(tmp_path)
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())
        tx.data_flow_started = True  # this transaction called start_data_flow

        with (
            patch.object(util_mod, "stop_data_flow") as mock_stop,
            patch("control.start.AsyncDaqControlClient"),
        ):
            await tx.__aenter__()
            await tx.__aexit__(RuntimeError, RuntimeError("hashpipe failed"), None)
            mock_stop.assert_called_once()

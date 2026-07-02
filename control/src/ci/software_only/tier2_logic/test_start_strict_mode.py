"""
test_start_strict_mode.py — Strict-mode resolution and rollback gating.

Ported from ci/software_only/tier2_logic/test_start_strict_mode.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.infra.workspace import Workspace
from control.start import (
    StartTransaction,
    _check_no_remote_hashpipe,
    _resolve_strict_mode,
)
from control.utils.run_state import RunStateManager, ValidationError

# ---------------------------------------------------------------------------
# _resolve_strict_mode
# ---------------------------------------------------------------------------

class TestResolveStrictMode:
    def test_explicit_true_overrides_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """explicit strict=True overrides env vars and container flag."""
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(True, cfg) is True  # type: ignore[arg-type]

    def test_explicit_false_overrides_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """explicit strict=False overrides PSETI_STRICT env var."""
        monkeypatch.setenv("PSETI_STRICT", "1")
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(False, cfg) is False  # type: ignore[arg-type]

    def test_env_var_takes_precedence_over_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PSETI_STRICT=0 overrides the container-based default."""
        monkeypatch.setenv("PSETI_STRICT", "0")
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(None, cfg) is False  # type: ignore[arg-type]

    def test_sw_tier_container_defaults_lenient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PSETI_TEST_TIER=tier4_chaos with container=True → lenient mode."""
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.setenv("PSETI_TEST_TIER", "tier4_chaos")
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(None, cfg) is False  # type: ignore[arg-type]

    def test_hw_sw_container_defaults_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No tier env + container=True without explicit override → strict."""
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = True
        assert _resolve_strict_mode(None, cfg) is True  # type: ignore[arg-type]

    def test_bare_metal_always_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-container deployment is always strict."""
        monkeypatch.delenv("PSETI_STRICT", raising=False)
        monkeypatch.delenv("PSETI_TEST_TIER", raising=False)
        cfg = MagicMock()
        cfg.head_node_container = False
        assert _resolve_strict_mode(None, cfg) is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _check_no_remote_hashpipe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pseti_workspace",
    [FleetSpec.minimal_unit()],
    indirect=True,
)
class TestCheckNoRemoteHashpipe:
    @pytest.mark.asyncio
    async def test_raises_if_hashpipe_running(self, pseti_workspace: Workspace) -> None:
        """_check_no_remote_hashpipe raises ValidationError when hashpipe is active."""
        cfg = pseti_workspace.topology.daq
        ip = str(cfg.daq_nodes[0].ip_addr)

        from ci.fixtures.adapters.fake_adapters import FakeNetworkClient
        net_client = FakeNetworkClient(reachable_nodes=[ip])
        net_client.status_responses[ip] = {"hashpipe_running": True, "hashpipe_pid": 999}

        with pytest.raises(ValidationError, match="Hashpipe already running"):
            await _check_no_remote_hashpipe(cfg, net_client, force_restart=False)

    @pytest.mark.asyncio
    async def test_passes_if_hashpipe_not_running(self, pseti_workspace: Workspace) -> None:
        """_check_no_remote_hashpipe does not raise when hashpipe is idle."""
        cfg = pseti_workspace.topology.daq
        ip = str(cfg.daq_nodes[0].ip_addr)

        from ci.fixtures.adapters.fake_adapters import FakeNetworkClient
        net_client = FakeNetworkClient(reachable_nodes=[ip])
        net_client.status_responses[ip] = {"hashpipe_running": False}

        await _check_no_remote_hashpipe(cfg, net_client, force_restart=False)  # must not raise

    @pytest.mark.asyncio
    async def test_force_restart_calls_stopdaq(self, pseti_workspace: Workspace) -> None:
        """force_restart=True issues StopDaq when hashpipe is running."""
        cfg = pseti_workspace.topology.daq
        ip = str(cfg.daq_nodes[0].ip_addr)

        from ci.fixtures.adapters.fake_adapters import FakeNetworkClient
        net_client = FakeNetworkClient(reachable_nodes=[ip])
        net_client.status_responses[ip] = {"hashpipe_running": True, "hashpipe_pid": 42}

        await _check_no_remote_hashpipe(cfg, net_client, force_restart=True)
        assert net_client.stop_calls.get(ip, 0) == 1


# ---------------------------------------------------------------------------
# data_flow_started flag correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pseti_workspace",
    [FleetSpec.minimal_unit()],
    indirect=True,
)
class TestDataFlowStartedFlag:
    def test_flag_starts_false(self, pseti_workspace: Workspace) -> None:
        """data_flow_started defaults to False on a new StartTransaction."""
        cfg = pseti_workspace.topology.daq
        state_mgr = RunStateManager()
        tx = StartTransaction(
            state_mgr, "test_run",
            cfg, MagicMock(), MagicMock()
        )
        assert tx.data_flow_started is False

    @pytest.mark.asyncio
    async def test_rollback_does_not_call_stop_data_flow_when_flag_false(
        self, pseti_workspace: Workspace
    ) -> None:
        """Rollback must NOT call stop_data_flow if data_flow_started is False.

        Guards against a failed start (pre-flight abort) halting data flow
        for a pre-existing valid observation on the same Quabos.
        """
        import control.utils.util as util_mod
        cfg = pseti_workspace.topology.daq
        state_mgr = RunStateManager()
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())
        tx.data_flow_started = False

        with patch.object(util_mod, "stop_data_flow") as mock_stop:
            await tx.__aenter__()
            await tx.__aexit__(ValidationError, ValidationError("fail"), None)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_calls_stop_data_flow_when_flag_true(
        self, pseti_workspace: Workspace
    ) -> None:
        """Rollback calls stop_data_flow when this transaction started data flow."""
        import control.utils.util as util_mod
        cfg = pseti_workspace.topology.daq
        state_mgr = RunStateManager()
        tx = StartTransaction(state_mgr, "test_run", cfg, MagicMock(), MagicMock())
        tx.data_flow_started = True

        with (
            patch.object(util_mod, "stop_data_flow") as mock_stop,
            patch("control.start.AsyncDaqControlClient"),
        ):
            await tx.__aenter__()
            await tx.__aexit__(RuntimeError, RuntimeError("hashpipe failed"), None)
            mock_stop.assert_called_once()

"""
tier2_logic/test_stale_ledger_healing.py — Stale ledger self-healing tests.

Ported from ci/software_only/tier2_logic/test_stale_ledger_healing.py.
Verifies that start_run auto-archives STARTING ledgers from dead PIDs
and that ACTIVE ledgers correctly block new starts.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from control.start import start_run
from control.utils.pydantic_config_models import RunStateLedger, RunStatus
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
class TestStaleLedgerHealing:
    """start_run stale-ledger detection and auto-healing."""

    @pytest.mark.asyncio
    async def test_when_starting_ledger_has_dead_pid_then_healed_and_run_proceeds(
        self, pseti_workspace: Workspace
    ) -> None:
        """STARTING ledger with a dead PID is archived; start_run continues normally."""
        head_data_dir = str(pseti_workspace.root / "head_data")
        daq_cfg = pseti_workspace.topology.daq.model_copy(  # type: ignore[union-attr]
            update={"head_node_data_dir": head_data_dir}
        )
        obs_cfg = pseti_workspace.topology.obs
        quabo_uids = pseti_workspace.topology.quabo_uids
        data_cfg = pseti_workspace.topology.data
        net_cfg = pseti_workspace.topology.network

        state_mgr = RunStateManager()

        # Seed a STARTING ledger with a PID that is guaranteed not to be running.
        ledger = RunStateLedger(
            run_name="stale_run",
            status=RunStatus.STARTING,
            start_time="2024-01-01T00:00:00Z",
            pid=999999,
            host=socket.gethostname(),
        )
        state_mgr.save_state(ledger)

        with (
            patch("control.start.config_file.validate_all", return_value=True),
            patch("control.start.util.is_local", return_value=True),
            patch("control.start.util.is_hk_recorder_running", return_value=False),
            patch("control.start.ph_baseline_file_ok", return_value=True),
            patch("control.start.make_run_dirs"),
            patch("control.start.config_file.associate"),
            patch("control.start.config_file.show_daq_assignments"),
            patch("control.start.util.write_run_name"),
            patch("control.start._check_daq_reachability"),
            patch("control.start._check_quabo_reachability"),
            patch("control.start.start_data_flow"),
            patch("control.start.util.start_hk_recorder"),
            patch("control.start.AsyncDaqControlClient"),
        ):
            await start_run(
                obs_cfg,  # type: ignore[arg-type]
                daq_cfg,
                quabo_uids,  # type: ignore[arg-type]
                data_cfg,  # type: ignore[arg-type]
                net_cfg,  # type: ignore[arg-type]
                no_hv=True,
                no_redis=True,
                no_data=True,
            )

        # The stale_run ledger should be archived — no longer the active state.
        current = state_mgr.load_state()
        assert current is None or current.run_name != "stale_run"

    @pytest.mark.asyncio
    async def test_when_active_ledger_exists_then_start_blocked_regardless_of_pid(
        self, pseti_workspace: Workspace
    ) -> None:
        """ACTIVE ledger is NOT healed by a dead PID; start_run returns None."""
        head_data_dir = str(pseti_workspace.root / "head_data")
        daq_cfg = pseti_workspace.topology.daq.model_copy(  # type: ignore[union-attr]
            update={"head_node_data_dir": head_data_dir}
        )
        obs_cfg = pseti_workspace.topology.obs
        quabo_uids = pseti_workspace.topology.quabo_uids
        data_cfg = pseti_workspace.topology.data
        net_cfg = pseti_workspace.topology.network

        state_mgr = RunStateManager()

        ledger = RunStateLedger(
            run_name="active_run",
            status=RunStatus.ACTIVE,
            start_time="2024-01-01T00:00:00Z",
            pid=999999,
            host=socket.gethostname(),
        )
        state_mgr.save_state(ledger)

        with (
            patch("control.start.config_file.validate_all", return_value=True),
            patch("control.start.util.is_local", return_value=True),
            patch("control.start.util.is_hk_recorder_running", return_value=False),
            patch("control.start.ph_baseline_file_ok", return_value=True),
        ):
            result = await start_run(
                obs_cfg,  # type: ignore[arg-type]
                daq_cfg,
                quabo_uids,  # type: ignore[arg-type]
                data_cfg,  # type: ignore[arg-type]
                net_cfg,  # type: ignore[arg-type]
                no_hv=True,
                no_redis=True,
                no_data=True,
            )

        assert result is None

        # The ACTIVE ledger must be preserved — not archived.
        preserved = state_mgr.load_state()
        assert preserved is not None
        assert preserved.run_name == "active_run"
        assert preserved.status == RunStatus.ACTIVE

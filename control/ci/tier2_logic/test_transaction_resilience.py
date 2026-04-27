"""
Tier 2 (Logic): Transaction Resilience and Data Safety Tests.

Probes the depth of non-trivial transactional root causes:
1. Validation resilience for Palomar/UCB topologies.
2. Rollback resilience under Read-only file systems (EROFS).
3. Orphaned Hashpipe cleanup safety checks.
4. Fundamental failure isolation (skipping transfer on error).
"""
from __future__ import annotations

import errno
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.start import StartTransaction
from control.stop import stop_run
from control.utils.run_state import RunStateLedger, RunStateManager

# ── 1. Validation Resilience ──────────────────────────────────────────────────

def test_realistic_palomar_topology_passes_validation() -> None:
    """Ensure that the Palomar 4-telescope topology passes all strict checks."""
    from ci.paths import PanoPathsTest
    palomar_root = PanoPathsTest.base_dir() / "configs" / "palomar"
    
    from control.utils import config_file
    
    with patch("control.utils.config_file.get_obs_config") as m_obs, \
         patch("control.utils.config_file.get_daq_config") as m_daq, \
         patch("control.utils.config_file.get_network_config") as m_net, \
         patch("control.utils.config_file.get_quabo_uids") as m_uids, \
         patch("control.utils.config_file.get_data_config") as m_data, \
         patch("control.utils.config_file.get_daemons_config") as m_daemons, \
         patch("control.start.ph_baseline_file_ok", return_value=True):
         
         # Load real files
         obs_cfg = config_file.get_obs_config(dir=palomar_root / "obs_config.json.quad")
         m_obs.return_value = obs_cfg
         
         # daq_config quad only has 1 node in the reference, but we need it to handle all modules
         daq_cfg = config_file.get_daq_config(dir=palomar_root / "daq_config.json.quad")
         # Coherence: Assign modules from obs to daq
         mids = []
         for dome in obs_cfg.domes:
             for mod in dome.modules:
                 mids.append(config_file.ip_addr_to_module_id(str(mod.ip_addr)))
         daq_cfg.daq_nodes[0].module_ids = mids
         m_daq.return_value = daq_cfg
         
         m_net.return_value = config_file.get_network_config(dir=palomar_root / "network_config.json")
         m_uids.return_value = MagicMock() # UIDs validation is separate
         m_daemons.return_value = MagicMock()
         
         # Ensure data_config matches obs_config overvoltage
         data_cfg = config_file.get_data_config(dir=palomar_root / "data_config_palomar.json")
         data_cfg.detector_overvoltage = obs_cfg.detector_overvoltage
         m_data.return_value = data_cfg
         
         passed = config_file.validate_all(check_network=False)
         assert passed, "Palomar quad-telescope config failed strict validation!"


# ── 2. Rollback IO Resilience ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_rollback_continues_on_erofs_archival_failure(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that StartTransaction rollback ladder completes even if /data is read-only."""
    from control.utils.pydantic_config_models import DaqConfig, QuaboUids
    
    state_mgr = RunStateManager(base_dir=tmp_path)
    daq_cfg = DaqConfig(
        head_node_ip_addr="127.0.0.1",
        head_node_data_dir=str(tmp_path / "data"),
        daq_nodes=[]
    )
    uids = QuaboUids(domes=[])
    net = MagicMock()
    
    # Mock os.makedirs to simulate Read-only file system
    with patch("os.makedirs", side_effect=OSError(errno.EROFS, "Read-only file system")), \
         patch("control.utils.util.kill_hk_recorder"), \
         patch("control.utils.util.stop_data_flow") as m_stop_flow:
         
        tx = StartTransaction(state_mgr, "run1", daq_cfg, uids, net)
        tx.data_flow_started = True # Force Step 3 to execute
        
        # Trigger rollback via __aexit__
        await tx.__aexit__(RuntimeError, RuntimeError("boom"), None)
        
        # 1. Check for the specific EROFS warning
        assert "Failed to archive partial artifacts (non-fatal)" in caplog.text
        # 2. Verify subsequent ladder steps (Step 3) were NOT skipped
        m_stop_flow.assert_called_once()


# ── 3. Orphaned Hashpipe Safety ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cleanup_refused_on_uncertain_liveness_without_force() -> None:
    """Proves that CleanupData rejects deletion if PID status is uncertain (corrupted file)."""
    # Unit-test the Servicer method directly.
    from panoseti_grpc.daq_control.server import DaqControlServicer
    from panoseti_grpc.generated import daq_control_pb2
    
    servicer = DaqControlServicer(grpc_enabled=False)
    # Inject a non-integer garbage PID (simulating corrupted pid file state)
    servicer.hashpipe_pid = "GARBAGE" # type: ignore
    
    req = daq_control_pb2.CleanupDataRequest(
        data_dir="/tmp",
        run_dir="test",
        module_id=[1],
        force=False
    )
    
    context = MagicMock()
    context.abort = AsyncMock() # Must be awaitable
    
    resp = await servicer.CleanupData(req, context)
    assert resp.success is False
    assert "status uncertain" in resp.message
    
    # Now try with force=True
    req.force = True
    resp = await servicer.CleanupData(req, context)
    # Fails now on directory not found, but it PASSED the PID gate
    assert "status uncertain" not in resp.message


# ── 4. Fundamental Teardown Failure Isolation ─────────────────────────────────

@pytest.mark.asyncio
async def test_stop_run_bypasses_enqueue_on_fundamental_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that StopTransaction does NOT enqueue a transfer job if a crash occurs in the ladder."""
    from control.utils.pydantic_config_models import DaqConfig, NetworkConfig, QuaboUids
    
    # Use environment override to force all RunStateManagers to use tmp_path
    monkeypatch.setenv("PSETI_STATE", str(tmp_path))
    
    state_mgr = RunStateManager() # Now uses tmp_path via env
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks").mkdir(parents=True, exist_ok=True)
    
    daq_cfg = DaqConfig(
        head_node_ip_addr="127.0.0.1",
        head_node_data_dir=str(tmp_path / "data"),
        daq_nodes=[]
    )
    run_name = "myrun.pffd"
    (tmp_path / "data" / run_name).mkdir(parents=True, exist_ok=True)
    
    # Pre-write an ACTIVE ledger
    ledger = RunStateLedger(
        run_name=run_name,
        status="ACTIVE",
        start_time="2026-01-01T00:00:00Z"
    )
    state_mgr.save_state(ledger)
    
    with patch("control.stop.RunStateManager", return_value=state_mgr), \
         patch("control.stop.TransferQueue") as m_tq_cls, \
         patch("control.utils.util.local_ip", side_effect=RuntimeError("Ladder Crash")):
         
         mock_tq = m_tq_cls.return_value
         
         # stop_run returns success status
         success = await stop_run(
             daq_cfg, NetworkConfig(), QuaboUids(domes=[]),
             run=run_name
         )
         
         assert success is False
         # CRITICAL: Enqueue must NEVER be called if an exception hit the transaction
         mock_tq.enqueue.assert_not_called()
         
         # Ledger must reflect the failure
         updated_ledger = state_mgr.load_state()
         assert updated_ledger is not None
         assert updated_ledger.status == "STOPPED_WITH_ERRORS"
         assert "Ladder Crash" in (updated_ledger.last_transfer_error or "")

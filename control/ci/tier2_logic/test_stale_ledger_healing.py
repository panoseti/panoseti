import os
import pathlib
import signal
import socket
import pytest
from unittest.mock import MagicMock, patch

from control.start import start_run
from control.utils.run_state import RunStateManager, RunStateLedger, ValidationError
from control.utils.pydantic_config_models import DaqConfig, ObsConfig, QuaboUids, DataConfig, NetworkConfig

@pytest.fixture
def mock_configs():
    daq_cfg = DaqConfig(
        head_node_data_dir="/tmp",
        head_node_ip_addr="127.0.0.1",
        daq_nodes=[]
    )
    obs_cfg = ObsConfig(name="test", domes=[])
    quids = QuaboUids(domes=[])
    data_cfg = DataConfig(run_type="test")
    net_cfg = NetworkConfig(modules=[], daq_nodes=[])
    return obs_cfg, daq_cfg, quids, data_cfg, net_cfg

@pytest.mark.asyncio
async def test_stale_starting_ledger_heals_with_dead_pid(tmp_path, monkeypatch, mock_configs):
    """If ledger is STARTING and PID is dead, start_run should auto-archive and proceed."""
    obs_cfg, daq_cfg, quids, data_cfg, net_cfg = mock_configs
    
    state_mgr = RunStateManager()
    # Create a fake ledger with STARTING status and a dead PID
    # We use a very large PID that is unlikely to exist
    fake_pid = 999999
    ledger = RunStateLedger(
        run_name="stale_run",
        status="STARTING",
        start_time="2024-01-01T00:00:00Z",
        pid=fake_pid,
        host=socket.gethostname()
    )
    state_mgr.save_state(ledger)
    
    # Mock dependencies to reach the ledger check
    with patch("control.start.config_file.validate_all", return_value=True), \
         patch("control.start.util.is_local", return_value=True), \
         patch("control.start.util.is_hk_recorder_running", return_value=False), \
         patch("control.start.ph_baseline_file_ok", return_value=True), \
         patch("control.start.make_run_dirs"), \
         patch("control.start.config_file.associate"), \
         patch("control.start.config_file.show_daq_assignments"), \
         patch("control.start.util.write_run_name"):
        
        # This should NOT raise ValidationError because it heals the STARTING ledger
        await start_run(obs_cfg, daq_cfg, quids, data_cfg, net_cfg, no_hv=True, no_redis=True, no_data=True)
        
        # Verify it was archived (moved out of the main state file)
        assert state_mgr.load_state().run_name != "stale_run"

@pytest.mark.asyncio
async def test_active_ledger_blocks_even_with_dead_pid(tmp_path, monkeypatch, mock_configs):
    """If ledger is ACTIVE, it should NOT auto-heal even if the start-process PID is dead."""
    obs_cfg, daq_cfg, quids, data_cfg, net_cfg = mock_configs
    
    state_mgr = RunStateManager()
    # Create a fake ledger with ACTIVE status and a dead PID
    fake_pid = 999999
    ledger = RunStateLedger(
        run_name="active_run",
        status="ACTIVE",
        start_time="2024-01-01T00:00:00Z",
        pid=fake_pid,
        host=socket.gethostname()
    )
    state_mgr.save_state(ledger)
    
    with patch("control.start.config_file.validate_all", return_value=True), \
         patch("control.start.util.is_local", return_value=True), \
         patch("control.start.util.is_hk_recorder_running", return_value=False), \
         patch("control.start.ph_baseline_file_ok", return_value=True):
        
        # start_run returns None on validation failure (it suppresses the ValidationError internally)
        result = await start_run(obs_cfg, daq_cfg, quids, data_cfg, net_cfg, no_hv=True, no_redis=True, no_data=True)
        assert result is None
        
        # Verify the ledger is still there and still ACTIVE
        current_state = state_mgr.load_state()
        assert current_state is not None
        assert current_state.run_name == "active_run"
        assert current_state.status == "ACTIVE"


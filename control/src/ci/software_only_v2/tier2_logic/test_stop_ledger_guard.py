"""
test_stop_ledger_guard.py — Ledger-based stop validation.

Ported from ci/software_only/tier2_logic/test_stop_ledger_guard.py.
"""

from __future__ import annotations

import logging
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from control.stop import stop_run
from control.utils.run_state import RunStateManager, RunStatus


@pytest.fixture
def mock_state_mgr(tmp_path: pathlib.Path) -> MagicMock:
    mgr = MagicMock(spec=RunStateManager)
    mgr.state_path = tmp_path / "ledger.toml"
    return mgr

@pytest.fixture
def mock_daq_config(tmp_path: pathlib.Path) -> MagicMock:
    node = MagicMock()
    node.ip_addr = "127.0.0.1"
    node.module_ids = [1]
    cfg = MagicMock()
    cfg.daq_nodes = [node]
    cfg.head_node_ip_addr = "127.0.0.1"
    cfg.head_node_container = False
    return cfg

@pytest.mark.asyncio
async def test_stop_refuses_if_already_finished(mock_state_mgr: MagicMock, mock_daq_config: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    ledger = MagicMock()
    ledger.run_name = "r1"
    ledger.status = RunStatus.RECORDING_ENDED
    mock_state_mgr.load_state.return_value = ledger
    
    with patch("control.utils.util.is_local", return_value=True), \
         patch("control.utils.util.read_run_name", return_value="r1"), \
         patch("control.stop.RunStateManager", return_value=mock_state_mgr), \
         caplog.at_level(logging.WARNING):
        
        res = await stop_run(
            daq_config=mock_daq_config, 
            network_config=MagicMock(),
            quabo_uids=MagicMock(),
            run="r1", 
            force_cleanup=False
        )
        
        assert "is in 'RECORDING_ENDED'" in caplog.text
        assert res is True  # stop_run returns True when it completes (even if aborted by validation)

@pytest.mark.asyncio
async def test_stop_proceeds_with_force_cleanup(mock_state_mgr: MagicMock, mock_daq_config: MagicMock) -> None:
    ledger = MagicMock()
    ledger.run_name = "r1"
    ledger.status = RunStatus.RECORDING_ENDED
    mock_state_mgr.load_state.return_value = ledger
    
    with patch("control.utils.util.is_local", return_value=True), \
         patch("control.utils.util.read_run_name", return_value="r1"), \
         patch("control.stop.RunStateManager", return_value=mock_state_mgr), \
         patch("control.stop.StopTransaction") as mock_tx_cls:
        
        # We need mock_tx to be an async context manager
        mock_tx = mock_tx_cls.return_value
        mock_tx.__aenter__.return_value = mock_tx
        mock_tx.run = "r1"
        mock_tx.success = True
        
        res = await stop_run(
            daq_config=mock_daq_config, 
            network_config=MagicMock(),
            quabo_uids=MagicMock(),
            run="r1", 
            force_cleanup=True
        )
        assert res is True

@pytest.mark.asyncio
async def test_stop_proceeds_if_active(mock_state_mgr: MagicMock, mock_daq_config: MagicMock) -> None:
    ledger = MagicMock()
    ledger.run_name = "r1"
    ledger.status = RunStatus.ACTIVE
    mock_state_mgr.load_state.return_value = ledger
    
    with patch("control.utils.util.is_local", return_value=True), \
         patch("control.utils.util.read_run_name", return_value="r1"), \
         patch("control.stop.RunStateManager", return_value=mock_state_mgr), \
         patch("control.stop.StopTransaction") as mock_tx_cls:
        
        mock_tx = mock_tx_cls.return_value
        mock_tx.__aenter__.return_value = mock_tx
        mock_tx.run = "r1"
        mock_tx.success = True
        
        res = await stop_run(
            daq_config=mock_daq_config, 
            network_config=MagicMock(),
            quabo_uids=MagicMock(),
            run="r1", 
            force_cleanup=False
        )
        assert res is True

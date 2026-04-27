"""Tier 2 (Logic): ExceptionGroup unwrap tests for start.py.

Verifies:
- _check_quabo_reachability logs and summary-raises multiple failures.
- StartTransaction.__aexit__ formats ExceptionGroup tracebacks for console and JSON.
"""
from __future__ import annotations

import json
import logging
import pathlib
from unittest.mock import MagicMock, patch

import anyio
import pytest

from control.start import (
    StartTransaction,
    _check_quabo_reachability,
)
from control.utils.run_state import RunStateManager, ValidationError


@pytest.fixture
def mock_state_mgr(tmp_path: pathlib.Path) -> MagicMock:
    mgr = MagicMock(spec=RunStateManager)
    mgr.state_path = tmp_path / "ledger.toml"
    mgr.load_state.return_value = MagicMock(run_name="r1", nodes=[], status="STARTING")
    return mgr

@pytest.fixture
def mock_daq_config(tmp_path: pathlib.Path) -> MagicMock:
    node = MagicMock()
    node.ip_addr = "10.0.0.1"
    node.module_ids = [1]
    cfg = MagicMock()
    cfg.daq_nodes = [node]
    cfg.head_node_data_dir = str(tmp_path / "head")
    return cfg

@pytest.fixture
def mock_quabo_uids() -> MagicMock:
    module = MagicMock()
    module.ip_addr = "192.168.1.10"
    quabo = MagicMock()
    quabo.uid = "q1"
    module.quabos = [quabo, MagicMock(uid=''), MagicMock(uid=''), MagicMock(uid='')]
    dome = MagicMock()
    dome.modules = [module]
    uids = MagicMock()
    uids.domes = [dome]
    return uids

@pytest.mark.asyncio
async def test_check_quabo_reachability_unwraps_exceptions(mock_quabo_uids: MagicMock) -> None:
    # Add a second Quabo to trigger multiple failures
    module2 = MagicMock()
    module2.ip_addr = "192.168.1.11"
    quabo2 = MagicMock()
    quabo2.uid = "q2"
    module2.quabos = [quabo2, MagicMock(uid=''), MagicMock(uid=''), MagicMock(uid='')]
    mock_quabo_uids.domes[0].modules.append(module2)

    network_config = MagicMock()

    # Mock _check_reachability to fail for both
    with patch("control.utils.config_validator._check_reachability") as mock_reach:
        mock_reach.side_effect = [
            (False, "port closed"),
            (False, "UDP timeout")
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            await _check_quabo_reachability(mock_quabo_uids, network_config, lenient=False)
        
        msg = str(exc_info.value)
        assert "port closed" in msg
        assert "UDP timeout" in msg

@pytest.mark.asyncio
async def test_start_transaction_aexit_formats_exception_group(
    mock_state_mgr: MagicMock,
    mock_daq_config: MagicMock,
    mock_quabo_uids: MagicMock,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path
) -> None:
    network_config = MagicMock()
    
    # Create an ExceptionGroup manually
    try:
        raise ExceptionGroup("test group", [
            ValueError("sub1"),
            TypeError("sub2")
        ])
    except ExceptionGroup as eg:
        exc_val = eg
        exc_tb = eg.__traceback__

    tx = StartTransaction(mock_state_mgr, "r1", mock_daq_config, mock_quabo_uids, network_config)
    
    # We want to test __aexit__ behavior when an ExceptionGroup is passed in
    with caplog.at_level(logging.ERROR):
        await tx.__aexit__(type(exc_val), exc_val, exc_tb)
    
    # Check console logs
    assert any("sub1" in record.message for record in caplog.records)
    assert any("sub2" in record.message for record in caplog.records)
    assert any("Traceback (most recent call last):" in record.message for record in caplog.records)

    # Check JSON failure context
    aborted_dir = tmp_path / "head" / "_aborted" / "r1"
    context_file = aborted_dir / "start_failure_context.json"
    assert context_file.exists()
    
    data = json.loads(await anyio.Path(context_file).read_text())
    assert "sub1" in data["traceback"]
    assert "sub2" in data["traceback"]
    assert "Traceback (most recent call last):" in data["traceback"]
